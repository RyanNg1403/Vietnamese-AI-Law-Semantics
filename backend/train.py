# train_bert.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from nltk.corpus import semcor
import nltk
import json
from tqdm import tqdm
from config import Config

# --- 1. DATASET CLASS ---
class WSDDataset(Dataset):
    def __init__(self, tagged_sents, label2id, tokenizer, max_len):
        self.sents = tagged_sents
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        sent_data = self.sents[idx]
        tokens = []
        labels = []
        
        # SemCor xử lý từng token
        for token in sent_data:
            if isinstance(token, nltk.tree.Tree):
                # Từ được gán nhãn
                word = token[0]
                try:
                    synset_name = token.label().synset().name()
                    label_id = self.label2id.get(synset_name, -100) # Nếu không có trong map thì bỏ qua
                except:
                    label_id = -100
                tokens.append(word)
                labels.append(label_id)
            else:
                # Từ thông thường (không gán nhãn)
                tokens.append(token)
                labels.append(-100) # -100 để PyTorch bỏ qua tính loss
        
        # Tokenize và căn chỉnh nhãn (WordPiece Tokenization)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Căn chỉnh nhãn: Chỉ gán nhãn cho sub-word đầu tiên của từ
        doc_labels = []
        arr_offset = encoding.offset_mapping[0].tolist()
        word_ids = encoding.word_ids()
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None: # Special tokens [CLS], [SEP]
                doc_labels.append(-100)
            elif word_idx != word_ids[i-1]: # Token đầu tiên của từ
                doc_labels.append(labels[word_idx])
            else: # Các sub-word tiếp theo
                doc_labels.append(-100)
                
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(doc_labels, dtype=torch.long)
        }

# --- 2. MAIN TRAINING FLOW ---
def train():
    # Load Label Map
    with open(Config.LABEL_MAP_PATH, 'r') as f:
        label2id = json.load(f)
    
    # Load Data (Lấy mẫu 5000 câu để demo, bỏ [:5000] nếu muốn chạy full SemCor)
    print("Loading SemCor data...")
    semcor_data = semcor.tagged_sents(tag='sem')
    
    # Khởi tạo Tokenizer & Model
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    # num_labels bằng len(label2id) nhưng trừ đi 1 vì key 'O' ta set là -100
    model = BertForTokenClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=len(label2id) 
    )
    model.to(Config.DEVICE)
    
    # Tạo DataLoader
    train_dataset = WSDDataset(semcor_data, label2id, tokenizer, Config.MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training Loop
    print("Start Training...")
    model.train()
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader)}")
    
    # Lưu mô hình
    print(f"Saving model to {Config.MODEL_SAVE_PATH}...")
    model.save_pretrained(Config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()
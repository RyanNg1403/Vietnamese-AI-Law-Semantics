# train_bert.py
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForTokenClassification, get_linear_schedule_with_warmup
from nltk.corpus import semcor
import nltk
import json
import pandas as pd
import argparse
from tqdm import tqdm
from config import Config
from sklearn.metrics import accuracy_score, f1_score

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
                word = str(token[0])
                try:
                    synset_name = token.label().synset().name()
                    label_id = self.label2id.get(synset_name, -100) # Nếu không có trong map thì bỏ qua
                except:
                    label_id = -100
                tokens.append(word)
                labels.append(label_id)
            else:
                # Từ thông thường (không gán nhãn)
                tokens.append(str(token))
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
            elif i == 0 or word_idx != word_ids[i-1]: # Token đầu tiên của từ
                doc_labels.append(labels[word_idx])
            else: # Các sub-word tiếp theo
                doc_labels.append(-100)
                
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(doc_labels, dtype=torch.long)
        }

# --- 1B. CSV DATASET CLASS (for fine-tuning data) ---
class CSVDataset(Dataset):
    def __init__(self, csv_path, label2id, tokenizer, max_len):
        """
        Dataset for CSV files with columns: Sentence_ID, Token, POS, Selected_Synset
        """
        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Group by sentence
        self.sentences = []
        for sent_id, group in self.df.groupby('Sentence_ID'):
            tokens = group['Token'].tolist()
            synsets = group['Selected_Synset'].fillna('').tolist()  # Fill NaN with empty string
            self.sentences.append((tokens, synsets))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        tokens, synsets = self.sentences[idx]
        labels = []
        
        # Map synsets to label IDs
        for synset in synsets:
            if synset and synset != '':  # If synset exists
                label_id = self.label2id.get(synset, -100)
            else:  # Function word or no synset
                label_id = -100
            labels.append(label_id)
        
        # Tokenize and align labels (same as WSDDataset)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Align labels: Only label first sub-word of each word
        doc_labels = []
        word_ids = encoding.word_ids()
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:  # Special tokens [CLS], [SEP]
                doc_labels.append(-100)
            elif i == 0 or word_idx != word_ids[i-1]:  # First sub-word of word
                doc_labels.append(labels[word_idx])
            else:  # Subsequent sub-words
                doc_labels.append(-100)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(doc_labels, dtype=torch.long)
        }

# --- 2. EVALUATION FUNCTION ---
def evaluate_on_test_set(model, tokenizer, label2id):
    """Quick evaluation on the legal test set"""
    try:
        model.eval()
        df = pd.read_csv(Config.LEGAL_TEST_FILE)
        df = df.dropna(subset=['Selected_Synset'])
        
        id2label = {v: k for k, v in label2id.items() if v != -100}
        y_true = []
        y_pred = []
        
        grouped = df.groupby('Sentence_ID')
        
        for sent_id, group in grouped:
            words = group['Token'].tolist()
            sentence_str = " ".join(words)
            
            encoding = tokenizer(
                sentence_str, 
                return_tensors="pt", 
                truncation=True, 
                max_length=Config.MAX_LEN
            )
            inputs = {k: v.to(Config.DEVICE) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            word_ids = encoding.word_ids(0)
            pred_labels_sent = predictions[0].tolist()
            
            current_word_idx = 0
            for idx, row in group.iterrows():
                gold_synset = row['Selected_Synset']
                
                found_pred = "O"
                for i, wid in enumerate(word_ids):
                    if wid == current_word_idx:
                        pred_id = pred_labels_sent[i]
                        if pred_id in id2label:
                            found_pred = id2label[pred_id]
                        break
                
                y_true.append(gold_synset)
                y_pred.append(found_pred)
                current_word_idx += 1
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        model.train()  # Set back to training mode
        return accuracy, f1_macro, f1_weighted
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        model.train()
        return None, None, None

# --- 3. MAIN TRAINING FLOW ---
def train(resume=False):
    # Load Label Map
    with open(Config.LABEL_MAP_PATH, 'r') as f:
        label2id = json.load(f)
    
    # Load Data (Lấy mẫu 5000 câu để demo, bỏ [:5000] nếu muốn chạy full SemCor)
    print("Loading SemCor data...")
    semcor_data = semcor.tagged_sents(tag='sem')
    print(f"Loaded {len(semcor_data)} sentences")
    
    # Khởi tạo Tokenizer & Model - Check for existing checkpoint
    import os
    checkpoint_exists = os.path.exists(Config.MODEL_SAVE_PATH) and os.path.exists(os.path.join(Config.MODEL_SAVE_PATH, 'config.json'))
    
    if resume and checkpoint_exists:
        print(f"\n🔄 Resuming from checkpoint at {Config.MODEL_SAVE_PATH}")
        print("Loading model from checkpoint to continue training...")
        tokenizer = BertTokenizerFast.from_pretrained(Config.MODEL_SAVE_PATH)
        model = BertForTokenClassification.from_pretrained(Config.MODEL_SAVE_PATH)
        print("✅ Successfully loaded checkpoint!\n")
    elif resume and not checkpoint_exists:
        print(f"\n⚠️  Resume requested but no checkpoint found at {Config.MODEL_SAVE_PATH}")
        print(f"Starting fresh training from {Config.MODEL_NAME}...\n")
        tokenizer = BertTokenizerFast.from_pretrained(Config.MODEL_NAME)
        model = BertForTokenClassification.from_pretrained(
            Config.MODEL_NAME, 
            num_labels=len(label2id) - 1
        )
    else:
        print(f"\n🆕 Starting fresh training from {Config.MODEL_NAME}...")
        tokenizer = BertTokenizerFast.from_pretrained(Config.MODEL_NAME)
        model = BertForTokenClassification.from_pretrained(
            Config.MODEL_NAME, 
            num_labels=len(label2id) - 1
        )
        print("✅ Model initialized!\n")
    
    model.to(Config.DEVICE)
    
    # Tạo DataLoader - Combine SemCor + Fine-tuning data
    print("📚 Loading training datasets...")
    semcor_dataset = WSDDataset(semcor_data, label2id, tokenizer, Config.MAX_LEN)
    print(f"  - SemCor: {len(semcor_dataset)} sentences")
    
    # Load fine-tuning CSV data if it exists
    import os
    if os.path.exists(Config.FINETUNE_TRAIN_FILE):
        csv_dataset = CSVDataset(Config.FINETUNE_TRAIN_FILE, label2id, tokenizer, Config.MAX_LEN)
        print(f"  - Fine-tuning CSV: {len(csv_dataset)} sentences")
        
        # CRITICAL FIX: Oversample small fine-tuning dataset to prevent dilution
        # Without this, legal data is overwhelmed by SemCor (743:1 ratio)
        csv_dataset_oversampled = ConcatDataset([csv_dataset] * Config.OVERSAMPLE_FACTOR)
        print(f"  - Oversampled CSV ({Config.OVERSAMPLE_FACTOR}x): {len(csv_dataset_oversampled)} examples")
        
        # Combine both datasets
        train_dataset = ConcatDataset([semcor_dataset, csv_dataset_oversampled])
        print(f"  - Combined total: {len(train_dataset)} sentences\n")
    else:
        print(f"  - Fine-tuning CSV not found, using only SemCor\n")
        train_dataset = semcor_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    
    # Training Loop
    print("Start Training...")
    model.train()
    
    global_batch_num = 0  # Track total batches across all epochs
    
    epoch_bar = tqdm(range(Config.EPOCHS), desc="Training Progress", position=0)
    for epoch in epoch_bar:
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False, position=1)
        
        for batch_idx, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            total_loss += loss.item()
            global_batch_num += 1
            
            loop.set_postfix(batch_loss=f"{loss.item():.4f}")
            
            # Evaluate every N batches
            if global_batch_num % Config.EVAL_EVERY_N_BATCHES == 0:
                tqdm.write(f"\n[Batch {global_batch_num}] Evaluating...")
                accuracy, f1_macro, f1_weighted = evaluate_on_test_set(model, tokenizer, label2id)
                if accuracy is not None:
                    tqdm.write(f"  Acc: {accuracy:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}\n")
            
        avg_loss = total_loss / len(train_loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        print(f"📊 Evaluating on test set...")
        accuracy, f1_macro, f1_weighted = evaluate_on_test_set(model, tokenizer, label2id)
        if accuracy is not None:
            print(f"   Accuracy: {accuracy:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")
        
        # Save checkpoint after each epoch
        print(f"💾 Saving checkpoint after epoch {epoch+1}...")
        model.save_pretrained(Config.MODEL_SAVE_PATH)
        tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
        print(f"✅ Checkpoint saved!")
    
    # Lưu mô hình cuối cùng
    print(f"\n🎉 Training completed!")
    print(f"Final model saved at: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT for Word Sense Disambiguation')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from existing checkpoint')
    args = parser.parse_args()
    
    train(resume=args.resume)
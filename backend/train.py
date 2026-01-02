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
import warnings
import re

# Suppress warnings
warnings.filterwarnings('ignore', message='.*Was asked to gather along dimension 0.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.*')

# --- CUSTOM COLLATE FUNCTION ---
def collate_fn(batch):
    """Custom collate function to handle pos_tags and word_ids"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # pos_tags and word_ids are lists, not tensors
    pos_tags = [item['pos_tags'] for item in batch]
    word_ids = [item['word_ids'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pos_tags': pos_tags,
        'word_ids': word_ids
    }

# --- POS CONSTRAINT HELPER ---
def get_pos_from_synset(synset_name):
    """Extract POS from synset name (e.g., 'developer.n.01' -> 'n')"""
    match = re.match(r'^.+\.([nvars])\.\d+$', synset_name)
    if match:
        return match.group(1)
    return None

def pos_tag_to_wn_pos(pos_tag):
    """Map NLTK POS tag to WordNet POS"""
    if pos_tag.startswith('N'):
        return 'n'
    elif pos_tag.startswith('V'):
        return 'v'
    elif pos_tag.startswith('J'):
        return 'a'
    elif pos_tag.startswith('R'):
        return 'r'
    return None

def create_pos_mask(label2id, pos_tags_per_token):
    """Create mask for valid synsets based on POS tags"""
    pos_mask = {}
    for synset, label_id in label2id.items():
        if label_id == -100:
            continue
        synset_pos = get_pos_from_synset(synset)
        pos_mask[synset] = synset_pos
    return pos_mask

def compute_pos_constraint_loss(logits, labels, label2id, batch_pos_tags, batch_word_ids, device):
    """
    Compute POS constraint loss: penalize predictions that don't match POS tag
    Args:
        logits: [batch_size, seq_len, num_labels]
        labels: [batch_size, seq_len]
        label2id: dict mapping synset -> label_id
        batch_pos_tags: list of lists, each inner list is pos_tags for one sample
        batch_word_ids: list of lists, each inner list is word_ids for one sample
    """
    if not Config.USE_POS_CONSTRAINT:
        return torch.tensor(0.0, device=device)
    
    batch_size, seq_len, num_labels = logits.shape
    pos_loss = torch.tensor(0.0, device=device)
    num_valid_tokens = 0
    
    # Create reverse mapping: label_id -> synset
    id2label = {v: k for k, v in label2id.items() if v != -100}
    
    for batch_idx in range(batch_size):
        pos_tags = batch_pos_tags[batch_idx] if batch_idx < len(batch_pos_tags) else []
        word_ids = batch_word_ids[batch_idx] if batch_idx < len(batch_word_ids) else []
        
        if not pos_tags or not word_ids:
            continue
        
        for token_idx in range(seq_len):
            if token_idx >= len(word_ids) or labels[batch_idx, token_idx] == -100:
                continue
            
            word_id = word_ids[token_idx]
            if word_id is None:
                continue
            
            # Get POS tag for this token
            if word_id < len(pos_tags):
                token_pos_tag = pos_tags[word_id]
                if not token_pos_tag:  # Empty POS tag
                    continue
                expected_wn_pos = pos_tag_to_wn_pos(token_pos_tag)
            else:
                continue
            
            if expected_wn_pos is None:
                continue
            
            # Get predicted label
            pred_label_id = torch.argmax(logits[batch_idx, token_idx]).item()
            if pred_label_id not in id2label:
                continue
            
            pred_synset = id2label[pred_label_id]
            pred_wn_pos = get_pos_from_synset(pred_synset)
            
            # Penalize if POS doesn't match
            if pred_wn_pos != expected_wn_pos:
                # Add penalty: negative log probability of correct POS synsets
                probs = torch.softmax(logits[batch_idx, token_idx], dim=0)
                pos_loss += -torch.log(probs[pred_label_id] + 1e-8)
                num_valid_tokens += 1
    
    if num_valid_tokens > 0:
        pos_loss = pos_loss / num_valid_tokens
    
    return pos_loss * Config.POS_CONSTRAINT_WEIGHT

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
                
        # For SemCor, we don't have POS tags, so return empty list
        doc_pos_tags = [''] * len(doc_labels)
                
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(doc_labels, dtype=torch.long),
            'pos_tags': doc_pos_tags,
            'word_ids': word_ids
        }

# --- 1B. CSV DATASET CLASS (for fine-tuning data) ---
class CSVDataset(Dataset):
    def __init__(self, csv_path, label2id, tokenizer, max_len):
        """
        Dataset for CSV files with columns: Sentence_ID, Token, POS, Selected_Synset
        Includes POS tags and compound word handling
        """
        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Group by sentence
        self.sentences = []
        for sent_id, group in self.df.groupby('Sentence_ID'):
            tokens = group['Token'].tolist()
            synsets = group['Selected_Synset'].fillna('').tolist()
            pos_tags = group['POS'].fillna('').tolist()  # Include POS tags
            
            # Handle compound words: replace hyphens with spaces for tokenization
            processed_tokens = []
            for token in tokens:
                # Replace hyphen/underscore with space for better tokenization
                if '-' in token or '_' in token:
                    processed_token = token.replace('-', ' ').replace('_', ' ')
                else:
                    processed_token = token
                processed_tokens.append(processed_token)
            
            self.sentences.append((processed_tokens, synsets, pos_tags))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        tokens, synsets, pos_tags = self.sentences[idx]
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
        doc_pos_tags = []  # Store POS tags aligned with word_ids
        word_ids = encoding.word_ids()
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:  # Special tokens [CLS], [SEP]
                doc_labels.append(-100)
                doc_pos_tags.append('')
            elif i == 0 or word_idx != word_ids[i-1]:  # First sub-word of word
                doc_labels.append(labels[word_idx])
                doc_pos_tags.append(pos_tags[word_idx] if word_idx < len(pos_tags) else '')
            else:  # Subsequent sub-words
                doc_labels.append(-100)
                doc_pos_tags.append('')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(doc_labels, dtype=torch.long),
            'pos_tags': doc_pos_tags,  # Include POS tags for constraint loss
            'word_ids': word_ids  # Include word_ids for POS alignment
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
    
    # Multi-GPU support
    if hasattr(Config, 'USE_MULTI_GPU') and Config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model, device_ids=Config.GPU_IDS)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # Mixed Precision Training setup
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda') if Config.USE_AMP else None
    
    # Learning rate scheduler with warmup
    # Adjust for gradient accumulation: scheduler steps less frequently
    num_training_steps = (len(train_loader) * Config.EPOCHS) // Config.GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    if Config.USE_AMP:
        print(f"🚀 Using Automatic Mixed Precision (FP16)")
    print(f"📊 Gradient accumulation: {Config.GRADIENT_ACCUMULATION_STEPS} steps (effective batch: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS})")
    if Config.USE_POS_CONSTRAINT:
        print(f"🎯 POS Constraint Loss enabled (weight: {Config.POS_CONSTRAINT_WEIGHT})")
    
    # Training Loop
    print("Start Training...")
    model.train()
    
    global_batch_num = 0  # Track total batches across all epochs
    
    # Track best metrics for model saving
    best_metrics = {
        'accuracy': 0.0,
        'f1_macro': 0.0,
        'f1_weighted': 0.0,
        'epoch': 0
    }
    
    epoch_bar = tqdm(range(Config.EPOCHS), desc="Training Progress", position=0)
    for epoch in epoch_bar:
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False, position=1)
        
        for batch_idx, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)
            
            # Get POS tags and word_ids if available (for CSV dataset)
            batch_pos_tags = batch.get('pos_tags', None)
            batch_word_ids = batch.get('word_ids', None)
            
            # Mixed Precision forward pass
            with autocast('cuda', enabled=Config.USE_AMP):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Add POS constraint loss if enabled and POS tags available
                if Config.USE_POS_CONSTRAINT and batch_pos_tags is not None and batch_word_ids is not None:
                    pos_loss = compute_pos_constraint_loss(
                        outputs.logits, labels, label2id, 
                        batch_pos_tags, batch_word_ids,
                        Config.DEVICE
                    )
                    loss = loss + pos_loss
                
                # DataParallel returns loss per GPU, need to average
                if hasattr(Config, 'USE_MULTI_GPU') and Config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
                    loss = loss.mean()
                
                # Scale loss for gradient accumulation
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with gradient scaling
            if Config.USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Only update weights every GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                if Config.USE_AMP:
                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()  # Update learning rate
                global_batch_num += 1
            
            total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS  # Unscale for logging
            
            loop.set_postfix(batch_loss=f"{loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}")
            
            # Evaluate every N batches (only on gradient update steps)
            if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0 and global_batch_num % Config.EVAL_EVERY_N_BATCHES == 0:
                tqdm.write(f"\n[Batch {global_batch_num}] Evaluating...")
                accuracy, f1_macro, f1_weighted = evaluate_on_test_set(model, tokenizer, label2id)
                if accuracy is not None:
                    tqdm.write(f"  Acc: {accuracy:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}\n")
        
        # Handle any remaining accumulated gradients at end of epoch
        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS != 0:
            if Config.USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        print(f"📊 Evaluating on test set...")
        accuracy, f1_macro, f1_weighted = evaluate_on_test_set(model, tokenizer, label2id)
        if accuracy is not None:
            print(f"   Accuracy: {accuracy:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")
            
            # Check if this epoch is better than best (at least 2 out of 3 metrics improved)
            improvements = 0
            if accuracy > best_metrics['accuracy']:
                improvements += 1
            if f1_macro > best_metrics['f1_macro']:
                improvements += 1
            if f1_weighted > best_metrics['f1_weighted']:
                improvements += 1
            
            if improvements >= 2:
                print(f"🎯 New best model! ({improvements}/3 metrics improved)")
                best_metrics['accuracy'] = accuracy
                best_metrics['f1_macro'] = f1_macro
                best_metrics['f1_weighted'] = f1_weighted
                best_metrics['epoch'] = epoch + 1
                
                # Save the best model
                print(f"💾 Saving best model from epoch {epoch+1}...")
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(Config.MODEL_SAVE_PATH)
                tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
                print(f"✅ Best model saved!")
            else:
                print(f"⏭️  Not saving (only {improvements}/3 metrics improved)")
        else:
            print("⚠️  Skipping save due to evaluation failure")
    
    # Training summary
    print(f"\n🎉 Training completed!")
    print(f"Best model from epoch {best_metrics['epoch']}:")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  F1-Macro: {best_metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted: {best_metrics['f1_weighted']:.4f}")
    print(f"Model saved at: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT for Word Sense Disambiguation')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from existing checkpoint')
    args = parser.parse_args()
    
    train(resume=args.resume)
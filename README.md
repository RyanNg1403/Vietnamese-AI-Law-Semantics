# Vietnamese Legal Text - Word Sense Disambiguation Project

BERT-based Word Sense Disambiguation (WSD) system for disambiguating word meanings using WordNet synsets.

---

## 📁 Project Structure

```
vietnamese-legal-text/
├── backend/
│   ├── preprocess.py      # Data preparation (text → draft CSV)
│   ├── postproces.py      # Generate Prolog knowledge base
│   ├── data_prep.py       # Label map generation
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── config.py          # Configuration settings
├── data/
│   ├── selected_paragraph.txt          # Raw input text
│   ├── gold_standard_completed.csv     # Manually annotated test data
│   ├── label_map.json                  # Synset ID mapping
│   └── rules.pl / knowledge_base.pl    # Prolog knowledge bases
└── models/
    └── bert_wsd_model/     # Saved model checkpoints
```

---

## ✅ Phase 1: Data Preparation (Completed)

### 1.1 Preprocessing (`preprocess.py`)
Converts raw text into a draft CSV with WordNet synset candidates:
- **Input**: `selected_paragraph.txt` (raw text)
- **Process**: 
  - Tokenizes sentences using NLTK
  - POS tagging (N, V, J tags only)
  - Retrieves up to 5 WordNet synset candidates per token
- **Output**: `draft_data.csv` (for manual annotation)

### 1.2 Postprocessing (`postproces.py`)
Generates Prolog knowledge base from annotated data:
- **Input**: `gold_standard_completed.csv` (manually annotated)
- **Process**:
  - Validates synset IDs against WordNet
  - Creates Prolog facts: `is_concept/1`, `has_synset/2`
  - Extracts hypernym relationships: `sub_class/2`
- **Output**: `knowledge_base.pl` (logical inference rules)

### 1.3 Data Files
- `selected_paragraph.txt` - Source text
- `gold_standard_completed.csv` - 65 annotated tokens (gold standard)
- `label_map.json` - Maps WordNet synsets to label IDs for training

---

## 🔄 Phase 2: Training (Current)

### Training Pipeline

**Model**: `BertForTokenClassification` fine-tuned on SemCor corpus (37,176 sentences)

**Key Features**:
- ✅ Automatic checkpoint resuming
- ✅ Epoch-by-epoch evaluation on test set
- ✅ Progress tracking with nested tqdm bars
- ✅ MPS (Apple Silicon) / CUDA / CPU support

### How to Run

#### Before training, install all required packages first:
```
# Create virtual env
python3 -m venv .venv  
# Activate it
source .venv/bin/activate   
# Install packages
pip install -r requirements.txt
```
#### Start Fresh Training
```bash
python3 backend/train.py
```

#### Resume from Checkpoint
```bash
python3 backend/train.py --resume
```

#### Configuration
Edit `backend/config.py`:
```python
EPOCHS = 10              # Number of training epochs
BATCH_SIZE = 16          # Batch size (reduce if OOM)
LEARNING_RATE = 2e-5     # Learning rate
MAX_LEN = 128            # Max sequence length
```

### Training Output
Each epoch displays:
- Average training loss
- Test set metrics: Accuracy, F1-Macro, F1-Weighted
- Checkpoint saved automatically

**Current Results (3 epochs)**:
- Accuracy: 44.62%
- F1-Macro: 0.26
- F1-Weighted: 0.46

---

## 📊 Evaluation

Run standalone evaluation:
```bash
python3 backend/evaluate.py
```

**Output**:
- Per-class precision, recall, F1 scores
- Overall accuracy and weighted/macro averages
- `evaluation_result.csv` - Token-level predictions vs gold standard

---

## 🚀 Improvement Strategies

### 1. Use a Better Base Model
**Current**: `bert-base-uncased` (English general text)

**Suggested**:
- `bert-base-multilingual-cased` - Better cross-domain transfer
- `vinai/phobert-base` - Vietnamese-specific (if label mapping feasible)

**Why**: Domain/language mismatch between SemCor training data and test domain can be mitigated with multilingual or domain-adapted models.

**Implementation**:
```python
# In config.py
MODEL_NAME = 'bert-base-multilingual-cased'
```

### 2. Add In-Domain Fine-Tuning Data
**Problem**: SemCor is general English text; test data is AI/tech domain.

**Solution**: Manually annotate 200-500 in-domain sentences similar to your test set, then:
1. Fine-tune on SemCor first (as current)
2. Continue fine-tuning on in-domain data
3. Evaluate on test set

**Expected Impact**: +10-20% accuracy improvement, especially for minority synsets.

---

## 📝 Notes

- Training saves checkpoints after each epoch to `models/bert_wsd_model/`
- The model uses SemCor (English WSD corpus) for pre-training
- Test data contains 65 manually annotated tokens from AI/tech domain
- Label alignment handles WordPiece tokenization correctly (first subword only)

---

## 🔧 Requirements

```bash
pip install torch transformers nltk pandas scikit-learn tqdm
```

**NLTK Data**:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('semcor')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
```

---

## 📈 Progress Tracking

- [x] Data preprocessing pipeline
- [x] Prolog knowledge base generation  
- [x] Training script with checkpointing
- [x] Evaluation pipeline
- [x] Initial baseline (3 epochs, 44.62% accuracy)
- [ ] Extended training (10+ epochs)
- [ ] Model architecture improvements
- [ ] In-domain data collection

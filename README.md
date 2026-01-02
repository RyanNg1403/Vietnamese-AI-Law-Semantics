# Vietnamese Legal Text - Word Sense Disambiguation Project

BERT-based Word Sense Disambiguation (WSD) system for disambiguating word meanings in legal text using WordNet synsets.

## Setup

### 1. Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required on first run)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('semcor'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

### 2. Download Models

Download the `legal_bert_wsd_model-pos-constrained` folder from [Google Drive](https://drive.google.com/drive/folders/1wERZZXeZAGtloVdAB4pOINZugpdYiwFK?usp=sharing) and place it in the `models/` directory:

```
models/
└── legal_bert_wsd_model-pos-constrained/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── ...
```

This matches the model path configured in `backend/config.py` (`MODEL_SAVE_PATH = './models/legal_bert_wsd_model-pos-constrained'`).

## Running the Application

### Backend

Start the FastAPI server:

```bash
cd backend
uvicorn backend.query_api:app --host 0.0.0.0 --port 3030 --reload
```

The API will run on `http://localhost:3030`.

### Frontend

Open `frontend/query.html` in your web browser, or serve it using a local server:

```bash
cd frontend
python -m http.server 8000
# Then open http://localhost:8000/query.html
```

The frontend will connect to the backend API automatically.

## Project Structure

### Backend Files

- **`config.py`** - Configuration settings (model paths, hyperparameters, device settings)
- **`query_api.py`** - FastAPI server for query processing and WSD inference
- **`train.py`** - Model training script (fine-tunes BERT on SemCor corpus)
- **`evaluate.py`** - Model evaluation script (computes accuracy, F1 scores)
- **`preprocess.py`** - Converts raw text to CSV with WordNet synset candidates
- **`postprocces.py`** - Generates Prolog knowledge base from annotated data
- **`data_prep.py`** - Creates label map from SemCor corpus synsets
- **`generate_fintune_data.py`** - Extracts and annotates training data from PDF
- **`mfs_baseline.py`** - Most Frequent Sense baseline implementation
- **`prolog_queries.py`** - Python implementation of Prolog knowledge queries

### Frontend Files

- **`query.html`** - Main query interface with pipeline visualization
- **`query-script.js`** - Frontend logic for query processing
- **`query-style.css`** - Styling for query interface
- **`index.html`** - Redirects to query.html

### Data Files

- **`data/gold_standard_completed.csv`** - Manually annotated test data
- **`data/label_map.json`** - Synset ID to label mapping
- **`data/knowledge_base.pl`** - Prolog knowledge base
- **`data/rules.pl`** - Prolog inference rules

## Training

Train the model:

```bash
python backend/train.py          # Start fresh training
python backend/train.py --resume # Resume from checkpoint
```

Edit `backend/config.py` to adjust hyperparameters (epochs, batch size, learning rate, etc.).

## Evaluation

Evaluate the model on test data:

```bash
python backend/evaluate.py
```

Outputs metrics and saves results to `evaluation_result.csv`.

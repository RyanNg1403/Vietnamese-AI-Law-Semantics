# config.py
class Config:
    # Đường dẫn
    MODEL_NAME = 'google-bert/bert-base-uncased'
    MODEL_SAVE_PATH = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/models/bert_base_wsd_model-finetuned'
    LABEL_MAP_PATH = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/data/label_map.json'
    
    # File dữ liệu của bạn
    LEGAL_TEST_FILE = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/data/gold_standard_completed.csv'
    FINETUNE_TRAIN_FILE = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/data/legal_finetuning_train_cleaned.csv'
    
    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 32  # Giảm xuống 8 nếu tràn RAM
    EPOCHS = 5       # SemCor khá lớn, 3 epoch là đủ
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01      # L2 regularization for AdamW
    MAX_GRAD_NORM = 1.0      # Gradient clipping threshold
    WARMUP_RATIO = 0.1       # 10% of training for warmup
    EVAL_EVERY_N_BATCHES = 5 # Run evaluation every N batches
    OVERSAMPLE_FACTOR = 100  # Oversample fine-tuning data to prevent dilution
    
    # Thiết bị
    DEVICE = 'mps' # 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
# config.py
import torch

class Config:
    # Đường dẫn
    MODEL_NAME = 'bert-base-uncased'
    MODEL_SAVE_PATH = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/models/bert_wsd_model'
    LABEL_MAP_PATH = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/data/label_map.json'
    
    # File dữ liệu của bạn
    LEGAL_TEST_FILE = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/data/gold_standard_completed.csv'
    
    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 16  # Giảm xuống 8 nếu tràn RAM
    EPOCHS = 3       # SemCor khá lớn, 3 epoch là đủ
    LEARNING_RATE = 2e-5
    
    # Thiết bị
    DEVICE = 'mps' # 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
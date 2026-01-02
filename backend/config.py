# config.py
class Config:
    # Đường dẫn
    # Using Legal-BERT: nlpaueb/legal-bert-base-uncased (better for legal domain)
    # Fallback: nlpaueb/bert-base-uncased-eurlex if legal-bert not available
    MODEL_NAME = 'nlpaueb/legal-bert-base-uncased'  # Try this first
    # MODEL_NAME = 'nlpaueb/bert-base-uncased-eurlex'  # Fallback option
    MODEL_SAVE_PATH = './models/legal_bert_wsd_model-pos-constrained'
    LABEL_MAP_PATH = './data/label_map.json'
    
    # File dữ liệu của bạn
    LEGAL_TEST_FILE = './data/gold_standard_completed.csv'
    FINETUNE_TRAIN_FILE = './data/legal_finetuning_train_cleaned.csv'
    
    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 64  # Giảm xuống 8 nếu tràn RAM
    EPOCHS = 15       # SemCor khá lớn, 3 epoch là đủ
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01      # L2 regularization for AdamW
    MAX_GRAD_NORM = 0.05      # Gradient clipping threshold
    WARMUP_RATIO = 0.05       # 10% of training for warmup
    EVAL_EVERY_N_BATCHES = 20 # Run evaluation every N batches
    OVERSAMPLE_FACTOR = 150  # Oversample fine-tuning data to prevent dilution
    
    # POS Constraint Loss
    USE_POS_CONSTRAINT = True  # Enable POS constraint in loss
    POS_CONSTRAINT_WEIGHT = 0.3  # Weight for POS constraint loss (0.0-1.0)
    
    # Training optimizations
    USE_AMP = True  # Automatic Mixed Precision (FP16) - 2x faster, 40% less memory
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = BATCH_SIZE * 4 = 512
    
    # Thiết bị
    DEVICE = 'mps' # Primary device (will be cuda:0 after CUDA_VISIBLE_DEVICES remapping)
    USE_MULTI_GPU = True  # Set to False for single GPU
    GPU_IDS = [0, 1]      # GPU indices (after CUDA_VISIBLE_DEVICES remapping)
    
    print(f"Using device: {DEVICE}")
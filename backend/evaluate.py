# evaluate.py
import torch
import pandas as pd
import json
from transformers import BertTokenizerFast, BertForTokenClassification
from config import Config
from sklearn.metrics import classification_report, accuracy_score

def evaluate_legal_text():
    # 1. Load Resources
    print("Loading model and data...")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(Config.MODEL_SAVE_PATH)
        model = BertForTokenClassification.from_pretrained(Config.MODEL_SAVE_PATH)
    except:
        print("Error: Model not found. Please run train_bert.py first.")
        return

    model.to(Config.DEVICE)
    model.eval()
    
    with open(Config.LABEL_MAP_PATH, 'r') as f:
        label2id = json.load(f)
    # Tạo map ngược: ID -> Synset Name để in kết quả
    id2label = {v: k for k, v in label2id.items() if v != -100}

    # 2. Load file CSV của bạn
    df = pd.read_csv(Config.LEGAL_TEST_FILE)
    # Lọc bỏ các dòng không có Gold Standard (nếu có)
    df = df.dropna(subset=['Selected_Synset'])
    
    y_true = []
    y_pred = []
    
    print("\n--- Running Inference on Legal Text ---")
    
    # Gom nhóm theo câu để đưa vào BERT ngữ cảnh đầy đủ
    grouped = df.groupby('Sentence_ID')
    
    for sent_id, group in grouped:
        # Tái tạo lại câu văn từ các token
        words = group['Token'].tolist()
        sentence_str = " ".join(words)
        
        # Tokenize
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
            
        # Mapping lại từ token BERT về word gốc
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        word_ids = encoding.word_ids(0)
        
        # Lấy nhãn dự đoán cho các từ quan trọng (có trong CSV của bạn)
        # Lưu ý: Logic này tương đối phức tạp do việc tách từ khác nhau
        # Để đơn giản hóa cho đồ án: Ta duyệt qua từng dòng trong CSV và tìm từ tương ứng
        
        pred_labels_sent = predictions[0].tolist()
        
        # Duyệt qua từng từ trong group CSV
        current_word_idx = 0
        for idx, row in group.iterrows():
            target_token = row['Token']
            gold_synset = row['Selected_Synset']
            
            # Tìm vị trí của từ này trong chuỗi input của BERT
            # (Cách đơn giản: Lấy sub-word đầu tiên của word_id tương ứng)
            # Vì ta duyệt tuần tự words và word_ids tăng dần
            
            found_pred = "O"
            for i, wid in enumerate(word_ids):
                if wid == current_word_idx:
                    pred_id = pred_labels_sent[i]
                    if pred_id in id2label:
                        found_pred = id2label[pred_id]
                    break # Chỉ lấy sub-word đầu tiên
            
            y_true.append(gold_synset)
            y_pred.append(found_pred)
            
            current_word_idx += 1 # Sang từ tiếp theo trong câu

    # 3. Báo cáo kết quả
    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nChi tiết (Classification Report):")
    # Chỉ in report cho các label có xuất hiện để đỡ rối
    labels_present = list(set(y_true + y_pred))
    print(classification_report(y_true, y_pred, labels=labels_present, zero_division=0))
    
    # Xuất file kết quả để xem sai ở đâu
    df_result = pd.DataFrame({'Gold': y_true, 'Predicted': y_pred})
    df_result['Match'] = df_result['Gold'] == df_result['Predicted']
    df_result.to_csv('evaluation_result.csv', index=False)
    print("\nĐã lưu chi tiết dự đoán vào 'evaluation_result.csv'")

if __name__ == "__main__":
    evaluate_legal_text()
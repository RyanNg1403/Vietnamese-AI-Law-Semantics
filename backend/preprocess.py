import nltk
import csv
from nltk.corpus import wordnet as wn

# Đảm bảo đã tải dữ liệu cần thiết
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

input_file = './data/selected_paragraph.txt'
output_csv = './data/draft_data.csv'

data_rows = []

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()
    sentences = nltk.sent_tokenize(text)

    for sent_id, sent in enumerate(sentences, 1):
        words = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(words)

        for token, pos in pos_tags:
            # Xử lý compound words (machine-based, fine-tune, etc.)
            is_compound = '-' in token or '_' in token
            original_token = token
            
            # Nếu là compound word, thử tìm synset của từ gốc
            if is_compound:
                # Tách compound: "machine-based" -> ["machine", "based"]
                parts = token.replace('_', '-').split('-')
                # Ưu tiên tìm synset của từ đầu tiên (thường là từ chính)
                base_word = parts[0].lower()
            else:
                base_word = token.lower()
            
            # Chỉ quan tâm Danh từ (N), Động từ (V), Tính từ (J)
            if pos.startswith(('N', 'V', 'J')):
                # Tìm synsets: ưu tiên từ gốc cho compound words
                synsets = wn.synsets(base_word)
                if not synsets and is_compound:
                    # Fallback: thử tìm synset của toàn bộ compound
                synsets = wn.synsets(token)
                
                candidates = [f"{s.name()} ({s.definition()})" for s in synsets]
                
                row = {
                    'Sentence_ID': sent_id,
                    'Token': original_token,  # Giữ nguyên token gốc
                    'POS': pos,
                    'Base_Word': base_word if is_compound else '',  # Từ gốc cho compound
                    'Is_Compound': 'Yes' if is_compound else 'No',
                    'Candidates': " | ".join(candidates[:5]), # Lấy 5 nghĩa đầu tiên để gợi ý
                    'Selected_Synset': '', # ĐỂ TRỐNG CHO BẠN ĐIỀN
                    'FOL_Predicate': ''    # ĐỂ TRỐNG CHO BẠN ĐIỀN
                }
                data_rows.append(row)

# Lưu ra file CSV để bạn sửa thủ công
keys = data_rows[0].keys()
with open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data_rows)

print("Đã tạo file draft_data.csv. Hãy mở file này để làm bước thủ công!")


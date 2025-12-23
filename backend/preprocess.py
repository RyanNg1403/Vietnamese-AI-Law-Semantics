import nltk
import csv
from nltk.corpus import wordnet as wn

# Đảm bảo đã tải dữ liệu cần thiết
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

input_file = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/data/selected_paragraph.txt'
output_csv = '/Users/PhatNguyen/Desktop/vietnamese-legal-text/data/draft_data.csv'

data_rows = []

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()
    sentences = nltk.sent_tokenize(text)

    for sent_id, sent in enumerate(sentences, 1):
        words = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(words)

        for token, pos in pos_tags:
            # Chỉ quan tâm Danh từ (N), Động từ (V), Tính từ (J)
            if pos.startswith(('N', 'V', 'J')):
                # Tìm synsets gợi ý
                synsets = wn.synsets(token)
                candidates = [f"{s.name()} ({s.definition()})" for s in synsets]
                
                row = {
                    'Sentence_ID': sent_id,
                    'Token': token,
                    'POS': pos,
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
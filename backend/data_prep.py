# data_prep.py
import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
import json
import os

# Tải dữ liệu cần thiết
print("Downloading NLTK data...")
nltk.download('semcor')
nltk.download('wordnet')

def create_label_map():
    print("Reading SemCor corpus... (This may take a while)")
    # Lấy tất cả các câu đã được gán nhãn
    semcor_sents = semcor.tagged_sents(tag='sem')
    
    unique_synsets = set()
    
    # Duyệt qua ngữ liệu để thu thập toàn bộ các nhãn Synset xuất hiện
    for sent in semcor_sents:
        for token in sent:
            # Token trong SemCor có dạng Tree nếu được gán nhãn
            if isinstance(token, nltk.tree.Tree):
                lemma_obj = token.label() # Lemma object
                if isinstance(lemma_obj, nltk.corpus.reader.wordnet.Lemma):
                    synset = lemma_obj.synset()
                    unique_synsets.add(synset.name())
    
    # Tạo mapping: Synset Name -> ID
    # Thêm nhãn 'O' (Outside) cho các từ không được gán nhãn
    labels = sorted(list(unique_synsets))
    label2id = {label: i for i, label in enumerate(labels)}
    label2id['O'] = -100 # Quy ước bỏ qua loss cho từ không gán nhãn
    
    print(f"Total unique synsets found: {len(labels)}")
    
    # Lưu ra file JSON để dùng lại khi train và test
    with open('label_map.json', 'w') as f:
        json.dump(label2id, f)
    
    print("Label map saved to 'label_map.json'")

if __name__ == "__main__":
    create_label_map()
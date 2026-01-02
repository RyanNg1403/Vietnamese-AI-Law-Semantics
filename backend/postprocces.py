import pandas as pd
from nltk.corpus import wordnet as wn
import nltk

# Tải dữ liệu WordNet nếu chưa có
try:
    wn.get_version()
except:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_valid_synset(synset_id):
    """Kiểm tra xem Synset ID có tồn tại trong WordNet không"""
    try:
        return wn.synset(synset_id)
    except:
        return None

def sanitize_predicate(text):
    """Chuẩn hóa tên vị từ cho Prolog (chữ thường, không dấu cách)"""
    if pd.isna(text) or text == "":
        return None
    text = str(text).lower().strip()
    text = text.replace(" ", "_").replace("-", "_")
    return text

def process_data(input_file, output_file):
    print(f"--- Đang đọc file {input_file} ---")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    # Tập hợp các facts để tránh trùng lặp
    prolog_facts = set()
    prolog_hierarchy = set()
    
    # Kiểm tra lỗi và xử lý từng dòng
    print("\n--- Bắt đầu kiểm tra và sinh tri thức ---")
    for index, row in df.iterrows():
        token = row['Token']
        synset_id = row.get('Selected_Synset')
        predicate = sanitize_predicate(row.get('FOL_Predicate'))

        # Bỏ qua nếu không có predicate hoặc synset (ví dụ: giới từ)
        if not predicate or pd.isna(synset_id):
            continue

        # 1. KIỂM TRA LỖI SYNSET
        synset = get_valid_synset(synset_id)
        if not synset:
            print(f"[CẢNH BÁO] Dòng {index+2}: Synset ID '{synset_id}' (của từ '{token}') không tồn tại trong WordNet. Hãy kiểm tra lại!")
            continue

        # 2. TẠO FACT CƠ BẢN
        # Ví dụ: is_concept(developer).
        prolog_facts.add(f"is_concept({predicate}).")
        
        # Lưu mapping để tra cứu: has_synset(developer, 'developer.n.01').
        prolog_facts.add(f"has_synset({predicate}, '{synset_id}').")

        # 3. MỞ RỘNG TRI THỨC (Hypernyms)
        # Tìm các lớp cha. Ví dụ: developer -> person
        hypernyms = synset.hypernyms()
        for hyper in hypernyms:
            # Lấy tên lemma đầu tiên của cha làm predicate cho cha
            # Ví dụ: person.n.01 -> person
            parent_name = hyper.lemmas()[0].name().lower()
            
            # Tạo luật: sub_class(developer, person).
            prolog_hierarchy.add(f"sub_class({predicate}, {parent_name}).")
            
            # Đệ quy: Thêm cả cha của cha (nếu muốn mở rộng sâu hơn)
            # Ở đây ta chỉ lấy 1 cấp để demo cho gọn
            
    # 4. GHI RA FILE PROLOG
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% --- GENERATED KNOWLEDGE BASE FROM CSV ---\n")
        f.write("% Basic Concepts\n")
        for fact in sorted(prolog_facts):
            f.write(fact + "\n")
            
        f.write("\n% Automatic Knowledge Expansion (WordNet Hypernyms)\n")
        f.write("% Rule: sub_class(Child, Parent) means Child IS A Parent.\n")
        for relation in sorted(prolog_hierarchy):
            f.write(relation + "\n")
            
    print(f"\n--- Hoàn tất! Kết quả đã lưu vào '{output_file}' ---")

# --- CẤU HÌNH ---
INPUT_CSV = './data/gold_standard_completed.csv'  # Tên file bạn đã sửa
OUTPUT_PL = './data/knowledge_base.pl'            # Tên file Prolog đầu ra

if __name__ == "__main__":
    process_data(INPUT_CSV, OUTPUT_PL)
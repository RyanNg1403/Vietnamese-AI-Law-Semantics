# generate_finetune_data.py
import csv
import re
import os
import time
# Bạn cần cài: pip install pypdf nltk openai
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from openai import OpenAI

# --- CẤU HÌNH ---
PDF_PATH = 'data/CS229.pdf'
OUTPUT_CSV = 'data/legal_finetuning_train.csv'
KEYWORDS = ['artificial intelligence', 'system', 'developer', 'provider', 
            'deployer', 'risk', 'incident', 'data', 'human']

# Get API key from environment variable
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")

client = OpenAI(api_key=API_KEY) if API_KEY else None

def extract_text_from_pdf(pdf_path):
    print(f"--- Đang đọc file {pdf_path} ---")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def filter_important_sentences(text):
    print("--- Đang tách câu và lọc theo từ khóa ---")
    # Tải tokenizer nếu chưa có
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt')

    sentences = sent_tokenize(text)
    selected_sentences = []
    
    for sent in sentences:
        # Làm sạch cơ bản
        clean_sent = sent.replace('\n', ' ').strip()
        # Logic B: Chỉ lấy câu chứa từ khóa quan trọng và độ dài vừa phải
        if 10 < len(clean_sent.split()) < 50: # Bỏ câu quá ngắn/dài
            if any(k in clean_sent.lower() for k in KEYWORDS):
                selected_sentences.append(clean_sent)
    
    # Lấy khoảng 50 câu tốt nhất để làm mẫu
    return selected_sentences[:50]

def llm_labeling(sentence):
    """
    Use OpenAI API to label tokens with WordNet synsets in context.
    Falls back to NLTK MFS if API is not available.
    """
    if not client:
        print("⚠️  No API key, falling back to NLTK MFS labeling")
        return fallback_mfs_labeling(sentence)
    
    try:
        # Call OpenAI API
        prompt = f"""Analyze this sentence from a legal document and label each token with its most appropriate WordNet synset ID.

Sentence: "{sentence}"

CRITICAL INSTRUCTIONS:
1. Tokenize the sentence ensuring NO words are skipped - include ALL tokens
2. For each token (including punctuation and function words), provide:
   - Token (the exact word/punctuation)
   - POS tag (NN, VB, JJ, DT, IN, etc.)
   - WordNet synset ID for content words (nouns, verbs, adjectives, adverbs)
   - Empty string "" for function words, articles, prepositions, conjunctions, punctuation
3. Do NOT skip any tokens - the output must reconstruct the full sentence exactly
4. Output ONLY a valid CSV format with columns: Token,POS,Selected_Synset
5. NO explanations, just the CSV data

Example output for "Artificial intelligence is a system":
Artificial,JJ,artificial.a.01
intelligence,NN,intelligence.n.01
is,VBZ,be.v.02
a,DT,
system,NN,system.n.01"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper and faster than gpt-4
            messages=[
                {"role": "system", "content": "You are a linguistic expert specializing in WordNet sense disambiguation for legal texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Deterministic output
            max_tokens=500
        )
        
        # Parse response
        csv_output = response.choices[0].message.content.strip()
        return parse_llm_csv_output(csv_output, sentence)
        
    except Exception as e:
        print(f"⚠️  API error: {e}. Falling back to NLTK.")
        return fallback_mfs_labeling(sentence)

def parse_llm_csv_output(csv_text, original_sentence):
    """Parse the CSV output from LLM"""
    labeled_rows = []
    lines = csv_text.strip().split('\n')
    
    for line in lines:
        # Skip header or empty lines
        if not line.strip() or 'Token' in line:
            continue
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            labeled_rows.append({
                'Token': parts[0],
                'POS': parts[1],
                'Selected_Synset': parts[2],
                'Sentence': original_sentence
            })
    
    return labeled_rows if labeled_rows else fallback_mfs_labeling(original_sentence)

def fallback_mfs_labeling(sentence):
    """Fallback to NLTK MFS if API fails - keeps ALL tokens"""
    from nltk.corpus import wordnet as wn
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    labeled_rows = []
    
    for token, pos in pos_tags:
        synset_id = ""
        # Only label content words (N, V, J, R for adverbs)
        if pos.startswith(('N', 'V', 'J', 'R')):
            synsets = wn.synsets(token)
            if synsets:
                synset_id = synsets[0].name()
        # Function words and punctuation get empty synset (will be -100 in training)
        
        labeled_rows.append({
            'Token': token,
            'POS': pos,
            'Selected_Synset': synset_id,  # Empty for function words
            'Sentence': sentence
        })
    
    return labeled_rows

def main():
    # 1. Trích xuất
    raw_text = extract_text_from_pdf(PDF_PATH)
    
    # 2. Lọc câu (Logic B)
    important_sentences = filter_important_sentences(raw_text)
    print(f"-> Đã tìm thấy {len(important_sentences)} câu tiềm năng.")
    
    # 3. Gán nhãn tự động với OpenAI API
    print(f"--- {'Using OpenAI API' if client else 'Using NLTK MFS fallback'} for labeling ---")
    all_data = []
    
    for i, sent in enumerate(important_sentences, 1):
        print(f"Processing sentence {i}/{len(important_sentences)}...")
        
        rows = llm_labeling(sent)
        
        # Thêm Sentence_ID để khớp format cũ
        for row in rows:
            row['Sentence_ID'] = i + 100 # ID bắt đầu từ 100 để tránh trùng file test
            del row['Sentence'] # Không cần lưu lại câu gốc trong từng dòng
            all_data.append(row)
        
        # Rate limiting: sleep to avoid hitting API limits
        if client and i < len(important_sentences):
            time.sleep(0.5)  # Wait 0.5s between requests

    # 4. Lưu file
    keys = ['Sentence_ID', 'Token', 'POS', 'Selected_Synset']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        # Chỉ ghi các cột cần thiết
        for row in all_data:
            writer.writerow({k: row.get(k, '') for k in keys})
            
    print(f"\n✅ [DONE] Created file '{OUTPUT_CSV}' with {len(all_data)} labeled tokens.")
    if client:
        print("📝 Labels generated using OpenAI GPT-4o-mini (high quality)")
        print("💡 Tip: Manually verify 10-20% of labels for best results")
    else:
        print("⚠️  Labels generated using NLTK MFS (lower quality)")
        print("💡 Tip: Set OPENAI_API_KEY for better results or manually verify labels")

if __name__ == "__main__":
    main()
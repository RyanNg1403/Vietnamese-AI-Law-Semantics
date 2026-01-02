# NGỮ NGHĨA HỌC TÍNH TOÁN
## ĐỒ ÁN 2 - GÁN NHÃN NGHĨA CỦA TỪ
### Chủ đề: Legal Text Semantics - Ngữ nghĩa văn bản pháp luật AI

---

## NỘI DUNG TRÌNH BÀY

1. GIỚI THIỆU ĐỒ ÁN
2. HUẤN LUYỆN MÔ HÌNH
3. BIỂU DIỄN TRI THỨC
4. TRUY VẤN TRI THỨC

---

# 1. GIỚI THIỆU ĐỒ ÁN

## 1.1 Mục tiêu đồ án

| Mục tiêu | Mô tả |
|----------|-------|
| **Biểu diễn FOL** | Dịch văn bản pháp luật sang logic vị từ bậc một |
| **Gán nhãn nghĩa** | Tự động xác định nghĩa từ (WSD) bằng WordNet synsets |
| **Bổ sung tri thức** | Mở rộng knowledge base qua WordNet hypernyms |
| **Truy vấn** | Hỗ trợ suy luận và trả lời câu hỏi |

**Ứng dụng**: Hệ thống QA pháp luật, truy vấn ngữ nghĩa, kiểm tra tuân thủ AI Act

---

## 1.2 Pipeline tổng quan

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Đoạn văn   │───▶│  Gán nhãn   │───▶│  Biểu diễn  │───▶│  Truy vấn   │
│  (8 câu)    │    │  WSD        │    │  FOL        │    │  Prolog     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                 │                  │                  │
       ▼                 ▼                  ▼                  ▼
   80 tokens      BERT + MFS         125 facts          8 queries
```

---

## 1.3 Ví dụ End-to-End

**Câu gốc**: "Developer is an organization or individual that designs an AI system."

| Bước | Input | Output |
|------|-------|--------|
| 1. Tokenize | Câu tiếng Anh | `Developer`, `organization`, `designs`, `system` |
| 2. WSD | Tokens | `developer.n.01`, `organization.n.01`, `design.n.01`, `system.n.01` |
| 3. FOL | Synsets | `is_developer(X,Y) :- organization(X), designs(X,Y), ai_system(Y).` |
| 4. Bổ sung | WordNet | `sub_class(developer, creator).` |
| 5. Truy vấn | `is_developer(X, chat_gpt_vn)` | **X = bkav_corp** |

---

## 1.4 Đoạn văn được chọn (8 câu)

**Nguồn**: EU AI Act - Article 3 (Definitions)

| # | Nội dung |
|---|----------|
| 1 | Artificial intelligence is the electronic implementation of human intellectual capabilities. |
| 2 | Artificial intelligence system is a machine-based system designed to perform AI capabilities with varying levels of autonomy. |
| 3 | Developer is an organization or individual that designs, builds, trains, tests, or fine-tunes an AI system. |
| 4 | Provider is an organization or individual that brings an AI system to the market. |
| 5 | Deployer is an organization or individual that uses an AI system for professional purposes. |
| 6 | User is an organization or individual that directly interacts with an AI system. |
| 7 | Serious incident is an event occurring during the operation of an AI system that causes significant damage. |
| 8 | The damage may affect human life, health, property, or the environment. |

---

## 1.5 Kết quả cần đạt

| Yêu cầu | Kết quả |
|---------|---------|
| Xác định nhãn nghĩa tự động | ✅ 80 tokens được gán nhãn |
| Biểu diễn logic vị từ bậc một | ✅ 8 định nghĩa FOL (6 roles + 2 concepts) |
| Bổ sung tri thức từ WordNet | ✅ 36 quan hệ hypernym |
| Truy vấn tri thức | ✅ 8 câu truy vấn verified |

---

# 2. HUẤN LUYỆN MÔ HÌNH

## 2.1 Ngữ liệu

| Dataset | Mô tả | Số lượng |
|---------|-------|----------|
| **SemCor** | Corpus WSD chuẩn (Miller et al., 1993) | 37,176 câu |
| **Legal Fine-tuning** | Dữ liệu domain pháp luật | ~50 câu |
| **Gold Standard** | Tập đánh giá thủ công (8 câu) | 80 tokens |

---

## 2.2 Phương pháp phân lớp

### Baseline: Most Frequent Sense (MFS)
- **Nguyên lý**: Chọn nghĩa phổ biến nhất trong WordNet
- **Ưu điểm**: Đơn giản, không cần training
- **Nhược điểm**: Không xét ngữ cảnh

### Mô hình đề xuất: BERT Token Classification
- **Base Model**: `nlpaueb/bert-base-uncased-eurlex` (pre-trained trên legal corpus)
- **Lý do chọn**: Hiểu ngữ cảnh, phù hợp domain pháp luật

---

## 2.3 Tiền xử lý ngữ liệu

| Bước | Công cụ | Mô tả |
|------|---------|-------|
| 1. Sentence Tokenize | NLTK `sent_tokenize` | Tách 8 câu |
| 2. Word Tokenize | NLTK `word_tokenize` | Tách từ |
| 3. POS Tagging | NLTK `pos_tag` | Gán nhãn từ loại |
| 4. Lọc | Custom | Giữ N (Noun), V (Verb), J (Adj) |
| 5. Synset Candidates | WordNet | Lấy 5 nghĩa đầu tiên |

**Output**: `draft_data.csv` → Gán nhãn thủ công → `gold_standard.csv`

---

## 2.4 Phương pháp rút trích đặc trưng

| Loại | Đặc trưng | Mô tả |
|------|-----------|-------|
| **Ngữ cảnh** | BERT Embeddings | Vector 768-dim có context |
| **Vị trí** | Positional Encoding | Vị trí token trong câu |
| **Subword** | WordPiece | Xử lý OOV, compound words |
| **POS** | Part-of-Speech tag | N, V, J filter |

**Label Alignment**: Chỉ gán nhãn cho sub-word đầu tiên của mỗi từ

---

## 2.5 Kết quả đánh giá

| Metric | MFS Baseline | BERT Model |
|--------|--------------|------------|
| **Accuracy** | **70.00%** (56/80) | 67.50% (54/80) |
| **Precision (Macro)** | 0.55 | 0.44 |
| **Recall (Macro)** | 0.55 | 0.44 |
| **F1-Weighted** | 0.71 | 0.68 |

---

## 2.6 Phân tích lỗi

### Lỗi MFS Baseline

| Token | Gold | MFS Predict | Nguyên nhân |
|-------|------|-------------|-------------|
| `is` | `be.v.02` | `be.v.01` | Sai sense số (02 vs 01) |
| `implementation` | `implementation.n.02` | `execution.n.06` | MFS chọn synset phổ biến nhất (execution.n.06 có lemma "implementation") |
| `individual` | `person.n.01` | `individual.a.01` | Sai POS (Noun vs Adj) |
| `designed` | `design.v.02` | `plan.v.03` | Synonym confusion |

### Lỗi BERT Model

| Token | Gold | BERT Predict | Nguyên nhân |
|-------|------|--------------|-------------|
| `machine-based` | `machine.n.01` | `machine-readable.a.01` | Compound word confusion |
| `perform` | `perform.v.02` | `system.n.01` | Context misalignment |
| `varying` | `varying.s.01` | `intelligence.n.01` | Label shift trong training |

---

## 2.7 Nhận xét

| Quan sát | Giải thích |
|----------|------------|
| MFS > BERT (70% vs 67.5%) | Legal text dùng nghĩa phổ biến, ít nhập nhằng |
| F1-Macro thấp (0.44) | Class imbalance: 45 synsets nhưng phân bố không đều |
| Domain mismatch | SemCor (general English) ≠ Legal terminology |

**Hướng cải thiện**:
- Fine-tune trực tiếp trên legal WSD corpus
- Tăng dữ liệu domain-specific
- Sử dụng Legal-BERT hoặc model lớn hơn

---

# 3. BIỂU DIỄN TRI THỨC

## 3.1 Ngôn ngữ bậc một (FOL)

| Thành phần | Ký hiệu | Ví dụ trong đồ án |
|------------|---------|-------------------|
| **Hằng (Constant)** | chữ thường | `bkav_corp`, `chat_gpt_vn`, `incident_01` |
| **Biến (Variable)** | chữ hoa | `X`, `Y`, `Actor`, `System` |
| **Vị từ 1-ngôi** | `p(x)` | `organization(bkav_corp)`, `system(chat_gpt_vn)` |
| **Vị từ 2-ngôi** | `p(x,y)` | `designs(bkav_corp, chat_gpt_vn)` |
| **Lượng từ** | ∀, ∃ | `∀x∀y (is_developer(x,y) ↔ ...)` |

---

## 3.2 Quy trình dịch thủ công

| Bước | Mô tả | Ví dụ |
|------|-------|-------|
| 1. Tách câu | Xác định đơn vị dịch | "Developer is an organization..." |
| 2. Xác định entity | Tìm chủ thể/đối tượng | Developer, organization, AI system |
| 3. Chọn predicate | Đặt tên quan hệ | `is_developer/2`, `organization/1` |
| 4. Viết FOL | Kết hợp với quantifier | `is_developer(X,Y) :- organization(X), ...` |

---

## 3.3 Dịch thủ công các định nghĩa

### Câu 3: Developer

**NL**: "Developer is an organization or individual that designs, builds, trains, tests, or fine-tunes an AI system."

**FOL**:
```
∀x∀y (is_developer(x,y) ↔ 
      (organization(x) ∨ individual(x)) ∧ 
      ai_system(y) ∧
      (designs(x,y) ∨ builds(x,y) ∨ trains(x,y) ∨ tests(x,y) ∨ fine_tunes(x,y)))
```

### Câu 7-8: Serious Incident

**NL**: "Serious incident is an event occurring during the operation of an AI system that causes significant damage. The damage may affect human life, health, property, or the environment."

**FOL**:
```
∀e (is_serious_incident(e) ↔ 
    event(e) ∧ 
    ∃s(ai_system(s) ∧ occurs_in(e,s)) ∧
    ∃t(causes_damage(e,t) ∧ t ∈ {human_life, health, property, environment}))
```

---

## 3.4 Bổ sung tri thức tự động (WordNet)

### Phương pháp

1. **Input**: Synset đã gán nhãn (ví dụ: `developer.n.01`)
2. **Truy vấn WordNet**: Lấy hypernym (lớp cha)
3. **Output**: Sinh fact `sub_class(child, parent)`

### Ví dụ chain hypernym

```
developer.n.01 → creator.n.02 → person.n.01
     ↓                ↓              ↓
sub_class(developer, creator)
sub_class(creator, person)      [extended]
```

### Kết quả trích xuất

| Synset gốc | Hypernym | Fact sinh ra |
|------------|----------|--------------|
| `developer.n.01` | `creator.n.02` | `sub_class(developer, creator).` |
| `user.n.01` | `person.n.01` | `sub_class(user, person).` |
| `provider.n.01` | `businessperson.n.01` | `sub_class(provider, businessperson).` |
| `incident.n.01` | `happening.n.01` | `sub_class(incident, happening).` |
| `system.n.01` | `instrumentality.n.03` | `sub_class(system, instrumentality).` |

---

## 3.5 Hiệu ứng bổ sung tri thức

### Trước khi bổ sung
```prolog
?- is_a(developer, person).
false.  % Không có quan hệ trực tiếp
```

### Sau khi bổ sung (suy luận bắc cầu)
```prolog
sub_class(developer, creator).
sub_class(creator, person).

is_a(X, Y) :- sub_class(X, Y).
is_a(X, Z) :- sub_class(X, Y), is_a(Y, Z).

?- is_a(developer, person).
true.   % Suy luận: developer → creator → person
```

**Lợi ích**: Hỗ trợ truy vấn gián tiếp, tăng coverage cho suy luận

---

## 3.6 Kết quả Knowledge Base

**File**: `knowledge_base.pl`

| Loại | Số lượng | Ví dụ |
|------|----------|-------|
| `is_concept/1` | 44 facts | `is_concept(developer).` |
| `has_synset/2` | 45 facts | `has_synset(developer, 'developer.n.01').` |
| `sub_class/2` | 36 relations | `sub_class(developer, creator).` |

**Tổng cộng**: 125 facts

---

# 4. TRUY VẤN TRI THỨC

## 4.1 Hệ luật Prolog

**File**: `rules.pl` - Mã hóa 8 định nghĩa từ Điều 3 EU AI Act (8 câu)

| # | Định nghĩa | Vị từ Prolog |
|---|------------|--------------|
| 1 | Artificial Intelligence | `artificial_intelligence_related(Concept)` |
| 2 | AI System | `ai_system(System)` |
| 3 | Developer | `is_developer(Actor, System)` |
| 4 | Provider | `is_provider(Actor, System)` |
| 5 | Deployer | `is_deployer(Actor, System)` |
| 6 | User | `is_user(Actor, System)` |
| 7 | Serious Incident | `is_serious_incident(Event)` |

---

## 4.2 Mock Data (Dữ liệu giả lập)

| Loại | Thực thể | Facts |
|------|----------|-------|
| **Organizations** | BKAV, FPT | `organization(bkav_corp).` `organization(fpt_software).` |
| **Individuals** | Nguyễn Văn An | `individual(nguyen_van_an).` |
| **State Agency** | Bộ Công An | `state_agency(bo_cong_an).` |
| **AI Systems** | ChatGPT VN, Camera | `system(chat_gpt_vn).` `machine_based(chat_gpt_vn).` `performs_ai_capabilities(chat_gpt_vn).` (Rule `ai_system/1` suy luận) |
| **Events** | Sự cố 01 | `event(incident_01).` `causes_damage(incident_01, reputation).` |

### Quan hệ

| Fact | Ý nghĩa |
|------|---------|
| `designs(bkav_corp, chat_gpt_vn).` | BKAV thiết kế ChatGPT VN |
| `trains(bkav_corp, chat_gpt_vn).` | BKAV huấn luyện ChatGPT VN |
| `brings_to_market(fpt_software, camera_traffic).` | FPT đưa Camera ra thị trường |
| `uses(nguyen_van_an, chat_gpt_vn).` | Ông An sử dụng ChatGPT |
| `purpose(bo_cong_an, camera_traffic, professional).` | Bộ CA dùng Camera cho công vụ |

---

## 4.3 Tập câu hỏi thử nghiệm

| # | Câu hỏi (NL) | Truy vấn FOL | Kết quả |
|---|--------------|--------------|---------|
| 1 | Ai là developer của ChatGPT VN? | `is_developer(X, chat_gpt_vn)` | **X = bkav_corp** |
| 2 | BKAV có phải provider không? | `is_provider(bkav_corp, _)` | **false** |
| 3 | FPT cung cấp hệ thống nào? | `is_provider(fpt_software, X)` | **X = camera_traffic** |
| 4 | Developer có phải là Person? | `is_a(developer, person)` | **true** |
| 5 | incident_01 có nghiêm trọng không? | `is_serious_incident(incident_01)` | **true** |
| 6 | Bộ CA có phải deployer? | `is_deployer(bo_cong_an, camera_traffic)` | **true** |
| 7 | Ông An có phải user? | `is_user(nguyen_van_an, chat_gpt_vn)` | **true** |
| 8 | User có phải Person? | `is_a(user, person)` | **true** |

---

## 4.4 Ví dụ suy luận nhiều bước

### Query: `is_a(developer, person)`

**Quá trình suy luận**:

```
Step 1: Kiểm tra sub_class(developer, person)? → false
Step 2: Tìm sub_class(developer, X)? → X = creator
Step 3: Đệ quy is_a(creator, person)?
Step 4: Kiểm tra sub_class(creator, person)? → true
Step 5: Trả về true
```

**Chain**: `developer` → `creator` → `person`

---

## 4.5 Phân tích kết quả

### Truy vấn thành công (8/8 = 100%)

| Query | Lý do thành công |
|-------|------------------|
| `is_developer(X, chat_gpt_vn)` | Mock data đầy đủ: `designs()` và `trains()` |
| `is_a(developer, person)` | WordNet hypernym chain hoạt động |
| `is_serious_incident(incident_01)` | `causes_damage(_, reputation)` ∈ damage types |

### Hạn chế hiện tại

| Vấn đề | Ví dụ | Hướng khắc phục |
|--------|-------|-----------------|
| Anaphora | "He designs..." → Ai? | Coreference resolution |
| Negation | "X is NOT a provider" | Thêm `\+` operator |
| Temporal | "X was developer in 2020" | Thêm timestamp |
| WSD errors | Sai synset → sai suy luận | Cải thiện model WSD |

---

## 4.6 Nhận xét

**Ưu điểm**:
- ✅ Mô hình hóa chính xác 8 định nghĩa pháp luật (6 roles + 2 concepts)
- ✅ Tự động mở rộng tri thức qua WordNet (36 relations)
- ✅ Hỗ trợ suy luận bắc cầu (transitive reasoning)
- ✅ 100% queries trả về kết quả đúng

**Hạn chế**:
- ⚠️ Phụ thuộc chất lượng WSD (67.5-70% accuracy)
- ⚠️ Cần mock data để demo
- ⚠️ Chưa xử lý phủ định, thời gian, anaphora

---

# TÓM TẮT

## Pipeline hoàn chỉnh

| Giai đoạn | Input | Phương pháp | Output |
|-----------|-------|-------------|--------|
| **1. Tiền xử lý** | `selected_paragraph.txt` (8 câu) | NLTK Tokenize + POS | `draft_data.csv` |
| **2. Gán nhãn** | Draft + Manual annotation | MFS: 70%, BERT: 67.5% | `gold_standard.csv` (80 tokens) |
| **3. Biểu diễn** | Gold standard | FOL + WordNet | `knowledge_base.pl` (125 facts) |
| **4. Truy vấn** | `rules.pl` + Mock data | Prolog inference | 8 queries verified (100%) |

## Đóng góp chính

1. **Pipeline end-to-end**: Văn bản pháp luật → Tri thức → Truy vấn
2. **WSD cho Legal domain**: So sánh MFS vs BERT trên EU AI Act
3. **Tự động mở rộng tri thức**: 36 hypernym relations từ WordNet
4. **Hệ luật Prolog**: Mã hóa 8 định nghĩa từ 8 câu Điều 3 EU AI Act

---

# THAM KHẢO

1. Miller, G. A. et al. (1993). *SemCor: A Semantic Concordance*
2. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*
3. EU AI Act - Article 3 Definitions (2024)
4. NLTK WordNet Documentation
5. Chalkidis, I. et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School*

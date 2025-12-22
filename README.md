Dưới đây là nội dung file `README.md` tóm tắt tổng quan cấu trúc đồ án của bạn. File này được thiết kế để trình bày một cách chuyên nghiệp, giúp giảng viên và các thành viên trong nhóm dễ dàng theo dõi.

---

# Đồ án: Hệ thống Gán nhãn nghĩa và Suy luận Pháp lý tiếng Việt

## 1. Giới thiệu tổng quan

Hệ thống hỗ trợ gán nhãn nghĩa (Word Sense Disambiguation - WSD) cho các thuật ngữ trong văn bản pháp luật tiếng Việt, sau đó chuyển đổi chúng sang ngôn ngữ logic bậc một (FOL) để thực hiện suy luận và trả lời câu hỏi tự động.

## 2. Công nghệ sử dụng

| Thành phần | Công nghệ |
| --- | --- |
| **Ngôn ngữ lập trình** | Python 3.10+, Prolog (SWI-Prolog) |
| **Xử lý ngôn ngữ tự nhiên** | PhoBERT / ViT5 (Gán nhãn nghĩa) |
| **Công cụ suy luận** | SWI-Prolog (kết nối qua thư viện `swiplserver`) |
| **Giao diện (Frontend)** | HTML5, CSS3, JavaScript |
| **Máy chủ (Backend)** | FastAPI hoặc Flask |

---

## 3. Cấu trúc thư mục

```text
LEGAL_WSD_PROJECT/
├── backend/                
│   ├── main.py             # API Server xử lý yêu cầu từ Frontend
│   ├── wsd_engine.py       # Mô hình AI (PhoBERT) gán nhãn nghĩa từ ngữ
│   ├── prolog_bridge.py    # Cầu nối gọi truy vấn từ Python sang Prolog
│   └── logic_files/        
│       ├── pheptoan.pl     # Định nghĩa toán tử logic (&, =>, ~, v.v.)
│       ├── lambda.pl       # Xử lý phép biến đổi Beta (Beta conversion)
│       ├── tuvung.pl       # Định nghĩa vị từ cho từ vựng pháp lý
│       └── mohinh.pl       # Chứa các sự kiện (Facts) và luật (Rules) cụ thể
├── frontend/               
│   ├── index.html          # Giao diện nhập liệu và hiển thị kết quả
│   ├── style.css           # Giao diện người dùng
│   └── script.js           # Xử lý logic gọi API và hiển thị
└── requirements.txt        # Danh sách các thư viện cần cài đặt

```

---

## 4. Quy trình xử lý (Workflow)

1. **Input:** Người dùng nhập một câu hoặc đoạn văn pháp lý (Ví dụ: "Bị cáo có hành vi trộm cắp").
2. **WSD & Parser:** * Mô hình AI xác định nghĩa của từ "trộm cắp" trong ngữ cảnh pháp lý.
* Chuyển câu văn sang cấu trúc logic bằng biểu thức Lambda.


3. **Knowledge Base:** Hệ thống kết hợp các luật đã có trong file `.pl` (Ví dụ: *Trộm cắp là vi phạm pháp luật*).
4. **Reasoning:** SWI-Prolog thực hiện suy luận dựa trên dữ liệu đầu vào và các luật pháp lý.
5. **Output:** Trả về kết quả xác nhận hoặc trả lời câu hỏi trên giao diện.

---

## 5. Cài đặt nhanh

1. **Cài đặt thư viện:** `pip install -r requirements.txt`
2. **Cài đặt Prolog:** Cài đặt [SWI-Prolog](https://www.swi-prolog.org/) vào máy tính.
3. **Chạy ứng dụng:** * Chạy Backend: `python backend/main.py`
* Mở file `frontend/index.html` trên trình duyệt.



---

**Bạn có muốn tôi viết chi tiết nội dung code cho file `prolog_bridge.py` để bạn chạy thử việc kết nối giữa Python và Prolog không?**
# Lab 19: Xây dựng hệ thống GraphRAG với Tech Company Corpus

Dự án này thực hiện các yêu cầu của Lab 19: Xây dựng một Pipeline GraphRAG, so sánh với Flat RAG truyền thống dựa trên bộ dữ liệu Tech Company Corpus.

## Cấu trúc thư mục

```text
.
├── data/
│   └── tech_corpus.txt    # Dữ liệu text chứa thông tin các công ty công nghệ
├── results/               # Thư mục chứa kết quả (hình ảnh graph, csv đánh giá)
├── src/
│   ├── config.py          # Cấu hình biến môi trường
│   ├── graph_builder.py   # Module trích xuất thực thể, quan hệ và xây dựng Graph
│   ├── flat_rag.py        # Module Flat RAG dùng Vector Database
│   ├── graph_rag.py       # Module Graph RAG truy vấn bằng đồ thị
│   └── evaluation.py      # Module chạy đánh giá 20 câu hỏi
├── main.py                # Script chạy toàn bộ pipeline
├── requirements.txt       # Danh sách thư viện
└── .env                   # File môi trường chứa OPENAI_API_KEY
```

## Cài đặt

1. Tạo môi trường ảo (tùy chọn) và cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Đổi tên file `.env.example` thành `.env` và điền `OPENAI_API_KEY` của bạn vào.

## Cách chạy

Chạy script chính để thực thi toàn bộ luồng từ xây dựng đồ thị, flat vector db đến lúc so sánh và đánh giá:

```bash
python main.py
```

# ViPRA 2026 - DSPy Prompt Optimizer

Hệ thống tự động tối ưu Prompt cho bài toán giải Vật lý tiếng Việt.

## Cấu trúc thư mục `src/` (Đã phân nhóm Pipeline)

```
src/
├── config.py                     # Cấu hình chung toàn hệ thống
├── test_modules.py               # Kiểm tra nhanh các module
│
├── [Nhóm RAG]
│   ├── rag_module.py             # Hybrid Search (BM25 + FAISS BGE-M3)
│   └── rag_data_prep.py          # Tiền xử lý dữ liệu cho RAG
│
├── [Nhóm Dùng chung (Result & Optimize)]
│   ├── result_optimize_solver.py     # DSPy Signature + PhysicsSolver (Loại A & B)
│   ├── result_optimize_metric.py     # Sandbox exec() và chấm điểm P1 theo luật BTC
│   └── result_optimize_translator.py # Dịch Vi->En (VinAI) + Placeholder Protection
│
├── [Nhóm Optimize Prompt]
│   ├── optimize_data_loader.py   # Load train.csv & dev.csv thành DSPy Examples
│   └── optimize_prompt.py        # Script tối ưu Prompt tự động (MIPROv2)
│
└── [Nhóm Result (Giải đề BTC)]
    ├── result_predict.py         # Script API / Inference xuất JSON chuẩn BTC
    ├── result_test_from_dev.py   # Chạy test hàng loạt trên tập Dev
    ├── result_llm_engine.py      # Engine sinh văn bản
    └── result_evaluate_with_llm.py # Đánh giá LLM-as-a-judge
```

## Hướng dẫn sử dụng

### Bước 1: Cấu hình model
Mở [config.py](config.py) và chỉnh `LM_MODEL_ID` + `LM_API_BASE` cho phù hợp với vLLM.

### Bước 2: Chạy tối ưu Prompt
```bash
uv run python src/optimize_prompt.py
```
> Kết quả được lưu tự động tại `output/optimized_prompt.json`.

### Bước 3: Thử giải bài toán mới
```bash
uv run python src/result_predict.py --id "TD401" --query "Tính năng lượng điện trường tích trữ trong tụ C khi C = 100 μF dưới U = 30 V"
```

## Vai trò của 2 file CSV

| File | Vai trò trong DSPy | Mô tả |
|---|---|---|
| `train_dataset.csv` | `trainset` | **Xưởng chế tạo** - Nguồn sinh biến thể Prompt và ví dụ Few-shot |
| `dev_dataset.csv` | `valset` | **Ban Giám Khảo** - Đánh giá và chốt hạ Prompt tốt nhất |

> `dev_dataset.csv` bị **cách ly hoàn toàn** khỏi quá trình sinh Prompt.  
> Đây là điểm mấu chốt giúp Prompt không bị "học vẹt" và tổng quát tốt khi thi thật.

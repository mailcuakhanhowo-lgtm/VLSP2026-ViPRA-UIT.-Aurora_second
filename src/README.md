# ViPRA 2026 - DSPy Prompt Optimizer

Hệ thống tự động tối ưu Prompt cho bài toán giải Vật lý tiếng Việt.

## Cấu trúc thư mục `src/`

```
src/
├── config.py        # Đường dẫn file CSV, tham số model và tối ưu
├── data_loader.py   # Load train.csv + dev.csv thành DSPy Examples
├── metric.py        # Hàm chấm điểm P1 theo đúng tiêu chí BTC (sai số 0.05)
├── solver.py        # DSPy Signature + Module PhysicsSolver
├── optimize.py      # Script tối ưu Prompt (MIPROv2)  <-- CHẠY CÁI NÀY TRƯỚC
└── predict.py       # Script inference khi đi thi
```

## Hướng dẫn sử dụng

### Bước 1: Cấu hình model
Mở [config.py](config.py) và chỉnh `LM_MODEL_ID` + `LM_API_BASE` cho phù hợp với vLLM của anh.

### Bước 2: Chạy tối ưu Prompt (bấm 1 lần, đi uống cà phê)
```bash
uv run python src/optimize.py
```
> Quá trình này chạy khoảng **1-2 tiếng** tùy GPU.  
> Kết quả được lưu tự động tại `output/optimized_prompt.json`.

### Bước 3: Thử giải bài toán mới
```bash
uv run python src/predict.py --id "TD401" --query "Tính năng lượng điện trường tích trữ trong tụ C khi C = 100 μF dưới U = 30 V"
```

## Vai trò của 2 file CSV

| File | Vai trò trong DSPy | Mô tả |
|---|---|---|
| `train_dataset.csv` | `trainset` | **Xưởng chế tạo** - Nguồn sinh biến thể Prompt và ví dụ Few-shot |
| `dev_dataset.csv` | `valset` | **Ban Giám Khảo** - Đánh giá và chốt hạ Prompt tốt nhất |

> `dev_dataset.csv` bị **cách ly hoàn toàn** khỏi quá trình sinh Prompt.  
> Đây là điểm mấu chốt giúp Prompt không bị "học vẹt" và tổng quát tốt khi thi thật.

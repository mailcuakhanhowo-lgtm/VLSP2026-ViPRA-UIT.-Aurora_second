"""
data_loader.py - Load và chuẩn hóa dữ liệu CSV thành DSPy Examples
"""
import csv
import dspy
from config import RAW_TRAIN_CSV, RAW_DEV_CSV, MAX_TRAIN_EXAMPLES, MAX_DEV_EXAMPLES


def _load_csv(filepath: str) -> list[dict]:
    """Đọc file CSV, xử lý BOM UTF-8 và trả về list dict."""
    rows = []
    with open(filepath, encoding="utf-8-sig") as f:  # utf-8-sig tự xử lý BOM \ufeff
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _normalize_id_key(row: dict) -> dict:
    """Đổi tên key id (có thể bị BOM thành \ufeffid) về 'id' chuẩn."""
    normalized = {}
    for k, v in row.items():
        clean_key = k.lstrip("\ufeff").strip()
        normalized[clean_key] = v.strip() if isinstance(v, str) else v
    return normalized


def load_dataset(filepath: str, max_examples: int = None) -> list[dspy.Example]:
    """
    Load file CSV và chuyển đổi sang danh sách DSPy Example.

    Mỗi Example có các trường:
        - query_id  : ID câu hỏi (VD: TD401)
        - query     : Nội dung đề bài tiếng Việt
        - cot       : Lời giải mẫu 4 bước (chỉ dùng để tham khảo / bootstrap)
        - answer    : Đáp án chuẩn (chuỗi số / biểu thức / văn bản)
        - unit      : Đơn vị (chuẩn ASCII), "-" hoặc "" nếu không có

    Tham số inputs=("query",) báo với DSPy chỉ "query" là đầu vào,
    còn lại là nhãn (labels) để chấm điểm.
    """
    raw_rows = _load_csv(filepath)
    examples = []

    for row in raw_rows:
        row = _normalize_id_key(row)
        # Chuẩn hóa unit: BTC dùng "-" cho câu không có đơn vị
        unit = row.get("unit", "").strip()
        if unit == "-":
            unit = ""

        # Chuẩn hóa context: lấy từ dữ liệu nếu có, nếu rỗng thì dùng Explicit Fallback Text
        context = row.get("context", "").strip()
        if not context:
            import config
            context = getattr(
                config,
                "RAG_FALLBACK_TEXT",
                "Không có tài liệu tham khảo bổ sung. Hãy sử dụng các định luật vật lý tiêu chuẩn."
            )

        ex = dspy.Example(
            query_id=row.get("id", ""),
            context=context,
            question=row.get("question", ""),  # field name khớp PhysicsRAG_PAL Signature
            answer=row.get("answer", ""),
            unit=unit,
        ).with_inputs("context", "question")

        examples.append(ex)

    if max_examples:
        examples = examples[:max_examples]

    return examples


def load_train() -> list[dspy.Example]:
    print(f"[DataLoader] Loading TRAIN: {RAW_TRAIN_CSV}")
    data = load_dataset(RAW_TRAIN_CSV, max_examples=MAX_TRAIN_EXAMPLES)
    print(f"[DataLoader] Train size: {len(data)} examples")
    return data


def load_dev() -> list[dspy.Example]:
    print(f"[DataLoader] Loading DEV: {RAW_DEV_CSV}")
    data = load_dataset(RAW_DEV_CSV, max_examples=MAX_DEV_EXAMPLES)
    print(f"[DataLoader] Dev size: {len(data)} examples")
    return data

"""
translator.py - Dịch đề bài Tiếng Việt → Tiếng Anh trước khi đưa vào LLM

Model: vinai/vinai-translate-vi2en-v2 (~400MB)
  - Do VinAI (Việt Nam) phát triển, chuyên biệt cho Tiếng Việt
  - Là model seq2seq (MarianMT-based), KHÔNG phải LLM
  - KHÔNG tính vào giới hạn 8B tham số của BTC
  - Tốc độ: ~0.05-0.3s/câu tùy thiết bị

Kỹ thuật Placeholder Protection:
  - Trước khi dịch: "đóng băng" số và ký hiệu vật lý quan trọng
    VD: "100 μF" → "__PH_0__", "X_L = 22 Ω" → "X_L = __PH_1__"
  - Sau khi dịch: khôi phục lại đúng nguyên bản
  - Đảm bảo model dịch không làm sai lệch các giá trị số học
"""

import re
import os
import sys
import logging
from typing import Optional

# Tắt log thừa của transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# Lazy import: chỉ load khi cần, tránh chậm startup
_tokenizer = None
_model = None
_device = None


# ============================================================
# PHẦN 1: PLACEHOLDER PROTECTION
# ============================================================

# Pattern để nhận diện các chuỗi cần bảo vệ (theo thứ tự ưu tiên)
_PROTECT_PATTERNS = [
    # Số kèm đơn vị vật lý (VD: "100 μF", "22.5 Ω", "3 × 10^-6 F")
    r'\d+(?:[.,]\d+)?\s*(?:×\s*10\^[+-]?\d+\s*)?'
    r'(?:μF|uF|nF|pF|μH|mH|μΩ|kΩ|MΩ|Ω|ohm|kV|mV|μV|'
    r'kA|mA|μA|kHz|MHz|GHz|mW|kW|MW|μJ|mJ|kJ|MJ|'
    r'Wb|mWb|μWb|T|mT|μT|G|N|kN|Pa|kPa|MPa|'
    r'rad/s|rpm|m/s|km/h|'
    r'V|A|W|J|F|H|Hz|s|m|C|K|kg|mol|cd|lm|lx)',

    # Ký hiệu khoa học thuần (VD: "10^-6", "3×10^8", "1.6×10^-19")
    r'\d+(?:[.,]\d+)?(?:\s*[×x]\s*10\^[+-]?\d+|\s*e[+-]?\d+)',

    # Biến vật lý có chỉ số (VD: "X_L", "U_C", "I_0", "R_1")
    r'[A-Za-z]{1,3}_[A-Za-z0-9]{1,3}(?:\s*=\s*\d+(?:[.,]\d+)?)?',

    # Số nguyên / thập phân đơn thuần quan trọng trong biểu thức
    r'(?<![A-Za-z])\d+(?:[.,]\d+)?(?:\s*%)?(?![A-Za-z_])',
]

_COMPILED_PATTERNS = [re.compile(p) for p in _PROTECT_PATTERNS]
_PLACEHOLDER_FMT = "__PH_{idx}__"
_PLACEHOLDER_RE  = re.compile(r"__PH_(\d+)__")


def _protect_values(text: str) -> tuple[str, dict]:
    """
    Thay thế các số và ký hiệu vật lý bằng placeholder.
    Trả về (text_với_placeholder, dict mapping idx→giá_trị_gốc).
    """
    protected = {}
    counter = [0]  # dùng list để có thể modify trong closure

    def replace_match(m: re.Match) -> str:
        idx = counter[0]
        counter[0] += 1
        placeholder = _PLACEHOLDER_FMT.format(idx=idx)
        protected[str(idx)] = m.group(0)
        return placeholder

    result = text
    # Áp dụng lần lượt từng pattern, bảo vệ match dài nhất trước
    # Dùng một lượt scan để tránh double-replace
    all_matches = []
    for pattern in _COMPILED_PATTERNS:
        for m in pattern.finditer(result):
            # Chỉ lấy nếu chưa nằm trong placeholder đã xử lý
            if "__PH_" not in result[m.start():m.end()]:
                all_matches.append((m.start(), m.end(), m.group(0)))

    # Sắp xếp từ cuối về đầu để không bị lệch vị trí khi thay thế
    all_matches.sort(key=lambda x: x[0], reverse=True)
    seen_ranges = []
    filtered = []
    for start, end, val in all_matches:
        # Bỏ qua nếu overlap với match đã chọn
        overlap = any(s <= start < e or s < end <= e for s, e in seen_ranges)
        if not overlap:
            filtered.append((start, end, val))
            seen_ranges.append((start, end))

    filtered.sort(key=lambda x: x[0], reverse=True)
    for start, end, val in filtered:
        idx = counter[0]
        counter[0] += 1
        placeholder = _PLACEHOLDER_FMT.format(idx=idx)
        protected[str(idx)] = val
        result = result[:start] + placeholder + result[end:]

    return result, protected


def _restore_values(translated: str, protected: dict) -> str:
    """
    Khôi phục placeholder về giá trị gốc trong bản dịch.
    Xử lý cả trường hợp model dịch thêm khoảng trắng quanh placeholder.
    """
    def restore_match(m: re.Match) -> str:
        idx = m.group(1)
        return protected.get(idx, m.group(0))

    # Đôi khi model dịch thêm dấu cách: "__ PH_0 __" → chuẩn hóa trước
    result = re.sub(r"__\s*PH_(\d+)\s*__", r"__PH_\1__", translated)
    return _PLACEHOLDER_RE.sub(restore_match, result)


# ============================================================
# PHẦN 2: LOAD MODEL (Lazy, Singleton)
# ============================================================

def _load_model_if_needed():
    """Load tokenizer và model lần đầu, giữ trong bộ nhớ cho các lần sau."""
    global _tokenizer, _model, _device
    if _model is not None:
        return  # Đã load rồi

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    import config

    device_str = config.TRANSLATION_DEVICE
    cache_dir  = config.TRANSLATION_CACHE_DIR
    model_id   = config.TRANSLATION_MODEL_ID

    print(f"[Translator] Loading model: {model_id}")
    print(f"[Translator] Cache dir   : {cache_dir}")
    print(f"[Translator] Device      : {device_str}")

    os.makedirs(cache_dir, exist_ok=True)

    _tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        src_lang="vi_VN",
    )
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    _device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    _model = _model.to(_device)
    _model.eval()

    print(f"[Translator] Model loaded successfully on {_device}!\n")


# ============================================================
# PHẦN 3: HÀM DỊCH CHÍNH
# ============================================================

def translate_vi_to_en(text: str, use_placeholder: bool = True) -> str:
    """
    Dịch một câu tiếng Việt sang tiếng Anh.

    Args:
        text: Câu tiếng Việt cần dịch
        use_placeholder: True (mặc định) → bảo vệ số và ký hiệu trước khi dịch

    Returns:
        Câu tiếng Anh đã được dịch
    """
    import config
    if not config.USE_TRANSLATION:
        return text  # Bỏ qua nếu tắt translation

    if not text or not text.strip():
        return text

    _load_model_if_needed()

    import torch

    # Bước 1: Bảo vệ số và ký hiệu
    if use_placeholder:
        text_to_translate, protected = _protect_values(text)
    else:
        text_to_translate = text
        protected = {}

    # Bước 2: Tokenize
    inputs = _tokenizer(
        text_to_translate,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(_device)

    # Bước 3: Sinh bản dịch
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            decoder_start_token_id=_tokenizer.lang_code_to_id["en_XX"],
            num_beams=4,
            max_length=config.TRANSLATION_MAX_LENGTH,
            early_stopping=True,
        )

    # Bước 4: Decode
    translated = _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Bước 5: Khôi phục placeholder
    if use_placeholder and protected:
        translated = _restore_values(translated, protected)

    return translated.strip()


def translate_batch(texts: list[str], use_placeholder: bool = True) -> list[str]:
    """
    Dịch nhiều câu cùng lúc (batch) — hiệu quả hơn khi dịch toàn bộ tập train/dev.

    Args:
        texts: Danh sách câu tiếng Việt
        use_placeholder: True → bảo vệ số và ký hiệu

    Returns:
        Danh sách câu tiếng Anh tương ứng
    """
    import config
    if not config.USE_TRANSLATION:
        return texts

    if not texts:
        return []

    _load_model_if_needed()

    import torch

    batch_size = config.TRANSLATION_BATCH_SIZE
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Bảo vệ từng câu trong batch
        if use_placeholder:
            protected_batch = [_protect_values(t) for t in batch]
            texts_to_translate = [p[0] for p in protected_batch]
            protected_dicts    = [p[1] for p in protected_batch]
        else:
            texts_to_translate = batch
            protected_dicts    = [{} for _ in batch]

        # Tokenize batch
        inputs = _tokenizer(
            texts_to_translate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(_device)

        # Sinh bản dịch batch
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                decoder_start_token_id=_tokenizer.lang_code_to_id["en_XX"],
                num_beams=4,
                max_length=config.TRANSLATION_MAX_LENGTH,
                early_stopping=True,
            )

        # Decode và khôi phục
        translated_batch = _tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for translated, protected in zip(translated_batch, protected_dicts):
            if use_placeholder and protected:
                translated = _restore_values(translated, protected)
            results.append(translated.strip())

        print(f"[Translator] Translated {min(i + batch_size, len(texts))}/{len(texts)} sentences...")

    return results


def preload():
    """Nạp model vào bộ nhớ trước (gọi 1 lần khi khởi động server để tránh delay)."""
    _load_model_if_needed()

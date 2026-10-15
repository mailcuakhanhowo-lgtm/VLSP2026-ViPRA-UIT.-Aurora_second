"""
rag_data_prep.py - Tiền xử lý dữ liệu cho RAG Pipeline
Bao gồm:
1. chunk_feynman_theory(): Đọc các file JSON trong feynman_vol_II,
   bóc tách, loại bỏ nhiễu HTML, chia đoạn (chunking ngữ nghĩa bảo vệ công thức toán),
   và xuất ra feynman_theory_chunks.jsonl phục vụ RAG.
2. process_train_dataset(): Làm sạch tập train CSV ra train_processed.jsonl
"""

import os
import sys
import re
import glob
import json
import argparse
import pandas as pd
import config

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# PHẦN 1: THUẬT TOÁN CHUNKING CHUYÊN BIỆT CHO LÝ THUYẾT VẬT LÝ
# ============================================================

def clean_feynman_text(raw_text: str) -> str:
    """
    Làm sạch các đoạn rác do quá trình crawl/parse web:
    - Bảng ký tự Hy Lạp sidebar
    - Khoảng trắng thừa, nhiều dòng trống liên tiếp
    - Ký tự gạch ngang/trừ Unicode
    """
    text = raw_text

    # Loại bỏ bảng liệt kê ký tự Hy Lạp ở cuối trang nếu có
    greek_pattern = r"Lower case Greek letters and commonly used capitals.*?(?=\n\n[A-Z]|\Z)"
    text = re.sub(greek_pattern, "", text, flags=re.DOTALL)

    # Chuẩn hóa khoảng trắng & ngắt dòng
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Chuẩn hóa dấu trừ/gạch nối
    text = text.replace("−", "-").replace("–", "-")

    return text.strip()


def split_into_semantic_paragraphs(text: str) -> list[str]:
    """
    Chia text thành các đoạn văn, đồng thời bảo vệ các khối LaTeX không bị đứt đoạn.
    (Giữ nguyên khối \\begin{equation}...\\end{equation} hoặc $$...$$)
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged_paragraphs = []
    buffer = []
    in_equation = False

    for p in raw_paragraphs:
        begins = len(re.findall(r"\\begin\{equation\}", p))
        ends = len(re.findall(r"\\end\{equation\}", p))

        if begins > ends:
            in_equation = True
            buffer.append(p)
        elif ends > begins:
            in_equation = False
            buffer.append(p)
            merged_paragraphs.append("\n\n".join(buffer))
            buffer = []
        elif in_equation:
            buffer.append(p)
        else:
            merged_paragraphs.append(p)

    if buffer:
        merged_paragraphs.append("\n\n".join(buffer))

    return merged_paragraphs


def chunk_paragraphs(
    paragraphs: list[str],
    header_prefix: str,
    target_words: int = config.CHUNK_TARGET_WORDS,
    min_words: int = config.CHUNK_MIN_WORDS,
    max_words: int = config.CHUNK_MAX_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Gom các đoạn văn thành từng chunk theo target_words và có overlap an toàn.
    Không bao giờ cắt xẻ đôi một công thức toán học.
    """
    chunks = []
    current_paras = []
    current_word_count = 0

    for p in paragraphs:
        p_words = len(p.split())
        
        # Nếu một đoạn văn bản thân nó quá dài (> max_words), tách theo câu
        if p_words > max_words:
            if current_paras:
                chunk_body = "\n\n".join(current_paras)
                chunks.append(f"{header_prefix}\n\n{chunk_body}")
                current_paras = []
                current_word_count = 0
            
            sentences = re.split(r'(?<=[.!?])\s+', p)
            sub_chunk = []
            sub_count = 0
            for s in sentences:
                s_count = len(s.split())
                if sub_count + s_count > target_words and sub_count >= min_words:
                    chunks.append(f"{header_prefix}\n\n{' '.join(sub_chunk)}")
                    overlap = sub_chunk[-2:] if len(sub_chunk) >= 2 else sub_chunk[-1:]
                    sub_chunk = list(overlap) + [s]
                    sub_count = sum(len(x.split()) for x in sub_chunk)
                else:
                    sub_chunk.append(s)
                    sub_count += s_count
            if sub_chunk:
                chunks.append(f"{header_prefix}\n\n{' '.join(sub_chunk)}")
            continue

        # Nếu thêm đoạn này vẫn trong ngưỡng cho phép
        if current_word_count + p_words <= target_words:
            current_paras.append(p)
            current_word_count += p_words
        elif current_word_count >= min_words:
            # Đủ ngưỡng mục tiêu -> Đóng chunk
            chunk_body = "\n\n".join(current_paras)
            chunks.append(f"{header_prefix}\n\n{chunk_body}")

            # Gối đầu (overlap): Giữ lại đoạn văn cuối cùng nếu kích thước hợp lý
            if current_paras and len(current_paras[-1].split()) <= overlap_words * 1.5:
                current_paras = [current_paras[-1], p]
                current_word_count = sum(len(x.split()) for x in current_paras)
            else:
                current_paras = [p]
                current_word_count = p_words
        else:
            current_paras.append(p)
            current_word_count += p_words

    # Đẩy chunk cuối cùng còn lại
    if current_paras:
        if current_word_count < min_words and chunks:
            last_chunk = chunks.pop()
            extra_body = "\n\n".join(current_paras)
            chunks.append(f"{last_chunk}\n\n{extra_body}")
        else:
            chunk_body = "\n\n".join(current_paras)
            chunks.append(f"{header_prefix}\n\n{chunk_body}")

    return chunks


# ============================================================
# PHẦN 2: XỬ LÝ TOÀN BỘ THƯ MỤC FEYNMAN
# ============================================================

def process_feynman_corpus(
    input_dir: str = config.FEYNMAN_RAW_DIR,
    output_jsonl: str = config.PROCESSED_THEORY_JSONL
):
    """
    Đọc tất cả 317 file JSON trong thư mục feynman_vol_II,
    làm sạch, cắt đoạn (chunking), và lưu ra file JSONL hoàn chỉnh.
    """
    print(f"[*] Bắt đầu xử lý Feynman Corpus từ: {input_dir}")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {input_dir}")

    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"[+] Tìm thấy {len(json_files)} file JSON bài giảng.")

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    all_chunks = []
    total_words = 0

    for file_idx, filepath in enumerate(json_files):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Lỗi đọc file {os.path.basename(filepath)}: {e}")
            continue

        chapter = str(data.get("chapter", "")).strip()
        section = str(data.get("section", "")).strip()
        theory_topic = str(data.get("theory", f"{chapter} > {section}")).strip()
        raw_content = str(data.get("content", ""))

        if not raw_content.strip():
            continue

        # 1. Làm sạch
        clean_text = clean_feynman_text(raw_content)

        # 2. Tạo Header định danh chủ đề vật lý cho chunk
        header_prefix = f"[{theory_topic}]"

        # 3. Phân đoạn ngữ nghĩa bảo vệ LaTeX
        paragraphs = split_into_semantic_paragraphs(clean_text)

        # 4. Gom chunking có gối đầu
        file_chunks = chunk_paragraphs(paragraphs, header_prefix)

        filename = os.path.splitext(os.path.basename(filepath))[0]

        for chunk_idx, chunk_text in enumerate(file_chunks):
            words = len(chunk_text.split())
            total_words += words
            has_latex = bool(re.search(r"\\(?:begin|mathbf|frac|sqrt|times|cdot|int|sum|\^|_)", chunk_text))

            record = {
                "id": f"{filename}_c{chunk_idx:02d}",
                "chapter": chapter,
                "section": section,
                "theory_topic": theory_topic,
                "text": chunk_text,
                "word_count": words,
                "has_latex": has_latex,
                "source": "Feynman Lectures on Physics Vol II"
            }
            all_chunks.append(record)

    # Ghi ra JSONL
    print(f"[*] Đang ghi {len(all_chunks)} chunks ra file: {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in all_chunks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_words = total_words / len(all_chunks) if all_chunks else 0
    print(f"[v] HOÀN TẤT CHUNKING FEYNMAN CORPUS:")
    print(f"    - Tổng số file JSON đầu vào : {len(json_files)}")
    print(f"    - Tổng số chunks tạo ra     : {len(all_chunks)}")
    print(f"    - Số từ trung bình mỗi chunk: {avg_words:.1f} words")
    print(f"    - File kết quả lưu tại       : {output_jsonl}\n")
    return all_chunks


# ============================================================
# PHẦN 3: XỬ LÝ TẬP DỮ LIỆU TRAIN CSV GỐC
# ============================================================

def process_train_dataset():
    raw_csv_path = config.RAW_TRAIN_CSV
    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_jsonl_path = config.PROCESSED_TRAIN_JSONL

    print(f"[*] Reading data from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, encoding='utf-8')
    print(f"[+] Initial row count: {len(df)}")

    df = df.dropna(subset=['cot', 'answer'])
    df['unit'] = df['unit'].fillna('')
    df = df.drop_duplicates(subset=['question'])
    print(f"[+] Row count after cleaning: {len(df)}")

    with open(train_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            clean_dict = {
                "id": str(row['id']),
                "question": str(row['question']).strip(),
                "cot": str(row['cot']).strip(),
                "answer": str(row['answer']).strip(),
                "unit": str(row['unit']).strip()
            }
            f.write(json.dumps(clean_dict, ensure_ascii=False) + '\n')

    print(f"[v] Train Data Preparation Completed: {train_jsonl_path}")


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ViPRA 2026 - RAG Data Preparation")
    parser.add_argument("--feynman", action="store_true", default=True, help="Xử lý và chunking dữ liệu Feynman Lectures")
    parser.add_argument("--train", action="store_true", default=False, help="Làm sạch tập train CSV của BTC")
    args = parser.parse_args()

    if args.feynman:
        process_feynman_corpus()
    if args.train:
        process_train_dataset()


if __name__ == "__main__":
    main()

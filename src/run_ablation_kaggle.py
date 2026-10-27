"""
run_ablation_kaggle.py - Script chạy thực nghiệm Progressive Additive Ablation Study (4 Rows) trên Kaggle / GPU Server.

Cách chạy trên Kaggle:
1. Chạy trên bộ Public Test 51 câu (Mặc định):
   python src/run_ablation_kaggle.py

2. Hoặc chỉ định đường dẫn file CSV bất kỳ (VD: tập Dev 339 câu):
   python src/run_ablation_kaggle.py --dataset /kaggle/input/vlsp-dataset-train-for-publicte/dev_dataset.csv
"""

import os
import sys
import re
import csv
import time
import json
import argparse
from typing import Dict, List

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from optimize_prompt import setup_lm
from evaluate_api_test import grade_prediction
from result_optimize_translator import preload as preload_translator
from ablation_row1_only_qwen import solve_row1
from ablation_row2_qwen_pal import solve_row2
from ablation_row3_qwen_rag_pal import solve_row3, init_row3_rag
from result_predict import solve as solve_row4, load_optimized_solver


def find_default_dataset() -> str:
    """Tự động dò tìm file test_dataset.csv (51 câu) trên Kaggle hoặc thư mục dự án."""
    candidates = [
        "/kaggle/input/pb-test/test_dataset.csv",
        "/kaggle/input/pb test/test_dataset.csv",
        os.path.join(BASE_DIR, "PBtest", "test_dataset.csv"),
        os.path.join(BASE_DIR, "datasets", "test_dataset.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Quét tự động trong /kaggle/input nếu tên folder khác
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f == "test_dataset.csv":
                    return os.path.join(root, f)
    return candidates[2]


def load_test_rows(csv_path: str, max_examples: int = None) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k.lstrip("\ufeff").strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            rows.append(clean_row)
    if max_examples:
        rows = rows[:max_examples]
    return rows


def score_p2_and_p3(pred_item: Dict) -> Dict[str, float]:
    """
    Đánh giá nhanh chất lượng P2 (Giải thích lý thuyết LaTeX, súc tích, không nháp số học)
    và P3 (Độ đầy đủ của các bước suy luận CoT trong reasoning.steps).
    """
    exp = str(pred_item.get("explanation", "")).strip()
    steps = pred_item.get("reasoning", {}).get("steps", []) if isinstance(pred_item.get("reasoning"), dict) else []

    # --- Đánh giá P2 ---
    if not exp or len(exp) < 15:
        p2_score = 0.0
    else:
        has_latex = bool(re.search(r"\$.*?\$|\\(?:frac|cdot|times|sqrt|omega|mu|pi|Delta)", exp))
        word_count = len(exp.split())
        is_concise = 1.0 if word_count <= 120 else 0.6
        # Kiểm tra xem có bị nhiễm phép tính nháp kiểu "= 100 * 1e-6 =" hoặc lặp 4 bước CoT không
        has_cot_dump = any(m in exp.lower() for m in ["bước 1:", "bước 2:", "bước 3:", "1. nhận diện"])
        has_scratchpad_math = bool(re.search(r"\d+\s*[\*\/]\s*\d+\s*=", exp))
        penalty = 0.7 if (has_cot_dump or has_scratchpad_math) else 1.0
        base_p2 = 1.0 if has_latex else 0.5
        p2_score = round(base_p2 * is_concise * penalty, 3)

    # --- Đánh giá P3 ---
    valid_steps = [
        s for s in steps
        if isinstance(s, str) and s.strip() and not s.startswith(("[Lỗi", "[Không"))
    ]
    if len(valid_steps) >= 3:
        p3_score = 1.0
    elif len(valid_steps) > 0:
        p3_score = 0.5
    else:
        p3_score = 0.0

    return {"p2_score": p2_score, "p3_score": p3_score}


def run_benchmark(csv_path: str, rows_to_run: List[int], max_examples: int = None):
    print("=" * 72)
    print("  CHƯƠNG TRÌNH CHẠY ABLATION STUDY (4 ROWS) - VIPRA VLSP 2026")
    print("=" * 72)
    print(f"[Dataset] Đường dẫn: {csv_path}")

    dataset = load_test_rows(csv_path, max_examples=max_examples)
    n_total = len(dataset)
    print(f"[Dataset] Tổng số câu hỏi kiểm thử: {n_total} câu\n")

    # 1. Khởi tạo LM 1 lần duy nhất
    setup_lm()

    # 2. Preload Translator (CPU)
    if getattr(config, "USE_TRANSLATION", True):
        try:
            preload_translator(direction="vi2en")
        except Exception as e:
            print(f"[Cảnh báo] Không thể preload translator: {e}")

    # 3. Pre-compute RAG context cho toàn bộ câu hỏi (để dùng chung cho Row 3 & Row 4, tiết kiệm thời gian)
    precomputed_contexts = {}
    if 3 in rows_to_run or 4 in rows_to_run:
        print("\n[RAG] Đang trích xuất trước ngữ cảnh Feynman (K) cho bộ câu hỏi...")
        rag = init_row3_rag()
        for item in dataset:
            qid = item.get("id", "")
            q_text = item.get("question", "")
            if rag:
                try:
                    precomputed_contexts[qid] = rag.get_context(q_text)
                except Exception:
                    precomputed_contexts[qid] = config.RAG_FALLBACK_TEXT
            else:
                precomputed_contexts[qid] = config.RAG_FALLBACK_TEXT
        print(f"[RAG] Hoàn tất trích xuất ngữ cảnh cho {len(precomputed_contexts)} câu.\n")

    row4_solver = load_optimized_solver() if 4 in rows_to_run else None

    row_names = {
        1: "Row 1: Only Qwen (Direct CoT)",
        2: "Row 2: Qwen + PAL (No RAG, Joint Exp)",
        3: "Row 3: Qwen + RAG + PAL (Joint Exp)",
        4: "Row 4: Full Framework (+ Decoupled Explainer)",
    }

    summary_table = []
    all_details = {}

    for r_id in rows_to_run:
        r_name = row_names.get(r_id, f"Row {r_id}")
        print("-" * 72)
        print(f">>> ĐANG CHẠY ĐÁNH GIÁ {r_name} ({n_total} câu)...")
        print("-" * 72)

        p1_hits = 0
        num_hits = 0
        unit_hits = 0
        p2_sum = 0.0
        p3_sum = 0.0
        latencies = []
        errors = 0
        row_items = []

        for idx, item in enumerate(dataset, 1):
            qid = item.get("id", f"Q{idx:03d}")
            question = item.get("question", "")
            gt_ans = item.get("answer", "")
            gt_unit = item.get("unit", "")

            t0 = time.time()
            try:
                if r_id == 1:
                    pred = solve_row1(qid, question)
                elif r_id == 2:
                    pred = solve_row2(qid, question)
                elif r_id == 3:
                    pred = solve_row3(qid, question, precomputed_context=precomputed_contexts.get(qid))
                elif r_id == 4:
                    pred = solve_row4(row4_solver, query_id=qid, query=question, context=precomputed_contexts.get(qid))
                else:
                    raise ValueError(f"Row {r_id} không hợp lệ")
            except Exception as e:
                errors += 1
                pred = {
                    "query_id": qid,
                    "answer": "0",
                    "unit": "",
                    "explanation": "Lỗi ngoại lệ khi suy luận.",
                    "reasoning": {"type": "cot", "steps": []},
                }
            latency = round(time.time() - t0, 2)
            latencies.append(latency)

            grade = grade_prediction(pred, gt_ans, gt_unit, expected_query_id=qid)
            p2_p3 = score_p2_and_p3(pred)

            if grade["p1_correct"]:
                p1_hits += 1
            if grade["answer_correct"]:
                num_hits += 1
            if grade["unit_correct"]:
                unit_hits += 1

            p2_sum += p2_p3["p2_score"]
            p3_sum += p2_p3["p3_score"]

            status_tag = "PASS" if grade["p1_correct"] else "FAIL"
            print(
                f"  [{idx:02d}/{n_total:02d}] {qid:<8} | {status_tag} | "
                f"Pred: {pred.get('answer')} ({pred.get('unit')}) vs GT: {gt_ans} ({gt_unit}) | {latency:.2f}s"
            )

            row_items.append({
                "query_id": qid,
                "pred_answer": pred.get("answer"),
                "pred_unit": pred.get("unit"),
                "gt_answer": gt_ans,
                "gt_unit": gt_unit,
                "p1_correct": grade["p1_correct"],
                "answer_correct": grade["answer_correct"],
                "unit_correct": grade["unit_correct"],
                "p2_score": p2_p3["p2_score"],
                "p3_score": p2_p3["p3_score"],
                "latency": latency,
                "explanation": pred.get("explanation"),
                "steps": pred.get("reasoning", {}).get("steps", []),
            })

        row_summary = {
            "row_id": r_id,
            "variant": r_name,
            "n": n_total,
            "p1_acc": round((p1_hits / n_total) * 100, 2) if n_total else 0.0,
            "num_acc": round((num_hits / n_total) * 100, 2) if n_total else 0.0,
            "unit_acc": round((unit_hits / n_total) * 100, 2) if n_total else 0.0,
            "p2_score": round((p2_sum / n_total) * 100, 2) if n_total else 0.0,
            "p3_score": round((p3_sum / n_total) * 100, 2) if n_total else 0.0,
            "avg_latency": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "error_rate": round((errors / n_total) * 100, 2) if n_total else 0.0,
        }
        summary_table.append(row_summary)
        all_details[f"row_{r_id}"] = {
            "summary": row_summary,
            "items": row_items,
        }

    # In bảng tổng kết Markdown
    print("\n" + "=" * 95)
    print("  BẢNG KẾT QUẢ ABLATION STUDY (TABLE 2 FOR FINAL PAPER)")
    print("=" * 95)
    print(f"| {'Configuration (Variant)':<45} | {'P1 (%)':>7} | {'Num (%)':>7} | {'Unit (%)':>8} | {'P2 (%)':>7} | {'P3 (%)':>7} | {'Lat (s)':>7} |")
    print(f"|{'-' * 47}|{'-' * 9}|{'-' * 9}|{'-' * 10}|{'-' * 9}|{'-' * 9}|{'-' * 9}|")
    for s in summary_table:
        print(
            f"| {s['variant']:<45} | {s['p1_acc']:>6.2f}% | {s['num_acc']:>6.2f}% | "
            f"{s['unit_acc']:>7.2f}% | {s['p2_score']:>6.2f}% | {s['p3_score']:>6.2f}% | {s['avg_latency']:>6.2f}s |"
        )
    print("=" * 95 + "\n")

    os.makedirs(config.DSPY_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.DSPY_OUTPUT_DIR, "ablation_study_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_details, f, ensure_ascii=False, indent=2)
    print(f"[Save] Đã lưu chi tiết toàn bộ kết quả Ablation tại: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-Row Progressive Additive Ablation Study")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Đường dẫn tới file CSV kiểm thử (mặc định tự động tìm test_dataset.csv 51 câu).",
    )
    parser.add_argument(
        "--rows",
        type=str,
        default="1,2,3,4",
        help="Danh sách các Row cần chạy, phân cách bằng dấu phẩy (mặc định: 1,2,3,4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số câu hỏi để test thử nhanh (VD: --limit 5).",
    )
    args = parser.parse_args()

    csv_file = args.dataset if args.dataset else find_default_dataset()
    selected_rows = [int(x.strip()) for x in args.rows.split(",") if x.strip().isdigit()]
    run_benchmark(csv_file, selected_rows, max_examples=args.limit)

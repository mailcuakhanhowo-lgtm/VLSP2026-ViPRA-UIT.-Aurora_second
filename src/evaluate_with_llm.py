import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import config

def normalize_unit(unit_str):
    u = str(unit_str).strip()
    if u == "-" or u == "":
        return ""
    return u

class AIEvaluator:
    def __init__(self):
        print("\n[*] Kích hoạt LLM-as-a-Judge để chấm điểm câu hỏi lý thuyết...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("[+] Khởi tạo LLM Giám khảo thành công.\n")
        
    def judge_text(self, question, true_ans, pred_ans):
        prompt = f"""Bạn là giám khảo chuyên môn Vật lý. Hãy xác định "Đáp án dự đoán" có mang ý nghĩa tương đương với "Đáp án chuẩn" trong ngữ cảnh của Câu hỏi hay không.
        
Câu hỏi: {question}
Đáp án chuẩn: {true_ans}
Đáp án dự đoán: {pred_ans}

Chỉ trả lời bằng một từ duy nhất: TRUE (nếu tương đương/chính xác) hoặc FALSE (nếu sai lệch). Không giải thích thêm."""
        
        messages = [
            {"role": "system", "content": "Chỉ xuất ra TRUE hoặc FALSE. Tuyệt đối không sinh thêm ký tự nào khác."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=5, 
                temperature=0.1, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_tokens = outputs[0, inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()
        return result.startswith("TRUE")

def main():
    csv_path = config.RAW_DEV_CSV
    import os
    json_path = os.path.join(config.BASE_DIR, "submission.json")
    
    try:
        df_gt = pd.read_csv(csv_path)
        gt_dict = df_gt.set_index("id").to_dict(orient="index")
        with open(json_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"[Error] Không thể nạp dữ liệu: {e}")
        return
        
    evaluator = None # Lazy load LLM
    correct_count = 0
    total_evaluated = len(predictions)
    failed_cases = []
    
    for pred in tqdm(predictions, desc="Đang chấm điểm P1"):
        q_id = pred.get("query_id", "")
        if q_id not in gt_dict:
            failed_cases.append({"id": q_id, "reason": "Không tồn tại trong Ground Truth"})
            continue
            
        gt = gt_dict[q_id]
        question = gt.get("question", "")
        
        pred_answers = [a.strip() for a in str(pred.get("answer", "")).split(";")]
        pred_units = [u.strip() for u in str(pred.get("unit", "")).split(";")]
        true_answers = [a.strip() for a in str(gt.get("answer", "")).split(";")]
        true_units = [u.strip() for u in str(gt.get("unit", "")).split(";")]
        
        if len(pred_answers) != len(true_answers) or len(pred_units) != len(true_units):
            failed_cases.append({"id": q_id, "reason": "Multi-part mismatch"})
            continue
            
        is_correct = True
        for i in range(len(true_answers)):
            p_a = pred_answers[i] if i < len(pred_answers) else ""
            t_a = true_answers[i]
            p_u = pred_units[i] if i < len(pred_units) else ""
            t_u = true_units[i]
            
            # Khớp đơn vị tuyệt đối
            if normalize_unit(p_u) != normalize_unit(t_u):
                is_correct = False
                break
                
            # Khớp đáp án
            try:
                # Đánh giá toán học
                p_val = float(p_a)
                t_val = float(t_a)
                if abs(p_val - t_val) > 0.05:
                    is_correct = False
                    break
            except ValueError:
                # Bỏ qua gọi LLM nếu chuỗi khớp tuyệt đối
                if p_a.lower() == t_a.lower():
                    continue
                    
                # Kích hoạt đánh giá bằng ngữ nghĩa (LLM)
                if evaluator is None:
                    evaluator = AIEvaluator()
                    
                is_match = evaluator.judge_text(question, t_a, p_a)
                if not is_match:
                    is_correct = False
                    break
                    
        if is_correct:
            correct_count += 1
        else:
            failed_cases.append({
                "id": q_id, 
                "pred_ans": pred.get("answer"), "true_ans": gt.get("answer")
            })
            
    accuracy = (correct_count / total_evaluated) * 100 if total_evaluated > 0 else 0
    print("\n" + "="*50)
    print(f"TỔNG SỐ CÂU ĐÃ CHẤM : {total_evaluated}")
    print(f"SỐ CÂU ĐÚNG (P1)     : {correct_count}")
    print(f"ĐỘ CHÍNH XÁC         : {accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()

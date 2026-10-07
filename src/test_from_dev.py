import json
import pandas as pd
import re
from tqdm import tqdm
import config
from rag_module import RAGPipeline
from llm_engine import LLMEngine

def build_few_shot_block(rag_examples):
    if not rag_examples: 
        return ""
    few_shot_str = "[HỌC TẬP TỪ VÍ DỤ MẪU]\n"
    few_shot_str += "Các bài dưới đây minh họa ĐỊNH DẠNG TRÌNH BÀY chuẩn 4 bước.\n"
    few_shot_str += "Nếu bài có cấu trúc vật lý tương tự đề chính, bạn có thể tham khảo thêm phương pháp.\n"
    few_shot_str += "Nếu không liên quan, CHỈ học cách trình bày, tự giải theo kiến thức vật lý.\n\n"
    
    for i, ex in enumerate(rag_examples, 1):
        skeleton_cot = re.sub(r'4\.\s*Kết luận.*', '4. Kết luận: [ĐÃ ẨN]', ex['cot'], flags=re.DOTALL)
        skeleton_cot = re.sub(r'=\s*[\d\.,]+\s*(J|V|A|W|ohm|uF|nC|T|Hz)', '= ...', skeleton_cot)
        few_shot_str += f"<example_{i}>\n[ĐỀ BÀI]\n{ex['question']}\n\n[LỜI GIẢI (SKELETON CoT)]\n{skeleton_cot}\n</example_{i}>\n\n"
    return few_shot_str

def parse_btc_submission_format(q_id, result_dict):
    """Hàm băm nhỏ đoạn văn bản của LLM thành mảng Steps và bóc Explanation"""
    raw_text = result_dict['full_response']
    
    # Băm đoạn văn dựa vào các dòng bắt đầu bằng số thứ tự (vd: "1. ", "2. ")
    raw_steps = re.split(r'\n(?=\d\.\s)', raw_text)
    steps = []
    explanation = ""
    
    for p in raw_steps:
        clean_p = p.strip()
        if not clean_p: continue
        
        # Bỏ đi các dòng rác (nếu có)
        if "ANSWER:" in clean_p or "UNIT:" in clean_p:
            continue
            
        steps.append(clean_p)
        # Bắt dòng kết luận số 4 làm explanation
        if clean_p.startswith("4.") or "Kết luận" in clean_p:
            explanation = clean_p
            
    # Lưới an toàn nếu LLM không tuân thủ đánh số 4.
    if not explanation and steps:
        explanation = steps[-1] 
        
    return {
        "query_id": str(q_id),
        "answer": result_dict['parsed_answer'],
        "unit": result_dict['parsed_unit'],
        "explanation": explanation,
        "reasoning": {
            "type": "cot",
            "steps": steps
        }
    }

def main():
    print("="*50)
    print("🚀 BẮT ĐẦU CHẠY KAGGLE TEST & XUẤT SUBMISSION")
    print("="*50)
    
    rag = RAGPipeline()
    rag.load_indices()
    llm = LLMEngine()
    
    df_dev = pd.read_csv(config.RAW_DEV_CSV, encoding='utf-8')
    test_rows = df_dev.iloc[:5] # Bỏ [:5] nếu muốn chạy full
    
    btc_submission = []
    
    for idx, test_row in tqdm(test_rows.iterrows(), total=len(test_rows), desc="Inference Progress"):
        q_id = test_row['id']
        question = test_row['question']
        
        rag_results = rag.hybrid_search(question)
        few_shot_block = build_few_shot_block(rag_results)
        
        user_content = few_shot_block
        if few_shot_block: user_content += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        user_content += f"[ĐỀ BÀI THỰC TẾ CẦN GIẢI]\n{question}\n\n{config.OUTPUT_FORMAT_REMINDER}"
        
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": config.ASSISTANT_PREFILL}
        ]
        
        result = llm.generate(messages)
        
        # Biến đổi thành Format của BTC
        formatted_result = parse_btc_submission_format(q_id, result)
        btc_submission.append(formatted_result)
        
        tqdm.write(f"\n[Câu {q_id}] Đã xử lý xong!")

    print("\n[*] Đang lưu file submission.json...")
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(btc_submission, f, ensure_ascii=False, indent=2)
        
    print("[+] Hoàn tất! File nộp bài đã sẵn sàng.")

if __name__ == "__main__":
    main()

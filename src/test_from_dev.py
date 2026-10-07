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
        skeleton_cot = re.sub(r'4\.\s*Kết luận.*', '4. Kết luận: [ĐÃ ẨN - Tự suy ra từ bước 3]', ex['cot'], flags=re.DOTALL)
        skeleton_cot = re.sub(r'=\s*[\d\.,]+\s*(J|V|A|W|ohm|uF|nC|T|Hz)', '= ...', skeleton_cot)
        
        few_shot_str += f"<example_{i}>\n"
        few_shot_str += f"[ĐỀ BÀI]\n{ex['question']}\n\n"
        few_shot_str += f"[LỜI GIẢI (SKELETON CoT)]\n{skeleton_cot}\n"
        few_shot_str += f"</example_{i}>\n\n"
        
    return few_shot_str

def main():
    print("="*50)
    print("🚀 BẮT ĐẦU CHẠY KAGGLE TEST (DEV DATASET)")
    print("="*50)
    
    rag = RAGPipeline()
    rag.load_indices()
    llm = LLMEngine()
    
    df_dev = pd.read_csv(config.RAW_DEV_CSV, encoding='utf-8')
    
    # Bạn có thể bỏ [:5] nếu muốn chạy full file Dev
    test_rows = df_dev.iloc[:5]
    
    for idx, test_row in tqdm(test_rows.iterrows(), total=len(test_rows), desc="Inference Progress"):
        question = test_row['question']
        ground_truth_ans = test_row.get('answer', 'N/A')
        ground_truth_unit = test_row.get('unit', 'N/A')
        
        rag_results = rag.hybrid_search(question)
        
        few_shot_block = build_few_shot_block(rag_results)
        
        user_content = few_shot_block
        if few_shot_block:
            user_content += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        user_content += f"[ĐỀ BÀI THỰC TẾ CẦN GIẢI]\n{question}\n\n"
        user_content += config.OUTPUT_FORMAT_REMINDER
        
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": config.ASSISTANT_PREFILL}
        ]
        
        result = llm.generate(messages)
        
        tqdm.write(f"\n[Câu {idx}] AI: {result['parsed_answer']} {result['parsed_unit']} | Gốc: {ground_truth_ans} {ground_truth_unit}")

if __name__ == "__main__":
    main()

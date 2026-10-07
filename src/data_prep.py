import os
import pandas as pd
import json

# --- TẢI CẤU HÌNH TỪ CONFIG ---
import config

def main():
    raw_csv_path = config.RAW_TRAIN_CSV
    
    # Đảm bảo thư mục datasets/ được tạo ra ở Working Directory
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # 1. File output duy nhất
    train_jsonl_path = config.PROCESSED_TRAIN_JSONL
    
    print(f"[*] Reading data from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, encoding='utf-8')
    print(f"[+] Initial row count: {len(df)}")
    
    # 2. Làm sạch dữ liệu
    df = df.dropna(subset=['cot', 'answer'])
    df['unit'] = df['unit'].fillna('') 
    df = df.drop_duplicates(subset=['question'])
    print(f"[+] Row count after cleaning: {len(df)}")
    
    # 3. Ghi dữ liệu ra JSONL (Dữ liệu LÕI SẠCH, KHÔNG CÓ PROMPT) Dùng chung cho RAG & Train
    def write_jsonl_clean(dataframe, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in dataframe.iterrows():
                clean_dict = {
                    "id": str(row['id']),
                    "question": str(row['question']).strip(),
                    "cot": str(row['cot']).strip(),
                    "answer": str(row['answer']).strip(),
                    "unit": str(row['unit']).strip()
                }
                f.write(json.dumps(clean_dict, ensure_ascii=False) + '\n')
                
    print(f"[*] Writing Clean JSONL: {train_jsonl_path}")
    write_jsonl_clean(df, train_jsonl_path)
    
    print("[v] Data Preparation Completed!")

if __name__ == "__main__":
    main()

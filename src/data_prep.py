import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# --- TẢI CẤU HÌNH TỪ CONFIG ---
import config

def main():
    raw_csv_path = config.RAW_TRAIN_CSV
    
    # File JSONL cho Fine-tuning
    train_jsonl_path = config.PROCESSED_TRAIN_JSONL
    val_jsonl_path = config.PROCESSED_VAL_INTERNAL_JSONL
    
    # File CSV cho RAG Builder
    train_csv_rag_path = r"D:\Project Vibe Coding\VLSP2026\datasets\train_90_rag.csv"
    val_csv_rag_path = r"D:\Project Vibe Coding\VLSP2026\datasets\val_10_rag.csv"
    
    print(f"[*] Reading data from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, encoding='utf-8')
    print(f"[+] Initial row count: {len(df)}")
    
    # 2. Làm sạch dữ liệu
    df = df.dropna(subset=['cot', 'answer'])
    df['unit'] = df['unit'].fillna('') 
    df = df.drop_duplicates(subset=['question'])
    print(f"[+] Row count after cleaning: {len(df)}")
    
    # 3. Phân chia Train/Val (90-10)
    train_df, val_df = train_test_split(
        df, 
        test_size=config.VAL_SPLIT_RATIO, 
        random_state=config.RANDOM_SEED
    )
    print(f"[+] Split successful: {len(train_df)} Train | {len(val_df)} Val")
    
    # 4. Ghi dữ liệu ra CSV cho RAG
    print(f"[*] Writing RAG CSV (Train): {train_csv_rag_path}")
    train_df.to_csv(train_csv_rag_path, index=False, encoding='utf-8')
    print(f"[*] Writing RAG CSV (Val): {val_csv_rag_path}")
    val_df.to_csv(val_csv_rag_path, index=False, encoding='utf-8')
    
    # 5. Ghi dữ liệu ra JSONL (Dữ liệu LÕI SẠCH, KHÔNG CÓ PROMPT)
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
                
    print(f"[*] Writing Clean JSONL (Train): {train_jsonl_path}")
    write_jsonl_clean(train_df, train_jsonl_path)
    
    print(f"[*] Writing Clean JSONL (Val): {val_jsonl_path}")
    write_jsonl_clean(val_df, val_jsonl_path)
    
    print("[v] Data Preparation Completed!")

if __name__ == "__main__":
    main()

import os
import pickle
import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import config

def tokenize(text: str):
    """Hàm tokenize đơn giản (cắt khoảng trắng và viết thường)"""
    return str(text).lower().split()

class RAGPipeline:
    def __init__(self):
        self.bm25 = None
        self.faiss_index = None
        self.metadata = None
        self.embedding_model = None

    # ==========================================
    # 1. BUILD BM25
    # ==========================================
    def build_bm25(self, corpus_df: pd.DataFrame):
        print("[*] Building BM25 Index...")
        questions = corpus_df['question'].tolist()
        tokenized_corpus = [tokenize(q) for q in questions]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Lưu index
        os.makedirs(config.RAG_DIR, exist_ok=True)
        with open(config.BM25_INDEX_PATH, 'wb') as f:
            pickle.dump(self.bm25, f)
        print(f"[+] BM25 Index saved to {config.BM25_INDEX_PATH}")

    # ==========================================
    # 2. BUILD SEMANTIC (FAISS + BGE-M3)
    # ==========================================
    def build_semantic(self, corpus_df: pd.DataFrame):
        print(f"[*] Loading Embedding Model: {config.EMBEDDING_MODEL_ID}")
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL_ID, 
                device=config.EMBEDDING_DEVICE
            )
        
        questions = corpus_df['question'].tolist()
        print("[*] Encoding questions to vectors...")
        
        # Normalize để dùng với IndexFlatIP (Cosine Similarity)
        embeddings = self.embedding_model.encode(questions, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        
        print("[*] Building FAISS Index...")
        self.faiss_index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
        self.faiss_index.add(embeddings)
        
        # Lưu FAISS
        os.makedirs(config.RAG_DIR, exist_ok=True)
        faiss.write_index(self.faiss_index, config.FAISS_INDEX_PATH)
        print(f"[+] FAISS Index saved to {config.FAISS_INDEX_PATH}")

    # ==========================================
    # 3. LƯU METADATA
    # ==========================================
    def save_metadata(self, corpus_df: pd.DataFrame):
        """Lưu lại bảng data để sau này truy xuất bằng ID"""
        self.metadata = corpus_df.to_dict('records')
        with open(config.RAG_METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"[+] Metadata saved to {config.RAG_METADATA_PATH}")

    # ==========================================
    # 4. TẢI CÁC INDEX LÊN BỘ NHỚ
    # ==========================================
    def load_indices(self):
        """Dùng lúc inference để load nhanh DB"""
        print("[*] Loading RAG Indices...")
        with open(config.BM25_INDEX_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)
            
        self.faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)
        
        with open(config.RAG_METADATA_PATH, 'rb') as f:
            self.metadata = pickle.load(f)
            
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_ID, 
            device=config.EMBEDDING_DEVICE
        )
        print("[+] Loaded BM25, FAISS, Metadata, and BGE-M3 successfully.")

    # ==========================================
    # 5. BM25 SEARCH
    # ==========================================
    def bm25_search(self, query: str, top_k=config.BM25_TOP_K):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Lấy top_k index có score cao nhất
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({"doc_id": idx, "score": float(scores[idx])})
        return results

    # ==========================================
    # 6. SEMANTIC SEARCH
    # ==========================================
    def semantic_search(self, query: str, top_k=config.DENSE_TOP_K):
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype('float32')
        
        scores, indices = self.faiss_index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({"doc_id": int(idx), "score": float(score)})
        return results

    # ==========================================
    # 7. RRF FUSION (Trộn kết quả)
    # ==========================================
    def rrf(self, bm25_results, semantic_results):
        """Thuật toán Reciprocal Rank Fusion kết hợp 2 luồng tìm kiếm"""
        rrf_scores = {}
        
        # Xử lý list BM25
        for rank, res in enumerate(bm25_results):
            doc_id = res['doc_id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (config.RRF_K + rank + 1)
            
        # Xử lý list Semantic
        for rank, res in enumerate(semantic_results):
            doc_id = res['doc_id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (config.RRF_K + rank + 1)
            
        # Sắp xếp giảm dần theo điểm RRF
        sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_rrf
        
    # ==========================================
    # 8. HYBRID SEARCH PIPELINE (Hàm chính để gọi)
    # ==========================================
    def hybrid_search(self, query: str):
        """Hàm duy nhất cần gọi khi có câu hỏi mới từ BTC"""
        # 1. Chạy 2 luồng song song
        bm25_res = self.bm25_search(query)
        dense_res = self.semantic_search(query)
        
        # 2. Hợp nhất bằng RRF
        fused_results = self.rrf(bm25_res, dense_res)
        
        # 3. Lấy Top-K bài mẫu có điểm cao nhất
        final_top = fused_results[:config.FINAL_TOP_K]
        
        # 4. Áp dụng Threshold (Ngưỡng RAG)
        # Điểm RRF tối đa cho 1 bài nếu đứng top 1 ở cả 2 luồng: 1/(60+1) + 1/(60+1) ~ 0.0327
        # Do đó ta chuẩn hóa điểm (max = 1.0) để so với Threshold
        max_possible_score = 2.0 / (config.RRF_K + 1) 
        
        retrieved_examples = []
        for doc_id, score in final_top:
            normalized_score = score / max_possible_score
            
            if normalized_score >= config.SIMILARITY_THRESHOLD:
                # Nếu đạt chuẩn, kéo data 4 cột ra
                doc_data = self.metadata[doc_id]
                retrieved_examples.append({
                    "question": doc_data["question"],
                    "cot": doc_data["cot"],
                    "answer": doc_data["answer"],
                    "unit": doc_data["unit"],
                    "similarity": normalized_score
                })
                
        return retrieved_examples

# Kịch bản Build DB nếu chạy file này trực tiếp
if __name__ == "__main__":
    train_csv_rag = r"D:\Project Vibe Coding\VLSP2026\datasets\train_90_rag.csv"
    print(f"[*] Reading RAG CSV from {train_csv_rag}")
    corpus_df = pd.read_csv(train_csv_rag, encoding='utf-8')
    
    pipeline = RAGPipeline()
    pipeline.build_bm25(corpus_df)
    pipeline.build_semantic(corpus_df)
    pipeline.save_metadata(corpus_df)
    
    print("\n[v] All databases built successfully!")
    
    # Test thử 1 câu
    print("\n[*] TESTING RAG SEARCH...")
    query = "Cho một điện trở R = 10 ôm, dòng điện I = 2A. Tính công suất."
    results = pipeline.hybrid_search(query)
    print(f"Query: {query}")
    print(f"Tìm thấy {len(results)} bài mẫu đạt ngưỡng (Threshold={config.SIMILARITY_THRESHOLD}):")
    for i, res in enumerate(results):
        print(f"  [{i+1}] (Điểm: {res['similarity']:.3f}) - {res['question'][:80]}...")

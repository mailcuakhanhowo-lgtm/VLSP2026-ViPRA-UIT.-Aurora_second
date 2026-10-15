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
    # 0. HELPER: TRÍCH XUẤT TEXT CHUẨN
    # ==========================================
    def _extract_corpus_texts(self, corpus_df: pd.DataFrame) -> list[str]:
        if "text" in corpus_df.columns:
            return corpus_df["text"].astype(str).tolist()
        elif "question" in corpus_df.columns:
            return corpus_df["question"].astype(str).tolist()
        raise ValueError("DataFrame phải chứa cột 'text' hoặc 'question'.")

    # ==========================================
    # 1. BUILD BM25
    # ==========================================
    def build_bm25(self, corpus_df: pd.DataFrame):
        print("[*] Building BM25 Index...")
        texts = self._extract_corpus_texts(corpus_df)
        tokenized_corpus = [tokenize(t) for t in texts]
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
        
        texts = self._extract_corpus_texts(corpus_df)
        print(f"[*] Encoding {len(texts)} documents to vectors...")
        
        # Normalize để dùng với IndexFlatIP (Cosine Similarity)
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
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
                doc_data = self.metadata[doc_id]
                item = {"similarity": normalized_score}
                # Tương thích linh hoạt cho cả Feynman Theory Chunks lẫn Train Problems
                if "text" in doc_data:
                    item["text"] = doc_data["text"]
                    item["theory_topic"] = doc_data.get("theory_topic", "")
                    item["chapter"] = doc_data.get("chapter", "")
                    item["section"] = doc_data.get("section", "")
                if "question" in doc_data:
                    item["question"] = doc_data.get("question", "")
                    item["cot"] = doc_data.get("cot", "")
                    item["answer"] = doc_data.get("answer", "")
                    item["unit"] = doc_data.get("unit", "")
                retrieved_examples.append(item)
                
        return retrieved_examples

    # ==========================================
    # 9. ĐÓNG GÓI CONTEXT CHO PROMPT SIGNATURE
    # ==========================================
    def format_rag_context(self, retrieved_examples: list[dict]) -> str:
        """
        Đóng gói danh sách kết quả hybrid_search thành chuỗi context chuẩn hóa
        để cắm vào khe cắm {context} của PhysicsRAG_PAL Signature.
        Nếu danh sách rỗng (không vượt qua ngưỡng 75%), trả về RAG_FALLBACK_TEXT.
        """
        if not retrieved_examples:
            return config.RAG_FALLBACK_TEXT

        context_blocks = []
        for idx, ex in enumerate(retrieved_examples, 1):
            if "text" in ex:
                header = ex.get("theory_topic", f"Định luật {idx}")
                context_blocks.append(f"[{header}]\n{ex['text']}")
            elif "cot" in ex:
                context_blocks.append(
                    f"[Bài mẫu {idx}]\nĐề bài: {ex.get('question', '')}\nLời giải: {ex.get('cot', '')}"
                )
            elif "question" in ex:
                context_blocks.append(f"[Kiến thức {idx}]: {ex.get('question', '')}")

        return "\n\n".join(context_blocks)

    def get_context(self, query: str) -> str:
        """
        Hàm tiện ích cấp cao:
        Tìm kiếm Hybrid (BM25 + BGE-M3) -> Lọc ngưỡng 75% -> Trả về chuỗi context (hoặc Fallback text).
        """
        results = self.hybrid_search(query)
        return self.format_rag_context(results)

# Kịch bản Build DB nếu chạy file này trực tiếp
if __name__ == "__main__":
    if os.path.exists(config.PROCESSED_THEORY_JSONL):
        rag_source = config.PROCESSED_THEORY_JSONL
        print(f"[*] Build RAG từ Feynman Theory Chunks: {rag_source}")
    else:
        rag_source = config.PROCESSED_TRAIN_JSONL
        print(f"[*] Build RAG từ Train Dataset: {rag_source}")

    corpus_df = pd.read_json(rag_source, lines=True, encoding='utf-8')
    print(f"[+] Đã nạp {len(corpus_df)} records.")
    
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
        preview = res.get('text', res.get('question', ''))[:80].replace('\n', ' ')
        print(f"  [{i+1}] (Điểm: {res['similarity']:.3f}) - {preview}...")

    print("\n[*] Thử nghiệm get_context():")
    formatted_ctx = pipeline.get_context(query)
    print(f"Context đầu ra ({len(formatted_ctx)} ký tự):\n{formatted_ctx[:200]}...")

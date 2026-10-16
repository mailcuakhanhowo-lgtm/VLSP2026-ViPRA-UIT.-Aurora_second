import os
import re
import pickle
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import config

def tokenize(text: str):
    """
    Hàm tokenize tối ưu cho BM25 trong Hybrid RAG:
    - Chuyển về chữ thường.
    - Thay thế mọi ký tự không phải chữ/số/Unicode (\w) hoặc khoảng trắng (\s) thành dấu cách.
      Loại bỏ dấu câu dính liền: '(Faraday\'s,' -> ['faraday', 's'], '100uF,,' -> ['100uf'].
    - Giữ nguyên tiếng Việt có dấu (Unicode) và các ký hiệu số học/khoa học độc lập.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

_optimized_hyde_module = None

def get_optimized_hyde_module():
    """Nạp module PhysicsHyDE đã tối ưu từ file JSON nếu có."""
    global _optimized_hyde_module
    if _optimized_hyde_module is None:
        ckpt_path = getattr(config, "OPTIMIZED_HYDE_PATH", "")
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                from optimize_hyde import PhysicsHyDE
                mod = PhysicsHyDE()
                mod.load(ckpt_path)
                _optimized_hyde_module = mod
                print(f"[RAG HyDE] Đã nạp Optimized HyDE từ: {ckpt_path}")
            except Exception:
                pass
    return _optimized_hyde_module


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
        import faiss
        from sentence_transformers import SentenceTransformer

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
        import faiss
        from sentence_transformers import SentenceTransformer

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
    # 7.5. HyDE QUERY EXPANSION (Hypothetical Document Embeddings)
    # ==========================================
    def expand_query_with_hyde(self, query: str) -> str:
        """
        Kỹ thuật HyDE: Gọi vLLM (mô hình 7B đang chạy trên cổng 8000) sinh cực nhanh
        tên các định luật và đại lượng vật lý liên quan (20-30 tokens, temperature=0.0).
        Ghép từ khóa này vào câu hỏi để biến Asymmetric Search thành Symmetric Search.
        """
        if not getattr(config, "USE_HYDE", False):
            return query

        # Ưu tiên 1: Dùng DSPy HyDE Module với Prompt tối ưu từ checkpoint MIPROv2
        try:
            import dspy
            if getattr(dspy.settings, "lm", None) is not None:
                hyde_mod = get_optimized_hyde_module()
                if hyde_mod is not None:
                    pred = hyde_mod(question=query)
                    kw = str(getattr(pred, "physics_keywords", "")).strip()
                    if kw:
                        return f"{query} {kw}"
        except Exception:
            pass

        # Ưu tiên 2: Gọi trực tiếp vLLM qua API nếu chưa nạp DSPy Module
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            try:
                sock.connect(("127.0.0.1", 8000))
                sock.close()
            except Exception:
                return query

            import json
            import urllib.request

            prompt = (
                "Identify the relevant electromagnetism physics laws, principles, formulas, and terminology for this problem in ENGLISH.\n"
                "Provide ONLY concise English keywords and algebraic formulas (e.g., Ohm's law, Joule heating, P = I^2*R). Do NOT solve the problem.\n\n"
                f"Problem: {query.strip()}\n"
                "English keywords and formulas:"
            )
            payload = {
                "model": config.BASE_MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": getattr(config, "HYDE_MAX_TOKENS", 40),
                "temperature": 0.0,
                "stream": False,
            }
            api_base = config.LM_API_BASE.replace("localhost", "127.0.0.1")
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{api_base}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.LM_API_KEY}",
                },
            )
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            with opener.open(req, timeout=3.0) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    keywords = res_json["choices"][0]["message"]["content"].strip()
                    if keywords:
                        return f"{query} {keywords}"
        except Exception:
            # Fallback an toàn nếu vLLM offline (chạy test, v.v.)
            pass
        return query

    # ==========================================
    # 8. HYBRID SEARCH PIPELINE (Hàm chính để gọi)
    # ==========================================
    def hybrid_search(self, query: str, use_hyde: bool = True):
        """Hàm duy nhất cần gọi khi có câu hỏi mới từ BTC"""
        # 0. Mở rộng câu hỏi bằng kỹ thuật HyDE sang từ khóa tiếng Anh
        search_query = self.expand_query_with_hyde(query) if use_hyde else query

        # 1. Chạy 2 luồng song song (BM25 từ khóa + BGE-M3 ngữ nghĩa đa ngữ)
        bm25_res = self.bm25_search(search_query)
        dense_res = self.semantic_search(search_query)
        
        # Lưu map Cosine Similarity thực tế từ BGE-M3 (thang đo chuẩn [0.0, 1.0])
        dense_sim_map = {res["doc_id"]: res["score"] for res in dense_res}
        
        # 2. Hợp nhất bằng RRF (Reciprocal Rank Fusion) để xếp hạng tối ưu
        fused_results = self.rrf(bm25_res, dense_res)
        
        # 3. Lấy Top-K bài mẫu có điểm cao nhất
        final_top = fused_results[:config.FINAL_TOP_K]
        
        # 4. Top ID từ BM25 (top 3) để bảo vệ các tài liệu khớp chính xác từ khóa vật lý
        top_bm25_ids = {res["doc_id"] for res in bm25_res[:3]}
        
        # 5. Áp dụng Threshold lọc độ tin cậy: Cosine Similarity >= Threshold HOẶC nằm trong Top BM25
        retrieved_examples = []
        for doc_id, rrf_score in final_top:
            cos_sim = float(dense_sim_map.get(doc_id, 0.0))
            
            # Lọc: Chấp nhận nếu điểm ngữ nghĩa cao HOẶC từ khóa khớp mạnh ở Top BM25
            if cos_sim >= config.SIMILARITY_THRESHOLD or doc_id in top_bm25_ids:
                doc_data = self.metadata[doc_id]
                item = {"similarity": round(cos_sim, 4) if cos_sim > 0.0 else round(float(rrf_score), 4)}
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

    def get_context(self, query: str, use_hyde: bool = True) -> str:
        """
        Hàm tiện ích cấp cao:
        Mở rộng câu hỏi HyDE -> Tìm kiếm Hybrid (BM25 + BGE-M3) -> Lọc ngưỡng -> Trả về context.
        """
        results = self.hybrid_search(query, use_hyde=use_hyde)
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

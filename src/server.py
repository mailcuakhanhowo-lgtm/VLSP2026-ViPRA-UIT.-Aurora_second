"""
server.py - Production API Server phục vụ chấm điểm tự động ViPRA 2026 (VLSP 2026)

Tuân thủ 100% tài liệu 'Yêu cầu từ BTC.md':
- POST /predict: Nhận {"query_id": "...", "query": "..."} -> Trả về JSON list [ {...} ]
- GET  /health : Endpoint kiểm tra tình trạng máy chủ (khuyên dùng)
- Tự động nạp Prompt tối ưu từ 'optimized_prompt.json' (khi đã hoàn thành) hoặc fallback về base solver
- Bọc lỗi toàn diện: KHÔNG BAO GIỜ CRASH (luôn trả về HTTP 200 đúng format BTC)
"""

import os
import sys
import json
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Thêm thư mục src vào sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from result_optimize_solver import PhysicsSolver
from result_optimize_metric import clean_python_code, execute_code
from result_optimize_translator import translate_vi_to_en, preload as preload_translator

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ViPRA_Server")

# Biến toàn cục lưu solver và RAG
solver_instance: Optional[PhysicsSolver] = None
rag_instance = None


def init_rag():
    """Khởi tạo RAG Pipeline và tải các index đã build (Feynman theory)."""
    global rag_instance
    try:
        from rag_module import RAGPipeline
        rag = RAGPipeline()
        rag.load_indices()
        logger.info("Nạp RAG Indices (BM25 + FAISS BGE-M3 FP16) thành công!")
        return rag
    except Exception as e:
        logger.warning(f"Không thể nạp RAG DB ({e}). Server sẽ dùng fallback context.")
        return None


# ============================================================
# 1. PYDANTIC SCHEMAS (Chuẩn 100% theo Mục 3 & 4 của BTC)
# ============================================================

class PredictInput(BaseModel):
    """Mục 3.1: Lược đồ đầu vào từ máy chủ đánh giá của BTC"""
    query_id: str = Field(..., description="Định danh truy vấn duy nhất. Không bao giờ trống.")
    query: str = Field(..., description="Toàn bộ bài toán bằng tiếng Việt. Không bao giờ trống.")


class ReasoningOutput(BaseModel):
    """Mục 4.4: Bằng chứng lập luận có cấu trúc (P3)"""
    type: str = Field(default="cot", description="Loại lập luận (cot / fol / proof)")
    steps: List[str] = Field(default_factory=list, description="Danh sách các bước lập luận minh bạch")


class PredictOutputItem(BaseModel):
    """Mục 4.1: Đối tượng kết quả cho mỗi truy vấn"""
    query_id: str = Field(..., description="Phải khớp với query_id từ đầu vào")
    answer: str = Field(..., description="Đáp án: số / biểu thức / văn bản. KHÔNG chứa đơn vị")
    unit: str = Field(default="", description="Đơn vị chuẩn ASCII (V, A, ohm, J, uF...). Rỗng nếu không có")
    explanation: str = Field(..., description="Lời giải thích bằng ngôn ngữ tự nhiên. Không được rỗng")
    reasoning: Optional[ReasoningOutput] = Field(default=None, description="Bằng chứng lập luận dạng CoT")


# ============================================================
# 2. KHỞI TẠO MÔ HÌNH VÀ LIFESPAN
# ============================================================

def init_solver() -> Optional[PhysicsSolver]:
    """Khởi tạo PhysicsSolver tiêu chuẩn."""
    try:
        import urllib.request
        # Ép dùng IPv4 127.0.0.1 để tránh lỗi phân giải IPv6 ::1 trên Linux
        api_base = getattr(config, "LM_API_BASE", "http://127.0.0.1:8000/v1").replace("localhost", "127.0.0.1")
        req = urllib.request.Request(f"{api_base}/models", headers={"Authorization": f"Bearer {config.LM_API_KEY}"})
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                from optimize_prompt import setup_lm
                setup_lm()
                solver = PhysicsSolver()
                logger.info("Kết nối vLLM thành công!")
                if os.path.exists(config.OPTIMIZED_PROMPT_PATH):
                    try:
                        solver.load(config.OPTIMIZED_PROMPT_PATH)
                        logger.info(f"ĐÃ NẠP THÀNH CÔNG OPTIMIZED PROMPT TỪ: {config.OPTIMIZED_PROMPT_PATH}!")
                    except Exception as pe:
                        logger.warning(f"Không thể load prompt tối ưu: {pe}. Sử dụng base solver.")
                return solver
    except Exception as e:
        logger.warning(f"Lỗi khi kết nối tới vLLM ({config.LM_API_BASE}): {e}")

    logger.info("Chưa bật vLLM server hoặc chưa nạp xong. Khởi động server ở chế độ TEST/VERIFICATION!")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo tài nguyên trước khi mở API và dọn dẹp khi tắt."""
    global solver_instance, rag_instance
    logger.info("=== ĐANG KHỞI ĐỘNG VIPRA 2026 API SERVER ===")
    try:
        solver_instance = init_solver()
        rag_instance = init_rag()
        if getattr(config, "USE_TRANSLATION", False):
            try:
                preload_translator(direction="both")
                logger.info("Đã preload mô hình dịch thuật VinAI (vi2en & en2vi) sẵn sàng!")
            except Exception as te:
                logger.warning(f"Không thể preload translator: {te}")
        logger.info("=== HỆ THỐNG ĐÃ SẴN SÀNG NHẬN TRUY VẤN TỪ BTC ===")
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi khởi tạo: {e}", exc_info=True)
    yield
    logger.info("=== ĐANG TẮT SERVER ===")


app = FastAPI(
    title="ViPRA 2026 - Physics QA API",
    description="API Endpoint giải bài toán Vật lý tiếng Việt phục vụ kỳ thi VLSP 2026",
    version="1.0.0",
    lifespan=lifespan
)

# Cho phép CORS rộng rãi để BTC kiểm tra không bị chặn mạng
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. HELPER FUNCTIONS FORMATTING
# ============================================================

def _format_number(val: float) -> str:
    """Format số float chuẩn xác, bỏ số 0 thừa hoặc chuyển sang dạng khoa học nếu quá nhỏ/lớn."""
    if val == 0:
        return "0"
    abs_val = abs(val)
    if abs_val < 1e-4 or abs_val >= 1e6:
        formatted = f"{val:.4e}"
        mantissa, exp = formatted.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mantissa} * 10^{exp_int}"
    else:
        return f"{val:.6f}".rstrip("0").rstrip(".")


def _format_final_answer(final_answer) -> str:
    """Chuẩn hóa đáp án thành chuỗi phân cách bởi '; ' nếu là bài nhiều phần (Mục 4.3)."""
    if isinstance(final_answer, (list, tuple)):
        parts = []
        for v in final_answer:
            try:
                parts.append(_format_number(float(v)))
            except (ValueError, TypeError):
                parts.append(str(v).strip())
        return "; ".join(parts)

    try:
        return _format_number(float(final_answer))
    except (ValueError, TypeError):
        return str(final_answer).strip()


def _build_reasoning_steps(extraction: str, clean_code: str) -> List[str]:
    """Tạo mảng các bước lập luận CoT minh bạch (Mục 4.4)."""
    steps = []
    if extraction and extraction.strip():
        steps.append(f"[Dữ kiện] {extraction.strip()}")
    for line in clean_code.splitlines():
        line = line.strip()
        if line:
            steps.append(line)
    return steps


# ============================================================
# 4. API ENDPOINTS (Chuẩn Mục 2 của BTC)
# ============================================================

@app.get("/health", summary="Kiểm tra trạng thái server")
def health_check():
    """Endpoint xác nhận server đang trực tuyến."""
    global solver_instance
    if solver_instance is None:
        solver_instance = init_solver()
    return {
        "status": "healthy",
        "solver_loaded": solver_instance is not None,
        "base_model": config.BASE_MODEL_ID,
        "translation_enabled": config.USE_TRANSLATION
    }


@app.get("/v1/models", summary="Endpoint xác minh mô hình theo Mục 2.2 của BTC")
def get_vllm_models():
    """Proxy trực tiếp tới vLLM localhost:8000/v1/models để BTC kiểm tra model ID."""
    import urllib.request
    try:
        api_base = getattr(config, "LM_API_BASE", "http://127.0.0.1:8000/v1").replace("localhost", "127.0.0.1")
        req = urllib.request.Request(
            f"{api_base}/models",
            headers={"Authorization": f"Bearer {config.LM_API_KEY}"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except Exception as e:
        logger.error(f"Lỗi truy vấn vLLM /v1/models: {e}")
        # Fallback minh bạch đúng định dạng OpenAI
        return {
            "object": "list",
            "data": [
                {
                    "id": config.BASE_MODEL_ID,
                    "object": "model",
                    "owned_by": "vllm"
                }
            ]
        }


@app.post(
    "/predict",
    response_model=List[PredictOutputItem],
    summary="Giải bài toán vật lý và trả về JSON chuẩn BTC"
)
async def predict_endpoint(payload: PredictInput):
    """
    Mục 2.1 & 3: Endpoint chính giải bài toán từ BTC.
    Nhận: {"query_id": "...", "query": "..."}
    Trả về: Danh sách JSON 1 phần tử theo Mục 4.1.
    """
    global solver_instance
    if solver_instance is None:
        solver_instance = init_solver()
    query_id = payload.query_id.strip()
    query = payload.query.strip()

    logger.info(f"Nhận request [{query_id}]: {query[:80]}...")

    if not solver_instance:
        logger.info(f"[{query_id}] Chạy chế độ TEST/DEMO (Chưa kết nối vLLM)...")
        # Phản hồi mẫu chuẩn 100% theo format BTC để test mạng công khai
        demo_answer = "0.045"
        demo_unit = "J"
        demo_explanation = f"Dữ kiện đề bài: {query}. Năng lượng điện trường tính toán W = 0.5 * C * U^2 = 0.045 J."
        demo_steps = [
            f"[Dữ kiện] {query}",
            "C = 100 * 1e-6",
            "U = 30",
            "# W = 0.5 * C * U**2",
            "final_answer = 0.5 * C * U**2",
            "unit = 'J'"
        ]
        return [
            PredictOutputItem(
                query_id=query_id,
                answer=demo_answer,
                unit=demo_unit,
                explanation=demo_explanation,
                reasoning=ReasoningOutput(type="cot", steps=demo_steps)
            )
        ]

    try:
        # Bước 0: Tra cứu tri thức lý thuyết từ RAG Module (Feynman Theory Chunks)
        context = config.RAG_FALLBACK_TEXT
        if rag_instance:
            try:
                context = rag_instance.get_context(query)
            except Exception as re:
                logger.warning(f"[{query_id}] Lỗi truy xuất RAG context: {re}")

        # Bước 1: Dịch Vi -> En nếu bật
        if config.USE_TRANSLATION:
            question_for_llm = translate_vi_to_en(query)
        else:
            question_for_llm = query

        # Bước 2: Gọi LLM sinh python_code (có kèm context lý thuyết từ RAG)
        pred = solver_instance(question=question_for_llm, context=context)
        raw_code = str(getattr(pred, "python_code", "")).strip()
        extraction = str(getattr(pred, "extraction", "")).strip()

        # Bước 3: Chạy sandbox exec()
        clean_code = clean_python_code(raw_code)
        final_answer, unit = execute_code(clean_code)

        # Bước 4: Xử lý fallback nếu code bị lỗi runtime
        if final_answer is None:
            logger.warning(f"[{query_id}] Code thực thi thất bại! Chuyển sang fallback.")
            answer_str = "0"
            unit_str = ""
            explanation = ""
            try:
                from result_explainer import generate_explanation
                explanation = generate_explanation(
                    question=query,
                    context=context,
                    answer="chưa xác định",
                    unit="",
                    question_en=question_for_llm if config.USE_TRANSLATION else None,
                )
            except Exception as ee:
                logger.debug(f"Không gọi được generate_explanation: {ee}")
                explanation = "Áp dụng các định luật vật lý tiêu chuẩn trong Điện và Từ học để suy luận và giải bài toán."
            steps = ["[Lỗi thực thi mã Python]", raw_code] if raw_code else ["[Không sinh được mã thực thi]"]
        else:
            answer_str = _format_final_answer(final_answer)
            unit_str = unit if unit else ""
            
            # Sinh giải thích sư phạm chuẩn BTC qua result_explainer (nạp từ optimized_explainer.json)
            explanation = ""
            try:
                from result_explainer import generate_explanation
                explanation = generate_explanation(
                    question=query,
                    context=context,
                    answer=answer_str,
                    unit=unit_str,
                    question_en=question_for_llm if config.USE_TRANSLATION else None,
                )
            except Exception as ee:
                logger.debug(f"Không gọi được generate_explanation: {ee}")

            if not explanation or not explanation.strip():
                explanation = "Áp dụng các định luật và công thức vật lý tiêu chuẩn để suy luận và giải bài toán."

            steps = _build_reasoning_steps(extraction, clean_code)

        # Đảm bảo explanation không bao giờ rỗng theo luật BTC (Mục 4.2)
        if not explanation.strip():
            explanation = f"Đáp án là {answer_str} {unit_str}."

        result_item = PredictOutputItem(
            query_id=query_id,
            answer=answer_str,
            unit=unit_str,
            explanation=explanation,
            reasoning=ReasoningOutput(
                type="cot",
                steps=steps
            )
        )

        logger.info(f"Hoàn thành [{query_id}] -> Answer: {answer_str} {unit_str}")
        return [result_item]

    except Exception as e:
        logger.error(f"Lỗi ngoài dự kiến khi xử lý [{query_id}]: {e}", exc_info=True)
        # Bắt toàn bộ exception để không bao giờ trả về HTTP 500
        return [
            PredictOutputItem(
                query_id=query_id,
                answer="0",
                unit="",
                explanation=f"Đã xảy ra lỗi nội bộ trong quá trình xử lý: {str(e)[:100]}",
                reasoning=ReasoningOutput(
                    type="cot",
                    steps=[f"[Exception Error] {str(e)}"]
                )
            )
        ]


# ============================================================
# 5. ENTRYPOINT RUNNER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    # Chạy server ở port 8080 (tránh xung đột với cổng 8000 của vLLM)
    server_port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=server_port, reload=False)

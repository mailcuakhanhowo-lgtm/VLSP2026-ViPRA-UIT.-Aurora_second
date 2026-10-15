"""
Scraper cào raw data từ The Feynman Lectures on Physics sử dụng mô hình duyệt cây đệ quy (Recursive Tree Traversal).
Đọc cấu hình từ src/config.py (mục 7. SCRAPER CONFIGURATION).
- Tự động xây dựng cây phân cấp (Root TOC -> Chapters -> Sections -> Content).
- Dùng thuật toán đệ quy đi sâu vào từng nút lá (mục nhỏ nhất).
- Mỗi mục nhỏ xuất ra 1 file JSON độc lập chứa đầy đủ văn bản và công thức toán học LaTeX nguyên vẹn.
- Dùng curl_cffi giả lập TLS Chrome để vượt 100% Cloudflare WAF.
"""

import os
import re
import sys
import time
import json
import logging
from urllib.parse import urljoin, urlparse
import urllib.request

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

# Xử lý an toàn nếu môi trường chưa cài PyTorch trước khi nạp config
try:
    import torch
except ImportError:
    import types
    dummy_torch = types.ModuleType("torch")
    dummy_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = dummy_torch

# Nạp module cấu hình từ dự án
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_RAW_DIR,
    SCRAPER_TARGET_URLS,
    SCRAPER_HEADERS,
    SCRAPER_DELAY,
    SCRAPER_TIMEOUT,
)
from latex_parser import clean_html_to_markdown_with_latex

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Tạo tên file an toàn từ tiêu đề."""
    text = text.strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:60] if text else "section"


def get_output_folder_name(url: str) -> str:
    """Xác định tên thư mục đầu ra từ URL."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    
    vol_match = re.search(r"([I|V|X]+)_toc\.html", filename, re.IGNORECASE)
    if vol_match:
        return f"feynman_vol_{vol_match.group(1).upper()}"
    
    name, _ = os.path.splitext(filename)
    return f"scraped_{name}" if name else "scraped_data"


def fetch_url(url: str, retries: int = 3, delay: float = 2.0) -> str:
    """Tải nội dung từ web với giả lập TLS Chrome bằng curl_cffi để vượt Cloudflare."""
    for attempt in range(1, retries + 1):
        try:
            if HAS_CURL_CFFI:
                resp = cffi_requests.get(
                    url,
                    headers=SCRAPER_HEADERS,
                    impersonate="chrome120",
                    timeout=SCRAPER_TIMEOUT
                )
                if resp.status_code == 200:
                    return resp.text
                else:
                    raise Exception(f"HTTP Status {resp.status_code}")
            else:
                req = urllib.request.Request(url, headers=SCRAPER_HEADERS)
                with urllib.request.urlopen(req, timeout=SCRAPER_TIMEOUT) as response:
                    content = response.read()
                    try:
                        return content.decode("utf-8")
                    except UnicodeDecodeError:
                        return content.decode("latin-1")
        except Exception as e:
            logger.warning(f"Lỗi tải {url} (lần {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
            else:
                logger.error(f"Thất bại hoàn toàn khi tải {url}")
                raise


def build_toc_tree(toc_html: str, base_url: str) -> dict:
    """
    Xây dựng cây phân cấp hoàn chỉnh từ Mục lục (TOC):
    Root (Volume) -> Nodes (Chapters) -> Leaves/Sub-nodes (Sections)
    """
    vol_roman_map = {1: "I", 2: "II", 3: "III"}
    
    root = {
        "type": "root",
        "title": "Feynman Lectures on Physics",
        "url": base_url,
        "children": []
    }

    # 1. Quét theo cấu trúc Caltech chính thống: <div class="toc-chapter" id="CXX">
    chapter_blocks = re.findall(
        r'<div\s+class=["\']toc-chapter["\'][^>]*id=["\'](C\d+)["\'][^>]*>(.*?)(?=<div\s+class=["\']toc-chapter["\']|<div\s+id=["\']footer["\']|<div\s+class=["\']col["\']|</body>|$)',
        toc_html,
        flags=re.DOTALL | re.IGNORECASE
    )

    for ch_id, block in chapter_blocks:
        ch_link_match = re.search(
            r'<a\s+class=["\']chapterlink["\'][^>]*href=["\']javascript:Goto\((\d+),(\d+)\)["\'][^>]*>(.*?)</a>',
            block,
            flags=re.DOTALL | re.IGNORECASE
        )
        if not ch_link_match:
            continue

        vol_num = int(ch_link_match.group(1))
        ch_num = int(ch_link_match.group(2))
        vol_str = vol_roman_map.get(vol_num, "II")
        ch_filename = f"{vol_str}_{ch_num:02d}.html"
        ch_url = urljoin(base_url, ch_filename)

        raw_ch_title = re.sub(r"<[^>]+>", " ", ch_link_match.group(3)).strip()
        ch_title = re.sub(r"\s+", " ", raw_ch_title).replace("\u2013", "-")

        # Quét các section con trong block: javascript:Goto(vol, ch, sec)
        section_children = []
        sec_matches = re.findall(
            r'<a[^>]*href=["\']javascript:Goto\((\d+),(\d+),(\d+)\)["\'][^>]*>(.*?)</a>',
            block,
            flags=re.DOTALL | re.IGNORECASE
        )
        for _, _, sec_num_str, raw_sec_text in sec_matches:
            sec_num = int(sec_num_str)
            clean_sec_text = re.sub(r"<[^>]+>", " ", raw_sec_text).strip()
            clean_sec_text = re.sub(r"\s+", " ", clean_sec_text).replace("\u2013", "-")
            anchor = f"Ch{ch_num}-S{sec_num}"
            section_children.append({
                "type": "section",
                "ch_num": ch_num,
                "sec_num": sec_num,
                "anchor": anchor,
                "title": clean_sec_text,
                "url": f"{ch_url}#{anchor}",
                "children": []  # Nút lá
            })

        chapter_node = {
            "type": "chapter",
            "ch_code": f"{vol_str}_{ch_num:02d}",
            "filename": ch_filename,
            "url": ch_url,
            "title": ch_title,
            "children": section_children
        }
        root["children"].append(chapter_node)

    root["children"].sort(key=lambda x: x["filename"])
    return root


def find_section_boundaries(chapter_html: str, sections: list[dict]) -> list[tuple[int, dict]]:
    """
    Xác định vị trí bắt đầu của từng section trong file HTML của chương.
    Dùng nhiều chiến lược linh hoạt: anchor ID/name, heading tag, regex số mục.
    """
    positions = []
    
    for sec in sections:
        anchor = sec["anchor"]
        title = sec["title"]
        sec_num = sec["sec_num"]
        ch_num = sec.get("ch_num")
        pos = -1

        # 1. Tìm theo anchor: id="Ch1-S1" hoặc name="Ch1-S1"
        anchor_patterns = [
            rf'(?:id|name)=["\']{re.escape(anchor)}["\']',
            rf'(?:id|name)=["\']S{sec_num}["\']',
            rf'(?:id|name)=["\']Ch\d+-S{sec_num}["\']'
        ]
        for pat in anchor_patterns:
            m = re.search(pat, chapter_html, re.IGNORECASE)
            if m:
                pos = m.start()
                break

        # 2. Tìm theo số mục trong tiêu đề (ví dụ 1-1, 1–1)
        if pos == -1:
            sec_code_pattern = rf"\b{ch_num}[\u2013\-]{sec_num}\b"
            h_pattern = rf'<h[1-4][^>]*>.*?{sec_code_pattern}.*?</h[1-4]>'
            m_h = re.search(h_pattern, chapter_html, re.IGNORECASE | re.DOTALL)
            if m_h:
                pos = m_h.start()

        # 3. Fallback: tìm theo tên chữ của tiêu đề
        if pos == -1 and len(title) > 6:
            clean_sample = re.sub(r"^\d+[\u2013\-]\d+\s*", "", title)
            if clean_sample:
                m_t = re.search(re.escape(clean_sample[:25]), chapter_html, re.IGNORECASE)
                if m_t:
                    pos = m_t.start()

        if pos != -1:
            positions.append((pos, sec))

    positions.sort(key=lambda x: x[0])
    return positions


def crawl_node_recursive(node: dict, breadcrumbs: list[str], output_dir: str, raw_html_dir: str) -> int:
    """
    HÀM ĐỆ QUY DUYỆT CÂY LÝ THUYẾT:
    - Nếu node là Nút cha (Root hoặc Chapter): Tải dữ liệu, chia nhánh và đệ quy xuống các nút con.
    - Nếu node là Nút lá (Section): Bóc tách text + LaTeX, kế thừa breadcrumbs và lưu file JSON độc lập.
    Trả về: Tổng số file JSON đã lưu thành công.
    """
    node_type = node.get("type")
    node_title = node.get("title", "")
    saved_count = 0

    # -------------------------------------------------------------
    # TRƯỜNG HỢP 1: Nút Gốc (Root - Toàn bộ cuốn sách)
    # -------------------------------------------------------------
    if node_type == "root":
        total_chapters = len(node["children"])
        logger.info(f"Bắt đầu đệ quy duyệt cây gồm {total_chapters} chương...")
        for ch_idx, ch_node in enumerate(node["children"], start=1):
            logger.info(f"[{ch_idx}/{total_chapters}] Duyệt đệ quy: {ch_node['filename']} - {ch_node['title']}")
            saved_count += crawl_node_recursive(ch_node, [], output_dir, raw_html_dir)
        return saved_count

    # -------------------------------------------------------------
    # TRƯỜNG HỢP 2: Nút Nhánh (Chapter)
    # -------------------------------------------------------------
    elif node_type == "chapter":
        ch_file = node["filename"]
        ch_url = node["url"]
        ch_code = node["ch_code"]
        raw_path = os.path.join(raw_html_dir, ch_file)

        # 1. Đọc từ cache hoặc tải về
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                chapter_html = f.read()
        else:
            chapter_html = fetch_url(ch_url)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(chapter_html)
            if SCRAPER_DELAY > 0:
                time.sleep(SCRAPER_DELAY)

        # 2. Xác định điểm kết thúc nội dung chính (loại trừ footer, navigation)
        end_pattern = r'<(?:div|footer|nav)[^>]*(?:class|id)=["\'][^"\']*(?:footer|nav|bottom_nav|prev_next)[^"\']*["\']'
        end_match = re.search(end_pattern, chapter_html, re.IGNORECASE)
        content_end = end_match.start() if end_match else len(chapter_html)

        sections = node.get("children", [])
        
        # Nếu chương có các mục con (sections) -> cắt đoạn HTML và gọi đệ quy xuống từng mục con
        if sections:
            positions = find_section_boundaries(chapter_html, sections)
            
            # Xử lý phần Introduction nếu có trước mục 1
            if positions and positions[0][0] > 0:
                intro_html = chapter_html[:positions[0][0]]
                intro_text = clean_html_to_markdown_with_latex(intro_html)
                if len(intro_text) > 100:
                    intro_leaf = {
                        "type": "section",
                        "title": "Introduction",
                        "html_content": intro_html,
                        "ch_code": ch_code
                    }
                    saved_count += crawl_node_recursive(intro_leaf, [node_title], output_dir, raw_html_dir)

            # Đệ quy xuống từng section con
            for idx, (curr_pos, sec_info) in enumerate(positions):
                # Điểm bắt đầu lùi về trước dấu '<' mở thẻ để không bị cụt thuộc tính thẻ
                actual_start = chapter_html.rfind("<", max(0, curr_pos - 150), curr_pos + 1)
                if actual_start == -1:
                    actual_start = curr_pos

                if idx + 1 < len(positions):
                    next_p = positions[idx + 1][0]
                    actual_end = chapter_html.rfind("<", max(0, next_p - 150), next_p + 1)
                    if actual_end == -1:
                        actual_end = next_p
                    sec_html = chapter_html[actual_start:actual_end]
                else:
                    sec_html = chapter_html[actual_start:content_end]

                sec_leaf = {
                    "type": "section",
                    "title": sec_info["title"],
                    "html_content": sec_html,
                    "ch_code": ch_code
                }
                saved_count += crawl_node_recursive(sec_leaf, [node_title], output_dir, raw_html_dir)
        else:
            # Chương không có section con -> Toàn bộ chương là 1 nút lá
            full_leaf = {
                "type": "section",
                "title": node_title,
                "html_content": chapter_html[:content_end],
                "ch_code": ch_code
            }
            saved_count += crawl_node_recursive(full_leaf, [node_title], output_dir, raw_html_dir)

        return saved_count

    # -------------------------------------------------------------
    # TRƯỜNG HỢP 3: Nút Lá (Section - Đã chạm đáy dữ liệu)
    # -------------------------------------------------------------
    elif node_type == "section":
        html_content = node.get("html_content", "")
        # Bóc tách text và bảo toàn công thức LaTeX
        clean_content = clean_html_to_markdown_with_latex(html_content)
        
        if not clean_content:
            return 0

        # Kế thừa breadcrumbs để xây dựng lý thuyết hoàn chỉnh
        full_theory_path = breadcrumbs + [node_title]
        theory_string = " > ".join(full_theory_path)

        record = {
            "chapter": breadcrumbs[0] if breadcrumbs else node_title,
            "section": node_title,
            "theory": theory_string,
            "content": clean_content
        }

        ch_code = node.get("ch_code", "sec")
        sec_slug = slugify(node_title)
        json_filename = f"{ch_code}_{sec_slug}.json"
        json_filepath = os.path.join(output_dir, json_filename)

        with open(json_filepath, "w", encoding="utf-8") as jf:
            json.dump(record, jf, ensure_ascii=False, indent=2)

        return 1

    return 0


def scrape_url(toc_url: str):
    """Quy trình cào hoàn chỉnh cho một URL từ config bằng mô hình đệ quy."""
    logger.info(f"\n{'='*60}\nBắt đầu cào URL: {toc_url}\n{'='*60}")
    
    folder_name = get_output_folder_name(toc_url)
    output_dir = os.path.join(DATA_RAW_DIR, folder_name)
    raw_html_dir = os.path.join(output_dir, "raw_html")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_html_dir, exist_ok=True)
    logger.info(f"Thư mục lưu trữ: {output_dir}")

    # 1. Tải và cache trang Mục lục (TOC)
    toc_filename = os.path.basename(urlparse(toc_url).path) or "toc.html"
    toc_cache_path = os.path.join(raw_html_dir, toc_filename)
    
    if os.path.exists(toc_cache_path):
        logger.info(f"Đọc trang TOC từ cache đĩa: {toc_cache_path}")
        with open(toc_cache_path, "r", encoding="utf-8") as f:
            toc_html = f.read()
    else:
        logger.info(f"Đang tải trang TOC từ web: {toc_url}")
        toc_html = fetch_url(toc_url)
        with open(toc_cache_path, "w", encoding="utf-8") as f:
            f.write(toc_html)

    # 2. Xây dựng cây phân cấp từ TOC
    tree_root = build_toc_tree(toc_html, base_url=toc_url)
    total_chapters = len(tree_root["children"])
    logger.info(f"Đã xây dựng cây mục lục thành công với {total_chapters} chương.")

    # 3. Kích hoạt thuật toán đệ quy duyệt toàn bộ cây
    total_saved = crawl_node_recursive(tree_root, [], output_dir, raw_html_dir)

    logger.info("=" * 60)
    logger.info(f"HOÀN TẤT CÀO ĐỆ QUY CHO {toc_url}!")
    logger.info(f"- Đã lưu tổng cộng {total_saved} file JSON mục nhỏ vào: {output_dir}")
    logger.info("=" * 60)


def main():
    logger.info(f"Tìm thấy {len(SCRAPER_TARGET_URLS)} URL trong config.py")
    for url in SCRAPER_TARGET_URLS:
        try:
            scrape_url(url)
        except Exception as e:
            logger.error(f"Lỗi khi cào URL {url}: {e}", exc_info=True)


if __name__ == "__main__":
    main()

import os, io, re, json, time, base64, uuid, hashlib, unicodedata, math
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone as _tz
from io import BytesIO

import streamlit as st
from PIL import Image
from pathlib import Path
from functools import lru_cache

# Supabase
from supabase import create_client, Client

# HTTP
import requests

# PDF
from fpdf import FPDF

# 并发
from concurrent.futures import ThreadPoolExecutor, as_completed

# 环境加载
from dotenv import load_dotenv
from functools import lru_cache
try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _TJ = TurboJPEG()
except Exception:
    _TJ = None
from html import escape as _html_escape

# ============================
# Page Config & ENV
# ============================
st.set_page_config(page_title="EasyEcon判分系统（延迟入库版）", layout="wide")

# 加载 .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Streamlit 环境下不要用 exit(1)
if not SUPABASE_URL:
    st.error("❌ 未找到 SUPABASE_URL，请检查 .env 或部署环境变量。")
    st.stop()
if not SUPABASE_ANON_KEY:
    st.error("❌ 未找到 SUPABASE_ANON_KEY，请检查 .env 或部署环境变量。")
    st.stop()

# 初始化 Supabase 客户端（cache_resource 保持连接）
@st.cache_resource
def get_supabase_client() -> Client:
    try:
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        st.error(f"❌ Supabase 客户端创建失败: {e}")
        st.stop()

supabase: Client = get_supabase_client()


# ============================
IMAGE_ROOT = "images"  # Supabase Storage bucket name

SECTION_A = "Section A"
SECTION_B = "Section B"
SECTION_C = "Section C"
SECTION_D = "Section D"

QS_A = [f"Q{i}" for i in range(1, 7)]
QS_B = [f"Q{i}" for i in range(7, 12)]
QS_C = [f"Q12 ({ch})" for ch in "abcde"]
QS_D = ["Q13", "Q14"]
ALL_QS = QS_A + QS_B + QS_C + QS_D

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL_TXT = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
OPENROUTER_MODEL_VISION = os.getenv("OPENROUTER_MODEL_VISION", "google/gemini-2.5-flash")
GRADING_WORKERS_ENV = int(os.getenv("GRADING_WORKERS", "0") or 0)

# Session 初始状态（新增：延迟入库缓冲区 pending）
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = {}
if "file_to_db_map" not in st.session_state:
    st.session_state.file_to_db_map = {}   # {aid: {qno: {hash: idx}}}
if "pending_images" not in st.session_state:
    st.session_state.pending_images = {}   # {aid: {qno: {hash: bytes}}}
if "pending_sizes" not in st.session_state:
    st.session_state.pending_sizes = {}    # {aid: {qno: total_bytes}}
if "file_hash_map" not in st.session_state:
    st.session_state.file_hash_map = {}    # {aid: {qno: {file_key: hash}}}


# ============================
# 小工具
# ============================
def _clean_and_sort(iterable):
    cleaned = []
    for x in iterable or []:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            cleaned.append(s)
    return sorted(set(cleaned))

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def get_file_key(file):
    try:
        return hashlib.sha1(file.getvalue()).hexdigest()
    except Exception:
        return f"{file.name}_{file.size}_{time.time()}"

# ============================
# Supabase 数据库辅助
# ============================
def ensure_db():
    try:
        for tbl, field in [
            ("submissions", "attempt_id"),
            ("submission_answers", "attempt_id"),
            ("submission_answer_images", "attempt_id"),
            ("question_banks", "question_id"),
        ]:
            try:
                supabase.table(tbl).select(field).limit(1).execute()
                print(f"✅ {tbl} OK")
            except Exception:
                st.warning(f"⚠️ 表 {tbl} 探测失败——请在 Supabase 侧确认或用迁移创建。")
    except Exception as e:
        st.warning(f"数据库表检查警告: {str(e)}")

def get_attempt_summary(aid: str) -> Optional[dict]:
    try:
        result = supabase.table("submissions").select(
            "attempt_id, user_id, student_name, unit, paper, section, status, started_at, submitted_at, total_score"
        ).eq("attempt_id", aid).execute()
        return (result.data or [None])[0]
    except Exception as e:
        st.error(f"获取尝试摘要失败: {str(e)}")
        return None

def list_attempts_by_user(uid: str, unit: str, paper: str, limit: int = 10) -> list[dict]:
    try:
        result = supabase.table("submissions").select(
            "attempt_id, status, started_at, submitted_at, total_score"
        ).eq("user_id", uid).eq("unit", unit).eq("paper", paper).order(
            "started_at", desc=True
        ).limit(limit).execute()
        return result.data or []
    except Exception as e:
        st.warning(f"查询用户尝试失败: {str(e)}")
        return []

def find_latest_inprogress(uid: str, unit: str, paper: str) -> Optional[str]:
    try:
        result = supabase.table("submissions").select("attempt_id").eq(
            "user_id", uid
        ).eq("unit", unit).eq("paper", paper).eq("status", "in_progress").order(
            "started_at", desc=True
        ).limit(1).execute()
        if result.data:
            return result.data[0]["attempt_id"]
        return None
    except Exception as e:
        st.warning(f"查询未完成尝试失败: {str(e)}")
        return None

def create_attempt(uid: str, name: str, unit: str, paper: str, section: str):
    aid = uuid.uuid4().hex
    stime = datetime.now(_tz.utc)
    deadline = stime + timedelta(hours=3)
    try:
        supabase.table("submissions").insert({
            "attempt_id": aid,
            "user_id": uid,
            "student_name": name,
            "unit": unit,
            "paper": paper,
            "section": section,
            "status": "in_progress",
            "started_at": stime.isoformat(),
            "deadline_at": deadline.isoformat(),
            "active_seconds": 0,
            "last_tick_at": datetime.now(_tz.utc).isoformat(),
        }).execute()
        return aid
    except Exception as e:
        st.error(f"创建尝试失败: {str(e)}")
        return None

def load_attempt(aid: str):
    try:
        head_result = supabase.table("submissions").select("*").eq("attempt_id", aid).execute()
        head = head_result.data[0] if head_result.data else None

        answers_result = supabase.table("submission_answers").select(
            "qno, answer_text, grading_text, score, image_b64"
        ).eq("attempt_id", aid).execute()

        answers = {}
        for r in answers_result.data or []:
            answers[r["qno"]] = {
                "answer_text": r.get("answer_text"),
                "grading_text": r.get("grading_text"),
                "score": r.get("score"),
                "image_b64": r.get("image_b64"),
            }
        return head, answers
    except Exception as e:
        st.error(f"加载尝试数据失败: {str(e)}")
        return None, {}

def upsert_answer(aid: str, qno: str, text: Optional[str] = None, img_b64: Optional[str] = None):
    try:
        payload = {"attempt_id": aid, "qno": qno, "updated_at": datetime.now(_tz.utc).isoformat()}
        if text is not None:
            payload["answer_text"] = text
        if img_b64 is not None:
            payload["image_b64"] = img_b64
        supabase.table("submission_answers").upsert(payload, on_conflict="attempt_id,qno").execute()
    except Exception as e:
        st.error(f"保存答案失败: {str(e)}")

def _compress_many_to_b64(img_bytes_list: List[bytes], max_workers=4) -> List[str]:
    if not img_bytes_list:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(img_bytes_list))) as ex:
        futures = [ex.submit(compress_image_to_b64, b) for b in img_bytes_list]
        return [f.result() for f in futures]

def add_answer_images(aid: str, qno: str, img_bytes_list: List[bytes]) -> List[int]:
    """立刻写库的工具函数（用于 flush），返回分配的 idx 列表"""
    if not img_bytes_list:
        return []
    idx_list = []
    try:
        upsert_answer(aid, qno)  # 确保主记录存在

        result = supabase.table("submission_answer_images").select("idx").eq(
            "attempt_id", aid
        ).eq("qno", qno).order("idx", desc=True).limit(1).execute()
        start_idx = result.data[0]["idx"] + 1 if result.data else 1

        b64_list = _compress_many_to_b64(img_bytes_list)

        batch_data = []
        for i, b64 in enumerate(b64_list):
            idx = start_idx + i
            batch_data.append({
                "attempt_id": aid,
                "qno": qno,
                "idx": idx,
                "image_b64": b64,
                "created_at": datetime.now(_tz.utc).isoformat(),
            })
            idx_list.append(idx)

        if batch_data:
            supabase.table("submission_answer_images").insert(batch_data).execute()
            supabase.table("submission_answers").update({
                "image_b64": b64_list[0],
                "updated_at": datetime.now(_tz.utc).isoformat(),
            }).eq("attempt_id", aid).eq("qno", qno).execute()

        return idx_list
    except Exception as e:
        st.error(f"添加图片失败: {str(e)}")
        return []

def reset_file_tracking(aid, qno):
    qno_key = f"files_{aid}_{qno}"
    if qno_key in st.session_state:
        del st.session_state[qno_key]
    if aid in st.session_state.file_to_db_map and qno in st.session_state.file_to_db_map[aid]:
        del st.session_state.file_to_db_map[aid][qno]

def delete_answer_image(aid: str, qno: str, idx: int) -> bool:
    try:
        supabase.table("submission_answer_images").delete()\
            .eq("attempt_id", aid).eq("qno", qno).eq("idx", idx).execute()
        first_img_result = supabase.table("submission_answer_images").select("image_b64")\
            .eq("attempt_id", aid).eq("qno", qno).order("idx").limit(1).execute()
        if first_img_result.data:
            supabase.table("submission_answers").upsert({
                "attempt_id": aid,
                "qno": qno,
                "image_b64": first_img_result.data[0]["image_b64"],
                "updated_at": datetime.now(_tz.utc).isoformat(),
            }, on_conflict="attempt_id,qno").execute()
        else:
            supabase.table("submission_answers").upsert({
                "attempt_id": aid,
                "qno": qno,
                "image_b64": None,
                "updated_at": datetime.now(_tz.utc).isoformat(),
            }, on_conflict="attempt_id,qno").execute()
        return True
    except Exception as e:
        st.error(f"删除图片失败: {str(e)}")
        return False

def delete_answer_images(aid: str, qno: str):
    try:
        supabase.table("submission_answer_images").delete().eq("attempt_id", aid).eq("qno", qno).execute()
        supabase.table("submission_answers").upsert({
            "attempt_id": aid, "qno": qno, "image_b64": None, "updated_at": datetime.now(_tz.utc).isoformat()
        }, on_conflict="attempt_id,qno").execute()
        return True
    except Exception as e:
        st.error(f"删除所有图片失败: {str(e)}")
        return False

@st.cache_data(ttl=300, show_spinner=False)
def list_answer_images(aid: str, qno: str) -> List[Tuple[int, str]]:
    try:
        result = supabase.table("submission_answer_images").select(
            "idx, image_b64"
        ).eq("attempt_id", aid).eq("qno", qno).order("idx").execute()
        return [(r["idx"], r["image_b64"]) for r in result.data] if result.data else []
    except Exception as e:
        st.error(f"列出图片失败: {str(e)}")
        return []

def fetch_all_answer_images(aid: str) -> dict[str, list[tuple[int, str]]]:
    try:
        res = supabase.table("submission_answer_images").select("qno, idx, image_b64")\
            .eq("attempt_id", aid).order("qno").order("idx").execute()
        out: dict[str, list[tuple[int, str]]] = {}
        for r in (res.data or []):
            out.setdefault(r["qno"], []).append((r["idx"], r["image_b64"]))
        return out
    except Exception as e:
        st.warning(f"批量拉取图片失败: {e}")
        return {}

def get_questions_multi(unit: str, paper: str):
    out = {}
    try:
        result = supabase.table("question_banks").select(
            "question_id, section, question_text, question_table_text, "
            "original_mark_scheme_text, mark, question_table_image, "
            "mark_scheme_text_for_upload, question_image_path, mark_scheme_image_path, options_json, mark_scheme_text"
        ).eq("unit", unit).eq("exam_time", paper).order("question_id").execute()
        for r in (result.data or []):
            out[r["question_id"]] = r
    except Exception as e:
        st.error(f"加载题库失败: {str(e)}")
    return out

@st.cache_data(ttl=300, show_spinner=False)
def list_available_unit_paper() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    try:
        result = supabase.table("question_banks").select("unit, exam_time").execute()
        for r in result.data or []:
            unit = r.get("unit")
            exam_time = r.get("exam_time")
            if not unit or not exam_time:
                continue
            out.setdefault(unit, [])
            if exam_time not in out[unit]:
                out[unit].append(exam_time)
        for unit in out:
            out[unit].sort()
        if not out:
            raise RuntimeError("empty")
    except Exception as e:
        out = {"U1": ["2019 01", "2019 05", "2022 10"], "U2": ["2019 01", "2019 05", "2022 10"]}
        st.warning(f"加载 Unit/Paper 选项失败：{e}（已使用兜底选项）")
    return out

# 1) get_image_from_path
@st.cache_data(ttl=3420, show_spinner=False)
def get_image_from_path(image_path: str) -> Optional[str]:
    if not image_path:
        return None
    try:
        key = _sb_norm_key(image_path)
        if not key:
            return None
        resp = supabase.storage.from_(IMAGE_ROOT).download(key)
        if resp:
            return compress_image_to_b64(resp)  # 下载后立即压缩 & base64
    except Exception as e:
        st.warning(f"无法加载图片 {image_path}: {e}")
    return None


# ============================
# 图像处理
# ============================
# 放在“图像处理”或“Supabase 辅助”附近，统一复用
def _sb_norm_key(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    s = str(p).strip().strip('/').strip('\\')
    # 去掉本地前缀（例如 "C:\..." 或 "Easy Econ\images\"）
    # 只保留从 bucket 根开始的相对路径
    # 先统一分隔符
    s = s.replace('\\', '/')
    # 如果包含 bucket 名自身（images/…），把它剥掉
    low = s.lower()
    if low.startswith('images/'):
        s = s[7:]  # 去掉 "images/"
    # 如果包含你本地项目根（如 "easy econ/images/…")，也剥掉
    if low.startswith('easy econ/images/'):
        s = s[len('easy econ/images/'):]
    # 再次清理首尾分隔符
    s = s.strip('/')

    # 空就返回 None
    return s or None

@lru_cache(maxsize=4096)
def _hash_bytes_for_cache(b: bytes) -> str:
    import hashlib
    return hashlib.sha1(b).hexdigest()

def _resize_with_pillow(img, max_size: int):
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img

def compress_image_to_b64(img_bytes: bytes, max_size: int = 1400, quality: int = 78) -> str:
    try:
        cache_key = (_hash_bytes_for_cache(img_bytes), max_size, quality)
    except Exception:
        cache_key = None

    if cache_key and hasattr(compress_image_to_b64, "_cache"):
        val = compress_image_to_b64._cache.get(cache_key)
        if val:
            return val
    else:
        compress_image_to_b64._cache = {}

    try:
        if _TJ:
            with Image.open(BytesIO(img_bytes)) as img:
                img = img.convert("RGB")
                img = _resize_with_pillow(img, max_size)
                rgb = img.tobytes()
                w, h = img.size
                out = _TJ.encode(rgb, w, h, TJPF_RGB, quality=quality, subsampling=2)
                b64 = base64.b64encode(out).decode("utf-8")
        else:
            with Image.open(BytesIO(img_bytes)) as img:
                img = img.convert("RGB")
                img = _resize_with_pillow(img, max_size)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=2)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        st.warning(f"图像压缩失败(降级原图)：{e}")
        b64 = base64.b64encode(img_bytes).decode("utf-8")

    if cache_key:
        compress_image_to_b64._cache[cache_key] = b64
    return b64

# ============================
# 延迟入库：缓冲 & 刷新
# ============================
def buffer_uploads(aid: str, qno: str, uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    """把新上传的图片只放入 session 缓冲区，不立即写库；并记录 file_key→hash"""
    if not uploaded_files:
        return 0
    aid_map = st.session_state.pending_images.setdefault(aid, {})
    q_map = aid_map.setdefault(qno, {})
    fh_map = st.session_state.file_hash_map.setdefault(aid, {}).setdefault(qno, {})
    total_sizes = st.session_state.pending_sizes.setdefault(aid, {}).setdefault(qno, 0)

    added = 0
    for f in uploaded_files:
        try:
            b = f.getvalue()
            h = sha1_bytes(b)
            fk = get_file_key(f)
            fh_map[fk] = h
            if h not in q_map:
                q_map[h] = b
                total_sizes += len(b)
                added += 1
        except Exception as e:
            st.warning(f"{qno} 读取文件失败: {e}")

    st.session_state.pending_sizes[aid][qno] = total_sizes
    return added


def pending_count(aid: str, qno: str) -> int:
    return len(st.session_state.pending_images.get(aid, {}).get(qno, {}))

def pending_bytes(aid: str, qno: str) -> int:
    return st.session_state.pending_sizes.get(aid, {}).get(qno, 0)

def clear_pending(aid: str, qno: str):
    st.session_state.pending_images.get(aid, {}).pop(qno, None)
    st.session_state.pending_sizes.get(aid, {}).pop(qno, None)
def remove_pending_image(aid: str, qno: str, h: str):
    """按哈希删除单张未入库图片，并更新 pending size"""
    q_map = st.session_state.pending_images.get(aid, {}).get(qno, {})
    if h in q_map:
        del q_map[h]
    total = sum(len(b) for b in q_map.values())
    st.session_state.pending_sizes.setdefault(aid, {})[qno] = total

def flush_all_pending_to_db(aid: str):
    """
    在评分时把所有 pending 图片写入数据库。
    使用 add_answer_images()（它内部会先 upsert submission_answers），
    同时建立 hash→idx 映射，供“点上传列表里的 X”时删除 DB 记录用。
    """
    aid_map = st.session_state.pending_images.get(aid, {})
    total = 0
    for qno, hashmap in list(aid_map.items()):
        if not hashmap:
            continue

        # 按确定顺序写库，便于和返回的 idx_list 一一对应
        items = list(hashmap.items())          # [(hash, bytes), ...]
        img_bytes_list = [b for (_h, b) in items]

        try:
            # ✅ 关键：用原有工具，内部会 upsert submission_answers
            idx_list = add_answer_images(aid, qno, img_bytes_list)
        except Exception as e:
            st.error(f"{qno} 写入图片失败：{e}")
            idx_list = []

        if idx_list:
            # 建立 hash→idx 映射
            h2idx = st.session_state.file_to_db_map.setdefault(aid, {}).setdefault(qno, {})
            for (h, _b), idx in zip(items, idx_list):
                h2idx[h] = idx

            total += len(idx_list)
            # 清空该题的 pending 缓存与尺寸统计
            clear_pending(aid, qno)
        else:
            # 未成功写库，保留 pending，方便稍后重试
            st.warning(f"{qno} 暂存图片未写入数据库（将保留在缓冲区以便重试）。")

    if total > 0:
        st.cache_data.clear()
    return total



# ============================
# 评分配置与函数
# ============================
def get_grading_prompt(qno: str, section: str) -> str:
    if section == SECTION_B:
        return (
            """You are a professional A-Level Economics examiner.

If the student's answer is extremely short, nonsensical, irrelevant, just repeats the question, or is not received by you, respond strictly with:
"This answer is invalid and cannot be graded. Please provide a complete and relevant response."

For valid answers, strictly adhere to the provided mark scheme. Only assess the criteria explicitly mentioned in the mark scheme (e.g., Knowledge, Application, Analysis). Do not evaluate any criteria not included in the mark scheme.

For each relevant criterion:
Assign a score based solely on the mark scheme's descriptors.
Provide concise feedback that directly references the student's answer (quote or paraphrase).
Do not add any additional commentary beyond what is required by the mark scheme.
Do not provide self-corrections or revised totals.

If not full marks, provide a concise model answer (prefixed with "Here's how your answer could be improved to achieve full marks:") that integrates correct content from the student and only adds the missing elements per the mark scheme.

Format response exactly:
Total: x/4.
Knowledge: x/y. Feedback: [...]
Application: x/y. Feedback: [...]
Analysis: x/y. Feedback: [...]
不要用 LaTeX，直接普通算式。"""
        )
    elif section == SECTION_C:
        if qno != "Q12 (e)":
            return """You are a professional A-Level Economics examiner.
Your task:
-If the student's answer is extremely short, nonsensical, or irrelevant or just repeat the question or is not received by you (e.g. just numbers like "00", or random characters), you should respond:
"This answer is invalid and cannot be graded. Please provide a complete and relevant response." Do not change this sentence.
Do not attempt to grade such answers.
- First, read the mark scheme and determine what marking criteria are used. It may include some or all of: Knowledge, Application,Analysis and Evaluation. Only grade the criteria that appear in the mark scheme.
- For each relevant criterion, assign a score and give **concise** feedback using **quotes or paraphrases** from the student's answer, if you didn't assign full marks please state the reason!

- Do **not** provide any self-corrections or revised totals. Only give your final judgement once, and stick to it.
- Finally, if not full marks, provide a model sample answer that would get full marks based on student's answer and your feedback and state"Here's how your answer could be improved to achieve full marks: ...". If full marks, the improved answer is not needed.

Please format your response exactly like this,if the the criteria does not appear in the mark scheme please do not show that criteria in the following format:

Total: x/y.
Knowledge: x/y. Feedback: ...
Application: x/y. Feedback: ...
Analysis: x/y. Feedback: ...
Evaluation: x/y.Feedback:...
Here's how your answer could be improved to achieve full marks:  <student's correct content + missing parts smoothly integrated into a full-mark answer+ be concise to contain what is needed for full marks>
"""
        else:
            return """
You are an A-Level Economics examiner.  
... (omitted for brevity in prompt content, identical to your original long prompt for Q12e) ...
"""
    else:
        return """
You are an A-Level Economics examiner. Your task is to strictly grade the student's short essay based on the materials in the questions and the mark scheme's "Indicative Content" guidance, the official marking criteria below. Given Students are Chinese, please use Chinese give your explanation.
... (omitted for brevity in prompt content, identical to your original long prompt for Section D) ...
"""

def grade_with_ai(question_info: dict, student_answer_text: str, student_images_b64: List[str]):
    if not OPENROUTER_API_KEY:
        return "⚠ AI grading skipped: OPENROUTER_API_KEY not set."

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "X-Title": "EasyEcon-AnswerUpload",
        }

        qno = question_info.get("question_id", "")
        section = question_info.get("section", "")
        question_text = question_info.get("question_text", "")
        question_table_text = question_info.get("question_table_text", "")
        mark_scheme_text = question_info.get("mark_scheme_text_for_upload", "")

        question_table_image = get_image_from_path(question_info.get("question_table_image"))
        question_image_b64 = get_image_from_path(question_info.get("question_image_path"))
        mark_scheme_image_b64 = get_image_from_path(question_info.get("mark_scheme_image_path"))

        grading_prompt = get_grading_prompt(qno, section)

        content = []
        if question_text:
            content.append({"type": "text", "text": "Question:\n" + question_text})
        if question_table_text:
            content.append({"type": "text", "text": "Question Table:\n" + question_table_text})
        if question_image_b64:
            content.append({"type": "text", "text": "Question Image:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{question_image_b64}", "detail": "high"}})
        if question_table_image:
            content.append({"type": "text", "text": "Question Table Image:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{question_table_image}", "detail": "high"}})
        if mark_scheme_text:
            content.append({"type": "text", "text": "Mark scheme:\n" + mark_scheme_text})
        if mark_scheme_image_b64:
            content.append({"type": "text", "text": "Mark Scheme Image:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{mark_scheme_image_b64}", "detail": "high"}})
        if student_answer_text:
            content.append({"type": "text", "text": "Student typed answer:\n" + student_answer_text})
        if student_images_b64:
            content.append({"type": "text", "text": "Student handwritten answer (please grade directly from these images):"})
            for b64 in student_images_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
        content.append({"type": "text", "text": grading_prompt})

        payload = {
            "model": OPENROUTER_MODEL_VISION if (student_images_b64 or question_image_b64 or question_table_image or mark_scheme_image_b64) else OPENROUTER_MODEL_TXT,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
            "provider": {"allow_fallbacks": False},
        }

        last = None
        for _ in range(3):
            try:
                with requests.Session() as s:
                    r = s.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                msg = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                return msg or "⚠ Empty response"
            except Exception as e:
                last = str(e)
                time.sleep(1.2)
        return f"❌ AI error: {last}"
    except Exception as e:
        return f"❌ Grading error: {str(e)}"

def parse_total(text: str, expected_max: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if not text:
        return None, expected_max
    m = re.search(
        r"""(?ix)
        total \s* :? \s* (\d+)
        \s* / \s* (\d+)
        (?: \s*marks? )?
    """,
        text,
    )
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, expected_max

# ============================
# UI辅助（只缓冲，不立即写库）
# ============================
def show_pending_badge(aid: str, qno: str):
    pc = pending_count(aid, qno)
    if pc > 0:
        size_mb = pending_bytes(aid, qno) / (1024 * 1024)
        st.info(f"🟡 暂存未入库：{pc} 张（约 {size_mb:.2f} MB）。评分时会自动写入数据库。")


def render_upload_ui(aid: str, qno: str):
    # --- 上传控件 ---
    uploaded_files = st.file_uploader(
        f"上传 {qno} 的答案照片 (可多张)",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key=f"upload_pending_{aid}_{qno}",
    )

    # === 利用“X”检测：对比前后 keys 集合，定位被点掉的文件 ===
    qno_key = f"files_{aid}_{qno}"
    prev_keys = st.session_state.get(qno_key, set())
    curr_keys = set()
    if uploaded_files:
        curr_keys = {get_file_key(f) for f in uploaded_files}

    removed_keys = prev_keys - curr_keys
    if removed_keys:
        fh_map = st.session_state.file_hash_map.get(aid, {}).get(qno, {}) or {}
        h2idx = st.session_state.file_to_db_map.get(aid, {}).get(qno, {}) or {}
        for fk in removed_keys:
            h = fh_map.pop(fk, None)
            if not h:
                continue
            # 先删 pending
            remove_pending_image(aid, qno, h)
            # 如该图已在评分时写入过 DB，则联动删 DB
            idx = h2idx.pop(h, None)
            if idx:
                delete_answer_image(aid, qno, idx)
        st.cache_data.clear()
        st.session_state[qno_key] = curr_keys
        st.rerun()

    # === 把新增文件放入 pending，并建立 file_key→hash 映射 ===
    if uploaded_files:
        added = buffer_uploads(aid, qno, uploaded_files)
        st.session_state[qno_key] = curr_keys
        if added > 0:
            st.success(f"已暂存 {added} 张新图片（未入库）")
            st.rerun()

    # === 只显示 Pending 预览 ===
    pend_map = st.session_state.pending_images.get(aid, {}).get(qno, {}) or {}
    if len(pend_map) > 0:
        st.write(f"🕒 未入库暂存图片（{len(pend_map)} 张预览）")
        cols = st.columns(3)
        for j, (_h, b) in enumerate(pend_map.items()):
            with cols[j % 3]:
                try:
                    st.image(Image.open(BytesIO(b)), caption=f"Pending {j+1}", use_container_width=True)
                except Exception:
                    st.write(f"Pending {j+1} (预览失败)")

    show_pending_badge(aid, qno)
    st.caption("提示：评分时会把未入库图片统一写入数据库。若已评分，点击上传列表中文件名右侧的 X 也会同步删除数据库里的对应图片。")




def is_graded(head, answers):
    if not head or head.get("status") != "submitted":
        return False
    for qno in ALL_QS:
        if qno in answers and answers[qno].get("score") is not None:
            return True
    return False

# ============================
# 评分并行器（评分前先 flush pending）
# ============================
def _grade_and_write_one(aid: str, qno: str, info: dict, ans_text: str, imgs_b64_list: List[str]):
    g = grade_with_ai(info, ans_text, imgs_b64_list)
    tot, _max = parse_total(g, info.get("mark"))
    try:
        supabase.table("submission_answers").upsert({
            "attempt_id": aid,
            "qno": qno,
            "answer_text": ans_text if ans_text else None,
            "grading_text": g,
            "score": tot,
            "updated_at": datetime.now(_tz.utc).isoformat(),
        }, on_conflict="attempt_id,qno").execute()
    except Exception as e:
        st.error(f"保存评分结果失败: {str(e)}")

def grade_attempt_parallel(aid: str, qidx: Dict[str, dict], max_workers: int = None):
    try:
        # ✅ 评分前统一把暂存图片写库
        flushed = flush_all_pending_to_db(aid)
        if flushed > 0:
            st.success(f"已将 {flushed} 张暂存图片写入数据库")

        head, answers = load_attempt(aid)
        # A 匹配
        for qno in QS_A:
            info = qidx.get(qno, {}) or {}
            sel = (answers.get(qno, {}) or {}).get("answer_text", "") or ""
            correct_key = (info.get("original_mark_scheme_text") or "").strip().upper()
            is_ok = (sel.strip().upper() == correct_key)
            score = 1 if is_ok else 0
            grading_text = f"selected={sel} | correct={correct_key} | ok={is_ok}"
            supabase.table("submission_answers").upsert({
                "attempt_id": aid,
                "qno": qno,
                "answer_text": sel or None,
                "grading_text": grading_text,
                "score": score,
                "updated_at": datetime.now(_tz.utc).isoformat(),
            }, on_conflict="attempt_id,qno").execute()

        head, answers = load_attempt(aid)

        jobs = []
        for qno in (QS_B + QS_C + QS_D):
            info = qidx.get(qno, {}) or {}
            ans_row = (answers.get(qno) or {})
            ans_text = ans_row.get("answer_text", "") or ""
            imgs = [b64 for (_i, b64) in list_answer_images(aid, qno)]
            if not (ans_text or imgs):
                continue
            jobs.append((qno, info, ans_text, imgs))

        if not jobs:
            st.info("没有可判分的主观题（B/C/D）。")
            try:
                total_result = supabase.table("submission_answers").select("score").eq("attempt_id", aid).execute()
                total_score = sum((r["score"] or 0) for r in (total_result.data or []))
                supabase.table("submissions").update({"total_score": total_score, "status": "submitted"})\
                    .eq("attempt_id", aid).execute()
            except Exception as e:
                st.error(f"更新总分失败: {str(e)}")
            return

        if max_workers is None:
            max_workers = GRADING_WORKERS_ENV or min(4, len(jobs))

        progress = st.progress(0.0, text=f"开始并行评分（{len(jobs)} 题）…")
        done = 0
        errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futmap = {ex.submit(_grade_and_write_one, aid, qno, info, ans_text, imgs): qno for (qno, info, ans_text, imgs) in jobs}
            for fut in as_completed(futmap):
                qno = futmap[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((qno, str(e)))
                done += 1
                progress.progress(done / len(jobs), text=f"已完成 {done}/{len(jobs)}")

        try:
            total_result = supabase.table("submission_answers").select("score").eq("attempt_id", aid).execute()
            total_score = sum((r["score"] or 0) for r in (total_result.data or []))
            supabase.table("submissions").update({"total_score": total_score, "status": "submitted"})\
                .eq("attempt_id", aid).execute()
        except Exception as e:
            st.error(f"更新总分失败: {str(e)}")

        if errors:
            st.warning("以下题目判分失败（已跳过，稍后可重试）：\n" + "\n".join(f"- {q}: {msg}" for q, msg in errors))
    except Exception as e:
        st.error(f"评分过程中出错: {str(e)}")

# ============================
# PDF 导出
# ============================
BASE = Path(__file__).resolve().parent
FONT_R = BASE / "assets/fonts/LXGWWenKai-Regular.ttf"
FONT_M = BASE / "assets/fonts/DejaVuSerif-Bold.ttf"   # 充当 Bold

def normalize_and_parse_total(grading_text: str, expected_max: Optional[int]):
    s, m = parse_total(grading_text or "", expected_max)
    return grading_text, s, m

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (ch == "\n") or ch.isprintable())
    return s

def _break_long_tokens(s: str, maxrun: int = 40) -> str:
    def _split(m):
        run = m.group(0)
        return " ".join(run[i: i + maxrun] for i in range(0, len(run), maxrun))
    return re.sub(rf"\S{{{maxrun+1},}}", _split, s)

def _safe_multicell(pdf: FPDF, w, h, txt, *args, **kwargs):
    t = _break_long_tokens(_clean_text(txt))
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(w, h, t, *args, **kwargs, new_x="LMARGIN", new_y="NEXT")
    except TypeError:
        pdf.multi_cell(w, h, t, *args, **kwargs)
        pdf.ln(h * 0.2)
def _ok(p: Path, min_bytes=1_000_000) -> bool:
    try:
        return p.exists() and p.stat().st_size >= min_bytes
    except Exception:
        return False

@lru_cache(maxsize=1)
def _cn_font_paths():
    # 优先：Regular 用 Regular；Bold 用 Medium
    # 回退：缺哪一个就用另一个兜底，保证不会因为 style="B" 崩
    if not (_ok(FONT_R) or _ok(FONT_M)):
        raise RuntimeError("中文字体缺失：请放入 assets/fonts/LXGWWenKai-Regular.ttf / -Medium.ttf")
    reg  = str(FONT_R if _ok(FONT_R) else FONT_M)
    bold = str(FONT_M if _ok(FONT_M) else (FONT_R if _ok(FONT_R) else FONT_M))
    return reg, bold
def _setup_cn_font(pdf):
    fam = "CN"
    if fam not in pdf.fonts:
        reg, bold = _cn_font_paths()
        pdf.add_font(fam, "",  reg,  uni=True)  # Regular→常规
        pdf.add_font(fam, "B", bold, uni=True)  # Medium→粗体（或回退）
    pdf.set_font(fam, size=12)
    return fam

def _content_area_h(pdf):
    return pdf.h - pdf.t_margin - pdf.b_margin

def _remain_h(pdf):
    return (pdf.h - pdf.b_margin) - pdf.get_y()

def _ensure_room(pdf, need_h: float):
    if _remain_h(pdf) < need_h:
        pdf.add_page()
        pdf.set_x(pdf.l_margin)

def _precompress_images_for_pdf(imgs_bytes: list[bytes], max_side: int, quality: int) -> list[bytes]:
    outs = []

    def _do(b):
        with Image.open(BytesIO(b)) as im:
            im = im.convert("RGB")
            # 确保最小尺寸，避免 0 高图
            if max(im.size) < 2:
                return b
            im = _resize_with_pillow(im, max_side)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=2)
            return buf.getvalue()

    if imgs_bytes:
        with ThreadPoolExecutor(max_workers=min(8, len(imgs_bytes))) as ex:
            outs = [f.result() for f in [ex.submit(_do, b) for b in imgs_bytes]]
    return outs

def _add_image_auto_paginate(pdf, img_path_or_bytes, target_w: float, padding: float = 2.0):
    if isinstance(img_path_or_bytes, (bytes, bytearray)):
        img = Image.open(BytesIO(img_path_or_bytes)).convert("RGB")
    else:
        img = Image.open(img_path_or_bytes).convert("RGB")
    w0, h0 = img.size
    if w0 == 0 or h0 == 0:
        return
    scale = target_w / float(w0)
    disp_w = target_w
    disp_h = h0 * scale
    usable_h = _content_area_h(pdf)

    if disp_h <= usable_h:
        _ensure_room(pdf, disp_h + padding)
        pdf.image(img, x=pdf.get_x(), y=pdf.get_y(), w=disp_w)
        pdf.ln(disp_h + padding)
        return

    slice_disp_h = usable_h
    slice_px_h = max(1, int(math.floor(slice_disp_h / scale)))

    top = 0
    while top < h0:
        bottom = min(h0, top + slice_px_h)
        crop = img.crop((0, top, w0, bottom))
        piece_h_disp = (bottom - top) * scale
        _ensure_room(pdf, piece_h_disp + padding)
        pdf.image(crop, x=pdf.get_x(), y=pdf.get_y(), w=disp_w)
        pdf.ln(piece_h_disp + padding)
        top = bottom

_WS = r"[ \t\r\n\f\v\u00A0\u2000-\u200B\u202F\u205F\u3000]"
_IMPROVE_RE = re.compile(rf"(?ix)^{_WS}*here{_WS}*['']?{_WS}*s{_WS}+how{_WS}+your{_WS}+answer{_WS}+could{_WS}+be{_WS}+improved{_WS}+to{_WS}+achieve{_WS}+full{_WS}+marks{_WS}*:?{_WS}*")
_INDICATIVE_RE = re.compile(rf"(?ix)^{_WS}*indicative{_WS}+content(?:{_WS}+guidance)?{_WS}*:?{_WS}*")
_LABEL_PREFIX_RE = re.compile(rf"(?ix)^{_WS}*(Knowledge|Application|Analysis|Evaluation){_WS}*:?{_WS}*(?:\d+(?:{_WS}*/{_WS}*\d+)?)?{_WS}*")
_BOLD_SPAN_RE = re.compile(
    rf"(?ix)(?:\bTotal\b{_WS}*:?{_WS}*\d+(?:{_WS}*/{_WS}*\d+)?)|(?:\b(?:Knowledge|Application|Analysis|Evaluation)\b{_WS}*:?{_WS}*(?:\d+(?:{_WS}*/{_WS}*\d+)?)?)|(?:\bindicative{_WS}+content(?:{_WS}+guidance)?\b{_WS}*:? )"
)

_LABEL_WORDS = ("Knowledge", "Application", "Analysis", "Evaluation")

def _has_at_least_three_labels(text: str) -> bool:
    if not text:
        return False
    found = sum(1 for w in _LABEL_WORDS if re.search(rf"\b{w}\b", text, re.I))
    return found >= 3

def _has_eval_and_indicative(text: str) -> bool:
    if not text:
        return False
    has_eval = re.search(r"\bEvaluation\b", text, re.I) is not None
    has_ind = re.search(rf"\bindicative{_WS}+content(?:{_WS}+guidance)?\b", text, re.I) is not None
    return has_eval and has_ind

def _wrap_text_by_width(pdf, text: str, max_w: float) -> list:
    text = (text or "").replace("\r", "")
    paragraphs = text.split("\n")
    lines = []
    for para in paragraphs:
        words = para.split(" ")
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip() if cur else w
            if pdf.get_string_width(cand) <= max_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                while pdf.get_string_width(w) > max_w and len(w) > 0:
                    lo, hi = 1, len(w)
                    cut = 1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if pdf.get_string_width(w[:mid]) <= max_w:
                            cut = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    lines.append(w[:cut])
                    w = w[cut:]
                cur = w
        lines.append(cur)
    return lines

def _render_inline_bold_spans(pdf, fam: str, content_w: float, line_h: float, text: str):
    idx = 0
    body = text or ""
    for m in _BOLD_SPAN_RE.finditer(body):
        if m.start() > idx:
            pdf.set_font(fam, "", 11)
            pdf.write(line_h, body[idx:m.start()])
        pdf.set_font(fam, "B", 11)
        pdf.write(line_h, body[m.start():m.end()])
        idx = m.end()
    if idx < len(body):
        pdf.set_font(fam, "", 11)
        pdf.write(line_h, body[idx:])

def _render_one_line_with_prefix_and_inline(pdf, fam: str, content_w: float, line_h: float, ln: str):
    text = ln or ""
    if _has_at_least_three_labels(text) or _has_eval_and_indicative(text):
        pdf.set_font(fam, "B", 11)
        pdf.cell(content_w, line_h, text, ln=1)
        return

    m_imp = _IMPROVE_RE.match(text)
    if m_imp:
        prefix = text[:m_imp.end()]
        rest = text[m_imp.end():]
        pdf.set_font(fam, "B", 11)
        pdf.write(line_h, prefix)
        pdf.set_font(fam, "", 11)
        _render_inline_bold_spans(pdf, fam, content_w, line_h, rest)
        pdf.ln(line_h)
        return

    m_ind = _INDICATIVE_RE.match(text)
    if m_ind:
        prefix = text[:m_ind.end()]
        rest = text[m_ind.end():]
        pdf.set_font(fam, "B", 11)
        pdf.write(line_h, prefix)
        pdf.set_font(fam, "", 11)
        _render_inline_bold_spans(pdf, fam, content_w, line_h, rest)
        pdf.ln(line_h)
        return

    m_lab = _LABEL_PREFIX_RE.match(text)
    if m_lab:
        prefix = text[:m_lab.end()]
        rest = text[m_lab.end():]
        pdf.set_font(fam, "B", 11)
        pdf.write(line_h, prefix)
        pdf.set_font(fam, "", 11)
        _render_inline_bold_spans(pdf, fam, content_w, line_h, rest)
        pdf.ln(line_h)
        return

    _render_inline_bold_spans(pdf, fam, content_w, line_h, text)
    pdf.ln(line_h)

def boxed_text_paginated(
    pdf,
    fam: str,
    title: str,
    text: str,
    box_pad: float = 2.0,
    line_h: float = 5.5,
    title_h: float = 6.0,
    title_bold: bool = True,
    imgs: list | None = None,
    img_w: float = 120.0,
    banner_h: float = 7.0,
    banner_bg: tuple = (235, 235, 235),
    cont_suffix: str = " (cont.)",
):
    if (not title) and (not text) and (not imgs):
        return

    inner_w = pdf.w - pdf.l_margin - pdf.r_margin
    content_w = inner_w - 2 * box_pad

    def draw_box(x, y, w, h):
        pdf.rect(x, y, w, h)

    pdf.set_font(fam, "B" if title_bold else "", 11)
    banner_lines = _wrap_text_by_width(pdf, (title or ""), content_w)
    banner_block_h = max(banner_h, title_h * len(banner_lines or [""]))

    pdf.set_font(fam, "", 11)
    text_lines = _wrap_text_by_width(pdf, text or "", content_w)
    if not text_lines:
        text_lines = [""]

    def begin_page_box(with_cont: bool = False):
        if _remain_h(pdf) < (banner_block_h + box_pad + line_h):
            pdf.add_page()
        pdf.set_x(pdf.l_margin)
        page_box_x = pdf.get_x()
        page_box_y = pdf.get_y()

        pdf.set_fill_color(*banner_bg)
        pdf.set_font(fam, "B" if title_bold else "", 11)
        lines = banner_lines[:] if not with_cont else _wrap_text_by_width(pdf, (title or "") + cont_suffix, content_w)

        pdf.rect(page_box_x, page_box_y, inner_w, banner_block_h, style="F")
        pdf.set_xy(page_box_x + box_pad, page_box_y + (banner_block_h - title_h * max(1, len(lines))) / 2.0)
        for ln in (lines or [""]):
            pdf.cell(content_w, title_h, ln, ln=1)
        pdf.set_font(fam, "", 11)

        pdf.set_xy(pdf.l_margin + box_pad, page_box_y + banner_block_h + box_pad)
        return page_box_x, page_box_y

    def close_page_box(page_box_x, page_box_y):
        box_h = (pdf.get_y() - page_box_y) + box_pad
        draw_box(page_box_x, page_box_y, inner_w, box_h)

    page_box_x, page_box_y = begin_page_box(with_cont=False)

    for ln in text_lines:
        if _remain_h(pdf) < (line_h + box_pad):
            close_page_box(page_box_x, page_box_y)
            pdf.add_page()
            page_box_x, page_box_y = begin_page_box(with_cont=True)
        _render_one_line_with_prefix_and_inline(pdf, fam, content_w, line_h, ln)

    if imgs:
        for p in imgs:
            if not p:
                continue
            close_page_box(page_box_x, page_box_y)
            if _remain_h(pdf) < (banner_block_h + box_pad + line_h):
                pdf.add_page()
            page_box_x, page_box_y = begin_page_box(with_cont=False)
            avail_w = content_w
            _add_image_auto_paginate(pdf, p, target_w=min(img_w, avail_w))
            close_page_box(page_box_x, page_box_y)
            if _remain_h(pdf) < (banner_block_h + box_pad + line_h):
                pdf.add_page()
            page_box_x, page_box_y = begin_page_box(with_cont=True)

    close_page_box(page_box_x, page_box_y)
    pdf.ln(2.0)

def insert_images_no_banner(pdf, imgs: list, img_w: float = 120.0, pad_top: float = 1.0, pad_bottom: float = 2.0):
    if not imgs:
        return
    inner_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.ln(pad_top)
    for p in imgs:
        if not p:
            continue
        avail_w = inner_w
        _add_image_auto_paginate(pdf, p, target_w=min(img_w, avail_w))
    pdf.ln(pad_bottom)

def clean_text_for_pdf(s: str) -> str:
    if not s:
        return ""
    s = s.replace('\\"', '"')
    s = s.replace("\\n", "\n")
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

def clean_options(raw: str) -> str:
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        items = None
    if isinstance(items, list):
        out = []
        for i, item in enumerate(items):
            s = str(item).strip().strip('"')
            if len(s) >= 2 and s[1] in [")", ".", " "]:
                out.append(f"{s[0].upper()}. {s[2:].lstrip(' .)')}")
            else:
                key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
                out.append(f"{key}. {s}")
        return "\n".join(out)
    txt = (raw or "").strip().strip("[]").replace('"', "")
    txt = txt.replace(",", "\n")
    return txt.strip()

def draw_score_table(pdf: FPDF, fam: str, rows, col_widths=(30, 45, 30), col_headers=("Question", "Section", "Score"), line_h=7):
    x = pdf.l_margin
    w_total = sum(col_widths)
    need_h = line_h * (1 + len(rows)) + 6
    if (pdf.get_y() + need_h) > pdf.page_break_trigger:
        pdf.add_page()
    y0 = pdf.get_y()
    pdf.set_draw_color(0, 0, 0)
    pdf.rect(x, y0, w_total, need_h)
    pdf.set_font(fam, "B", 11)
    for i, head in enumerate(col_headers):
        pdf.set_xy(x + sum(col_widths[:i]), y0)
        pdf.cell(col_widths[i], line_h, head, border=1, align="C")
    pdf.ln(line_h)
    pdf.set_font(fam, "", 11)
    for r in rows:
        for i, cell in enumerate(r):
            pdf.set_xy(x + sum(col_widths[:i]), pdf.get_y())
            pdf.cell(col_widths[i], line_h, str(cell), border=1)
        pdf.ln(line_h)
    pdf.ln(4)

def export_pdf(head, answers, qidx, fast_mode: bool = True, qs=None, include_pending: Optional[dict] = None):
    IMG_W_MARKSCHEME = 100.0 if fast_mode else 120.0
    IMG_W_STUDENT = 80.0 if fast_mode else 100.0
    JPEG_MAXSIDE = 1000 if fast_mode else 1400
    JPEG_QUALITY = 72 if fast_mode else 78

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    fam = _setup_cn_font(pdf)
    pdf.add_page()

    pdf.set_font(fam, "B", 13)
    _safe_multicell(pdf, 0, 8, f"Mock Exam {head.get('unit', '')} {head.get('paper', '')} (Section A + B + C + D)")
    pdf.set_font(fam, "", 11)
    _safe_multicell(pdf, 0, 6, f"Student: {head.get('student_name', '')}")
    
    # 在常量 QS_A/… 定义之后，加一个工具函数：
    def _ordered_qs(qidx: dict[str, dict]) -> list[str]:
        prefer = QS_A + QS_B + QS_C + QS_D
        ordered = [q for q in prefer if q in qidx]
        # 兜底：如果还有额外题号（非常规命名），按自然顺序追加
        extras = [q for q in qidx.keys() if q not in ordered]
        extras.sort()
        return ordered + extras


    order_qs = qs if qs is not None else _ordered_qs(qidx)

    rows = []
    for qno in order_qs:
        info = qidx.get(qno, {}) or {}
        sec = info.get("section", "")
        if sec == SECTION_A:
            scr = answers.get(qno, {}).get("score")
            score_str = "" if scr is None else str(int(scr))
        else:
            gtxt = (answers.get(qno, {}) or {}).get("grading_text", "")
            norm_txt, ssum, msum = normalize_and_parse_total(gtxt, expected_max=qidx.get(qno, {}).get("mark"))
            if norm_txt and norm_txt != gtxt:
                answers[qno]["grading_text"] = norm_txt
            score_str = f"{int(ssum)}/{int(msum)}" if (ssum is not None and msum) else "—"
        rows.append((qno, sec, score_str))
    if rows:
        draw_score_table(pdf, fam, rows)

    all_imgs = fetch_all_answer_images(head.get("attempt_id", ""))

    for qno in order_qs:
        info = qidx.get(qno, {}) or {}
        ans = answers.get(qno, {}) or {}
        section_label = info.get("section", "")

        pdf.ln(2)
        pdf.set_font(fam, "B", 12)
        _safe_multicell(pdf, 0, 7, f"{section_label} — {qno}")

        pdf.set_font(fam, "", 11)
        qtext = clean_text_for_pdf(info.get("question_text") or "")
        _safe_multicell(pdf, 0, 6, qtext)

        opts_raw = info.get("options_json")
        if opts_raw:
            pdf.set_font(fam, "", 11)
            opts_clean = clean_text_for_pdf(clean_options(opts_raw))
            _safe_multicell(pdf, 0, 6, opts_clean)

        # 2) export_pdf 中的题干表格图
        question_table_img_val = info.get("question_table_image")
        if question_table_img_val:
            try:
                key = _sb_norm_key(question_table_img_val)
                if key:
                    qtab_bytes = supabase.storage.from_(IMAGE_ROOT).download(key)
                    avail_w = pdf.w - pdf.l_margin - pdf.r_margin
                    _add_image_auto_paginate(pdf, qtab_bytes, target_w=min(120, avail_w))
            except Exception as e:
                print("下载题干表格图失败:", e)


        # 学生答案文本
        pdf.set_font(fam, "B", 11)
        _safe_multicell(pdf, 0, 6, "Your Answer:")
        pdf.set_font(fam, "", 11)
        try:
            ans_json_raw = ans.get("answer_text") or ""
            aj = json.loads(ans_json_raw) if ans_json_raw else {}
            if isinstance(aj, dict) and "ans" in aj and isinstance(aj["ans"], dict):
                aj = aj["ans"]
            stu_ans_text = clean_text_for_pdf(aj.get("written_answer"))
        except Exception:
            stu_ans_text = clean_text_for_pdf(ans.get("answer_text") or "")
        _safe_multicell(pdf, 0, 6, stu_ans_text)
        pdf.ln(2)

        # 学生上传图：已入库 + 可选包含 pending（不强制写库）
        imgs_bytes = []
        if ans.get("image_b64"):
            try:
                imgs_bytes.append(base64.b64decode(ans["image_b64"]))
            except Exception:
                pass
        for _idx, b64 in all_imgs.get(qno, []):
            try:
                imgs_bytes.append(base64.b64decode(b64))
            except Exception:
                pass
        if include_pending and qno in include_pending:
            # 仅用于导出，不入库
            for b in include_pending[qno].values():
                imgs_bytes.append(b)

        _seen = set()
        imgs_bytes_dedup = []
        for b in imgs_bytes:
            try:
                b64c = compress_image_to_b64(b, max_size=1420, quality=85)
            except Exception:
                b64c = base64.b64encode(b).decode("utf-8")
            if b64c not in _seen:
                _seen.add(b64c)
                imgs_bytes_dedup.append(b)

        if imgs_bytes_dedup:
            precomp = _precompress_images_for_pdf(imgs_bytes_dedup, JPEG_MAXSIDE, JPEG_QUALITY)
            avail_w = pdf.w - pdf.l_margin - pdf.r_margin
            for b in precomp:
                _add_image_auto_paginate(pdf, b, target_w=min(IMG_W_STUDENT, avail_w * 0.55))

        # Feedback & Mark Scheme
        if section_label in (SECTION_B, SECTION_C, SECTION_D):
            fb_text = clean_text_for_pdf(ans.get("grading_text") or "")
            boxed_text_paginated(pdf, fam, "Feedback:", fb_text)

            ms_text = clean_text_for_pdf(info.get("original_mark_scheme_text") or "")
            ms_img_rel = info.get("mark_scheme_image_path")

            if ms_text.strip():
                boxed_text_paginated(pdf, fam, "Mark Scheme:", ms_text)
            if ms_img_rel:
                try:
                    key = _sb_norm_key(ms_img_rel)
                    if key:
                        ms_bytes = supabase.storage.from_(IMAGE_ROOT).download(key)
                        insert_images_no_banner(pdf, [ms_bytes], img_w=120.0)
                except Exception as e:
                    print("下载 Mark Scheme 图失败:", e)

        if section_label == SECTION_A:
            corr = clean_text_for_pdf((info.get("original_mark_scheme_text") or "").strip())
            pdf.set_font(fam, "B", 11)
            _safe_multicell(pdf, 0, 6, "Correct Answer:")
            pdf.set_font(fam, "", 11)
            _safe_multicell(pdf, 0, 6, corr)

            exp = clean_text_for_pdf(info.get("mark_scheme_text") or "")
            boxed_text_paginated(pdf, fam, "Answer Explanation:", exp)

            ms_img_path = info.get("mark_scheme_image_path")
            if ms_img_path:
                try:
                    key = _sb_norm_key(ms_img_path)
                    if key:
                        ms_bytes2 = supabase.storage.from_(IMAGE_ROOT).download(key)
                        avail_w = pdf.w - pdf.l_margin - pdf.r_margin
                        _add_image_auto_paginate(pdf, ms_bytes2, target_w=min(120, avail_w))
                except Exception as e:
                    print("下载 Mark Scheme 图失败:", e)

    # 生成 PDF 字节
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        data = bytes(out)
    else:
        data = out.encode("latin-1", "ignore")
    if not data or len(data) < 100:
        fallback = FPDF()
        fallback.add_page()
        fallback.set_font("Arial", size=12)
        fallback.cell(0, 10, "Empty Report Placeholder")
        out2 = fallback.output(dest="S")
        data = out2 if isinstance(out2, (bytes, bytearray)) else out2.encode("latin-1", "ignore")
    return data

# ============================
# UI
# ============================
ensure_db()

st.title("EasyEcon判分系统（延迟入库版）")

with st.sidebar:
    st.subheader("考生信息")
    uid = st.text_input("用户ID", value=st.session_state.get("user_id", ""))
    name = st.text_input("姓名", value=st.session_state.get("student_name", ""))

    if "_pending_unit" in st.session_state:
        st.session_state.selected_unit = st.session_state.pop("_pending_unit") or st.session_state.get("selected_unit")
    if "_pending_paper" in st.session_state:
        st.session_state.selected_paper = st.session_state.pop("_pending_paper") or st.session_state.get("selected_paper")

    up_map = list_available_unit_paper()

    units_sorted = _clean_and_sort(up_map.keys()) or ["U2"]
    if "selected_unit" not in st.session_state or st.session_state.selected_unit not in units_sorted:
        st.session_state.selected_unit = units_sorted[0]

    def _on_unit_change():
        ps = _clean_and_sort(up_map.get(st.session_state.selected_unit, [])) or ["2019 05"]
        st.session_state.selected_paper = ps[0]

    unit = st.selectbox("选择 Unit", units_sorted, key="selected_unit", on_change=_on_unit_change)

    papers_sorted = _clean_and_sort(up_map.get(unit, [])) or ["2019 05"]
    if "selected_paper" not in st.session_state or st.session_state.selected_paper not in papers_sorted:
        st.session_state.selected_paper = papers_sorted[0]

    paper = st.selectbox("选择 Paper（Exam Time）", papers_sorted, key="selected_paper")

    if st.button("进入评分系统", use_container_width=True):
        aid_new = create_attempt(uid, name, st.session_state.selected_unit, st.session_state.selected_paper, "A+B+C+D")
        if aid_new:
            st.session_state.update(user_id=uid, student_name=name, attempt_id=aid_new)
            st.rerun()

    st.markdown("---")
    st.subheader("恢复上一次判分")

    restore_aid = st.text_input("粘贴 attempt_id 恢复", value="")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("用 attempt_id 恢复", use_container_width=True, key="btn_restore_by_id"):
            if restore_aid.strip():
                meta = get_attempt_summary(restore_aid.strip())
                if meta:
                    st.session_state._pending_unit = meta.get("unit")
                    st.session_state._pending_paper = meta.get("paper")
                    st.session_state.update(
                        attempt_id=meta["attempt_id"],
                        user_id=meta.get("user_id", st.session_state.get("user_id", "")),
                        student_name=meta.get("student_name", st.session_state.get("student_name", "")),
                    )
                    st.success(f"已恢复：{meta['student_name']} / {meta['attempt_id']}")
                    st.rerun()
                else:
                    st.error("未找到该 attempt_id 对应的记录。请确认后重试。")
            else:
                st.warning("请先粘贴有效的 attempt_id。")

    recent = []
    if st.session_state.get("user_id"):
        try:
            recent = list_attempts_by_user(st.session_state["user_id"], unit, paper, limit=10)
        except Exception as e:
            st.warning(f"查询最近记录失败：{e}")

    if recent:
        sel = st.selectbox(
            "或选择最近记录恢复",
            options=["(请选择)"] + [
                f"{r['attempt_id']} — {r['status']} — {(r.get('total_score') and int(r['total_score'])) or '-'}分 — {(r.get('submitted_at') or r.get('started_at'))}"
                for r in recent
            ],
            index=0,
        )
        with col_r2:
            if st.button("恢复所选记录", use_container_width=True, key="btn_restore_pick"):
                if sel and sel != "(请选择)":
                    pick_idx = [i for i, x in enumerate(recent) if x["attempt_id"] in sel]
                    if pick_idx:
                        chosen = recent[pick_idx[0]]
                        meta = get_attempt_summary(chosen["attempt_id"])
                        if meta:
                            st.session_state._pending_unit = meta.get("unit")
                            st.session_state._pending_paper = meta.get("paper")
                        st.session_state.attempt_id = chosen["attempt_id"]
                        st.success(f"已恢复：{chosen['attempt_id']}")
                        st.rerun()
                else:
                    st.warning("请先从下拉框选择一条记录。")

    latest_ip = None
    try:
        if st.session_state.get("user_id"):
            latest_ip = find_latest_inprogress(st.session_state["user_id"], unit, paper)
    except Exception as e:
        st.warning(f"查询未完成记录失败：{e}")

    if latest_ip:
        if st.button("继续上次未完成（in_progress）", use_container_width=True, key="btn_resume_inprogress"):
            meta = get_attempt_summary(latest_ip)
            if meta:
                st.session_state._pending_unit = meta.get("unit")
                st.session_state._pending_paper = meta.get("paper")
            st.session_state["attempt_id"] = latest_ip
            st.success(f"已恢复未完成记录：{latest_ip}")
            st.rerun()

aid = st.session_state.get("attempt_id")
if not aid:
    st.info("请先在侧边栏输入你的ID和姓名，并选择 Unit/Paper 创建记录。")
    st.stop()

head, answers = load_attempt(aid)
if not head:
    st.error("无法加载尝试数据，请重新创建。")
    st.stop()

UNIT = head.get("unit")
PAPER = head.get("paper")
st.subheader(f"当前试卷：{UNIT} — {PAPER}")

qidx = get_questions_multi(UNIT, PAPER)

graded = is_graded(head, answers)

st.write("请为每道题目上传您的答案。Section A为选择题，请直接选择答案。其他题目可以上传手写答案的照片。")

# ============================
# Section A (选择题)
# ============================
st.divider()
st.header("Section A - 选择题")
st.write("请为每道题选择正确的答案 (A-D)")

with st.form(key=f"section_a_form_{aid}"):
    for qno in QS_A:
        existing_answer = (answers.get(qno, {}) or {}).get("answer_text", "") or ""
        options = ["", "A", "B", "C", "D"]
        default_idx = options.index(existing_answer) if existing_answer in options else 0
        selected = st.selectbox(f"{qno}:", options, index=default_idx, key=f"select_{aid}_{qno}")
        if selected:
            st.session_state[f"{aid}_{qno}_answer"] = selected
    submit_a = st.form_submit_button("保存 Section A 答案")

if submit_a:
    with st.spinner("正在保存选择题答案..."):
        for qno in QS_A:
            answer = st.session_state.get(f"{aid}_{qno}_answer", "")
            if answer:
                upsert_answer(aid, qno, text=answer)
        st.success("Section A 答案已保存")

# ============================
# Section B, C, D (手写答案) — 延迟入库
# ============================
st.divider()
st.header("Section B, C, D - 手写答案（延迟入库）")
st.write("上传后将先暂存，评分时会统一写入数据库。")
DIAGRAM_HINT_MD = (
    "📝 **若本题需要画图，请按下述提示作图（会影响判分）**\n"
    "- 每条曲线旁**用文字**标注名称：`Demand (D)`、`Supply (S)`、`AD`、`AS`、`LRAS/SRAS`、`PPF`。\n"
    "- **移动/旋转方向**务必**用箭头**清楚标出：`AD/AS →/←`、`S/D →/←`、`PPF outward → / inward ←`。\n"
    "- 涉及**均衡变化**建议**用箭头**标注均衡**价格**和**数量**变化的方向，如：`P1 → P2`,`Q1 → Q2`。\n"
)

tabs = st.tabs(["Section B", "Section C", "Section D"])

with tabs[0]:
    st.subheader("Section B")
    for qno in QS_B:
        with st.expander(f"{qno}", expanded=False):
            st.info(DIAGRAM_HINT_MD)
            render_upload_ui(aid, qno)

with tabs[1]:
    st.subheader("Section C")
    for qno in QS_C:
        with st.expander(f"{qno}", expanded=False):
            st.info(DIAGRAM_HINT_MD)
            render_upload_ui(aid, qno)

with tabs[2]:
    st.subheader("Section D")
    for qno in QS_D:
        with st.expander(f"{qno}", expanded=False):
            st.info(DIAGRAM_HINT_MD)
            render_upload_ui(aid, qno)

# ============================
# 评分部分
# ============================
st.divider()
st.header("评分")
st.write("上传完所有答案后，点击下方按钮进行评分。（会先把所有暂存图片写入数据库）")

if st.button("开始评分", use_container_width=True):
    st.cache_data.clear()
    with st.spinner("正在评分，请稍候..."):
        grade_attempt_parallel(aid, qidx)
    st.success("评分完成")
    st.rerun()

# ============================
# 分数预览
# ============================
st.divider()
st.header("分数预览")
head, answers = load_attempt(aid)
graded = is_graded(head, answers)

total_score = head.get("total_score") if head else None
if total_score is not None:
    st.subheader(f"总分: {int(total_score)}")

section_tabs = st.tabs(["Section A", "Section B", "Section C", "Section D"])

def _wrapped_block(label: str, text: str):
    if not (text or "").strip():
        return
    with st.expander(label):
        st.markdown(
            f'<div style="white-space:pre-wrap; word-wrap:break-word; line-height:1.5;">{_html_escape(text)}</div>',
            unsafe_allow_html=True
        )


with section_tabs[0]:
    for qno in QS_A:
        row = answers.get(qno) or {}
        score = row.get("score")
        answer = row.get("answer_text", "") or ""
        grading = row.get("grading_text", "") or ""
        st.markdown(f"**{qno}** — 答案: {answer} — 得分: {'' if score is None else int(score)}")
        if grading:
            _wrapped_block("判分细节", grading)
        if graded and qno in qidx:
            ms_text = qidx[qno].get("original_mark_scheme_text", "")
            if (ms_text or "").strip():
                _wrapped_block("查看标准答案", ms_text)


def _show_section_preview(section_qs):
    for qno in section_qs:
        row = answers.get(qno) or {}
        score = row.get("score")
        grading = row.get("grading_text", "") or ""
        st.markdown(f"**{qno}** — 得分: {'' if score is None else int(score)}")
        if grading:
            _wrapped_block("Feedback", grading)

        # 仅保留“查看标准答案”，并支持自动换行
        if graded and qno in qidx:
            ms_text = qidx[qno].get("original_mark_scheme_text", "")
            if (ms_text or "").strip():
                _wrapped_block("查看标准答案", ms_text)


with section_tabs[1]:
    _show_section_preview(QS_B)
with section_tabs[2]:
    _show_section_preview(QS_C)
with section_tabs[3]:
    _show_section_preview(QS_D)

# ============================
# 导出 PDF
# ============================
def _answers_signature(head, answers, qidx, include_pending=False) -> str:
    sig_src = {
        "attempt_id": head.get("attempt_id"),
        "total_score": head.get("total_score"),
        "answers": {
            k: {
                "answer_text": (answers.get(k) or {}).get("answer_text"),
                "score": (answers.get(k) or {}).get("score"),
                "grading_text": (answers.get(k) or {}).get("grading_text"),
                # 不把 image_b64 全量入签名，避免过大：只取每题的图片数量 & 暂存张数
                "img_count": len(list_answer_images(head.get("attempt_id", ""), k)),
                "pending": (len(st.session_state.pending_images.get(head.get("attempt_id", ""), {}).get(k, {}))
                            if include_pending else 0),
            }
            for k in qidx.keys()
        },
    }
    raw = json.dumps(sig_src, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

if "pdf_cache" not in st.session_state:
    st.session_state.pdf_cache = {}  # { (sig, fast_mode, write_before): bytes }

# ============================
# 导出 PDF（仅快速模式）
# ============================
st.divider()
st.header("导出 PDF 报告")

# 固定仅使用快速模式
fast_mode = True
st.caption("提示：导出使用快速模式（更小图、更快生成）。评分阶段已将暂存图片写入数据库；导出不会再次写入。")

if st.button("📄 生成 PDF", use_container_width=True):
    with st.spinner("正在生成PDF…"):
        head2, answers2 = load_attempt(aid)

        # 不包含 pending；不再写库
        include_pending_map = None

        sig = _answers_signature(head2, answers2, qidx, include_pending=False)
        cache_key = (sig, fast_mode, True)  # 第三个布尔位仅用于区分缓存键

        pdf_bytes = st.session_state.pdf_cache.get(cache_key)
        if not pdf_bytes:
            pdf_bytes = export_pdf(head2, answers2, qidx, fast_mode=fast_mode, include_pending=include_pending_map)
            if pdf_bytes and len(pdf_bytes) > 0:
                st.session_state.pdf_cache[cache_key] = pdf_bytes

        if not pdf_bytes or len(pdf_bytes) == 0:
            st.error("生成 PDF 失败：得到空字节流。请重试或检查题目/答案数据。")
        else:
            file_name = f"{head2.get('student_name', '')}_{head2.get('attempt_id', '')}.pdf" or "report.pdf"
            st.download_button(
                label="⬇️ 下载 PDF",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
                key=f"pdf_dl_{aid}_fast_{int(time.time())}",
            )
            st.success("PDF 已生成，可点击按钮下载！")
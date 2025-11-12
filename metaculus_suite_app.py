import os
import io
import re
import csv
import time
import json
import hashlib
import zipfile
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# ================================
# Global config & shared constants
# ================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

BASE = "https://www.metaculus.com"
API2 = f"{BASE}/api2"
API  = f"{BASE}/api"

# ---- Metaculus token (hardcoded as requested) ----
METACULUS_TOKEN = "a5217e06384668c082997ec2d5b0c68945497b43".strip()

UA_QS = {
    "User-Agent": "metaculus-question-factors/1.0 (+python-requests)",
    "Authorization": f"Token {METACULUS_TOKEN}",
}
UA_COM = {
    "User-Agent": "metaculus-comments-llm-scorer/1.1 (+python-requests)",
    "Authorization": f"Token {METACULUS_TOKEN}",
}
UA_DLZIP = {
    "User-Agent": "metaculus-project-dl/1.0 (+python-requests)",
    "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
}

HTTP_QS = requests.Session()
HTTP_COM = requests.Session()
HTTP_DL  = requests.Session()

PREFERRED_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "anthropic/claude-3.5-sonnet",
    "qwen/qwen-2.5-32b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-large-2411",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]

# ================================
# Shared helpers
# ================================

def ascii_safe(s: str) -> str:
    try:
        return s.encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return "".join(ch for ch in s if ord(ch) < 256)


def get_openrouter_key() -> str:
    v = st.session_state.get("OPENROUTER_API_KEY_OVERRIDE", "").strip() if "OPENROUTER_API_KEY_OVERRIDE" in st.session_state else ""
    if not v:
        try:
            if "OPENROUTER_API_KEY" in st.secrets:
                v = str(st.secrets["OPENROUTER_API_KEY"]).strip()
        except Exception:
            pass
    if not v:
        v = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return v


def or_headers(x_title: str = "Metaculus Suite", referer: str = "https://localhost", user_agent: str = "metaculus-suite/1.0") -> Dict[str, str]:
    key = get_openrouter_key()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": ascii_safe(referer),
        "X-Title": ascii_safe(x_title),
        "User-Agent": ascii_safe(user_agent),
    }


@st.cache_data(show_spinner=False, ttl=300)
def list_models_clean() -> List[Dict[str, Any]]:
    """Fetch OpenRouter models (returns [] if key missing/unavailable)."""
    try:
        r = requests.get(OPENROUTER_MODELS, headers=or_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        ms = data.get("data") or data.get("models") or []
    except Exception:
        return []
    out = []
    for m in ms:
        out.append(
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "context_length": m.get("context_length") or m.get("max_context_length"),
                "pricing": m.get("pricing") or {},
                "tags": m.get("tags") or [],
                "arch": m.get("architecture"),
            }
        )
    return out


def pick_model(user_choice: Optional[str] = None) -> str:
    if user_choice:
        return user_choice
    env_model = os.environ.get("OPENROUTER_MODEL", "").strip()
    if env_model:
        return env_model
    ms = list_models_clean()
    if ms:
        ids = {m.get("id"): m for m in ms if m.get("id")}
        for mid in PREFERRED_MODELS:
            if mid in ids:
                return mid
        best_id, best_price = None, 1e9
        for m in ms:
            id_ = (m.get("id") or "").lower()
            tags = " ".join((m.get("tags") or [])).lower()
            arch = (m.get("arch") or "").lower()
            if ("instruct" in id_) or ("instruct" in tags) or ("instruct" in arch):
                pr = (m.get("pricing") or {})
                p = pr.get("prompt") or pr.get("input") or 0.0
                try:
                    p = float(p) if p else 0.0
                except Exception:
                    p = 0.0
                if p < best_price:
                    best_price, best_id = p, (m.get("id") or "")
        if best_id:
            return best_id
    return PREFERRED_MODELS[0]


def _resolve_model_from_sidebar(base_key: str, fallback: Optional[str] = None) -> str:
    custom_key = f"MODEL_{base_key}_CUSTOM"
    dd_key = f"MODEL_{base_key}"
    custom = st.session_state.get(custom_key, "")
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    dd = st.session_state.get(dd_key, "")
    if isinstance(dd, str) and dd.strip():
        return dd.strip()
    return pick_model(fallback)


def call_openrouter(messages: List[Dict[str, str]], model: str, max_tokens: int = 220, temperature: float = 0.0,
                    retries: int = 3, expect: str = "object", title_hint: str = "Metaculus Comment Scorer",
                    ua_hint: str = "metaculus-comments-llm-scorer/1.1") -> Any:
    payload = {"model": model, "messages": messages, "temperature": temperature, "top_p": 1, "max_tokens": max_tokens}
    last = None
    for k in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=or_headers(title_hint, user_agent=ua_hint), json=payload, timeout=120)
            if r.status_code == 404:
                raise RuntimeError("404 No endpoints for model")
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", "2") or 2)
                time.sleep(min(retry_after, 10))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            ch = data.get("choices") or []
            if not ch:
                raise RuntimeError("No choices in response")
            content = ch[0].get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("Empty content")
            return parse_json_relaxed(content, expect=expect)
        except Exception as e:
            last = e
            time.sleep(0.7 * (k + 1))
    raise RuntimeError(f"[openrouter] giving up after retries: {repr(last)}")


def parse_json_relaxed(s: str, expect: str = "auto") -> Any:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            s = m.group(1).strip()

    def balanced_slice(s: str, open_char: str, close_char: str) -> Optional[str]:
        start = s.find(open_char)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None

    if expect in ("array", "auto"):
        blk = balanced_slice(s, "[", "]")
        if blk:
            try:
                return json.loads(blk)
            except Exception:
                pass
    blk = balanced_slice(s, "{", "}")
    if blk:
        try:
            return json.loads(blk)
        except Exception:
            pass
    objs = []
    for m in re.finditer(r"\{.*?\}", s, flags=re.DOTALL):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            continue
    if objs:
        return objs if len(objs) > 1 else objs[0]
    raise ValueError("Could not parse JSON from model output")


def start_new_run():
    for k in [
        "score_df",
        "score_agg_q",
        "score_agg_author",
        "OPENROUTER_API_KEY_OVERRIDE",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    list_models_clean.clear()
    st.rerun()

# ================================
# Metaculus fetchers (shared)
# ================================

def _get(http: requests.Session, url: str, params: Optional[Dict[str, Any]], ua: Dict[str, str]) -> Dict[str, Any]:
    r = http.get(url, params=params or {}, headers=ua, timeout=30)
    if r.status_code == 429:
        wait = float(r.headers.get("Retry-After", "1") or 1)
        time.sleep(min(wait, 10))
        r = http.get(url, params=params or {}, headers=ua, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=300)
def fetch_recent_questions(n_subjects: int = 10, page_limit: int = 80) -> List[Dict[str, Any]]:
    data = _get(HTTP_QS, f"{API2}/questions/", {"status": "open", "limit": page_limit}, UA_QS)
    results = data.get("results") or data.get("data") or []

    def ts(q):
        return q.get("open_time") or q.get("created_at") or q.get("scheduled_close_time") or ""

    results.sort(key=ts, reverse=True)
    out = []
    for q in results[:n_subjects]:
        qid = q.get("id")
        if not qid:
            continue
        out.append(
            {
                "id": qid,
                "title": q.get("title", ""),
                "url": q.get("page_url") or q.get("url") or f"{BASE}/questions/{qid}/",
                "body": q.get("description") or q.get("body") or q.get("background") or q.get("text") or "",
            }
        )
    return out


def fetch_question_by_id(qid: int) -> Optional[Dict[str, Any]]:
    try:
        q = _get(HTTP_QS, f"{API2}/questions/{qid}/", None, UA_QS)
        if not q or "id" not in q:
            return None
        return {
            "id": q["id"],
            "title": q.get("title", f"Question {qid}"),
            "url": q.get("page_url") or q.get("url") or f"{BASE}/questions/{qid}/",
            "body": q.get("description") or q.get("body") or q.get("background") or q.get("text") or "",
        }
    except Exception:
        return None


def fetch_comments_for_post(post_id: int, page_limit: int = 120) -> List[Dict[str, Any]]:
    base = f"{API}/comments/"
    params = {"post": post_id, "limit": page_limit, "offset": 0, "sort": "-created_at", "is_private": "false"}
    out, url = [], base
    while url:
        data = _get(HTTP_COM, url, params if url == base else None, UA_COM)
        batch = data.get("results") or []
        out += batch
        nxt = data.get("next")
        if nxt:
            url = nxt
            time.sleep(0.2)
        else:
            if batch:
                params["offset"] = params.get("offset", 0) + params.get("limit", page_limit)
                url = base
                time.sleep(0.2)
            else:
                break
    return out


# -------- Artelar helpers (Tournament via project ZIP) --------

def _decode_bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", "replace")


def download_project_csv(project_id: int) -> bytes:
    """Download the data ZIP for a Metaculus project and return the bytes for question_data.csv."""
    url = f"{BASE}/api/projects/{int(project_id)}/download-data/"
    headers = dict(UA_DLZIP)
    headers["Authorization"] = f"Token {METACULUS_TOKEN}"
    params = {"include_comments": "false", "include_scores": "false"}
    r = HTTP_DL.get(url, headers=headers, params=params, timeout=90)
    if r.status_code != 200:
        msg_preview = (r.text or "")[:300]
        msg_preview = msg_preview.replace('\n', ' ').replace('\r', ' ')
        raise RuntimeError(f"Download refused ({r.status_code}): {msg_preview}")
    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        names = zf.namelist()
        target = next((n for n in names if n.lower().endswith("question_data.csv")), None)
        if not target:
            raise RuntimeError(f"'question_data.csv' missing in archive. Found: {names}")
        return zf.read(target)


def parse_question_csv_rows(data: bytes) -> List[Dict[str, Any]]:
    """Return list of dicts: {post_id:int, question_id:Optional[int], title:str, url:str} from question_data.csv."""
    text = _decode_bytes_to_text(data)
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []

    def parse_int(x):
        if x is None:
            return None
        s = str(x).strip().replace("\u00a0", "").replace(",", "")
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            m = re.search(r"\d+", s)
            return int(m.group(0)) if m else None

    for row in reader:
        qid = parse_int(row.get("Question ID") or row.get("question_id") or row.get("id"))
        pid = parse_int(row.get("Post ID") or row.get("post_id"))
        title = (row.get("Question Title") or row.get("title") or "").strip()
        url = (row.get("Question URL") or row.get("url") or "").strip()
        if not url and qid:
            url = f"{BASE}/questions/{qid}/"
        if not url and pid:
            url = f"{BASE}/posts/{pid}/"
        if pid is None:
            continue
        rows.append({"post_id": pid, "question_id": qid, "title": title, "url": url})
    return rows

# ================================
# Comment Scorer (LLM)
# ================================

SYSTEM_PROMPT_SCORE = (
    "You are a strict rater for a forecasting forum.\n\n"
    "Return ONLY valid JSON with keys:\n"
    "- score (integer 1..6)\n"
    "- rationale (string, <=180 chars)\n"
    "- flags (object with booleans: off_topic, toxicity, low_effort, has_evidence, likely_ai)\n"
    "- evidence_urls (array of http/https URLs; <=5, deduplicated; MUST be [] if has_evidence=false)\n\n"
    "Primary goal: reward deep, decision-useful analysis that helps update a forecast, even if it cites few/no links. Do NOT overvalue link counts.\n\n"
    "Scoring rubric (weighting guidance):\n"
    "- Depth & originality (40%): causal mechanisms, scenarios, uncertainty decomposition, alternative hypotheses, sensitivity to assumptions.\n"
    "- Forecast linkage (25%): explicit/implicit update logic (e.g., direction/size/timing).\n"
    "- Quantification & falsifiability (15%): numbers, ranges, base rates, likelihoods.\n"
    "- Evidence use (10%): credible sources are a plus but not required if reasoning is strong.\n"
    "- Clarity & civility (10%): coherent, non-toxic, concise.\n\n"
    "Anchor points:\n"
    "1 = trivial/emoji/thanks/toxic; no update value\n"
    "2 = notes an issue or intuition but no clear impact on forecast\n"
    "3 = some relevant info OR sources but weak update logic\n"
    "4 = solid reasoning OR quantified estimate; clear link to forecast\n"
    "5 = strong, structured analysis with scenarios and/or quantified update\n"
    "6 = excellent, original synthesis; decomposes uncertainty and gives a justified update\n\n"
    "Edge rules:\n"
    "- Be conservative with score inflation; reserve 6 for standout analysis.\n"
    "- Penalize source dumps with no update logic.\n"
    "- Long quotes or AI-ish boilerplate -> likely_ai=true.\n"
    "- Rationale <=180 chars; output JSON only.\n"
)


FEWSHOTS_SCORE = [
    {"role": "user", "content": "TEXT: Thanks for sharing!"},
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "score": 1,
                "rationale": "Trivial acknowledgement only.",
                "flags": {
                    "off_topic": False,
                    "toxicity": False,
                    "low_effort": True,
                    "has_evidence": False,
                    "likely_ai": False,
                },
                "evidence_urls": [],
            }
        ),
    },
    {"role": "user", "content": "TEXT: Anyone who thinks this will happen is an idiot."},
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "score": 1,
                "rationale": "Toxic with no evidence.",
                "flags": {
                    "off_topic": False,
                    "toxicity": True,
                    "low_effort": True,
                    "has_evidence": False,
                    "likely_ai": False,
                },
                "evidence_urls": [],
            }
        ),
    },
    {"role": "user", "content": "TEXT: Turnout fell 3–5% vs 2020 in key counties (CSV). I estimate P(win)=0.56."},
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "score": 4,
                "rationale": "Quantified comparison with evidence pointer.",
                "flags": {
                    "off_topic": False,
                    "toxicity": False,
                    "low_effort": False,
                    "has_evidence": True,
                    "likely_ai": False,
                },
                "evidence_urls": [],
            }
        ),
    },
]


def build_msgs_score(qtitle: str, qurl: str, text: str, cid: int, aid: Optional[int], votes: Optional[int]) -> List[Dict[str, str]]:
    u = (
        "Rate this comment for quality.\n\n"
        f"QUESTION_TITLE: {qtitle}\nQUESTION_URL: {qurl}\n\n"
        f"COMMENT_ID: {cid}\nAUTHOR_ID: {aid}\nVOTE_SCORE: {votes}\nTEXT:\n{text}\n\n"
        'Return JSON: {"score":1|2|3|4|5,"rationale":"...",'
        '"flags":{"off_topic":bool,"toxicity":bool,"low_effort":bool,"has_evidence":bool,"likely_ai":bool},'
        '"evidence_urls":["..."]}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT_SCORE}] + FEWSHOTS_SCORE + [{"role": "user", "content": u}]


_cache_score: Dict[str, Dict[str, Any]] = {}


def score_with_llm(qtitle: str, qurl: str, c: Dict[str, Any], model: str) -> Dict[str, Any]:
    text = (c.get("text") or "").strip()
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key not in _cache_score:
        msgs = build_msgs_score(qtitle, qurl, text, c.get("id"), (c.get("author") or {}).get("id"), c.get("vote_score"))
        resp = call_openrouter(
            msgs,
            model,
            max_tokens=220,
            temperature=0.0,
            expect="object",
            title_hint="Metaculus Comment Scorer",
            ua_hint="metaculus-comments-llm-scorer/1.1",
        )
        _cache_score[key] = resp
    return _cache_score[key]

# ================================
# Downloads fragment (prevents full rerun on download)
# ================================

def _downloads_ui(df: pd.DataFrame, agg_q: pd.DataFrame, agg_author: pd.DataFrame):
    st.success(f"{len(df)} comments scored across {df['market_id'].nunique()} question(s).")
    st.dataframe(df, use_container_width=True, height=400)

    st.markdown("#### Aggregated — by question")
    if isinstance(agg_q, pd.DataFrame) and not agg_q.empty:
        st.dataframe(agg_q, use_container_width=True)
    else:
        st.info("No question-level aggregation available yet.")

    st.markdown("#### Aggregated — by commenter")
    if isinstance(agg_author, pd.DataFrame) and not agg_author.empty:
        st.dataframe(agg_author, use_container_width=True)
    else:
        st.info("No commenter-level aggregation available yet.")

    def to_csv_bytes(frame: pd.DataFrame) -> bytes:
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "💾 Download RAW CSV",
        data=to_csv_bytes(df),
        file_name="metaculus_comment_scores_raw.csv",
        mime="text/csv",
        key="dl_raw_csv",
    )
    c2.download_button(
        "💾 Download by QUESTION CSV",
        data=to_csv_bytes(agg_q),
        file_name="metaculus_comment_scores_by_question.csv",
        mime="text/csv",
        key="dl_by_q_csv",
    )
    c3.download_button(
        "💾 Download by COMMENTER CSV",
        data=to_csv_bytes(agg_author),
        file_name="metaculus_comment_scores_by_commenter.csv",
        mime="text/csv",
        key="dl_by_author_csv",
    )

    st.markdown("---")
    st.button("🔁 Press here for a new run", key="newrun_score_bottom", on_click=start_new_run)


# Prefer fragment if available
if hasattr(st, "fragment"):
    @st.fragment
    def downloads_fragment(df: pd.DataFrame, agg_q: pd.DataFrame, agg_author: pd.DataFrame):
        _downloads_ui(df, agg_q, agg_author)
else:
    def downloads_fragment(df: pd.DataFrame, agg_q: pd.DataFrame, agg_author: pd.DataFrame):
        _downloads_ui(df, agg_q, agg_author)

# ================================
# Comment Scorer UI/logic
# ================================

def run_comment_scorer():
    st.subheader("🔍 Comment Scorer")
    st.caption("Fetch Metaculus comments and rate quality with an LLM. Exports raw and aggregated CSVs.")

    colA, colB = st.columns([2, 1])
    with colA:
        mode = st.radio(
            "Mode",
            [
                "Score recent questions",
                "Score specific IDs",
                "Tournament (Artelar CSV)",  # replaces old numeric tournament mode
            ],
            horizontal=True,
        )
    with colB:
        if st.button("🔁 Press here for a new run", key="newrun_score_top"):
            start_new_run()

    model_choice = _resolve_model_from_sidebar("DEFAULT")
    comments_limit = st.number_input(
        "Max comments per question",
        min_value=10,
        max_value=500,
        value=120,
        step=10,
    )

    with st.form("scorer_form", clear_on_submit=False):
        qids: List[int] = []
        project_id: Optional[int] = None
        n_from_csv: Optional[int] = None

        if mode == "Score recent questions":
            n = st.number_input(
                "Number of recent open questions",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key="recent_n",
            )

        elif mode == "Score specific IDs":
            qids_str = st.text_area(
                "Metaculus Question IDs (comma or space separated)",
                placeholder="Example: 12345, 67890, 13579",
                key="ids_str",
            )
            if qids_str.strip():
                for chunk in qids_str.replace(",", " ").split():
                    try:
                        qids.append(int(chunk))
                    except ValueError:
                        pass

        elif mode == "Tournament (Artelar CSV)":
            project_id = st.number_input(
                "Tournament / Project ID (numeric)",
                min_value=1,
                step=1,
                value=32821,
                help="Metaculus project/tournament numeric ID.",
            )
            n_from_csv = st.number_input(
                "How many questions to score (from CSV order)",
                min_value=1,
                value=10,
                step=1,
                help="We take the first N questions from question_data.csv",
            )

        submitted = st.form_submit_button("▶️ Run scoring", type="primary")

    if submitted:
        rows: List[Dict[str, Any]] = []
        try:
            model = model_choice

            # ---- Resolve subjects based on mode ----
            if mode == "Score recent questions":
                subjects = fetch_recent_questions(
                    n_subjects=int(st.session_state.get("recent_n", 10)), page_limit=80
                )

            elif mode == "Score specific IDs":
                subjects = []
                for q in qids:
                    s = fetch_question_by_id(q)
                    if s:
                        subjects.append(s)

            elif mode == "Tournament (Artelar CSV)":
                if not project_id:
                    st.warning("Please provide a valid tournament/project ID.")
                    subjects = []
                else:
                    # 1) Download project ZIP → question_data.csv
                    csv_bytes = download_project_csv(int(project_id))
                    rows_all = parse_question_csv_rows(csv_bytes)
                    if not rows_all:
                        st.warning("No usable rows in question_data.csv (missing Post ID).")
                        subjects = []
                    else:
                        # 2) Take first N rows
                        k = int(n_from_csv or 1)
                        rows_sel = rows_all[:k]
                        df_ids = pd.DataFrame(rows_sel)
                        with st.expander("Selected questions (from CSV)", expanded=False):
                            st.dataframe(df_ids, use_container_width=True, height=260)
                            st.download_button(
                                "⬇️ Download selected (CSV)",
                                data=df_ids.to_csv(index=False).encode("utf-8"),
                                file_name="tournament_questions.csv",
                                mime="text/csv",
                                key="dl_tourn_ids_csv",
                            )
                        # 3) Convert to subjects compatible with scorer
                        subjects = [
                            {
                                "id": r.get("question_id") or r.get("post_id"),
                                "title": r.get("title") or (f"Post {r.get('post_id')}") ,
                                "url": r.get("url") or (f"{BASE}/posts/{r.get('post_id')}/"),
                                # body not required for scoring prompt
                            }
                            for r in rows_sel
                        ]
                        # Keep auxiliary mapping from subject id to post_id (to ensure we fetch comments by post)
                        subj_to_post: Dict[int, int] = {}
                        for r in rows_sel:
                            key_id = (r.get("question_id") or r.get("post_id"))
                            if key_id is not None:
                                subj_to_post[int(key_id)] = int(r.get("post_id"))

            else:
                subjects = []

            if not subjects:
                st.warning("No questions found.")
            else:
                # ---- Scoring loop ----
                prog = st.progress(0.0)
                status = st.empty()
                total = len(subjects)
                for i, s in enumerate(subjects, 1):
                    status.info(f"Processing {i}/{total}: [{s['id']}] {s['title']}")

                    # In Artelar mode we must fetch comments by POST ID if available
                    if mode == "Tournament (Artelar CSV)":
                        # map subject to post_id; fallback to using subject id directly
                        sid = int(s["id"]) if s.get("id") is not None else None
                        post_id = None
                        if sid is not None and 'subj_to_post' in locals():
                            post_id = subj_to_post.get(sid)
                        if not post_id:
                            # fallback: attempt to use id as post id
                            post_id = sid
                        comments = fetch_comments_for_post(int(post_id), page_limit=int(comments_limit)) if post_id else []
                    else:
                        # In the other modes, Metaculus comment endpoint expects 'post' (same as question id for questions)
                        comments = fetch_comments_for_post(int(s["id"]), page_limit=int(comments_limit))

                    for c in comments:
                        text = " ".join((c.get("text") or "").split())
                        if not text:
                            continue
                        a = c.get("author") or {}
                        resp = score_with_llm(s.get("title", ""), s.get("url", ""), c, model)
                        try:
                            score = int(max(1, min(5, round(float(resp.get("score", 3))))))
                        except Exception:
                            score = 3
                        rows.append(
                            {
                                "poster_id": a.get("id"),
                                "poster_username": (a.get("username") or a.get("name") or ""),
                                "comment_id": c.get("id"),
                                "market_id": s["id"],
                                "ai_score": score,
                                "rationale": resp.get("rationale", ""),
                                "flags": json.dumps(resp.get("flags") or {}),
                                "evidence_urls": ";".join(resp.get("evidence_urls") or []),
                                "comment_text": text,
                                "question_title": s.get("title", ""),
                                "question_url": s.get("url", ""),
                            }
                        )
                    prog.progress(i / total)
                status.success("Done.")
        except Exception as e:
            st.error(f"Error: {e}")
            rows = []

        # ---- Aggregate + persistent state for downloads ----
        if rows:
            df = pd.DataFrame(rows)
            agg_q = (
                df.groupby(["market_id", "question_title", "question_url"])
                .agg(
                    n_comments=("ai_score", "size"),
                    avg_score=("ai_score", "mean"),
                    p_low=("ai_score", lambda x: (x <= 2).mean()),
                    p_high=("ai_score", lambda x: (x >= 4).mean()),
                )
                .reset_index()
            )
            agg_author = (
                df.groupby(["poster_id", "poster_username"])
                .agg(
                    n_comments=("ai_score", "size"),
                    avg_score=("ai_score", "mean"),
                    p_low=("ai_score", lambda x: (x <= 2).mean()),
                    p_high=("ai_score", lambda x: (x >= 4).mean()),
                )
                .reset_index()
            )
            st.session_state["score_df"] = df
            st.session_state["score_agg_q"] = agg_q
            st.session_state["score_agg_author"] = agg_author

    # Always render existing results without recomputing
    if "score_df" in st.session_state and isinstance(st.session_state["score_df"], pd.DataFrame):
        df = st.session_state["score_df"]
        agg_q = st.session_state.get("score_agg_q", pd.DataFrame())
        agg_author = st.session_state.get("score_agg_author", pd.DataFrame())
        downloads_fragment(df, agg_q, agg_author)
    else:
        st.info("No comments were scored.")


# ================================
# Stubs for the other modules (unchanged)
# ================================

def run_question_factors():
    st.subheader("📈 Question Factors (LLM)")
    st.info("This section is unchanged in this patch.")


def run_question_generator():
    st.subheader("🧪 AI Question Generator")
    st.info("This section is unchanged in this patch.")


# ================================
# App Shell
# ================================

st.set_page_config(
    page_title="Metaculus Suite (Comments • Q-Gen • Factors)",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Metaculus Suite — Comments • Question Generation • Factors")

with st.sidebar:
    st.header("🔐 API & Models")
    DEBUG = st.checkbox("Debug mode (verbose)", value=False)
    st.caption("Enter your OpenRouter key here. It overrides env/secrets for this session.")
    api_key_input = st.text_input("OpenRouter API key", type="password", key="OPENROUTER_API_KEY_SIDEBAR")
    if api_key_input:
        st.session_state["OPENROUTER_API_KEY_OVERRIDE"] = api_key_input.strip()

    cols = st.columns([1, 1])
    if cols[0].button("🔄 Refresh model list", use_container_width=True):
        list_models_clean.clear()
    if cols[1].button("🧹 Reset all state", use_container_width=True):
        start_new_run()

    try:
        models = list_models_clean()
    except Exception:
        models = []
    model_ids = sorted([m.get("id") for m in models if m.get("id")]) or []
    default_model_id = pick_model(None)

    st.divider()
    st.subheader("Defaults (Scorer & Factors)")
    st.selectbox(
        "Default model",
        options=model_ids or [default_model_id],
        index=(model_ids or [default_model_id]).index(default_model_id)
        if (model_ids or [default_model_id])
        else 0,
        key="MODEL_DEFAULT",
        help="Used by Comment Scorer and Question Factors unless a custom model ID is set below.",
    )
    st.text_input(
        "Custom model ID (optional)",
        key="MODEL_DEFAULT_CUSTOM",
        placeholder="e.g., openai/gpt-4.1 or anthropic/claude-3.5-sonnet",
    )

    st.divider()
    st.subheader("Question Generator models")
    c1, c2, c3 = st.columns(3)
    c1.selectbox(
        "Generator",
        options=model_ids or [default_model_id],
        index=(model_ids or [default_model_id]).index(default_model_id)
        if (model_ids or [default_model_id])
        else 0,
        key="MODEL_GEN",
    )
    c2.selectbox(
        "Critique",
        options=model_ids or [default_model_id],
        index=(model_ids or [default_model_id]).index(default_model_id)
        if (model_ids or [default_model_id])
        else 0,
        key="MODEL_CRIT",
    )
    c3.selectbox(
        "Judge",
        options=model_ids or [default_model_id],
        index=(model_ids or [default_model_id]).index(default_model_id)
        if (model_ids or [default_model_id])
        else 0,
        key="MODEL_JUDGE",
    )
    c1.text_input("Custom generator ID", key="MODEL_GEN_CUSTOM", placeholder="Optional override")
    c2.text_input("Custom critique ID", key="MODEL_CRIT_CUSTOM", placeholder="Optional override")
    c3.text_input("Custom judge ID", key="MODEL_JUDGE_CUSTOM", placeholder="Optional override")

    st.caption("Tip: custom IDs let you call any model you have access to, even if it isn't listed.")

mode = st.radio(
    "Choose a module",
    ["Comment Scorer", "Question Generator", "Question Factors"],
    horizontal=True,
)

if mode == "Comment Scorer":
    run_comment_scorer()
elif mode == "Question Generator":
    run_question_generator()
else:
    run_question_factors()

st.caption("Tip: add OPENROUTER_API_KEY in app secrets (Settings → Secrets) or enter it here.")


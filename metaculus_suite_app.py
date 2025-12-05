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


def or_headers(
    x_title: str = "Metaculus Suite",
    referer: str = "https://localhost",
    user_agent: str = "metaculus-suite/1.0",
) -> Dict[str, str]:
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


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 220,
    temperature: float = 0.0,
    retries: int = 3,
    expect: str = "object",  # "object" | "array" | "raw"
    title_hint: str = "Metaculus Comment Scorer",
    ua_hint: str = "metaculus-comments-llm-scorer/1.1",
) -> Any:
    """
    Wrapper around the OpenRouter chat API.

    - If expect == "raw": returns the raw text content (no JSON parsing).
    - Else: parse JSON using parse_json_relaxed with the given expectation.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    last = None
    for k in range(retries):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=or_headers(title_hint, user_agent=ua_hint),
                json=payload,
                timeout=120,
            )
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
            if expect == "raw":
                return content
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

    # Try fenced code block
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
    global _cache_score
    _cache_score.clear()  # reset du cache de scoring pour permettre le changement de modèle

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

# >>> FIRST STAGE: BIG / VERBOSE MODEL WITH EXIGEANT SCORING CRITERIA <<<
SYSTEM_PROMPT_SCORE = """
You are a narrow scoring module inside a larger pipeline.
Your ONLY job is to rate Metaculus comments for quality in the AI Pathways Tournament.

You are NOT a general-purpose assistant.
Do NOT brainstorm, speculate, or explore side topics. DO NOT THINK, DO NOT THINK, OUTPUT RESULTS AS FAST AS POSSIBLE. BE EFFICIENT. BE INSTANT.

Your output in this FIRST STAGE may be FREE-FORM NATURAL LANGUAGE.
You DO NOT need to output JSON here.

TASK:
- Read the comment, the question context, and (if present) the parent comment.
- If a parent comment is provided, treat the comment as a reply and assess:
  - how well it answers the parent,
  - how accurately it engages with the parent's claims,
  - and whether it productively advances the discussion.
- Decide on:
  - score: integer 1..6
  - rationale: short textual justification (<=180 chars ideally, absolutely no long paragraphs)
  - flags: off_topic, toxicity, low_effort, has_evidence, likely_ai
  - evidence_urls: any http/https URLs referenced or clearly implied

STRICT FORMAT AND BREVITY:
- Keep the answer short and structured.
- Aim for at most 6–8 short lines total.
- NO headings, NO bullet lists, NO meta commentary.
- Do NOT think out loud; just produce the final decision.

FOCUS (STRUCTURE FOR THE SECOND STAGE):
- Be clear and explicit about the rating you choose.
- Mention the score explicitly as "score = X".
- Mention flags explicitly, e.g. "flags: off_topic=false, toxicity=false, low_effort=true, has_evidence=false, likely_ai=false".
- List evidence URLs explicitly, or say "evidence_urls: []" if none.

SCORING WEIGHTS (THESE ARE STRICT AND MUST BE TAKEN SERIOUSLY):
The comments should be ranked based on how well they:
- Showcase clear, deep, and insightful reasoning, delving into the relevant mechanisms that affect the event in question. (40%)
- Offer useful insights about the overall scenario(s), the interactions between questions, or the relationship between the questions and the scenario(s). (30%)
- Provide valuable information and are based on the available evidence. (20%)
- Challenge the community's prediction or assumptions held by other forecasters. (10%)

ANCHOR POINTS (1–6 SCALE):
The comments do not need to have all the following characteristics; one strong attribute can sometimes compensate for weaker ones.
Use these anchors when deciding the score:

1 = Trivial, badly written, or completely unreasonable comment with no predictive value.
2 = Brief or slightly confused comment offering only surface value.
3 = Good comment with rational arguments or potentially useful information.
4 = Very good comment which explains solid reasoning in some detail and provides actionable information.
5 = Excellent comment with meaningful analysis, presenting a deep dive of the available information and arguments, and drawing original conclusions from it.
6 = Outstanding synthesis comment which clearly decomposes uncertainty, connects multiple questions or scenarios, and gives a compelling reason to significantly update forecasts.

ADDITIONAL CONSTRAINTS:
- Be conservative with high scores (5 and especially 6). Reserve them for comments that are clearly above the tournament median in insight and usefulness.
- Penalize comments that are long but vague, generic, or boilerplate.
- Penalize pure link dumps with little or no reasoning or forecast impact.
- Toxic or uncivil comments should receive low scores and toxicity=true.
- When in doubt between two adjacent scores, pick the lower one quickly rather than overthinking.

FLAGS INTERPRETATION:
- off_topic: true if the comment is largely unrelated to the question or the AI Pathways scenario.
- toxicity: true if the comment is hostile, insulting, or clearly uncivil.
- low_effort: true if the comment is very short, trivial, or adds almost nothing.
- has_evidence: true if the comment brings specific data, references, links, or clearly factual information.
- likely_ai: true if the comment is long, generic, and boilerplate-sounding with little specificity or real engagement.

You must stay strictly on task: rate, justify briefly, set flags, list URLs. Nothing else.
"""


# SECOND STAGE: small model → STRICT JSON CONVERTER
SYSTEM_PROMPT_JSON_CONVERT = """
You convert free-form rating text into STRICT JSON with this exact schema:
{
  "score": 1|2|3|4|5|6,
  "rationale": "<string, <=180 chars>",
  "flags": {
    "off_topic": true|false,
    "toxicity": true|false,
    "low_effort": true|false,
    "has_evidence": true|false,
    "likely_ai": true|false
  },
  "evidence_urls": ["<string>", "..."]
}

HARD RULES:
1. OUTPUT STRICT JSON ONLY.
   - No explanations
   - No comments
   - No Markdown
   - No code fences
   - No extra keys
   - No trailing commas

2. DO NOT THINK "OUT LOUD".
   - No chain-of-thought in the output.
   - No meta commentary.

3. If some information is missing in the raw text:
   - Use a safe default:
     - score: 3 if unclear
     - rationale: ""
     - flags: false for all unless clearly indicated
     - evidence_urls: [] if none clearly extractable

4. You MUST:
   - Parse any explicit "score = X" or similar notation.
   - Parse any obvious booleans for the flags.
   - Collect any http/https URLs as evidence_urls (deduplicate).

5. Final answer:
   - ONE JSON object
   - EXACTLY matching the schema above
   - No natural language outside JSON.
"""

FEWSHOTS_SCORE = [
    {"role": "user", "content": "TEXT: Thanks for sharing!"},
    {
        "role": "assistant",
        "content": "score = 1\nrationale: Trivial acknowledgement only.\nflags: off_topic=false, toxicity=false, low_effort=true, has_evidence=false, likely_ai=false\nevidence_urls: []",
    },
    {"role": "user", "content": "TEXT: Anyone who thinks this will happen is an idiot."},
    {
        "role": "assistant",
        "content": "score = 1\nrationale: Toxic with no evidence.\nflags: off_topic=false, toxicity=true, low_effort=true, has_evidence=false, likely_ai=false\nevidence_urls: []",
    },
    {"role": "user", "content": "TEXT: Turnout fell 3–5% vs 2020 in key counties (CSV). I estimate P(win)=0.56."},
    {
        "role": "assistant",
        "content": "score = 5\nrationale: Quantified comparison with evidence pointer.\nflags: off_topic=false, toxicity=false, low_effort=false, has_evidence=true, likely_ai=false\nevidence_urls: []",
    },
]


def build_msgs_score(
    qtitle: str,
    qurl: str,
    text: str,
    cid: int,
    aid: Optional[int],
    votes: Optional[int],
    parent_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    First-stage message: big/verbose model does free-form rating.
    Includes optional parent comment text to evaluate reply quality.
    """
    parent_block = ""
    if parent_text:
        parent_block = f"PARENT_COMMENT_TEXT:\n{parent_text}\n\n"

    u = (
        "Rate this comment for quality in FREE-FORM TEXT (FIRST STAGE).\n\n"
        f"QUESTION_TITLE: {qtitle}\nQUESTION_URL: {qurl}\n\n"
        f"{parent_block}"
        f"COMMENT_ID: {cid}\nAUTHOR_ID: {aid}\nVOTE_SCORE: {votes}\nTEXT:\n{text}\n\n"
        "You MUST:\n"
        "- Explicitly write the score as 'score = X' (X in 1..6).\n"
        "- Provide a short rationale line starting with 'rationale:'.\n"
        "- Provide flags in one line starting with 'flags:' and including all booleans.\n"
        "- Provide 'evidence_urls: [...]' listing URLs or [].\n\n"
        "Do NOT output JSON in this stage.\n"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT_SCORE}] + FEWSHOTS_SCORE + [{"role": "user", "content": u}]


def build_msgs_json_convert(raw_text: str) -> List[Dict[str, str]]:
    """
    Second-stage message: small model converts raw free-form into strict JSON.
    """
    u = (
        "Here is the raw rating text from another model:\n\n"
        f"{raw_text}\n\n"
        "Convert this into ONE JSON object following the schema exactly. "
        "Remember: JSON only, no Markdown, no explanations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_JSON_CONVERT},
        {"role": "user", "content": u},
    ]


_cache_score: Dict[str, Dict[str, Any]] = {}


def score_with_llm(
    qtitle: str,
    qurl: str,
    c: Dict[str, Any],
    model: str,
    parent_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Two-stage scoring pipeline:
    1) Big model (chosen by user) does free-form rating text.
    2) Small model (hard-coded gpt-4o-mini) converts that text into strict JSON.

    Now uses optional parent_text to judge how well the comment replies to its parent.
    """
    text = (c.get("text") or "").strip()
    parent_sig = (parent_text or "").strip()
    cache_key_raw = f"{model}||{text}||{parent_sig}"
    key = hashlib.sha256(cache_key_raw.encode("utf-8")).hexdigest()

    if key not in _cache_score:
        # ---- Stage 1: big / verbose model, raw free-form ----
        msgs_stage1 = build_msgs_score(
            qtitle,
            qurl,
            text,
            c.get("id"),
            (c.get("author") or {}).get("id"),
            c.get("vote_score"),
            parent_text=parent_text,
        )
        raw_output = call_openrouter(
            msgs_stage1,
            model,
            max_tokens=300,
            temperature=0.0,
            expect="raw",  # <== important: no JSON parsing here
            title_hint="Metaculus Comment Scorer – Stage 1 (Raw)",
            ua_hint="metaculus-comments-llm-scorer-stage1/1.0",
        )

        # ---- Stage 2: small robust model → strict JSON ----
        converter_model = "openai/gpt-4o-mini"
        msgs_stage2 = build_msgs_json_convert(raw_output)
        json_output = call_openrouter(
            msgs_stage2,
            converter_model,
            max_tokens=220,
            temperature=0.0,
            expect="object",
            title_hint="Metaculus Comment Scorer – JSON Convert",
            ua_hint="metaculus-comments-json-convert/1.0",
        )

        _cache_score[key] = json_output

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
    st.caption(f"🧠 Model used for STAGE 1 (raw rating): `{model_choice}`")
    st.caption("📏 STAGE 2 (JSON convert) uses `openai/gpt-4o-mini` hard-coded.")

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
                                "title": r.get("title") or (f"Post {r.get('post_id')}"),
                                "url": r.get("url") or (f"{BASE}/posts/{r.get('post_id')}/"),
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
                        sid = int(s["id"]) if s.get("id") is not None else None
                        post_id = None
                        if sid is not None and 'subj_to_post' in locals():
                            post_id = subj_to_post.get(sid)
                        if not post_id:
                            post_id = sid
                        comments = fetch_comments_for_post(int(post_id), page_limit=int(comments_limit)) if post_id else []
                    else:
                        comments = fetch_comments_for_post(int(s["id"]), page_limit=int(comments_limit))

                    for c in comments:
                        text = " ".join((c.get("text") or "").split())
                        if not text:
                            continue

                        # ---- NEW: fetch parent comment text if available ----
                        parent_text = None
                        parent_id = c.get("parent") or c.get("parent_comment") or c.get("in_reply_to")
                        if parent_id:
                            for cp in comments:
                                if cp.get("id") == parent_id:
                                    parent_text = (cp.get("text") or "").strip()
                                    break

                        a = c.get("author") or {}
                        resp = score_with_llm(
                            s.get("title", ""),
                            s.get("url", ""),
                            c,
                            model,
                            parent_text=parent_text,
                        )
                        try:
                            score = int(max(1, min(6, round(float(resp.get("score", 3))))))
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
        help="Used by Comment Scorer (stage 1) and Question Factors unless a custom model ID is set below.",
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import csv
import time
import json
import hashlib
import itertools
import random
import tempfile
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# ================================
# Global config & shared constants
# ================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

API2 = "https://www.metaculus.com/api2"
API = "https://www.metaculus.com/api"

UA_QS   = {"User-Agent": "metaculus-question-factors/1.0 (+python-requests)"}
UA_COM  = {"User-Agent": "metaculus-comments-llm-scorer/0.8 (+python-requests)"}
UA_QGEN = {"User-Agent": "metaculus-ai-qgen/1.2 (+python-requests)"}

HTTP_QS = requests.Session()
HTTP_COM = requests.Session()
HTTP_QGEN = requests.Session()

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
    # priority: session override -> Streamlit secrets -> env var
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
        # fallback: cheapest instruct-ish
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
    """
    Read model selection from sidebar controls:
    - prefer custom text field (MODEL_<BASE>_CUSTOM) if nonempty
    - else use dropdown (MODEL_<BASE>)
    - else fallback -> pick_model()
    """
    custom_key = f"MODEL_{base_key}_CUSTOM"
    dd_key = f"MODEL_{base_key}"
    custom = st.session_state.get(custom_key, "")
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    dd = st.session_state.get(dd_key, "")
    if isinstance(dd, str) and dd.strip():
        return dd.strip()
    return pick_model(fallback)

def call_openrouter(messages: List[Dict[str, str]], model: str, max_tokens: int = 1200, temperature: float = 0.0, retries: int = 3, expect: str = "auto", title_hint: str = "Metaculus Suite", ua_hint: str = "metaculus-suite/1.0") -> Any:
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
    # fenced
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL|re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            s = m.group(1).strip()
    # bracket slices
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
                    return s[start:i+1]
        return None
    if expect in ("array","auto"):
        blk = balanced_slice(s,"[","]")
        if blk:
            try: return json.loads(blk)
            except Exception: pass
    blk = balanced_slice(s,"{","}")
    if blk:
        try: return json.loads(blk)
        except Exception: pass
    # last resort: collect many objects
    objs = []
    for m in re.finditer(r"\{.*?\}", s, flags=re.DOTALL):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            continue
    if objs:
        return objs if len(objs)>1 else objs[0]
    raise ValueError("Could not parse JSON from model output")

def start_new_run():
    for k in list(st.session_state.keys()):
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
    def ts(q): return q.get("open_time") or q.get("created_at") or q.get("scheduled_close_time") or ""
    results.sort(key=ts, reverse=True)
    out = []
    for q in results[:n_subjects]:
        qid = q.get("id")
        if not qid: continue
        out.append(
            {
                "id": qid,
                "title": q.get("title", ""),
                "url": q.get("page_url") or q.get("url") or f"https://www.metaculus.com/questions/{qid}/",
                "body": q.get("description") or q.get("body") or q.get("background") or q.get("text") or "",
            }
        )
    return out

def fetch_question_by_id(qid: int) -> Optional[Dict[str, Any]]:
    try:
        q = _get(HTTP_QS, f"{API2}/questions/{qid}/", None, UA_QS)
        if not q or "id" not in q: return None
        return {
            "id": q["id"],
            "title": q.get("title", f"Question {qid}"),
            "url": q.get("page_url") or q.get("url") or f"https://www.metaculus.com/questions/{qid}/",
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

# ================================
# Comment Scorer (module A)
# ================================

SYSTEM_PROMPT_SCORE = (
    "You are a strict rater for a forecasting forum.\n\n"
    "Return ONLY valid JSON with keys:\n"
    "- score (integer 1..5)\n"
    "- rationale (string, <=180 chars)\n"
    "- flags (object with booleans: off_topic, toxicity, low_effort, has_evidence, likely_ai)\n"
    "- evidence_urls (array of http/https URLs; <=5, deduplicated; MUST be [] if has_evidence=false)\n\n"
    "Core principle: A good comment ties facts/arguments to how to update the forecast "
    "(priors/base rates, mechanisms, scenarios, timelines, probabilities). Listing facts without linking them "
    "to the forecast deserves a low score.\n\n"
    "Rubric:\n"
    "1 = Toxic/off-topic or irrelevant factual claims.\n"
    "2 = Low effort: generic, facts with no explicit update logic; unclear stance.\n"
    "3 = Adequate: takes a stance and makes at least one explicit link from facts to forecast, but shallow.\n"
    "4 = Good: clear reasoning that updates the forecast; uses base rates/mechanisms/scenarios; cites data.\n"
    "5 = Excellent: structured, novel insight; quantifies impact; multiple credible sources; transparent uncertainty.\n\n"
    "Flags:\n"
    "- off_topic: not about the forecast.\n"
    "- toxicity: insults/harassment/slurs.\n"
    "- low_effort: <50 words or fact list with no explicit forecast linkage.\n"
    "- has_evidence: true only if concrete sources/data are cited (prefer URLs).\n"
    "- likely_ai: style suggests AI; this does NOT cap the score.\n\n"
    "Evidence:\n"
    "- If sources are cited, set has_evidence=true and include up to 5 URLs in evidence_urls.\n"
    "- If no concrete source, has_evidence=false and evidence_urls=[].\n\n"
    "Edge rules:\n"
    "- Be conservative if uncertain.\n"
    "- Rationale must state how the forecast should be updated (direction/size/conditions) in <=180 chars.\n"
    "- Do NOT include any text outside the JSON.\n"
)

FEWSHOTS_SCORE = [
    {"role": "user", "content": "TEXT: Thanks for sharing!"},
    {"role": "assistant", "content": json.dumps({"score":1,"rationale":"Trivial acknowledgement only.","flags":{"off_topic":False,"toxicity":False,"low_effort":True,"has_evidence":False,"likely_ai":False},"evidence_urls":[]})},
    {"role": "user", "content": "TEXT: Anyone who thinks this will happen is an idiot."},
    {"role": "assistant", "content": json.dumps({"score":1,"rationale":"Toxic with no evidence.","flags":{"off_topic":False,"toxicity":True,"low_effort":True,"has_evidence":False,"likely_ai":False},"evidence_urls":[]})},
    {"role": "user", "content": "TEXT: Turnout fell 3–5% vs 2020 in key counties (CSV). I estimate P(win)=0.56."},
    {"role": "assistant", "content": json.dumps({"score":4,"rationale":"Quantified comparison with evidence pointer.","flags":{"off_topic":False,"toxicity":False,"low_effort":False,"has_evidence":True,"likely_ai":False},"evidence_urls":[]})},
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
        resp = call_openrouter(msgs, model, max_tokens=220, temperature=0.0, expect="object", title_hint="Metaculus Comment Scorer", ua_hint="metaculus-comments-llm-scorer/0.8")
        _cache_score[key] = resp
    return _cache_score[key]

def run_comment_scorer():
    st.subheader("🔍 Comment Scorer")
    st.caption("Fetch Metaculus comments and rate quality with an LLM. Exports raw and aggregated CSVs.")
    colA, colB = st.columns([2,1])
    with colA:
        mode = st.radio("Mode", ["Score recent questions", "Score specific IDs"], horizontal=True)
    with colB:
        if st.button("🧹 Start a new run"):
            start_new_run()

    # Use sidebar default model (or its custom override)
    model_choice = _resolve_model_from_sidebar("DEFAULT")

    comments_limit = st.number_input("Max comments per question", min_value=10, max_value=500, value=120, step=10)

    qids: List[int] = []
    if mode == "Score recent questions":
        n = st.number_input("Number of recent questions", min_value=1, max_value=100, value=10, step=1)
    else:
        qids_str = st.text_area("Metaculus IDs (comma or space separated)", placeholder="Example: 12345, 67890, 13579")
        if qids_str.strip():
            for chunk in qids_str.replace(",", " ").split():
                try:
                    qids.append(int(chunk))
                except ValueError:
                    pass

    if st.button("▶️ Run scoring", type="primary"):
        rows: List[Dict[str, Any]] = []
        try:
            model = model_choice
            if mode == "Score recent questions":
                subjects = fetch_recent_questions(n_subjects=int(n), page_limit=80)
            else:
                subjects = []
                for q in qids:
                    s = fetch_question_by_id(q)
                    if s: subjects.append(s)
            if not subjects:
                st.warning("No questions found.")
            else:
                prog = st.progress(0.0)
                status = st.empty()
                total = len(subjects)
                for i, s in enumerate(subjects, 1):
                    status.info(f"Processing {i}/{total}: [{s['id']}] {s['title']}")
                    comments = fetch_comments_for_post(s["id"], page_limit=int(comments_limit))
                    for c in comments:
                        text = " ".join((c.get("text") or "").split())
                        if not text: continue
                        a = c.get("author") or {}
                        resp = score_with_llm(s["title"], s["url"], c, model)
                        try:
                            score = int(max(1, min(5, round(float(resp.get('score', 3))))))
                        except Exception:
                            score = 3
                        rows.append(
                            {
                                "poster_id": a.get("id"),
                                "comment_id": c.get("id"),
                                "market_id": s["id"],
                                "ai_score": score,
                                "rationale": resp.get("rationale",""),
                                "flags": json.dumps(resp.get("flags") or {}),
                                "evidence_urls": ";".join(resp.get("evidence_urls") or []),
                                "comment_text": text,
                                "question_title": s.get("title",""),
                                "question_url": s.get("url",""),
                            }
                        )
                    prog.progress(i/total)
                status.success("Done.")
        except Exception as e:
            st.error(f"Error: {e}")
            rows = []

        if rows:
            df = pd.DataFrame(rows)
            st.success(f"{len(rows)} comments scored across {df['market_id'].nunique()} question(s).")
            st.dataframe(df, use_container_width=True, height=400)

            # Aggregations
            agg = (
                df.groupby(["market_id","question_title","question_url"])
                  .agg(n_comments=("ai_score","size"),
                       avg_score=("ai_score","mean"),
                       p_low=("ai_score", lambda x: (x<=2).mean()),
                       p_high=("ai_score", lambda x: (x>=4).mean()))
                  .reset_index()
            )
            st.markdown("#### Aggregated (by question)")
            st.dataframe(agg, use_container_width=True)

            # Downloads
            def to_csv_bytes(frame: pd.DataFrame) -> bytes:
                buf = io.StringIO()
                frame.to_csv(buf, index=False)
                return buf.getvalue().encode("utf-8")

            c1, c2 = st.columns(2)
            c1.download_button("💾 Download raw comment scores CSV", data=to_csv_bytes(df), file_name="metaculus_comment_scores_raw.csv", mime="text/csv")
            c2.download_button("💾 Download aggregated scores CSV", data=to_csv_bytes(agg), file_name="metaculus_comment_scores_aggregated.csv", mime="text/csv")
        else:
            st.info("No comments were scored.")

# ================================
# Question Factors (module B)
# ================================

SYSTEM_PROMPT_FACTORS = (
    "You are an expert forecasting analyst. Given a Metaculus question title and URL, "
    "RETURN ONLY a valid JSON object with keys: question_id (int|null), question_title (string), "
    "factors (array). Return between 3 and 5 factors. Each factor must be an object with keys: "
    "'factor' (short phrase, <=10 words), 'rationale' (<=220 chars), and 'confidence' (float between 0.0 and 1.0). "
    "**Rationale MUST be concrete** and include, in <=220 chars, these three items: "
    "(A) one observable indicator to monitor (e.g. 'monthly FX reserves', 'number of senior resignations'); "
    "(B) a suggested data source (e.g. 'IMF reports', 'official press release', 'Reuters'); "
    "(C) a plausible numeric threshold or pattern that would materially change the outlook (e.g. 'reserves drop >10% in 2 months'). "
    "Do NOT include any explanatory text outside the JSON. Do NOT invent clickable URLs — if you name sources, use generic well-known names only. "
    "Return only the JSON object and nothing else."
)

FEWSHOTS_FACTORS = [
    {
        "role": "user",
        "content": "TITLE: Will the NASA Administrator still be in office on December 31, 2025? URL: https://example/123",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "question_id": 123,
                "question_title": "Will the NASA Administrator still be in office on December 31, 2025?",
                "factors": [
                    {
                        "factor": "Political support",
                        "rationale": "Indicator: public endorsements & key committee statements; Source: Congressional record/major press; Threshold: 2+ committee chairs publicly call for removal -> sharply lowers chances.",
                        "confidence": 0.65,
                    },
                    {
                        "factor": "Agency turnover",
                        "rationale": "Indicator: number of senior resignations in 3 months; Source: agency press releases / Reuters; Threshold: >=2 senior exec resignations within 90 days signals instability.",
                        "confidence": 0.55,
                    },
                    {
                        "factor": "Legal/investigations",
                        "rationale": "Indicator: formal investigations or DOJ referral; Source: Inspector General/DOJ announcements; Threshold: public referral or indictment -> major negative impact.",
                        "confidence": 0.6,
                    },
                ],
            }
        ),
    },
]

_cache_factors: Dict[str, Dict[str, Any]] = {}

def parse_json_strict(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        a = s.find("{"); b = s.rfind("}")
        if a != -1 and b != -1:
            return json.loads(s[a:b+1])
        raise

def call_openrouter_factors(messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    payload = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1, "max_tokens": 450, "response_format": {"type":"json_object"}}
    last = None
    for k in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=or_headers("Metaculus Question Factors", user_agent="metaculus-question-factors/1.0"), json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get("Retry-After","2") or 2), 10)); continue
            r.raise_for_status()
            data = r.json()
            ch = data.get("choices") or []
            content = ch[0].get("message", {}).get("content","")
            return parse_json_strict(content)
        except Exception as e:
            last = e; time.sleep(0.6*(k+1))
    st.error(f"OpenRouter call failed: {last!r}")
    return {"question_id": None, "question_title": "", "factors": []}

def build_factors_msgs(qid: int, qtitle: str, qurl: str) -> List[Dict[str, str]]:
    u = (f"TITLE: {qtitle}\nURL: {qurl}\nQUESTION_ID: {qid}\n\n"
         "Produce JSON: {\"question_id\": int, \"question_title\": str, \"factors\":[{\"factor\":\"...\",\"rationale\":\"...\",\"confidence\":0.0}, ...]}\n"
         "Return 3-5 factors.")
    return [{"role":"system","content":SYSTEM_PROMPT_FACTORS}] + FEWSHOTS_FACTORS + [{"role":"user","content":u}]

def get_question_factors_with_llm(qid: int, qtitle: str, qurl: str, model: str) -> Dict[str, Any]:
    text = f"{qid}|{qtitle}"
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = _cache_factors.get(key)
    if cached: return cached
    msgs = build_factors_msgs(qid, qtitle, qurl)
    resp = call_openrouter_factors(msgs, model)
    if not isinstance(resp, dict):
        resp = {"question_id": qid, "question_title": qtitle, "factors": []}
    if "question_id" not in resp: resp["question_id"] = qid
    if "question_title" not in resp: resp["question_title"] = qtitle
    facs = resp.get("factors") or []
    normalized = []
    for f in facs[:5]:
        if not isinstance(f, dict): continue
        factor = (f.get("factor") or "").strip()
        rationale = (f.get("rationale") or "").strip()
        try: confidence = float(f.get("confidence", 0.0))
        except Exception: confidence = 0.0
        normalized.append({"factor": factor, "rationale": rationale, "confidence": max(0.0, min(1.0, confidence))})
    resp["factors"] = normalized
    _cache_factors[key] = resp
    return resp

def rows_from_subject(subject: Dict[str, Any], factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rank, f in enumerate(factors, 1):
        out.append(
            {
                "market_id": subject["id"],
                "title": subject["title"],
                "factor_rank": rank,
                "factor": f.get("factor"),
                "rationale": f.get("rationale"),
                "confidence": f.get("confidence"),
                "url": subject["url"],
            }
        )
    return out

def run_question_factors():
    st.subheader("📈 Question Factors (LLM)")
    st.caption("Generate concise, concrete forecasting factors for Metaculus questions. Export CSV.")
    colA, colB = st.columns([2,1])
    with colA:
        mode = st.radio("Input mode", ["Recent questions", "Specific IDs"], horizontal=True)
    with colB:
        if st.button("🧹 Start a new run"):
            start_new_run()

    # Use sidebar default model
    model_choice = _resolve_model_from_sidebar("DEFAULT")

    if mode == "Recent questions":
        n = st.number_input("How many recent questions?", min_value=1, max_value=50, value=3, step=1)
    else:
        qids_text = st.text_input("Comma-separated Metaculus IDs", placeholder="e.g., 14016, 13999, 12024")

    if st.button("▶️ Generate factors", type="primary"):
        try:
            used_model = model_choice
            if mode == "Recent questions":
                subjects = fetch_recent_questions(n_subjects=int(n), page_limit=80)
            else:
                ids = [int(x.strip()) for x in (qids_text or "").split(",") if x.strip()]
                subjects = []
                for qid in ids:
                    s = fetch_question_by_id(qid)
                    if s: subjects.append(s)
            if not subjects:
                st.warning("No subjects retrieved. Check your inputs.")
            else:
                rows: List[Dict[str, Any]] = []
                prog = st.progress(0.0)
                for i, s in enumerate(subjects, 1):
                    st.info(f"Processing {i}/{len(subjects)}: [{s['id']}] {s['title']}")
                    resp = get_question_factors_with_llm(s["id"], s["title"], s["url"], used_model)
                    facts = resp.get("factors") or []
                    rows.extend(rows_from_subject(s, facts))
                    prog.progress(i/len(subjects))
                if rows:
                    df = pd.DataFrame(rows, columns=["market_id","title","factor_rank","factor","rationale","confidence","url"])
                    st.success(f"Generated factors for {df['market_id'].nunique()} question(s).")
                    st.dataframe(df, use_container_width=True)
                    buf = io.StringIO(); df.to_csv(buf, index=False)
                    st.download_button("💾 Download CSV", data=buf.getvalue().encode("utf-8"), file_name="metaculus_question_factors.csv", mime="text/csv")
                else:
                    st.info("No factors returned.")
        except Exception as e:
            st.error(f"Run failed: {e!r}")

# ================================
# Question Generator (module C)
# ================================

GEN_SYS = "You are a senior Metaculus question writer. Return STRICT JSON only."
GEN_USER_TMPL = """Task: Generate {n} candidate forecasting questions matching Metaculus style.

Topic brief (3–6 lines):
{brief}

Domain tags: {tags} | Target horizon (if relevant): {horizon}

For EACH candidate, output an object with:
"title", "body", "resolution_criteria", "timeframe":{{"start":"...","end":"...","timezone":"UTC"}},
"canonical_source": ["Publisher names or URLs allowed"], "answer_type": "binary|numeric|date|multiple",
"proposed_bins_or_ranges": "...(if numeric)", "difficulty": "low|med|high",
"rationale": "why decision-relevant and non-trivial", "policy_notes": "safety/legal notes".

Constraints:
- Outcomes must be independently verifiable from public sources; cite canonical_source (publishers allowed; URLs optional).
- Include explicit end dates (UTC) and exact resolution checks; avoid vague terms unless thresholded.
- Title ≤ 100 chars; body 2–5 concise sentences.
- Return a STRICT JSON array of {n} objects; no commentary, no markdown fences. If you add prose/fences, output will be discarded.
Few-shot good examples:
{good_examples}

Few-shot bad/avoid examples (with reasons to avoid):
{bad_examples}
"""

CRIT_SYS = "You are a meticulous Metaculus question editor. Return STRICT JSON."
CRIT_USER_TMPL = """Given this candidate JSON, rate 1–5 on each dimension:
clarity, falsifiability, operationalization, usefulness, safety.
List 3 concrete edits to raise any score <5. Then return a revised candidate.

Return:
{{
 "scores": {{...}},
 "edits": ["...","...","..."],
 "revised_candidate": {{...}}
}}

Candidate:
{candidate_json}
"""

JUDGE_SYS = "You are a strict Metaculus adjudicator. Return STRICT JSON only."
JUDGE_USER_TMPL = """Apply this rubric (1–5 each): clarity, falsifiability, operationalization, usefulness, safety.
Give overall (mean) and short notes. Return:
{{"scores":{{"clarity":int,"falsifiability":int,"operationalization":int,"usefulness":int,"safety":int}},"overall":X.X,"blockers":["..."],"notes":"..."}}

Candidate:
{candidate_json}
"""

PAIRWISE_SYS = "You are a strict adjudicator. Return STRICT JSON only."
PAIRWISE_USER_TMPL = """Compare Candidate A vs B for expected forecasting value to Metaculus users,
holding to the rubric. Pick a winner in {{"winner":"A"|"B","reason":"≤2 lines"}}.

A:
{A}

B:
{B}
"""

def load_examples_csv(path: str, k_good: int = 3, k_bad: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not path or not os.path.exists(path):
        return [], []
    goods, bads = [], []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    def is_good(r: Dict[str, Any]) -> bool:
        t = (r.get("resolution_criteria") or r.get("resolution") or "").lower()
        return ("utc" in t or " by " in t or " on " in t) and ("will " in (r.get("title", "").lower()))

    def row2obj(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": r.get("title") or r.get("question_title") or "",
            "body": r.get("body") or r.get("background") or "",
            "resolution_criteria": r.get("resolution_criteria") or r.get("resolution") or "",
            "timeframe": r.get("timeframe") or r.get("end") or "",
            "answer_type": r.get("answer_type") or "",
        }

    random.shuffle(rows)
    for r in rows:
        o = row2obj(r)
        if not o["title"]:
            continue
        if is_good(r) and len(goods) < k_good:
            goods.append(o)
        elif not is_good(r) and len(bads) < k_bad:
            bads.append(o)
        if len(goods) >= k_good and len(bads) >= k_bad:
            break
    return goods, bads

def generate_candidates(brief: str, tags: List[str], horizon: str, n: int, good: List[Dict[str, Any]], bad: List[Dict[str, Any]], model: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    if dry_run:
        out = []
        for i in range(n):
            out.append(
                {
                    "title": f"[MOCK] {brief[:40]} — Q{i+1}",
                    "body": "Context lines. Why it matters. Actors involved.",
                    "resolution_criteria": "On 2030-12-31 23:59:59 UTC, check source X for Y; YES if Z; otherwise NO.",
                    "timeframe": {"start": "2026-01-01 00:00:00", "end": "2030-12-31 23:59:59", "timezone": "UTC"},
                    "canonical_source": ["Reuters", "official press release"],
                    "answer_type": "binary",
                    "proposed_bins_or_ranges": "",
                    "difficulty": "med",
                    "rationale": "Decision-relevant; not trivially predictable.",
                    "policy_notes": "",
                }
            )
        return out
    good_str = json.dumps(good, ensure_ascii=False) if good else "[]"
    bad_str = json.dumps(bad, ensure_ascii=False) if bad else "[]"
    user = GEN_USER_TMPL.format(n=n, brief=brief, tags=",".join(tags), horizon=horizon, good_examples=good_str, bad_examples=bad_str)
    resp = call_openrouter(
        [{"role":"system","content":GEN_SYS},{"role":"user","content":user}],
        model=model, max_tokens=4000, temperature=0.5, expect="array", title_hint="Metaculus AI Question Generation", ua_hint="metaculus-ai-qgen/1.2"
    )
    if isinstance(resp, list): return resp
    if isinstance(resp, dict) and "candidates" in resp and isinstance(resp["candidates"], list): return resp["candidates"]
    if isinstance(resp, dict): return [resp]
    raise RuntimeError("Generation returned unexpected shape")

def critique_and_revise(cand: Dict[str, Any], model: str, dry_run: bool = False) -> Tuple[Dict[str, Any], Dict[str, int]]:
    if dry_run:
        return cand, {"clarity":4,"falsifiability":4,"operationalization":4,"usefulness":4,"safety":5}
    user = CRIT_USER_TMPL.format(candidate_json=json.dumps(cand, ensure_ascii=False))
    resp = call_openrouter([{"role":"system","content":CRIT_SYS},{"role":"user","content":user}], model=model, max_tokens=2000, temperature=0.1, expect="object", title_hint="Metaculus AI QGEN Critique", ua_hint="metaculus-ai-qgen/1.2")
    revised = resp.get("revised_candidate") or cand
    scores = {k.lower(): int(round(float(v))) for k, v in (resp.get("scores") or {}).items()}
    return revised, scores

def judge(cand: Dict[str, Any], model: str, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        base = 3.6 + random.random()*1.0
        return {"scores":{"clarity":4,"falsifiability":4,"operationalization":4,"usefulness":4,"safety":5},"overall":round(min(5.0, base),2),"blockers":[],"notes":"mock"}
    user = JUDGE_USER_TMPL.format(candidate_json=json.dumps(cand, ensure_ascii=False))
    resp = call_openrouter([{"role":"system","content":JUDGE_SYS},{"role":"user","content":user}], model=model, max_tokens=1200, temperature=0.0, expect="object", title_hint="Metaculus AI QGEN Judge", ua_hint="metaculus-ai-qgen/1.2")
    resp["overall"] = float(resp.get("overall", 0.0))
    return resp

def pairwise_battle(A: Dict[str, Any], B: Dict[str, Any], model: str, dry_run: bool = False) -> str:
    if dry_run:
        return "A" if random.random()<0.5 else "B"
    user = PAIRWISE_USER_TMPL.format(A=json.dumps(A, ensure_ascii=False), B=json.dumps(B, ensure_ascii=False))
    resp = call_openrouter([{"role":"system","content":PAIRWISE_SYS},{"role":"user","content":user}], model=model, max_tokens=400, temperature=0.0, expect="object", title_hint="Metaculus AI QGEN Pairwise", ua_hint="metaculus-ai-qgen/1.2")
    return resp.get("winner","A")

def run_pipeline_in_memory(brief: str, tags: List[str], horizon: str, n: int = 10, examples_csv: Optional[str] = None, top_k: int = 5, dry_run: bool = False, models: Dict[str,str] = None) -> Dict[str, Any]:
    gen_model = models.get("gen") if models else pick_model()
    crit_model = models.get("crit") if models else pick_model()
    judge_model = models.get("judge") if models else pick_model()
    good, bad = ([], [])
    if examples_csv:
        try:
            good, bad = load_examples_csv(examples_csv, k_good=3, k_bad=2)
        except Exception:
            good, bad = [], []
    cands = generate_candidates(brief, tags, horizon, n, good, bad, gen_model, dry_run=dry_run)
    revised = []; crit_scores = []
    for c in cands:
        r, s = critique_and_revise(c, crit_model, dry_run=dry_run)
        revised.append(r); crit_scores.append(s)
    judgements = [judge(c, judge_model, dry_run=dry_run) for c in revised]
    idx = sorted(range(len(judgements)), key=lambda i: -judgements[i].get("overall", 0.0))[: max(2, top_k)]
    top = [revised[i] for i in idx]; top_scores = [judgements[i] for i in idx]
    wins = {i:0 for i in range(len(top))}
    for i, j in itertools.combinations(range(len(top)), 2):
        w = pairwise_battle(top[i], top[j], judge_model, dry_run=dry_run)
        if w == "A": wins[i]+=1
        elif w == "B": wins[j]+=1
    ranked = [x for _, x in sorted(((wins[i], i) for i in range(len(top))), reverse=True)]
    top = [top[i] for i in ranked]; top_scores = [top_scores[i] for i in ranked]
    return {"gen_model":gen_model, "crit_model":crit_model, "judge_model":judge_model, "candidates":top, "judgements":top_scores}

def run_question_generator():
    st.subheader("🧪 AI Question Generator")
    st.caption("Generate, critique, judge and rank Metaculus-style questions. Export CSV & JSONL.")

    # Read models from sidebar
    gen_model  = _resolve_model_from_sidebar("GEN")
    crit_model = _resolve_model_from_sidebar("CRIT")
    judge_model= _resolve_model_from_sidebar("JUDGE")

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        brief = st.text_area("Topic brief (3–6 lines)", height=150, placeholder="e.g. medium-term AI benchmarks, EU/US regulation, deployment race dynamics...")
    with c2:
        tags_str = st.text_input("Domain tags (comma-separated)", value="ai,policy,geopolitics")
    with c3:
        horizon = st.text_input("Horizon / resolution description", value="resolve by 2030-12-31 UTC")

    n = st.slider("Number of candidates", min_value=3, max_value=30, value=10, step=1)
    top_k = st.slider("Top K after ranking", min_value=2, max_value=10, value=5, step=1)
    dry_run = st.checkbox("Dry run (no API calls, mock output)", value=False)

    ex_col1, ex_col2 = st.columns(2)
    scrape_n = ex_col1.slider("Scrape N Metaculus examples (0 = none)", min_value=0, max_value=50, value=0, step=5)
    examples_file = ex_col2.file_uploader("Or upload Metaculus example questions CSV", type=["csv"])

    if st.button("▶️ Generate & rank", type="primary"):
        if not brief.strip():
            st.warning("Please provide at least a short topic brief.")
        else:
            examples_path = None
            if scrape_n > 0:
                try:
                    qs = fetch_recent_questions(n_subjects=max(10, scrape_n), page_limit=max(80, scrape_n))
                    fieldnames = ["id","url","title","body","resolution_criteria","timeframe","answer_type"]
                    fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
                    with open(path,"w",newline="",encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
                        for q in qs:
                            w.writerow({
                                "id": q.get("id",""),
                                "url": q.get("url",""),
                                "title": q.get("title",""),
                                "body": q.get("body",""),
                                "resolution_criteria": "",
                                "timeframe": "",
                                "answer_type": "",
                            })
                    examples_path = path
                except Exception as e:
                    st.error(f"Metaculus scraping error: {e}"); st.stop()
            elif examples_file is not None:
                fd, path = tempfile.mkstemp(suffix=".csv")
                with os.fdopen(fd, "wb") as f:
                    f.write(examples_file.read())
                examples_path = path

            with st.spinner("Running generation, critique, judging and ranking..."):
                try:
                    res = run_pipeline_in_memory(
                        brief=brief,
                        tags=[t.strip() for t in tags_str.split(",") if t.strip()],
                        horizon=horizon,
                        n=n,
                        examples_csv=examples_path,
                        top_k=top_k,
                        dry_run=dry_run,
                        models={"gen": gen_model, "crit": crit_model, "judge": judge_model},
                    )
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    return
            st.success("Done")

            gen_m = res["gen_model"]; crit_m = res["crit_model"]; judge_m = res["judge_model"]
            st.markdown(f"**Models used** — Generator: `{gen_m}` · Critique: `{crit_m}` · Judge: `{judge_m}`")

            cands = res["candidates"]; judgements = res["judgements"]
            rows = []
            for c, s in zip(cands, judgements):
                rows.append({
                    "overall": s.get("overall", 0.0),
                    "clarity": (s.get("scores") or {}).get("clarity"),
                    "falsifiability": (s.get("scores") or {}).get("falsifiability"),
                    "operationalization": (s.get("scores") or {}).get("operationalization"),
                    "usefulness": (s.get("scores") or {}).get("usefulness"),
                    "safety": (s.get("scores") or {}).get("safety"),
                    "title": c.get("title",""),
                    "body": c.get("body",""),
                    "resolution_criteria": c.get("resolution_criteria",""),
                    "timeframe_start": (c.get("timeframe") or {}).get("start",""),
                    "timeframe_end": (c.get("timeframe") or {}).get("end",""),
                    "timezone": (c.get("timeframe") or {}).get("timezone",""),
                    "answer_type": c.get("answer_type",""),
                    "proposed_bins_or_ranges": c.get("proposed_bins_or_ranges",""),
                    "canonical_source": "; ".join(c.get("canonical_source") or []),
                    "difficulty": c.get("difficulty",""),
                    "rationale": c.get("rationale",""),
                    "policy_notes": c.get("policy_notes",""),
                    "judge_notes": s.get("notes",""),
                })
            if rows:
                df = pd.DataFrame(rows).sort_values("overall", ascending=False)
                st.dataframe(df, use_container_width=True, height=500)
                top_row = df.iloc[0]
                st.markdown("### Top candidate")
                st.markdown(f"**Title:** {top_row['title']}")
                st.markdown(f"**Body:** {top_row['body']}")
                st.markdown(f"**Resolution criteria:** {top_row['resolution_criteria']}")
                st.markdown(f"**Timeframe:** {top_row['timeframe_start']} → {top_row['timeframe_end']} ({top_row.get('timezone','UTC')})")
                st.markdown(f"**Answer type:** {top_row['answer_type']}")
                st.markdown(f"**Rationale:** {top_row['rationale']}")
                if top_row.get("policy_notes"):
                    st.markdown(f"**Policy notes:** {top_row['policy_notes']}")
                if top_row.get("judge_notes"):
                    st.markdown(f"**Judge notes:** {top_row['judge_notes']}")

                # Downloads
                csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
                csv_bytes = csv_buf.getvalue().encode("utf-8")

                records = [{"candidate": c, "judge": s} for c, s in zip(cands, judgements)]
                jsonl_buf = io.StringIO()
                for r in records:
                    jsonl_buf.write(json.dumps(r, ensure_ascii=False) + "\n")
                jsonl_bytes = jsonl_buf.getvalue().encode("utf-8")

                c1, c2 = st.columns(2)
                c1.download_button("💾 Download CSV", data=csv_bytes, file_name="metaculus_ai_qgen_top.csv", mime="text/csv")
                c2.download_button("💾 Download JSONL", data=jsonl_bytes, file_name="metaculus_ai_qgen_top.jsonl", mime="application/json")
            else:
                st.info("No candidates generated.")

# ================================
# App Shell
# ================================

st.set_page_config(page_title="Metaculus Suite (Comments • Q-Gen • Factors)", page_icon="🔮", layout="wide")

st.title("🔮 Metaculus Suite — Comments • Question Generation • Factors")

with st.sidebar:
    st.header("🔐 API & Models")

    # API key (session override)
    st.caption("Enter your OpenRouter key here. It overrides env/secrets for this session.")
    api_key_input = st.text_input("OpenRouter API key", type="password", key="OPENROUTER_API_KEY_SIDEBAR")
    if api_key_input:
        st.session_state["OPENROUTER_API_KEY_OVERRIDE"] = api_key_input.strip()

    cols = st.columns([1,1])
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
        index=(model_ids or [default_model_id]).index(default_model_id) if (model_ids or [default_model_id]) else 0,
        key="MODEL_DEFAULT",
        help="Used by Comment Scorer and Question Factors unless a custom model ID is set below."
    )
    st.text_input("Custom model ID (optional)", key="MODEL_DEFAULT_CUSTOM", placeholder="e.g., openai/gpt-4.1 or anthropic/claude-3.5-sonnet")

    st.divider()
    st.subheader("Question Generator models")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Generator", options=model_ids or [default_model_id],
                 index=(model_ids or [default_model_id]).index(default_model_id) if (model_ids or [default_model_id]) else 0,
                 key="MODEL_GEN")
    c2.selectbox("Critique",  options=model_ids or [default_model_id],
                 index=(model_ids or [default_model_id]).index(default_model_id) if (model_ids or [default_model_id]) else 0,
                 key="MODEL_CRIT")
    c3.selectbox("Judge",     options=model_ids or [default_model_id],
                 index=(model_ids or [default_model_id]).index(default_model_id) if (model_ids or [default_model_id]) else 0,
                 key="MODEL_JUDGE")
    c1.text_input("Custom generator ID", key="MODEL_GEN_CUSTOM", placeholder="Optional override")
    c2.text_input("Custom critique ID",  key="MODEL_CRIT_CUSTOM", placeholder="Optional override")
    c3.text_input("Custom judge ID",     key="MODEL_JUDGE_CUSTOM", placeholder="Optional override")

    st.caption("Tip: custom IDs let you call any model you have access to, even if it isn't listed.")

mode = st.radio("Choose a module", ["Comment Scorer", "Question Generator", "Question Factors"], horizontal=True)

if mode == "Comment Scorer":
    run_comment_scorer()
elif mode == "Question Generator":
    run_question_generator()
else:
    run_question_factors()
st.caption("Tip:add OPENROUTER_API_KEY in app secrets (Settings → Secrets) or enter it here. This app calls Metaculus public APIs and OpenRouter.")

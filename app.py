from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from rule_engine import (
    DIMENSIONS, analyze_pipeline, build_owl_mermaid, extract_formula_weights,
    load_ontology, load_rules, summarize_rules,
    # [v1.2.2 axiom 흡수]
    sync_frames_with_rules, get_active_frames, PER_DIM_CAP,
    compute_temporal_shift_penalty, compute_weighted_distortion,
    describe_temporal_shift,
    # [v1.2.2 axiom #4] OWL 그래프 추론
    audit_article_group, audit_temporal_pair, heuristic_extract_symbol,
    # [v4 보강] OWL reasoning trace table
    build_reasoning_trace,
)
# Cloud-friendly optional LLM/RAG imports.
# Streamlit Community Cloud 배포판은 sentence-transformers/torch 같은 무거운 로컬 sLLM·Dense RAG 의존성을
# 기본 requirements.txt에 포함하지 않는다. sllm_extractor가 없거나 heavy dependency 때문에 import 실패하면
# 앱은 휴리스틱/룰 기반 모드로 계속 실행되고, 로컬 배포판에서만 선택 기능을 활성화한다.
try:
    from sllm_extractor import (
        DEFAULT_MODEL,
        extract_features_with_sllm,
        compare_rag_results,
        _rule_text,
    )
    SLLM_EXTRACTOR_AVAILABLE = True
    SLLM_EXTRACTOR_IMPORT_ERROR = None
except Exception as _sllm_import_error:  # pragma: no cover - deployment guard
    DEFAULT_MODEL = "gpt-4o-mini"
    SLLM_EXTRACTOR_AVAILABLE = False
    SLLM_EXTRACTOR_IMPORT_ERROR = _sllm_import_error

    def _rule_text(rule):
        return " ".join(
            str(rule.get(k, ""))
            for k in ["rule_id", "target_frame", "llm_instruction_ko", "frame_definition_ko", "schema_description_ko"]
            if rule.get(k)
        )

    def extract_features_with_sllm(*args, **kwargs):
        raise RuntimeError(
            "sllm_extractor 모듈 또는 선택 의존성이 로드되지 않았습니다. "
            "Streamlit Cloud 배포판에서는 기본 휴리스틱/룰 기반 모드를 사용하세요. "
            "로컬에서 LLM/sLLM 모드를 쓰려면 requirements-sllm.txt를 설치하세요."
        )

    def compare_rag_results(articles, rules_data, top_k=8, dense_model=None, rule_embeddings=None, min_sparse_score=0.001, min_dense_score=0.15):
        # Minimal Sparse-only fallback for cloud deployments.
        article_text = " ".join(
            " ".join(str(a.get(k, "")) for k in ["title", "subtitle", "summary", "body", "text", "content"] if a.get(k))
            for a in articles
        )
        article_tokens = set(re.findall(r"[0-9A-Za-z가-힣_]+", article_text.lower()))
        sparse_scores = []
        for rule in rules_data.get("rules", []):
            rt = _rule_text(rule).lower()
            rule_tokens = set(re.findall(r"[0-9A-Za-z가-힣_]+", rt))
            if not rule_tokens or not article_tokens:
                score = 0.0
            else:
                score = len(article_tokens & rule_tokens) / max(1, len(article_tokens | rule_tokens))
            if score >= min_sparse_score:
                sparse_scores.append((score, rule))
        sparse_scores.sort(key=lambda x: x[0], reverse=True)
        sparse_top = sparse_scores[:top_k]
        return {
            "sparse_available": bool(sparse_top),
            "dense_available": False,
            "sparse_top": sparse_top,
            "dense_top": [],
            "common_ids": [],
            "sparse_only_ids": [r.get("rule_id") for _, r in sparse_top],
            "dense_only_ids": [],
            "min_sparse_score": min_sparse_score,
            "min_dense_score": min_dense_score,
            "sparse_filtered_count": len(sparse_scores),
            "dense_filtered_count": 0,
        }

BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_PATH = BASE_DIR / "ontology" / "context_sync_app_centered_ontology_1024.owl"
RULES_PATH = BASE_DIR / "ontology" / "news_rules_1024.json"
SAMPLE_PATH = BASE_DIR / "sample_data" / "sample_articles.json"
SCENARIOS_PATH = BASE_DIR / "sample_data" / "sample_scenarios.json"

st.set_page_config(page_title="Context-Sync News Analyzer", page_icon="◎", layout="wide")

@st.cache_data(show_spinner=False)
def cached_rules(path: str):
    return load_rules(Path(path))

@st.cache_resource(show_spinner=False)
def cached_ontology(path: str):
    return load_ontology(Path(path))

@st.cache_resource(show_spinner="임베딩 모델 다운로드/로드 중입니다 (최초 1회 약 400MB)...")
def load_dense_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("jhgan/ko-sroberta-multitask")
    except Exception as e:
        return None

@st.cache_resource(show_spinner="규칙 벡터화 중입니다...")
def compute_rule_embeddings(_model, rules_list):
    return _model.encode([_rule_text(r) for r in rules_list])

rules_data = cached_rules(str(RULES_PATH))
ontology = cached_ontology(str(ONTOLOGY_PATH))
rules = rules_data.get("rules", [])
summary = summarize_rules(rules)
weights = extract_formula_weights(rules_data.get("aggregation_formula"))

# [v1.2.2 axiom 흡수] #1 sync_frames_with_rules: OWL의 풍부한 어휘 ↔ JSON 룰셋의 부분집합 동기화
active_frames = get_active_frames(rules_data)

st.title("Context-Sync 뉴스 시계열 논조 분석")
st.caption("OWL 의미 기준층 (ontology-calibrated) + JSON 1024개 규칙 + Python rule engine + axiom 정규화 점수 + 선택형 LLM(GPT/sLLM) 추출기 + Streamlit 시연 앱")

with st.sidebar:
    st.header("시스템 상태")
    st.metric("규칙 수", f"{len(rules):,}")
    st.metric("OWL triples", f"{len(ontology):,}")
    st.write("**Ruleset**", rules_data.get("ruleset_id", "-"))
    st.write("**Version**", rules_data.get("version", "-"))

    # [v1.2.4] baseline-restored + explanation mitigation
    st.divider()
    st.subheader("Axiom Audit Engine")
    st.write(f"**Active frames**: {len(active_frames)}개")
    st.write(f"**per_dim_cap**: {PER_DIM_CAP}  *(baseline)*")
    with st.expander("dimension_weights (v2.0 baseline)"):
        for d, w in weights.items():
            st.write(f"- `{d}`: {w}")
        st.caption(
            "OWL baseline: 0.34/0.24/0.18/0.14/0.10, per_dim_cap=45. "
            "cap 45는 'major shift 단독(~22.7점)'과 '설명 없는 silent pivot(34점 상한 수렴)'을 점수 폭으로 분리. "
            "시뮬레이터에서 가중치 프리셋(시계열 엄격 0.40 등)과 cap(40 sensitive / 45 default / 60 conservative)을 직접 조작 가능."
        )
    st.caption(
        "ontology-calibrated reasoning: OWL 어휘 통제 + JSON 룰 + 5차원 정규화 "
        "+ explanation mitigation (변경 사유 충실 시 부가 페널티 부분 완화)"
    )

    st.write("**GitHub 친화 구조**")
    st.code("코드 + OWL + JSON만 저장\n대용량 기사/임베딩은 repo 제외", language="text")

    st.divider()
    st.subheader("LLM 추출기 (Cloud: OpenAI 중심)")
    cloud_mode = os.getenv("STREAMLIT_CLOUD", "1") == "1"
    use_sllm = st.toggle(
        "기사 지표를 LLM으로 추출",
        value=False,
        disabled=not SLLM_EXTRACTOR_AVAILABLE,
        help="Cloud 배포판은 OpenAI API 중심입니다. 로컬 Hugging Face sLLM은 requirements-sllm.txt 설치 후 실행하세요.",
    )
    if not SLLM_EXTRACTOR_AVAILABLE:
        st.warning("LLM 추출 모듈을 불러오지 못해 휴리스틱/룰 기반 모드로 실행합니다. 로컬 sLLM은 GitHub에서 내려받아 requirements-sllm.txt 설치 후 사용하세요.")
    sllm_model = st.text_input(
        "모델명",
        value=DEFAULT_MODEL,
        help="Streamlit Cloud에서는 OpenAI GPT 계열 사용을 권장합니다. Hugging Face 로컬 sLLM은 로컬 실행용입니다.",
    )
    # Safe secrets access to avoid StreamlitSecretNotFoundError when secrets.toml is missing
    default_key = ""
    try:
        if hasattr(st, "secrets"):
            default_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=default_key,
        help="GPT 모델 이용 시 필요합니다. Streamlit Cloud Secrets 또는 사이드바 입력을 사용할 수 있습니다.",
    )
    sllm_max_tokens = st.slider("LLM max_new_tokens", 80, 500, 220)
    use_owl_audit_llm_summary = st.toggle(
        "OWL graph audit LLM 요약 생성",
        value=False,
        disabled=not use_sllm,
        help=(
            "LLM 추출기를 켠 경우에만 main graph audit 1건을 별도로 요약합니다. "
            "기사 본문은 다시 보내지 않고 OWL reasoning trace / audit reasons / fired_rules 요약만 사용해 비용을 제한합니다."
        ),
    )
    if not use_sllm:
        st.caption("OWL graph audit LLM 요약은 LLM 추출기를 켰을 때만 선택할 수 있습니다.")
    st.caption("2차 배포 기준: Cloud에서는 OpenAI API 기반 추출까지만 권장합니다. Dense RAG/sentence-transformers/로컬 sLLM은 로컬 실행 옵션입니다.")

    st.divider()
    if cloud_mode:
        use_dense_rag = False
        st.info("Dense RAG / sentence-transformers 시각화는 Cloud 배포판에서 비활성화했습니다. GitHub 저장소를 내려받아 로컬에서 requirements-sllm.txt를 설치하면 사용할 수 있습니다.")
    else:
        use_dense_rag = st.toggle("🚀 [Local Beta] 밀집 벡터(Dense) RAG 및 시각화", value=False)
        if use_dense_rag:
            st.caption("`jhgan/ko-sroberta-multitask` 모델을 사용하여 RAG 및 2D 시각화를 수행합니다.")

    st.divider()
    top_n = st.slider("표시할 트리거 규칙 수", 3, 30, 12)
    available_profiles = ["전체"] + summary.get("profiles", [])
    available_contexts = ["전체"] + summary.get("contexts", [])
    selected_profile = st.selectbox("프로필 필터", available_profiles)
    selected_context = st.selectbox("맥락 필터", available_contexts)
    profile_filter = None if selected_profile == "전체" else selected_profile
    context_filter = None if selected_context == "전체" else selected_context


def load_sample_text() -> str:
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH.read_text(encoding="utf-8")
    return "[]"


def parse_articles(raw: str):
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "articles" in data:
            data = data["articles"]
        if not isinstance(data, list):
            raise ValueError("JSON 최상위는 기사 배열이어야 합니다.")
        return data, None
    except Exception as e:
        return [], str(e)



def _meta_content(soup, *keys: str) -> str:
    """Return the first matching meta content from property/name keys."""
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    return ""


def _clean_date(value: str) -> str:
    """Extract YYYY-MM-DD from common date strings."""
    if not value:
        return ""
    m = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", str(value))
    if not m:
        return ""
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def fetch_article_from_url(url: str, topic_fallback: str = "") -> dict:
    """기사 URL에서 title/body/outlet/date/topic을 휴리스틱으로 추출한다.

    주의:
    - 언론사 페이지 구조와 robots/차단 정책에 따라 실패할 수 있다.
    - 실패 시 직접 입력/JSON 입력으로 보완하는 것을 전제로 한 보조 기능이다.
    """
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError("URL 자동 추출에는 beautifulsoup4가 필요합니다. `pip install beautifulsoup4` 후 다시 실행하세요.") from exc

    url = url.strip()
    if not url:
        raise ValueError("URL이 비어 있습니다.")
    if not re.match(r"^https?://", url):
        url = "https://" + url

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ContextSyncCurator/1.0; +https://github.com/downteam7-crypto/context-sync.curator1)"
        },
    )
    with urlopen(req, timeout=15) as resp:
        raw = resp.read()
        content_type = resp.headers.get("Content-Type", "")
    # 대부분의 한국어 뉴스는 utf-8 또는 euc-kr/cp949 계열이다.
    encoding = "utf-8"
    m = re.search(r"charset=([\w\-]+)", content_type, re.IGNORECASE)
    if m:
        encoding = m.group(1)
    try:
        html_text = raw.decode(encoding, errors="replace")
    except Exception:
        html_text = raw.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html_text, "html.parser")

    # Title
    title = _meta_content(soup, "og:title", "twitter:title")
    if not title and soup.find("title"):
        title = soup.find("title").get_text(" ", strip=True)
    for sep in [" - ", " | ", " : "]:
        if sep in title and len(title.split(sep)[0].strip()) >= 4:
            title = title.split(sep)[0].strip()

    # Subtitle / lead / summary
    subtitle = _meta_content(soup, "og:description", "twitter:description", "description", "article:tag")
    if not subtitle:
        for sel in ["h2", ".subtitle", ".sub_title", ".summary", ".lead", ".article_summary", ".news_summary"]:
            node = soup.select_one(sel)
            if node:
                subtitle = node.get_text(" ", strip=True)
                if subtitle:
                    break
    subtitle = re.sub(r"\s+", " ", subtitle).strip()

    # Outlet
    outlet = _meta_content(soup, "og:site_name", "application-name")
    if not outlet:
        domain = urlparse(url).netloc.replace("www.", "")
        outlet = "네이버뉴스" if "naver.com" in domain else ("다음뉴스" if "daum.net" in domain else domain)

    # Date
    date_str = _clean_date(
        _meta_content(
            soup,
            "article:published_time",
            "article:modified_time",
            "og:pubdate",
            "pubdate",
            "publish-date",
            "date",
            "DC.date",
        )
    )
    if not date_str:
        date_str = _clean_date(html_text) or date.today().strftime("%Y-%m-%d")

    # Topic/category
    topic = (
        _meta_content(soup, "article:section", "section", "category", "news_keywords")
        or topic_fallback.strip()
    )
    if "," in topic:
        topic = topic.split(",")[0].strip()
    if not topic:
        topic = topic_fallback.strip() or "url-import"

    # Body: choose the longest plausible article body among common selectors.
    selectors = [
        "article",
        "[itemprop='articleBody']",
        "div#articleBody",
        "div#articleBodyContents",
        "div#newsct_article",
        "div.article_body",
        "div.news_body",
        "div.news_body_area",
        "div.story-news",
        "section",
    ]
    candidates = []
    for sel in selectors:
        node = soup.select_one(sel)
        if not node:
            continue
        node = BeautifulSoup(str(node), "html.parser")
        for bad in node(["script", "style", "iframe", "ins", "aside", "nav", "footer", "header", "button"]):
            bad.decompose()
        text = node.get_text("\n", strip=True)
        text = re.sub(r"\n{2,}", "\n", text).strip()
        if len(text) >= 80:
            candidates.append(text)
    if candidates:
        body = max(candidates, key=len)
    else:
        p_texts = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) >= 20]
        body = "\n".join(p_texts)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    if not title and body:
        title = body.splitlines()[0][:80]
    if not body or len(body) < 80:
        raise ValueError("본문 추출에 실패했습니다. 해당 사이트가 본문을 차단했거나 페이지 구조가 맞지 않습니다.")

    return {
        "title": title,
        "subtitle": subtitle,
        "summary": subtitle,
        "body": body,
        "date": date_str,
        "outlet": outlet,
        "topic": topic,
        "url": url,
    }



def _article_date_value(article: dict):
    """날짜 문자열을 정렬 가능한 datetime으로 변환한다."""
    return pd.to_datetime(article.get("date") or article.get("published_at") or "", errors="coerce")


def _article_text(article: dict) -> str:
    """audit_temporal_pair에 넣을 기사 텍스트를 합성한다."""
    title = str(article.get("title", "") or "")
    subtitle = str(article.get("subtitle") or article.get("summary") or article.get("description") or article.get("lead") or "")
    body = str(article.get("body") or article.get("text") or article.get("content") or "")
    return " ".join(p for p in [title, title, subtitle, body] if p)


def build_adjacent_pair_audits(articles: list[dict], rules_data: dict, graph) -> list[dict]:
    """같은 outlet+topic 그룹 안에서 인접 기사 쌍을 모두 graph audit한다.

    first↔last audit은 전체 drift를 보여주지만 중간 반전을 놓칠 수 있다.
    이 함수는 1→2, 2→3, ... 인접 구간을 별도 계산해 최대 변곡 구간을 보여준다.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for art in articles:
        key = (str(art.get("outlet", "?")), str(art.get("topic", "?")))
        groups[key].append(art)

    audits = []
    for (outlet, topic), group in groups.items():
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda a: (_article_date_value(a) if not pd.isna(_article_date_value(a)) else pd.Timestamp.min))
        for i in range(1, len(group_sorted)):
            past = group_sorted[i - 1]
            present = group_sorted[i]
            audit = audit_temporal_pair(
                _article_text(past),
                _article_text(present),
                rules_data,
                graph=graph,
            )
            audit["_group_key"] = f"{outlet} / {topic}"
            audit["_pair_index"] = i
            audit["_pair_label"] = f"{i}→{i + 1}"
            audit["past_article"] = {
                "title": past.get("title", ""),
                "date": past.get("date") or past.get("published_at") or "",
                "outlet": past.get("outlet", ""),
            }
            audit["present_article"] = {
                "title": present.get("title", ""),
                "date": present.get("date") or present.get("published_at") or "",
                "outlet": present.get("outlet", ""),
            }
            audits.append(audit)

    audits.sort(key=lambda a: float(a.get("weighted_distortion", 0) or 0), reverse=True)
    return audits


def render_rule_heatmap(fired_rules: list[dict], candidate_rules: list[dict]):
    """fired/candidate rule 분포를 dimension × source, schema_type × source로 요약 표시."""
    rows = []
    for source, ruleset in [("actual_fired", fired_rules), ("candidate", candidate_rules)]:
        for r in ruleset or []:
            rows.append({
                "source": source,
                "dimension": r.get("dimension", "unknown"),
                "schema_type": r.get("schema_type", "unknown"),
                "target_frame": r.get("target_frame", "unknown"),
                "penalty_abs": abs(float(r.get("axiom_penalty", 0) or 0)),
                "count": 1,
            })
    if not rows:
        st.info("표시할 rule heatmap 데이터가 없습니다.")
        return

    hm_df = pd.DataFrame(rows)
    st.markdown("**Dimension × rule source**")
    dim_pivot = hm_df.pivot_table(index="dimension", columns="source", values="count", aggfunc="sum", fill_value=0)
    st.dataframe(dim_pivot, width="stretch")

    st.markdown("**Schema type × rule source**")
    schema_pivot = hm_df.pivot_table(index="schema_type", columns="source", values="count", aggfunc="sum", fill_value=0)
    st.dataframe(schema_pivot, width="stretch")

    st.markdown("**Penalty sum by dimension**")
    penalty_df = hm_df.groupby("dimension", as_index=False)["penalty_abs"].sum().sort_values("penalty_abs", ascending=False)
    st.bar_chart(penalty_df.set_index("dimension")["penalty_abs"])



def _short_text(value, limit: int = 110) -> str:
    """UI 표 안에서 너무 긴 룰 설명을 짧게 접는다."""
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _join_rule_items(value, max_items: int = 3, limit: int = 120) -> str:
    """positive_cues / negative_indicators / expected_evidence 등을 표용 문자열로 변환."""
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        shown = items[:max_items]
        suffix = f" 외 {len(items) - max_items}개" if len(items) > max_items else ""
        return _short_text(" / ".join(shown) + suffix, limit)
    if isinstance(value, dict):
        return _short_text(json.dumps(value, ensure_ascii=False), limit)
    return _short_text(value, limit)


def _value_signature(value) -> str:
    """룰별 차이 필드 계산을 위한 안정적 문자열 표현."""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _candidate_rule_row(rule: dict) -> dict:
    """후보 룰의 '실제로 다른 부분'을 한 줄로 비교할 수 있게 만든다."""
    return {
        "rule_id": rule.get("rule_id", ""),
        "fired": rule.get("is_fired", ""),
        "schema": rule.get("schema_id", ""),
        "frame": rule.get("target_frame", ""),
        "dimension": rule.get("dimension", ""),
        "context": rule.get("context", ""),
        "profile": rule.get("profile", ""),
        "severity": rule.get("severity_band", ""),
        "value_anchor": rule.get("value_anchor", ""),
        "axiom_penalty": rule.get("axiom_penalty", 0),
        "strength": rule.get("rule_strength", 0),
        "intensity": rule.get("intensity", 0),
        "risk_weight": rule.get("risk_weight", ""),
        "distortion_weight": rule.get("distortion_weight", ""),
        "consensus_weight": rule.get("consensus_weight", ""),
        "positive_cues": _join_rule_items(rule.get("positive_cues", []), max_items=3),
        "negative_indicators": _join_rule_items(rule.get("negative_indicators", []), max_items=2),
    }


def _diff_fields_for_rules(ruleset: list[dict]) -> list[str]:
    """동일 schema/frame 묶음 안에서 실제로 값이 갈리는 필드를 찾는다."""
    fields = [
        "context", "profile", "severity_band", "value_anchor", "dimension",
        "risk_weight", "distortion_weight", "consensus_weight",
        "positive_cues", "negative_indicators", "expected_evidence_ko",
        "axiom_penalty", "rule_strength", "intensity", "is_fired",
    ]
    diff_fields = []
    for field in fields:
        values = {_value_signature(r.get(field, "")) for r in ruleset}
        if len(values) > 1:
            diff_fields.append(field)
    return diff_fields


def _llm_mode_label(model_name: str, cloud_mode: bool) -> str:
    """LLM 부가설명의 실행 환경 라벨을 만든다."""
    model = (model_name or "").lower()
    is_openai = model.startswith("gpt-") or "gpt" in model
    if cloud_mode and is_openai:
        return "Cloud OpenAI LLM"
    if cloud_mode and not is_openai:
        return "Cloud LLM"
    if is_openai:
        return "Local OpenAI LLM"
    return "Local Hugging Face sLLM"


def _rag_mode_label(rag_mode: str) -> str:
    if rag_mode == "dense":
        return "Dense RAG (sentence-transformers 의미 임베딩)"
    return "Sparse RAG (Jaccard 어절/키워드 매칭)"


def render_feature_rag_rule_selection_note(scope: str = "feature"):
    """Feature/RAG 설명에서 두 종류의 룰 선택 공리를 명시한다."""
    if scope == "feature":
        st.caption(
            "여기서 표시되는 RAG 주입 규칙은 LLM이 5개 feature 값을 추출하기 전에 참고한 검색 결과입니다. "
            "선택 기준은 기사 텍스트와 룰 텍스트의 유사도(Sparse Jaccard 또는 Dense cosine similarity)이며, "
            "해당 룰이 실제로 fired 되었는지나 feature activation 임계치를 통과했는지는 아직 반영하지 않습니다. "
            "따라서 3.7의 feature 기반 후보 규칙 목록과 완전히 일치하지 않을 수 있습니다."
        )
    else:
        st.caption(
            "주의: 이 후보 룰 목록은 RAG 검색 결과가 아닙니다. 5개 feature 값이 산출된 뒤, "
            "dimension score, rule weight, severity 등을 조합한 feature activation 임계치를 통과한 규칙입니다. "
            "반면 2. 지표 분해의 RAG 주입 규칙은 LLM feature 추출 전에 텍스트 유사도로 고른 참고 규칙입니다. "
            "두 목록은 같은 Feature/RAG 계열 보조층에 속하지만, 룰 선택 공리가 다르므로 일부 차이가 나는 것이 정상입니다."
        )


def _preview_text(text: str, limit: int = 700) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _llm_referenced_rule_ids(sllm_meta: Optional[dict], ruleset: list[dict]) -> list[str]:
    """LLM RAG prompt에 실제로 포함된 rule_id를 현재 묶음 기준으로 추린다."""
    if not sllm_meta:
        return []
    rag_text = str(sllm_meta.get("rag_selected_rules", "") or "")
    ids = []
    for r in ruleset:
        rid = str(r.get("rule_id", "") or "")
        if rid and rid in rag_text:
            ids.append(rid)
    return ids


def render_llm_supplement(sllm_meta: Optional[dict], cloud_mode: bool, group_rules: Optional[list[dict]] = None):
    """LLM 추출 성공 시 후보 룰 해석 보조 설명을 표시한다.

    Cloud에서는 OpenAI LLM 해설만, Local에서는 OpenAI LLM 또는 Hugging Face sLLM 해설을
    실행 모델명에 따라 구분해 보여준다. 점수 계산에는 관여하지 않는다.
    """
    if not sllm_meta:
        return

    model = str(sllm_meta.get("model", "-") or "-")
    mode_label = _llm_mode_label(model, cloud_mode)
    rag_label = _rag_mode_label(str(sllm_meta.get("rag_mode", "") or ""))
    reason = str(sllm_meta.get("reason", "") or "").strip()

    if group_rules is None:
        st.info(
            f"🤖 **Feature/RAG 기반 LLM 보조 해설 ({mode_label})**  \n"
            f"모델: `{model}` · RAG 모드: {rag_label}  \n"
            "이 설명은 LLM이 기사 묶음과 RAG 주입 규칙을 바탕으로 5개 feature를 추출할 때 생성한 보조 설명입니다. "
            "최종 axiom_distortion의 직접 근거는 위 3.5의 OWL graph audit / reasoning trace / actual fired_rules를 따릅니다."
        )
        render_feature_rag_rule_selection_note(scope="feature")
        if reason:
            st.markdown("**Feature/RAG LLM 요약 해설**")
            st.write(reason)
        if sllm_meta.get("rag_selected_rules"):
            with st.expander("Feature/RAG LLM이 프롬프트에서 참고한 RAG 규칙", expanded=False):
                st.code(str(sllm_meta.get("rag_selected_rules", "")), language="text")
        if sllm_meta.get("raw_response"):
            with st.expander("Feature/RAG LLM 원문 응답", expanded=False):
                st.code(str(sllm_meta.get("raw_response", "")), language="json")
        return

    referenced = _llm_referenced_rule_ids(sllm_meta, group_rules)
    if referenced or reason:
        st.markdown("**🤖 이 후보 룰 묶음에 대한 Feature/RAG 기반 LLM 보조 해석**")
        if referenced:
            st.caption("LLM RAG prompt에 포함된 관련 rule_id: " + ", ".join(f"`{rid}`" for rid in referenced))
        if reason:
            st.caption(_preview_text(reason, 350))
        st.caption("주의: 이 문장은 LLM feature extraction 과정에서 생성된 설명입니다. OWL graph audit의 독립 추론 결과가 아니며, penalty 계산을 추가로 바꾸지 않습니다.")
        render_feature_rag_rule_selection_note(scope="candidate")


def _safe_json_extract(text: str) -> dict:
    """LLM 응답에서 JSON 객체를 느슨하게 추출한다. 실패하면 빈 dict."""
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass
    return {}


def _compact_graph_audit_for_llm(audit: dict, max_trace_rows: int = 12, max_rules: int = 10, max_reasons: int = 8) -> dict:
    """기사 본문 없이 OWL graph audit 결과만 LLM 요약용으로 압축한다."""
    details = audit.get("details", {}) or {}
    past = details.get("past", {}) or {}
    present = details.get("present", {}) or {}
    trace = build_reasoning_trace(audit) or []
    fired_rules = audit.get("fired_rules", []) or []
    return {
        "group_key": audit.get("_group_key", "-"),
        "article_count": audit.get("article_count", 0),
        "past_article": {
            "title": (audit.get("past_article", {}) or {}).get("title", ""),
            "date": (audit.get("past_article", {}) or {}).get("date", ""),
        },
        "present_article": {
            "title": (audit.get("present_article", {}) or {}).get("title", ""),
            "date": (audit.get("present_article", {}) or {}).get("date", ""),
        },
        "past_signal": {
            "promoted_value": past.get("promoted_value"),
            "detected_frame": past.get("detected_frame"),
            "stance_polarity": past.get("stance_polarity"),
        },
        "present_signal": {
            "promoted_value": present.get("promoted_value"),
            "detected_frame": present.get("detected_frame"),
            "stance_polarity": present.get("stance_polarity"),
        },
        "scores": {
            "logic_score": audit.get("logic_score"),
            "weighted_distortion": audit.get("weighted_distortion"),
            "coherence_score": audit.get("score"),
            "verdict": audit.get("anchor_verdict"),
        },
        "dimension_breakdown": audit.get("dimension_breakdown", {}),
        "dimension_weights": audit.get("dimension_weights", {}),
        "explanation_mitigation": details.get("explanation_mitigation"),
        "reasoning_trace": trace[:max_trace_rows],
        "audit_reasons": [str(r) for r in (audit.get("reasons", []) or [])[:max_reasons]],
        "fired_rules": [
            {
                "rule_id": r.get("rule_id"),
                "schema_id": r.get("schema_id"),
                "target_frame": r.get("target_frame"),
                "dimension": r.get("dimension"),
                "axiom_penalty": r.get("axiom_penalty"),
                "rule_strength": r.get("rule_strength"),
                "intensity": r.get("intensity"),
            }
            for r in fired_rules[:max_rules]
        ],
    }


def generate_owl_graph_audit_llm_summary(
    audit: dict,
    model_name: str,
    api_key: str = "",
    max_tokens: int = 420,
    temperature: float = 0.1,
    cloud_mode: bool = True,
) -> str:
    """main OWL graph audit 1건을 LLM으로 요약한다.

    비용 통제를 위해 기사 본문은 다시 보내지 않고, graph audit의 구조화 결과만 전달한다.
    OpenAI GPT 계열은 Cloud/Local 모두 지원하고, Hugging Face sLLM은 Local에서만 시도한다.
    """
    model = (model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    is_openai = model.lower().startswith("gpt-") or "gpt" in model.lower()
    audit_payload = _compact_graph_audit_for_llm(audit)
    prompt = (
        "아래는 뉴스 시계열 분석기의 OWL graph audit 결과입니다. "
        "기사 본문을 새로 해석하지 말고, 제공된 reasoning_trace / audit_reasons / fired_rules / dimension_breakdown 안의 정보만 사용하세요.\n"
        "사람이 읽기 쉬운 한국어 요약을 JSON 하나로 출력하세요. 새 판단이나 추측을 추가하지 마세요.\n\n"
        "출력 형식:\n"
        "{\n"
        '  "summary": "3~5문장 요약",\n'
        '  "key_points": ["핵심 근거 1", "핵심 근거 2", "핵심 근거 3"],\n'
        '  "caution": "이 요약은 계산에 관여하지 않는 설명 보조층이라는 주의 문구"\n'
        "}\n\n"
        "OWL graph audit JSON:\n"
        + json.dumps(audit_payload, ensure_ascii=False, indent=2)
    )

    if is_openai:
        active_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not active_api_key:
            raise ValueError("OWL graph audit 요약을 생성하려면 OpenAI API Key가 필요합니다.")
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {active_api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You summarize deterministic OWL graph audit traces. "
                        "Return exactly one valid JSON object in Korean. Do not add facts outside the supplied audit."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API 호출 실패 (상태 코드 {response.status_code}): {response.text}")
        raw = response.json()["choices"][0]["message"]["content"]
    else:
        if cloud_mode:
            raise RuntimeError("Cloud 배포판에서는 OWL graph audit 요약에 OpenAI GPT 계열 모델만 지원합니다.")
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("로컬 sLLM 요약을 사용하려면 `pip install -r requirements-sllm.txt`가 필요합니다.") from e
        generator = pipeline("text-generation", model=model, tokenizer=model, trust_remote_code=True)
        outputs = generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            return_full_text=False,
        )
        raw = outputs[0].get("generated_text", "") if outputs else ""

    parsed = _safe_json_extract(raw)
    if parsed:
        parts = []
        if parsed.get("summary"):
            parts.append(str(parsed.get("summary")))
        key_points = parsed.get("key_points") or []
        if isinstance(key_points, list) and key_points:
            parts.append("\n".join(f"- {p}" for p in key_points[:5]))
        if parsed.get("caution"):
            parts.append(f"\n주의: {parsed.get('caution')}")
        return "\n\n".join(p for p in parts if p).strip() or json.dumps(parsed, ensure_ascii=False, indent=2)
    return str(raw or "").strip()


def render_candidate_rule_explanations(candidate_rules: list[dict], sllm_meta: Optional[dict] = None, cloud_mode: bool = True):
    """후보 룰을 기계적 나열 대신 schema/frame별로 묶고, 룰별 차이를 명시한다."""
    if not candidate_rules:
        st.info("표시할 feature 기반 후보 규칙이 없습니다.")
        return

    st.caption(
        "동일한 Schema/Frame 아래의 룰은 프레임 정의와 스키마 설명이 반복될 수 있습니다. "
        "아래 표는 공통 설명은 한 번만 보여주고, rule_id별로 실제로 달라지는 context/profile/cue/severity/weight를 비교합니다."
    )

    render_llm_supplement(sllm_meta, cloud_mode)

    overview_rows = []
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for r in candidate_rules:
        key = (str(r.get("schema_id", "-")), str(r.get("target_frame", "-")), str(r.get("dimension", "-")))
        grouped.setdefault(key, []).append(r)
        overview_rows.append(_candidate_rule_row(r))

    overview_df = pd.DataFrame(overview_rows)
    if not overview_df.empty:
        preferred_cols = [
            "rule_id", "fired", "schema", "frame", "dimension", "context", "profile", "severity",
            "value_anchor", "axiom_penalty", "strength", "intensity", "positive_cues", "negative_indicators",
        ]
        st.markdown("**후보 룰 요약표 — 룰별 차이 중심**")
        st.dataframe(overview_df[[c for c in preferred_cols if c in overview_df.columns]], width="stretch", hide_index=True)

    group_summary = []
    for (schema_id, frame, dimension), ruleset in grouped.items():
        penalties = [float(r.get("axiom_penalty", 0) or 0) for r in ruleset]
        contexts = sorted({str(r.get("context", "-")) for r in ruleset})
        profiles = sorted({str(r.get("profile", "-")) for r in ruleset})
        severities = sorted({str(r.get("severity_band", "-")) for r in ruleset})
        group_summary.append({
            "schema/frame": f"{schema_id} / {frame}",
            "dimension": dimension,
            "rules": len(ruleset),
            "fired": sum(1 for r in ruleset if bool(r.get("is_fired"))),
            "penalty_range": f"{min(penalties):.1f} ~ {max(penalties):.1f}" if penalties else "-",
            "contexts": ", ".join(contexts[:4]) + (f" 외 {len(contexts)-4}개" if len(contexts) > 4 else ""),
            "profiles": ", ".join(profiles[:4]) + (f" 외 {len(profiles)-4}개" if len(profiles) > 4 else ""),
            "severities": ", ".join(severities),
        })
    st.markdown("**Schema/Frame 묶음 요약**")
    st.dataframe(pd.DataFrame(group_summary), width="stretch", hide_index=True)

    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (sum(1 for r in item[1] if bool(r.get("is_fired"))), len(item[1])),
        reverse=True,
    )

    for (schema_id, frame, dimension), ruleset in sorted_groups:
        first = ruleset[0]
        fired_count = sum(1 for r in ruleset if bool(r.get("is_fired")))
        diff_fields = _diff_fields_for_rules(ruleset)
        with st.expander(
            f"🧩 {schema_id} / {frame} / {dimension} — {len(ruleset)}개 후보, fired {fired_count}개",
            expanded=fired_count > 0,
        ):
            st.markdown("**공통 프레임/스키마 설명**")
            if first.get("frame_definition_ko"):
                st.write(f"- 프레임 정의: {first.get('frame_definition_ko')}")
            if first.get("schema_description_ko"):
                st.write(f"- 스키마 설명: {first.get('schema_description_ko')}")
            if first.get("score_hint"):
                hint = first.get("score_hint")
                inc = _join_rule_items(hint.get("increase_when", []) if isinstance(hint, dict) else "", max_items=3)
                dec = _join_rule_items(hint.get("decrease_when", []) if isinstance(hint, dict) else "", max_items=3)
                if inc or dec:
                    st.caption(f"점수 상승 조건: {inc}")
                    st.caption(f"점수 완화 조건: {dec}")

            render_llm_supplement(sllm_meta, cloud_mode, group_rules=ruleset)

            st.markdown("**이 묶음에서 rule_id별로 달라지는 항목**")
            if diff_fields:
                st.write(" · ".join(f"`{field}`" for field in diff_fields))
            else:
                st.caption("이 묶음의 후보 룰은 표시 가능한 주요 필드가 거의 동일합니다.")

            detail_df = pd.DataFrame([_candidate_rule_row(r) for r in ruleset])
            detail_cols = [
                "rule_id", "fired", "context", "profile", "severity", "value_anchor",
                "axiom_penalty", "strength", "intensity", "risk_weight", "distortion_weight",
                "consensus_weight", "positive_cues", "negative_indicators",
            ]
            st.dataframe(detail_df[[c for c in detail_cols if c in detail_df.columns]], width="stretch", hide_index=True)

            st.caption(
                "해석: 같은 schema/frame이면 프레임 정의는 동일하게 반복됩니다. "
                "따라서 이 표에서는 각 rule_id가 어떤 맥락(context), 프로필(profile), 강도(severity), "
                "cue/indicator 조합으로 달라지는지를 중심으로 읽으면 됩니다."
            )


tab1, tab2, tab3, tab4, tab5 = st.tabs(["기사 분석", "수동 시뮬레이터", "규칙/온톨로지 탐색", "LLM 설정", "GitHub 배포 구조"])

with tab1:
    st.subheader("1. 기사 묶음 입력")
    st.write("같은 언론사/같은 사안의 기사들을 날짜순으로 비교하면 시계열 논조 이동을 더 잘 볼 수 있습니다.")

    # ── 입력 방식 선택 ────────────────────────────────────────────────────────
    input_mode = st.radio(
        "입력 방식",
        ["📝 기사 직접 입력 (자동 JSON 변환)", "🔗 기사 URL 입력 (자동 추출)", "{ } JSON 직접 입력"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 모드 A: 기사 직접 입력 → 자동 JSON 변환
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if input_mode.startswith("📝"):
        if "article_list" not in st.session_state:
            st.session_state.article_list = []

        # ── [A] 미니 검증셋 원클릭 예시 버튼 ─────────────────────────────────
        if SCENARIOS_PATH.exists():
            scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8")).get("scenarios", [])
            if scenarios:
                st.markdown("##### 🎯 예시 시나리오 (원클릭 로드)")
                sc_cols = st.columns(len(scenarios))
                for idx, sc in enumerate(scenarios):
                    with sc_cols[idx]:
                        if st.button(sc["label"], key=f"sc_{sc['id']}", width="stretch"):
                            st.session_state.article_list = sc["articles"]
                            st.rerun()
                        st.caption(f"기대: {sc.get('expected_verdict', '-')}")
                st.divider()

        with st.form("article_form", clear_on_submit=True):
            st.markdown("##### ✏️ 기사 추가")
            fc1, fc2, fc3 = st.columns([2, 2, 1])
            f_outlet = fc1.text_input("매체명 (outlet)", placeholder="예: 조선일보, KBS",
                                      help="같은 outlet+topic 그룹끼리 시계열 이동을 계산합니다.")
            f_topic  = fc2.text_input("사안 키워드 (topic)", placeholder="예: 정책X, 반도체법")
            f_date   = fc3.text_input("날짜 (date)", placeholder="2026-01-01")
            f_title  = st.text_input("제목 *", placeholder="기사 제목을 입력하세요")
            f_subtitle = st.text_input("부제/요약 (선택)", placeholder="기사 부제, 리드문, 요약문이 있으면 입력하세요")
            f_body   = st.text_area("본문 *", placeholder="기사 본문을 여기에 붙여넣으세요. 길이 제한 없음.", height=200)
            submitted = st.form_submit_button("➕ 기사 추가", width="stretch", type="primary")

        if submitted:
            if not f_title.strip() and not f_body.strip():
                st.warning("제목 또는 본문을 입력해 주세요.")
            else:
                entry: dict = {"title": f_title.strip(), "subtitle": f_subtitle.strip(), "summary": f_subtitle.strip(), "body": f_body.strip()}
                if f_outlet.strip():
                    entry["outlet"] = f_outlet.strip()
                if f_topic.strip():
                    entry["topic"] = f_topic.strip()
                if f_date.strip():
                    entry["date"] = f_date.strip()
                st.session_state.article_list.append(entry)
                st.success(f"기사 추가됨 (현재 {len(st.session_state.article_list)}개)")

        if st.session_state.article_list:
            preview_rows = [
                {
                    "#": i + 1,
                    "매체": a.get("outlet", "-"),
                    "사안": a.get("topic", "-"),
                    "날짜": a.get("date", "-"),
                    "제목": a.get("title", "")[:40] + ("…" if len(a.get("title", "")) > 40 else ""),
                    "본문길이": f"{len(a.get('body', ''))}자",
                }
                for i, a in enumerate(st.session_state.article_list)
            ]
            st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)

            col_clr, col_json = st.columns([1, 3])
            if col_clr.button("🗑️ 전체 초기화", width="stretch"):
                st.session_state.article_list = []
                st.rerun()
            with col_json.expander("📋 자동 생성된 JSON 보기"):
                st.code(json.dumps(st.session_state.article_list, ensure_ascii=False, indent=2), language="json")

            articles = st.session_state.article_list
            err = None
        else:
            st.info("위 폼에서 기사를 하나 이상 추가한 뒤 분석을 실행하세요.")
            articles, err = [], "기사가 없습니다"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 모드 B: URL 입력 → 기사 자동 추출
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif input_mode.startswith("🔗"):
        if "article_list" not in st.session_state:
            st.session_state.article_list = []

        st.markdown("##### 🔗 기사 URL 자동 추출")
        st.caption("URL만으로 제목·본문·매체·날짜·topic을 추출합니다. 사이트 구조/차단 정책에 따라 실패할 수 있으며, 실패 시 직접 입력 또는 JSON 입력을 사용하세요.")

        default_topic = st.text_input(
            "공통 사안 키워드 fallback (topic)",
            placeholder="예: 에너지전환, 교육개혁, 신약허가",
            help="기사 페이지에서 category/topic을 찾지 못하면 이 값을 사용합니다. 같은 topic이어야 시계열 그룹으로 묶입니다.",
        )
        url_text = st.text_area(
            "기사 URL 목록",
            placeholder="https://example.com/news/1\nhttps://example.com/news/2",
            height=130,
            help="여러 개를 넣을 때는 줄바꿈으로 구분하세요.",
        )

        c_fetch, c_clear = st.columns([2, 1])
        if c_fetch.button("🌐 URL에서 기사 불러오기", width="stretch", type="primary"):
            urls = [u.strip() for u in url_text.splitlines() if u.strip()]
            if not urls:
                st.warning("URL을 하나 이상 입력해 주세요.")
            else:
                added = 0
                failures = []
                with st.spinner(f"기사 {len(urls)}개를 가져오는 중입니다..."):
                    for u in urls:
                        try:
                            article = fetch_article_from_url(u, topic_fallback=default_topic)
                            st.session_state.article_list.append(article)
                            added += 1
                        except Exception as exc:
                            failures.append({"url": u, "error": str(exc)})
                if added:
                    st.success(f"{added}개 기사 추출 완료 (현재 {len(st.session_state.article_list)}개)")
                if failures:
                    st.warning(f"{len(failures)}개 URL은 추출에 실패했습니다.")
                    st.dataframe(pd.DataFrame(failures), width="stretch", hide_index=True)

        if c_clear.button("🗑️ URL/직접 입력 기사 초기화", width="stretch"):
            st.session_state.article_list = []
            st.rerun()

        if st.session_state.article_list:
            preview_rows = [
                {
                    "#": i + 1,
                    "매체": a.get("outlet", "-"),
                    "사안": a.get("topic", "-"),
                    "날짜": a.get("date", "-"),
                    "제목": a.get("title", "")[:50] + ("…" if len(a.get("title", "")) > 50 else ""),
                    "본문길이": f"{len(a.get('body', ''))}자",
                    "URL": a.get("url", "")[:45] + ("…" if len(a.get("url", "")) > 45 else ""),
                }
                for i, a in enumerate(st.session_state.article_list)
            ]
            st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)
            with st.expander("📋 URL 추출 결과 JSON 보기"):
                st.code(json.dumps(st.session_state.article_list, ensure_ascii=False, indent=2), language="json")
            articles = st.session_state.article_list
            err = None
        else:
            st.info("URL에서 기사를 불러온 뒤 분석을 실행하세요.")
            articles, err = [], "기사가 없습니다"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 모드 C: JSON 직접 입력 (기존 방식)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    else:
        raw_json = st.text_area(
            "기사 JSON 배열",
            value=load_sample_text(),
            height=360,
            help="필수는 아니지만 outlet, topic, date, title, body 필드를 권장합니다.",
        )
        articles, err = parse_articles(raw_json)
        if err:
            st.error(f"JSON 파싱 오류: {err}")
        else:
            st.success(f"기사 {len(articles)}개 로드됨")

    # ── 분석 실행 (두 모드 공통) ──────────────────────────────────────────────
    st.divider()
    if st.button("🔍 시계열 논조 분석 실행", type="primary", disabled=bool(err) or not articles,
                 width="stretch"):
        sllm_meta = None
        manual_features = None

        # [v4 보강] Dense 모델/임베딩 로드 — use_sllm 여부와 무관하게 RAG 비교에 활용
        dense_model = None
        rule_embeddings = None
        if use_dense_rag:
            try:
                dense_model = load_dense_model()
                if dense_model:
                    rule_embeddings = compute_rule_embeddings(dense_model, rules)
            except Exception:
                pass

        if use_sllm:
            with st.spinner("LLM(GPT/sLLM)이 기사 묶음에서 5개 지표를 추출하는 중입니다. 로컬 sLLM은 최초 실행 시 다운로드로 인해 다소 시간이 걸릴 수 있습니다."):
                try:
                    manual_features, sllm_meta = extract_features_with_sllm(
                        articles,
                        rules_data,
                        model_name=sllm_model.strip() or DEFAULT_MODEL,
                        max_new_tokens=sllm_max_tokens,
                        dense_model=dense_model,
                        rule_embeddings=rule_embeddings,
                        api_key=openai_api_key,
                    )
                except Exception as e:
                    st.error(f"LLM 추출 실패: {e}")
                    st.info("휴리스틱 추출기로 자동 전환합니다. 로컬 sLLM을 활성화하려면 `pip install -r requirements-sllm.txt` 설치를 확인하시고, GPT 계열을 사용하시려면 API 키 및 라이브러리 설정을 확인하세요.")
                    manual_features = None

        # [v4 보강] RAG Evidence Panel — Sparse vs Dense 병렬 비교
        if articles:
            with st.expander("🔍 RAG Evidence Panel — Sparse vs Dense 병렬 비교", expanded=False):
                st.caption(
                    "**이중 RAG**: 같은 기사에 대해 Sparse(Jaccard 키워드 매칭)와 Dense(ko-sroberta 의미 임베딩)를 *동시에* 실행해 "
                    "어떤 룰을 가져오는지 *나란히* 비교합니다. 두 검색의 차이가 *그 자체로 정보*입니다."
                )
                rag_cmp = compare_rag_results(
                    articles, rules_data,
                    top_k=8,
                    dense_model=dense_model,
                    rule_embeddings=rule_embeddings,
                    min_sparse_score=0.001,
                    min_dense_score=0.15,
                )
                st.caption(
                    f"낮은 관련도 후보는 필터링합니다: "
                    f"Sparse score ≥ {rag_cmp.get('min_sparse_score', 0):.3f}, "
                    f"Dense similarity ≥ {rag_cmp.get('min_dense_score', 0):.2f}. "
                    f"필터링 수: Sparse {rag_cmp.get('sparse_filtered_count', 0)}개, "
                    f"Dense {rag_cmp.get('dense_filtered_count', 0)}개."
                )
                if not rag_cmp["sparse_available"]:
                    st.info("Sparse 검색 결과가 없거나 최소 관련도 기준을 통과한 룰이 없습니다.")
                else:
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        st.markdown("##### 📚 Sparse (Jaccard 키워드)")
                        st.caption("고유명사·정책명·통계 같은 *정확 매칭*에 강함")
                        sp_df = pd.DataFrame([
                            {
                                "rule_id": r.get("rule_id"),
                                "score": round(s, 3),
                                "dimension": r.get("dimension"),
                                "target_frame": r.get("target_frame"),
                                "schema_id": r.get("schema_id"),
                            }
                            for s, r in rag_cmp["sparse_top"]
                        ])
                        st.dataframe(sp_df, width="stretch", hide_index=True)
                    with rcol2:
                        st.markdown("##### 🧠 Dense (ko-sroberta 의미 임베딩)")
                        st.caption("의미적 유사성·구조적 패턴 매칭에 강함")
                        if rag_cmp["dense_available"]:
                            de_df = pd.DataFrame([
                                {
                                    "rule_id": r.get("rule_id"),
                                    "similarity": round(s, 3),
                                    "dimension": r.get("dimension"),
                                    "target_frame": r.get("target_frame"),
                                    "schema_id": r.get("schema_id"),
                                }
                                for s, r in rag_cmp["dense_top"]
                            ])
                            st.dataframe(de_df, width="stretch", hide_index=True)
                        else:
                            st.warning(
                                "Dense RAG가 비활성화되어 있거나, 최소 similarity 기준을 통과한 룰이 없습니다. "
                                "사이드바에서 `[Beta] 밀집 벡터(Dense) RAG`를 활성화하거나 입력 기사/룰셋의 관련도를 확인하세요."
                            )

                    # 교집합 / 차집합 분석
                    if rag_cmp["dense_available"]:
                        st.markdown("##### 🔗 두 검색의 대비")
                        ccols = st.columns(3)
                        with ccols[0]:
                            st.markdown(f"**공통 (Sparse ∩ Dense)** — {len(rag_cmp['common_ids'])}개")
                            if rag_cmp["common_ids"]:
                                for rid in rag_cmp["common_ids"]:
                                    st.write(f"• `{rid}`")
                            else:
                                st.caption("(없음)")
                        with ccols[1]:
                            st.markdown(f"**Sparse only** — {len(rag_cmp['sparse_only_ids'])}개")
                            if rag_cmp["sparse_only_ids"]:
                                for rid in rag_cmp["sparse_only_ids"][:5]:
                                    st.write(f"• `{rid}`")
                                if len(rag_cmp["sparse_only_ids"]) > 5:
                                    st.caption(f"... 외 {len(rag_cmp['sparse_only_ids'])-5}개")
                            else:
                                st.caption("(없음)")
                        with ccols[2]:
                            st.markdown(f"**Dense only** — {len(rag_cmp['dense_only_ids'])}개")
                            if rag_cmp["dense_only_ids"]:
                                for rid in rag_cmp["dense_only_ids"][:5]:
                                    st.write(f"• `{rid}`")
                                if len(rag_cmp["dense_only_ids"]) > 5:
                                    st.caption(f"... 외 {len(rag_cmp['dense_only_ids'])-5}개")
                            else:
                                st.caption("(없음)")
                        st.caption(
                            f"**해석**: 공통 룰은 *두 방식 모두 강하게 신호*. Sparse only는 *키워드 일치 룰*. "
                            f"Dense only는 *의미적으로 가깝지만 키워드는 다른 룰* — 본 시스템이 *놓치기 쉬운* 룰을 잡아냄."
                        )

        result = analyze_pipeline(
            articles,
            rules_data,
            top_n=top_n,
            profile=profile_filter,
            context=context_filter,
            manual_features=manual_features,
            owl_graph=ontology,
        )
        # [개선] graph audit 결과를 메인 axiom_distortion으로 승격
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "최종 왜곡도 (axiom)",
            f"{result.get('axiom_distortion', 0):.1f}",
            help="graph audit weighted_distortion 우선. graph audit이 불가능할 때만 feature-lite fallback을 사용합니다.",
        )
        c2.metric(
            "정합성 점수",
            f"{result.get('coherence_score', 0):.1f}",
            help="100 - axiom_distortion. 높을수록 시계열 정합성이 높습니다.",
        )
        c3.metric("최종 판정 (axiom)", result.get("axiom_verdict", "-"))
        c4.metric(
            "v1 보조 왜곡도",
            f"{result.get('v1_distortion_score', result.get('score', 0)):.1f}",
            help="features × weights 단순 가중합. 최종 판정이 아니라 비교용 보조지표입니다.",
        )

        st.caption(
            "**5단계 정합성 밴드**: 85 ~ 100 안정적 정합 · 70 ~ 84 기준 부합 · "
            "55 ~ 69 주의 필요 · 40 ~ 54 중점 검토 필요 · 0 ~ 39 기준 이탈"
        )

        st.caption(
            f"**axiom source**: `{result.get('axiom_source', '-')}`  |  "
            f"**primary group**: {result.get('primary_group_key', '-')}  |  "
            f"**polarity shift**: {result.get('polarity_shift', 0):.2f} "
            f"({result.get('polarity_shift_label', '-')})  |  "
            f"**temporal penalty**: {result.get('temporal_penalty', 0)}"
        )
        st.caption(
            f"v1 verdict: {result.get('v1_verdict', result.get('verdict', '-'))}  |  "
            f"axiom-lite fallback distortion: {result.get('axiom_lite_distortion', 0):.1f} "
            f"({result.get('axiom_lite_verdict', '-')})"
        )

        if result.get("verdict_reason"):
            st.info(f"📌 판정 핵심 근거: {result['verdict_reason']}")

        # [v4.3] Stage 0 Validity Red Card — 5차원 점수보다 우선하는 상위 자격 심사
        if result.get("validity_violation"):
            red = result.get("validity_red_card", {}) or {}
            st.error(
                "⛔ Stage 0 Validity Red Card: 최소 사실/윤리 기준 위반이 감지되어 "
                "최종 왜곡도는 100, 정합성 점수는 0으로 override되었습니다."
            )
            if red.get("hits"):
                rc_df = pd.DataFrame([
                    {
                        "article": h.get("article_index"),
                        "title": h.get("title"),
                        "category": h.get("category"),
                        "label": h.get("label"),
                        "matched_text": h.get("matched_text"),
                        "reason": h.get("reason"),
                    }
                    for h in red.get("hits", [])
                ])
                st.dataframe(rc_df, width="stretch", hide_index=True)
            trace = result.get("validity_trace", [])
            if trace:
                with st.expander("🧬 Stage 0 Validity Reasoning Trace", expanded=True):
                    st.dataframe(pd.DataFrame(trace), width="stretch", hide_index=True)

        if result.get("dimension_breakdown"):
            with st.expander("⚙️ 최종 axiom 차원별 페널티 분해 (graph audit 우선)", expanded=True):
                st.caption(
                    "이 표는 최종 axiom_distortion에 직접 반영되는 graph audit 기반 페널티 분해입니다. "
                    "LLM의 Feature/RAG 추출 근거는 아래 `2. 지표 분해` 섹션에서 별도로 확인합니다."
                )
                bd = result["dimension_breakdown"]
                bd_df = pd.DataFrame([
                    {
                        "dimension": dim,
                        "누적 페널티": bd.get(dim, 0),
                        "정규화 (페널티/cap × 100)": round(min(100, abs(bd.get(dim, 0)) / PER_DIM_CAP * 100), 1),
                        "가중치": result["weights"].get(dim, 0),
                        "기여 distortion": round(min(100, abs(bd.get(dim, 0)) / PER_DIM_CAP * 100) * result["weights"].get(dim, 0), 2),
                    }
                    for dim in DIMENSIONS
                ])
                st.dataframe(bd_df, width="stretch", hide_index=True)
                st.bar_chart(bd_df.set_index("dimension")["기여 distortion"])

        st.subheader("2. 지표 분해")
        feature_df = pd.DataFrame(
            [{"dimension": k, "score": v, "weight": result["weights"].get(k, 0)} for k, v in result["features"].items()]
        )
        st.dataframe(feature_df, width="stretch", hide_index=True)
        st.bar_chart(feature_df.set_index("dimension")["score"])

        if sllm_meta:
            with st.expander("Feature/RAG LLM 추출 근거 / 원문 응답", expanded=False):
                st.caption(
                    "이 근거는 LLM이 기사 묶음과 RAG 주입 규칙을 바탕으로 5개 feature 값을 추출할 때 생성한 설명입니다. "
                    "최종 axiom_distortion의 직접 근거는 3.5의 OWL graph audit / reasoning trace / actual fired_rules를 따릅니다."
                )
                render_feature_rag_rule_selection_note(scope="feature")
                st.write("**모델**", sllm_meta.get("model"))
                st.write(
                    "**RAG 모드**",
                    "Dense (의미 임베딩)" if sllm_meta.get("rag_mode") == "dense" else "Sparse (Jaccard 어절 겹침)",
                )
                st.write("**Feature/RAG 추출 근거**", sllm_meta.get("reason", ""))
                rag_rules = sllm_meta.get("rag_selected_rules", "")
                if rag_rules:
                    st.write("**RAG가 선택해 feature 추출 프롬프트에 주입한 후보 규칙**")
                    st.code(rag_rules, language="text")
                st.write("**LLM 원문 응답**")
                st.code(sllm_meta.get("raw_response", ""), language="json")

        if result["series"]:
            st.subheader("3. 기사별 시계열 신호")
            series_df = pd.DataFrame(result["series"]).copy()
            article_rows_df = pd.DataFrame(result["article_rows"])
            st.dataframe(article_rows_df, width="stretch", hide_index=True)
            if "sentiment" in series_df:
                try:
                    import plotly.express as px

                    plot_df = series_df.copy()
                    plot_df["date_sort"] = pd.to_datetime(plot_df.get("date", ""), errors="coerce")
                    plot_df = plot_df.sort_values(["date_sort", "date"], na_position="last").reset_index(drop=True)
                    plot_df["article_no"] = plot_df.index + 1
                    plot_df["delta"] = plot_df["sentiment"].diff().abs().fillna(0)
                    max_delta_idx = int(plot_df["delta"].idxmax()) if len(plot_df) > 1 else 0

                    fig = px.line(
                        plot_df,
                        x="article_no",
                        y="sentiment",
                        markers=True,
                        hover_data=["date", "title", "delta"],
                    )
                    fig.update_layout(
                        xaxis_title="기사 순서(날짜순)",
                        yaxis_title="sentiment (-1~+1)",
                        margin=dict(l=20, r=20, t=30, b=20),
                    )
                    if len(plot_df) > 1 and plot_df.loc[max_delta_idx, "delta"] > 0:
                        fig.add_annotation(
                            x=plot_df.loc[max_delta_idx, "article_no"],
                            y=plot_df.loc[max_delta_idx, "sentiment"],
                            text=f"max Δ={plot_df.loc[max_delta_idx, 'delta']:.2f}",
                            showarrow=True,
                            arrowhead=2,
                        )
                    st.plotly_chart(fig, width="stretch")

                    delta_rows = []
                    for i in range(1, len(plot_df)):
                        delta_rows.append({
                            "pair": f"{i}→{i + 1}",
                            "from": plot_df.loc[i - 1, "title"],
                            "to": plot_df.loc[i, "title"],
                            "sentiment_delta": round(abs(plot_df.loc[i, "sentiment"] - plot_df.loc[i - 1, "sentiment"]), 3),
                        })
                    if delta_rows:
                        with st.expander("📈 인접 기사 sentiment 변화량", expanded=False):
                            st.dataframe(pd.DataFrame(delta_rows), width="stretch", hide_index=True)
                except Exception:
                    st.line_chart(series_df.set_index("date")["sentiment"])

        # ─────────────────────────────────────────────────────────
        # [개선] OWL graph audit 결과 표시 — analyze_pipeline에서 이미 계산한 결과 재사용
        # ─────────────────────────────────────────────────────────
        graph_audits = result.get("graph_audits", [])
        if articles and len(articles) >= 2:
            st.subheader("3.5. OWL 그래프 추론 및 actual fired_rules")
            st.caption(
                "graph audit 결과가 메인 `axiom_distortion`의 1순위 입력입니다. "
                "이 섹션에서는 같은 outlet+topic 그룹의 시간순 첫 기사와 마지막 기사를 비교하고, "
                "최종 점수에 직접 기여한 `actual axiom fired_rules`까지 함께 묶어 보여줍니다. "
                "아래 후보 규칙 해설 섹션(3.7)은 중복 표기가 아니라, feature 기반 후보 룰의 "
                "schema/frame/context/cue 차이를 설명하는 보조 영역입니다."
            )

            if not graph_audits:
                st.info("그래프 추론 가능한 그룹이 없습니다. 이 경우 최종 axiom 점수는 feature-lite fallback을 사용합니다. 같은 outlet+topic의 기사가 2개 이상 필요합니다.")
            else:
                summary_rows = []
                for idx, audit in enumerate(graph_audits):
                    summary_rows.append({
                        "대표": "★ main" if idx == 0 else "",
                        "그룹 (outlet/topic)": audit.get("_group_key", "-"),
                        "기사 수": audit.get("article_count", 0),
                        "PAST 가치": audit["details"]["past"].get("promoted_value"),
                        "PRESENT 가치": audit["details"]["present"].get("promoted_value"),
                        "PRESENT 프레임": audit["details"]["present"].get("detected_frame"),
                        "polarity Δ": round(abs(audit["details"]["past"].get("stance_polarity", 0)
                                                - audit["details"]["present"].get("stance_polarity", 0)), 3),
                        "logic_score": audit.get("logic_score"),
                        "weighted_distortion": audit.get("weighted_distortion"),
                        "coherence_score": audit.get("score"),
                        "verdict": audit.get("anchor_verdict"),
                        "fired_rules": len(audit.get("fired_rules", [])),
                    })
                st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

                for idx, audit in enumerate(graph_audits):
                    label_prefix = "★ MAIN — " if idx == 0 else ""
                    with st.expander(
                        f"📋 {label_prefix}{audit.get('_group_key', '-')} — "
                        f"distortion {audit.get('weighted_distortion', 0):.1f}, "
                        f"coherence {audit.get('score', 0):.1f} ({audit.get('anchor_verdict', '-')})",
                        expanded=(idx == 0),
                    ):
                        c_past, c_present = st.columns(2)
                        with c_past:
                            st.markdown("**📰 PAST (시간순 첫 기사)**")
                            st.caption(audit["past_article"].get("title", ""))
                            st.caption(f"📅 {audit['past_article'].get('date', '')}")
                            past_d = audit["details"]["past"]
                            st.write(f"• promoted_value: `{past_d.get('promoted_value')}`")
                            st.write(f"• detected_frame: `{past_d.get('detected_frame')}`")
                            st.write(f"• stance_polarity: `{past_d.get('stance_polarity')}`")
                        with c_present:
                            st.markdown("**📰 PRESENT (시간순 마지막 기사)**")
                            st.caption(audit["present_article"].get("title", ""))
                            st.caption(f"📅 {audit['present_article'].get('date', '')}")
                            pres_d = audit["details"]["present"]
                            st.write(f"• promoted_value: `{pres_d.get('promoted_value')}`")
                            st.write(f"• detected_frame: `{pres_d.get('detected_frame')}`")
                            st.write(f"• stance_polarity: `{pres_d.get('stance_polarity')}`")

                        if idx == 0 and use_owl_audit_llm_summary:
                            with st.expander("🤖 OWL graph audit 전용 LLM 요약해설", expanded=True):
                                st.caption(
                                    "이 요약은 3.5의 OWL reasoning trace / audit reasons / actual fired_rules를 사람이 읽기 쉽게 풀어쓴 것입니다. "
                                    "기사 본문을 다시 해석하지 않으며, 최종 점수 계산에는 관여하지 않습니다."
                                )
                                try:
                                    with st.spinner("main OWL graph audit를 LLM으로 요약하는 중입니다..."):
                                        audit_summary_text = generate_owl_graph_audit_llm_summary(
                                            audit,
                                            model_name=sllm_model.strip() or DEFAULT_MODEL,
                                            api_key=openai_api_key,
                                            max_tokens=min(max(int(sllm_max_tokens), 220), 500),
                                            cloud_mode=cloud_mode,
                                        )
                                    st.write(audit_summary_text)
                                except Exception as exc:
                                    st.warning(f"OWL graph audit LLM 요약 생성 실패: {exc}")
                                    st.caption("요약 생성에 실패해도 graph audit 계산과 최종 점수에는 영향이 없습니다.")

                        mitigation = audit.get("details", {}).get("explanation_mitigation")
                        if mitigation:
                            factor = mitigation.get("mitigation_factor", 1.0)
                            if factor < 1.0:
                                st.success(
                                    f"설명 보정 활성화: {mitigation.get('level')} "
                                    f"(signal={mitigation.get('explanation_signal')}, factor={factor}) — "
                                    f"{mitigation.get('reason')}"
                                )
                            else:
                                st.caption(
                                    f"설명 보정 비활성: signal={mitigation.get('explanation_signal')}, "
                                    f"factor={factor} — {mitigation.get('reason')}"
                                )

                        st.markdown("**차원별 페널티 분해**")
                        bd_df = pd.DataFrame([
                            {
                                "dimension": d,
                                "누적 페널티": audit["dimension_breakdown"].get(d, 0),
                                "정규화": round(min(100, abs(audit["dimension_breakdown"].get(d, 0)) / PER_DIM_CAP * 100), 1),
                                "가중치": audit["dimension_weights"].get(d, 0),
                                "기여 distortion": round(min(100, abs(audit["dimension_breakdown"].get(d, 0)) / PER_DIM_CAP * 100) * audit["dimension_weights"].get(d, 0), 2),
                            }
                            for d in DIMENSIONS
                        ])
                        st.dataframe(bd_df, width="stretch", hide_index=True)

                        # [v4 보강] OWL Reasoning Trace Table — ontology-calibrated reasoning의 논리 사슬
                        trace = build_reasoning_trace(audit)
                        if trace:
                            st.markdown("**🧬 OWL Reasoning Trace (ontology-calibrated reasoning 논리 사슬)**")
                            st.caption(
                                "audit_temporal_pair의 *각 추론 단계*를 구조화한 표. "
                                "ValueAnchor → OWL relation → calibratesDimension → penalty → fired_rule의 *논리 사슬*이 추적됨."
                            )
                            trace_df = pd.DataFrame(trace)
                            st.dataframe(trace_df, width="stretch", hide_index=True)

                        st.markdown("**audit reasons (그래프 추론 + 룰 발화 추적)**")
                        for r in audit.get("reasons", []):
                            st.markdown(f"- {r}")

                        if audit.get("fired_rules"):
                            st.markdown(f"**actual axiom fired_rules ({len(audit['fired_rules'])}개)**")
                            fr_df = pd.DataFrame([
                                {
                                    "rule_id": r.get("rule_id"),
                                    "schema_id": r.get("schema_id"),
                                    "target_frame": r.get("target_frame"),
                                    "dimension": r.get("dimension"),
                                    "axiom_penalty": r.get("axiom_penalty"),
                                    "intensity": r.get("intensity"),
                                    "rule_strength": r.get("rule_strength"),
                                }
                                for r in audit["fired_rules"][:10]
                            ])
                            st.dataframe(fr_df, width="stretch", hide_index=True)

        # ─────────────────────────────────────────────────────────
        # [4단계 보강] adjacent pair audit — 중간 변곡 구간 탐지
        # ─────────────────────────────────────────────────────────
        adjacent_audits = build_adjacent_pair_audits(articles, rules_data, ontology) if articles and len(articles) >= 2 else []
        if adjacent_audits:
            st.subheader("3.6. 인접 구간 graph audit (full sequence 보강)")
            st.caption("first↔last audit이 놓칠 수 있는 중간 반전을 찾기 위해 같은 outlet+topic 그룹의 1→2, 2→3, ... 구간을 모두 검사합니다.")
            adj_rows = []
            for idx, audit in enumerate(adjacent_audits):
                past_d = audit.get("details", {}).get("past", {})
                pres_d = audit.get("details", {}).get("present", {})
                adj_rows.append({
                    "대표": "★ max" if idx == 0 else "",
                    "그룹": audit.get("_group_key", "-"),
                    "구간": audit.get("_pair_label", "-"),
                    "PAST 날짜": audit.get("past_article", {}).get("date", ""),
                    "PRESENT 날짜": audit.get("present_article", {}).get("date", ""),
                    "polarity Δ": round(abs(float(past_d.get("stance_polarity", 0) or 0) - float(pres_d.get("stance_polarity", 0) or 0)), 3),
                    "weighted_distortion": audit.get("weighted_distortion", 0),
                    "coherence_score": audit.get("score", 0),
                    "verdict": audit.get("anchor_verdict", "-"),
                    "fired_rules": len(audit.get("fired_rules", [])),
                })
            st.dataframe(pd.DataFrame(adj_rows), width="stretch", hide_index=True)

            with st.expander("📌 최대 인접 변곡 구간 상세", expanded=False):
                audit = adjacent_audits[0]
                st.write(f"**그룹**: {audit.get('_group_key', '-')}, **구간**: {audit.get('_pair_label', '-')}")
                c_prev, c_next = st.columns(2)
                with c_prev:
                    st.markdown("**이전 기사**")
                    st.caption(audit.get("past_article", {}).get("title", ""))
                    st.caption(audit.get("past_article", {}).get("date", ""))
                with c_next:
                    st.markdown("**다음 기사**")
                    st.caption(audit.get("present_article", {}).get("title", ""))
                    st.caption(audit.get("present_article", {}).get("date", ""))
                st.markdown("**audit reasons**")
                for reason in audit.get("reasons", []):
                    st.markdown(f"- {reason}")

        st.subheader("3.7. 후보 규칙 해설 및 RuleSchema 발화 분포")
        st.caption(
            "위 3.5/3.6 섹션은 최종 axiom 점수에 직접 기여한 actual fired_rules를 graph audit 맥락에서 보여줍니다. "
            "이 섹션은 같은 룰을 다시 나열하기보다, feature 기반 후보 규칙이 어떤 schema/frame/context/cue 차이로 갈리는지와 "
            "차원별 발화 분포를 해설하는 보조 영역입니다. LLM 요약이 표시되는 경우에도 이는 Feature/RAG 추출 과정의 보조 설명이며, "
            "OWL graph audit 전용 요약은 3.5의 별도 토글/expander에서 분리해 표시합니다."
        )
        render_feature_rag_rule_selection_note(scope="candidate")
        fired_df = pd.DataFrame(result.get("axiom_fired_rules", []))
        candidate_df = pd.DataFrame(result.get("candidate_rules", result.get("matched_rules", [])))

        if not fired_df.empty:
            st.success(
                f"Actual axiom fired_rules {len(fired_df)}개가 최종 axiom_distortion에 직접 기여했습니다. "
                "상세 표는 위 `3.5. OWL 그래프 추론 및 actual fired_rules`의 각 graph audit expander 안에서 확인하세요."
            )
            with st.expander("Actual axiom fired_rules 전체표 보기 (중복 방지용 접힘)", expanded=False):
                hide_cols = [
                    "llm_instruction_ko", "positive_cues", "negative_indicators",
                    "frame_definition_ko", "schema_description_ko", "expected_evidence_ko", "score_hint"
                ]
                st.caption(
                    "이 표는 검산·디버깅용 전체 목록입니다. 기본 화면에서는 graph audit 흐름 안에서 fired rule을 확인하도록 접어 두었습니다."
                )
                st.dataframe(fired_df.drop(columns=hide_cols, errors="ignore"), width="stretch", hide_index=True)
        else:
            st.info(
                "최종 graph audit에서 actual fired_rules가 없거나 graph audit이 불가능했습니다. "
                "아래 후보 규칙은 feature 기반 보조 참고용입니다."
            )

        if not candidate_df.empty:
            with st.expander("📋 Feature 기반 후보 규칙 해설 — 룰별 차이 중심", expanded=True):
                render_candidate_rule_explanations(
                    result.get("candidate_rules", result.get("matched_rules", [])),
                    sllm_meta=sllm_meta,
                    cloud_mode=cloud_mode,
                )
        else:
            st.info("표시할 feature 기반 후보 규칙이 없습니다.")

        with st.expander("🧭 RuleSchema / Dimension 발화 heatmap", expanded=False):
            render_rule_heatmap(
                result.get("axiom_fired_rules", []),
                result.get("candidate_rules", result.get("matched_rules", [])),
            )

        # ── [D] 결과 Export ───────────────────────────────────────────────────
        st.subheader("4. 결과 다운로드")
        exp_c1, exp_c2 = st.columns(2)
        # JSON export
        export_json = json.dumps({
            "final_axiom_distortion": result.get("axiom_distortion"),
            "coherence_score": result.get("coherence_score"),
            "axiom_verdict": result.get("axiom_verdict"),
            "axiom_source": result.get("axiom_source"),
            "validity_violation": result.get("validity_violation", False),
            "validity_red_card": result.get("validity_red_card"),
            "validity_trace": result.get("validity_trace", []),
            "primary_group_key": result.get("primary_group_key"),
            "v1_distortion_score": result.get("v1_distortion_score", result.get("score")),
            "v1_verdict": result.get("v1_verdict", result.get("verdict")),
            "axiom_lite_distortion": result.get("axiom_lite_distortion"),
            "axiom_lite_verdict": result.get("axiom_lite_verdict"),
            "polarity_shift": result.get("polarity_shift"),
            "temporal_penalty": result.get("temporal_penalty"),
            "features": result["features"],
            "weights": result["weights"],
            "dimension_breakdown": result.get("dimension_breakdown"),
            "axiom_fired_rules": result.get("axiom_fired_rules", []),
            "candidate_rules": result.get("candidate_rules", result.get("matched_rules", [])),
            "graph_audits": result.get("graph_audits", []),
            "article_rows": result["article_rows"],
        }, ensure_ascii=False, indent=2)
        exp_c1.download_button(
            "📥 JSON 다운로드",
            data=export_json,
            file_name="context_sync_result.json",
            mime="application/json",
            width="stretch",
        )
        # CSV export
        csv_rows = result["article_rows"].copy() if result["article_rows"] else []
        for row in csv_rows:
            row["final_axiom_distortion"] = result.get("axiom_distortion")
            row["coherence_score"] = result.get("coherence_score")
            row["axiom_verdict"] = result.get("axiom_verdict")
            row["v1_distortion_score"] = result.get("v1_distortion_score", result.get("score"))
            row["v1_verdict"] = result.get("v1_verdict", result.get("verdict"))
            for dim, val in result["features"].items():
                row[f"feat_{dim}"] = val
        export_csv = pd.DataFrame(csv_rows).to_csv(index=False)
        exp_c2.download_button(
            "📥 CSV 다운로드",
            data=export_csv,
            file_name="context_sync_result.csv",
            mime="text/csv",
            width="stretch",
        )

        # ── [E] 의미 공간 시각화 (Dense Embedding) ────────────────────────────────
        if use_dense_rag and result.get("matched_rules"):
            st.subheader("6. 의미 공간(Embedding Space) 2D 시각화")
            dense_model = load_dense_model()
            if dense_model:
                try:
                    import plotly.express as px
                    from sklearn.decomposition import PCA

                    article_text = " ".join(" ".join(str(a.get(k, "")) for k in ["title", "body"] if a.get(k)) for a in articles)
                    matched = result["matched_rules"]
                    
                    # 텍스트 수집 (기사 1개 + 매칭된 룰 N개)
                    texts = [article_text] + [_rule_text(r) for r in matched]
                    labels = ["📰 입력 기사"] + [f"🎯 {r.get('rule_id')}" for r in matched]
                    
                    # 툴팁에 보여줄 텍스트 (줄바꿈 추가로 보기 좋게)
                    hover_texts = [article_text[:100] + "..."] + [r.get("llm_instruction_ko", r.get("rule_id"))[:100] + "..." for r in matched]
                    types = ["Article"] + ["Rule"] * len(matched)
                    
                    # 임베딩 및 PCA (2차원으로 축소)
                    with st.spinner("시각화를 위한 벡터 차원 축소 중..."):
                        embs = dense_model.encode(texts)
                        if len(embs) > 1:  # PCA는 최소 2개 이상의 데이터 필요
                            # 컴포넌트 수는 데이터 수보다 클 수 없음
                            n_components = min(2, len(embs))
                            pca = PCA(n_components=n_components)
                            coords = pca.fit_transform(embs)
                            
                            # 데이터가 2개여서 1차원으로만 축소된 경우 y축을 0으로 처리
                            x_coords = coords[:, 0]
                            y_coords = coords[:, 1] if n_components == 2 else [0] * len(coords)
                            
                            df_plot = pd.DataFrame({
                                "x": x_coords,
                                "y": y_coords,
                                "label": labels,
                                "hover": hover_texts,
                                "Type": types
                            })
                            
                            fig = px.scatter(
                                df_plot, x="x", y="y", color="Type", text="label", hover_data=["hover"],
                                color_discrete_map={"Article": "#e74c3c", "Rule": "#3498db"}
                            )
                            fig.update_traces(textposition='top center', marker=dict(size=12))
                            fig.update_layout(
                                xaxis_title="PCA Dimension 1",
                                yaxis_title="PCA Dimension 2",
                                showlegend=True,
                                margin=dict(l=20, r=20, t=30, b=20)
                            )
                            st.plotly_chart(fig, width="stretch")
                            st.caption("PCA(주성분 분석)를 통해 차원을 2D로 축소했습니다. 거리가 가까울수록 의미적으로 유사함을 나타냅니다.")
                except Exception as e:
                    st.error(f"시각화 중 오류가 발생했습니다: {e}")

with tab2:
    st.subheader("수동 지표 시뮬레이터")
    st.caption("v2.0 baseline 5차원 평가 + 가중치 직접 조작. 기본값은 OWL baseline(0.34/cap45), 시계열 엄격 모드는 대안 프로파일로 시연합니다.")

    sim_mode = st.radio(
        "조작 대상",
        ["📊 차원 값 (관측치 features)", "⚖️ 가중치 (시스템 사상 weights)"],
        horizontal=True,
        help="features는 '이 기사를 어떻게 본다'의 조작. weights는 '본 시스템이 어떤 차원을 더 중시한다'의 조작.",
    )

    # 공통: 기본 manual_features (5차원)
    if "sim_features" not in st.session_state:
        st.session_state.sim_features = {
            "temporal_shift": 40,
            "frame_effect": 30,
            "context_omission": 20,
            "consensus_deviation": 10,
            "evidence_quality": 15,
        }

    if sim_mode.startswith("📊"):
        # ─── 차원 값 (features) 조작 모드 ───
        st.markdown("##### 5차원 features 슬라이더")
        cols = st.columns(5)
        manual_features = {
            "temporal_shift": cols[0].slider("시계열 논조 이동", 0, 100, st.session_state.sim_features["temporal_shift"], key="sl_ts"),
            "frame_effect": cols[1].slider("프레임 효과", 0, 100, st.session_state.sim_features["frame_effect"], key="sl_fe"),
            "context_omission": cols[2].slider("맥락 누락", 0, 100, st.session_state.sim_features["context_omission"], key="sl_co"),
            "consensus_deviation": cols[3].slider("기준 이탈", 0, 100, st.session_state.sim_features["consensus_deviation"], key="sl_cd"),
            "evidence_quality": cols[4].slider("증거 품질", 0, 100, st.session_state.sim_features["evidence_quality"], key="sl_eq", help="높을수록 증거 부재 = 왜곡 큼"),
        }
        st.session_state.sim_features = manual_features
        custom_weights = None  # 기본 가중치 사용
    else:
        # ─── 가중치 (weights) 직접 조작 모드 ───
        st.markdown("##### dimension_weights 직접 조작")

        # 프리셋
        preset = st.selectbox(
            "프리셋",
            ["v2.0 baseline (OWL 기본값)", "시계열 엄격 (0.40/cap45)", "사실 우선", "프레임 우선", "다원 이성"],
            help="다른 가치 입장으로 본 시스템을 보면 어떻게 바뀌는지 시연. 슬라이더를 직접 만져도 됨.",
        )
        PRESETS = {
            "v2.0 baseline (OWL 기본값)": (0.34, 0.24, 0.18, 0.14, 0.10),
            "시계열 엄격 (0.40/cap45)":     (0.40, 0.22, 0.15, 0.13, 0.10),
            "사실 우선":                   (0.20, 0.18, 0.18, 0.14, 0.30),
            "프레임 우선":                 (0.25, 0.40, 0.13, 0.12, 0.10),
            "다원 이성":                   (0.25, 0.20, 0.15, 0.30, 0.10),
        }
        ts0, fe0, co0, cd0, eq0 = PRESETS[preset]

        wcols = st.columns(5)
        custom_weights = {
            "temporal_shift": wcols[0].slider("temporal_shift", 0.0, 1.0, ts0, 0.01, key=f"w_ts_{preset}"),
            "frame_effect": wcols[1].slider("frame_effect", 0.0, 1.0, fe0, 0.01, key=f"w_fe_{preset}"),
            "context_omission": wcols[2].slider("context_omission", 0.0, 1.0, co0, 0.01, key=f"w_co_{preset}"),
            "consensus_deviation": wcols[3].slider("consensus_deviation", 0.0, 1.0, cd0, 0.01, key=f"w_cd_{preset}"),
            "evidence_quality": wcols[4].slider("evidence_quality", 0.0, 1.0, eq0, 0.01, key=f"w_eq_{preset}"),
        }
        total = sum(custom_weights.values())
        if abs(total - 1.0) > 0.01:
            st.error(f"⚠️ 가중치 합 = {total:.3f}. **1.0이 되도록 조정**하세요. (현재 합으로 계산은 진행되지만, 사상적 정합성이 깨집니다.)")
        else:
            st.success(f"✓ 가중치 합 = {total:.3f}")

        # features는 기본값 유지 (관측치 시뮬레이션용)
        st.markdown("##### features (관측치 — 가중치 효과 비교용 기준값)")
        with st.expander("features 조정 (선택)", expanded=False):
            fcols = st.columns(5)
            manual_features = {
                "temporal_shift": fcols[0].slider("시계열 논조 이동", 0, 100, st.session_state.sim_features["temporal_shift"], key="fsl_ts"),
                "frame_effect": fcols[1].slider("프레임 효과", 0, 100, st.session_state.sim_features["frame_effect"], key="fsl_fe"),
                "context_omission": fcols[2].slider("맥락 누락", 0, 100, st.session_state.sim_features["context_omission"], key="fsl_co"),
                "consensus_deviation": fcols[3].slider("기준 이탈", 0, 100, st.session_state.sim_features["consensus_deviation"], key="fsl_cd"),
                "evidence_quality": fcols[4].slider("증거 품질", 0, 100, st.session_state.sim_features["evidence_quality"], key="fsl_eq"),
            }
            st.session_state.sim_features = manual_features

    if st.button("수동 규칙 엔진 실행", type="primary"):
        # custom_weights가 있으면 rules_data를 임시 복사해 weights override
        if custom_weights is not None:
            # aggregation_formula를 custom_weights 기준으로 재생성
            custom_formula = " + ".join(f"{v:.2f}*{k}" for k, v in custom_weights.items())
            rules_data_for_sim = dict(rules_data)
            rules_data_for_sim["aggregation_formula"] = f"final_distortion = {custom_formula}"
        else:
            rules_data_for_sim = rules_data

        result = analyze_pipeline(
            [],
            rules_data_for_sim,
            top_n=top_n,
            profile=profile_filter,
            context=context_filter,
            manual_features=manual_features,
            owl_graph=ontology,
        )

        # 메트릭 표시
        mcols = st.columns(3)
        mcols[0].metric("v1 보조 왜곡도", f"{result.get('v1_distortion_score', result['score']):.2f}", help=str(result["weights"]))
        mcols[1].metric("axiom-lite 왜곡도", f"{result.get('axiom_distortion', 0):.2f}")
        mcols[2].metric("후보 규칙", f"{len(result.get('candidate_rules', result['matched_rules']))}개")

        # 가중치 + features 분해 표시
        wf_df = pd.DataFrame([
            {
                "dimension": dim,
                "feature": manual_features.get(dim, 0),
                "weight": result["weights"].get(dim, 0),
                "기여도": round(manual_features.get(dim, 0) * result["weights"].get(dim, 0), 2),
            }
            for dim in ["temporal_shift", "frame_effect", "context_omission", "consensus_deviation", "evidence_quality"]
        ])
        st.markdown("##### 차원별 기여도 분해")
        st.dataframe(wf_df, width="stretch", hide_index=True)
        st.bar_chart(wf_df.set_index("dimension")["기여도"])

        # 판정 근거
        if result.get("verdict_reason"):
            st.info(f"📌 판정 핵심 근거: {result['verdict_reason']}")
        # 활성 규칙
        if result["matched_rules"]:
            with st.expander("후보 규칙 목록", expanded=False):
                st.dataframe(pd.DataFrame(result["matched_rules"]), width="stretch", hide_index=True)


    st.divider()
    st.subheader("Axiom cap 정규화 시뮬레이터")
    st.caption("raw penalty가 per_dim_cap을 거쳐 weighted_distortion으로 바뀌는 과정을 직접 확인합니다. 예: temporal_shift -30, cap 45 → 정규화 66.7 × weight 0.34 = 22.7점 왜곡.")
    cap_cols = st.columns(6)
    cap_value = cap_cols[0].slider("per_dim_cap", 30.0, 80.0, float(PER_DIM_CAP), 1.0, key="cap_sim_value")
    raw_breakdown = {
        "temporal_shift": cap_cols[1].slider("temporal raw", -60.0, 0.0, -30.0, 1.0, key="cap_raw_ts"),
        "frame_effect": cap_cols[2].slider("frame raw", -60.0, 0.0, -10.0, 1.0, key="cap_raw_fe"),
        "context_omission": cap_cols[3].slider("context raw", -60.0, 0.0, -8.0, 1.0, key="cap_raw_co"),
        "consensus_deviation": cap_cols[4].slider("consensus raw", -60.0, 0.0, -5.0, 1.0, key="cap_raw_cd"),
        "evidence_quality": cap_cols[5].slider("evidence raw", -60.0, 0.0, -3.0, 1.0, key="cap_raw_eq"),
    }
    sim_weights = custom_weights if custom_weights is not None else weights
    sim_distortion = compute_weighted_distortion(raw_breakdown, sim_weights, per_dim_cap=cap_value)
    sim_rows = []
    for dim in DIMENSIONS:
        normalized = min(100.0, abs(raw_breakdown.get(dim, 0.0)) / cap_value * 100.0)
        sim_rows.append({
            "dimension": dim,
            "raw_penalty": raw_breakdown.get(dim, 0.0),
            "normalized": round(normalized, 1),
            "weight": sim_weights.get(dim, 0),
            "contribution": round(normalized * sim_weights.get(dim, 0), 2),
        })
    sim_df = pd.DataFrame(sim_rows)
    sim_m1, sim_m2 = st.columns(2)
    sim_m1.metric("weighted_distortion", f"{sim_distortion:.1f}")
    sim_m2.metric("coherence_score", f"{100 - sim_distortion:.1f}")
    st.dataframe(sim_df, width="stretch", hide_index=True)
    st.bar_chart(sim_df.set_index("dimension")["contribution"])

with tab3:
    st.subheader("규칙/온톨로지 탐색")
    s1, s2, s3 = st.columns(3)
    s1.metric("전체 규칙", f"{summary['count']:,}")
    s2.metric("차원 수", len(summary["by_dimension"]))
    s3.metric("프레임 Top10 수", len(summary["by_frame_top10"]))

    st.write("**차원별 규칙 수**")
    st.dataframe(pd.DataFrame(summary["by_dimension"].items(), columns=["dimension", "count"]), width="stretch", hide_index=True)
    st.write("**Schema type별 규칙 수**")
    st.dataframe(pd.DataFrame(summary["by_schema_type"].items(), columns=["schema_type", "count"]), width="stretch", hide_index=True)
    st.write("**상위 프레임**")
    st.dataframe(pd.DataFrame(summary["by_frame_top10"].items(), columns=["target_frame", "count"]), width="stretch", hide_index=True)

    query = st.text_input("규칙 검색", placeholder="예: SilentPivot, temporal_shift, Election")
    if query:
        q = query.lower()
        filtered = [r for r in rules if q in json.dumps(r, ensure_ascii=False).lower()]
        st.write(f"검색 결과: {len(filtered)}개")
        st.dataframe(pd.DataFrame(filtered[:100]), width="stretch", hide_index=True)

    with st.expander("OWL 일부 정보"):
        st.write(f"파일: `{ONTOLOGY_PATH.name}`")
        st.write(f"RDF triples: `{len(ontology)}`")
        st.write("이 앱에서는 OWL을 무거운 추론 엔진이 아니라 의미 기준층/스키마 레이어로 사용합니다.")

    # ── [C] OWL 계층 구조 시각화 (Mermaid) ────────────────────────────────────
    st.subheader("OWL 계층 구조 시각화")
    st.caption("Layer → ValueAnchor → Frame → RuleSchema 관계를 자동으로 추출한 다이어그램입니다.")
    mermaid_code = build_owl_mermaid(ontology)
    st.markdown(f"```mermaid\n{mermaid_code}\n```")

with tab4:
    st.subheader("LLM (GPT/sLLM) 연결 방식")
    st.write("이 앱은 기본적으로 휴리스틱 추출기로 바로 실행되고, 선택적으로 OpenAI GPT 모델 또는 Hugging Face 기반 로컬/오픈소스 sLLM 추출기를 활성화할 수 있습니다.")
    st.code(
        """# 기본 실행 및 GPT 이용 (필요시 openai 설치 및 API 키 설정 필요)
pip install -r requirements.txt
streamlit run app.py

# 로컬/오픈소스 sLLM 모드까지 사용
pip install -r requirements-sllm.txt
streamlit run app.py
""",
        language="bash",
    )
    st.write("로컬 sLLM 모델 가중치는 GitHub 저장소에 포함하지 않으며, 첫 실행 때 Hugging Face 캐시에 다운로드됩니다.")
    st.write("기본 로컬 모델명은 `Qwen/Qwen2.5-0.5B-Instruct`이며, OpenAI GPT 모델(예: `gpt-4o`, `gpt-4-turbo`) 등도 연동하여 사용할 수 있습니다.")
    st.warning("Streamlit Community Cloud 무료 환경 등에서는 로컬 sLLM 연동 시 메모리 한계나 실행 지연이 있을 수 있으므로 GPT API를 사용하거나 Ollama, HF Inference API 등 외부 API로 위임하는 것을 권장합니다.")

with tab5:
    st.subheader("GitHub / Streamlit Cloud 배포 구조")
    st.caption("현재 공개 저장소와 Streamlit Community Cloud 2차 배포 기준을 반영한 구조입니다.")

    st.markdown(
        "- **Live Demo**: <https://context-sync-curator1.streamlit.app/>\n"
        "- **GitHub Repository**: <https://github.com/downteam7-crypto/context-sync.curator1>\n"
        "- **Cloud main file path**: `app.py`\n"
        "- **Cloud 기본 실행 모드**: OpenAI API + OWL/JSON rule engine 중심의 경량 배포"
    )

    st.markdown("#### 저장소 구조")
    st.code(
        """context-sync.curator1/
├─ app.py                         # Streamlit 메인 앱
├─ rule_engine.py                 # OWL/JSON 기반 axiom audit 계산 엔진
├─ sllm_extractor.py              # OpenAI GPT 추출기 + 로컬 sLLM/Dense RAG 옵션
├─ requirements.txt               # Streamlit Cloud 경량 의존성
├─ requirements-sllm.txt          # 로컬 sLLM / Dense RAG 고급 의존성
├─ README.md
├─ LICENSE
├─ .env.example
├─ .gitignore
├─ .streamlit/
│  └─ config.toml
├─ ontology/
│  ├─ context_sync_app_centered_ontology_1024.owl
│  └─ news_rules_1024.json
├─ sample_data/
│  ├─ sample_articles.json
│  └─ sample_scenarios.json
├─ docs/
│  ├─ 01_problem_framing.md
│  ├─ 02_cognition_and_metacognition.md
│  ├─ 03_hybrid_ontology.md
│  └─ 04_background.md
├─ legacy/
│  └─ 로드맵_3단계/
│     └─ v1.0-3.5stage/
└─ comparison/
   ├─ README.md
   └─ axiom_tracker_pure_llm.py
""",
        language="text",
    )

    st.markdown("#### Cloud 배포 범위")
    st.write(
        "Streamlit Cloud에서는 `requirements.txt`만 설치해 앱을 가볍게 실행합니다. "
        "`torch`, `transformers`, `sentence-transformers`, `scikit-learn` 기반의 Dense RAG/로컬 sLLM 기능은 "
        "`requirements-sllm.txt`로 분리해 로컬 실행 옵션으로 둡니다."
    )
    st.code(
        """# Cloud
streamlit run app.py

# Local advanced mode
pip install -r requirements.txt
pip install -r requirements-sllm.txt
STREAMLIT_CLOUD=0 streamlit run app.py
""",
        language="bash",
    )

    st.markdown("#### 비밀키 / 대용량 데이터 원칙")
    st.write(
        "OpenAI API Key는 GitHub에 올리지 않고 Streamlit Cloud의 Secrets 또는 로컬 `.env`에 둡니다. "
        "대용량 기사 원문, 임베딩 DB, 캐시, 가상환경, `__pycache__`/`.pyc` 파일은 저장소에서 제외합니다."
    )
    st.info(
        "Cloud 데모는 경량 배포판입니다. Dense RAG, sentence-transformers 시각화, 로컬 Hugging Face sLLM은 "
        "GitHub 저장소를 내려받아 로컬에서 실행할 때 사용하는 고급 옵션입니다."
    )

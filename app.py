from __future__ import annotations

import json
import os
from pathlib import Path

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
from sllm_extractor import DEFAULT_MODEL, extract_features_with_sllm, compare_rag_results

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
    from sllm_extractor import _rule_text
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
    with st.expander("dimension_weights (v1.2.4 baseline)"):
        for d, w in weights.items():
            st.write(f"- `{d}`: {w}")
        st.caption("OWL baseline 복귀: 0.34/0.24/0.18/0.14/0.10. 시계열 엄격 모드(0.40/cap45)는 시뮬레이터 프리셋에서 선택 가능.")
    st.caption(
        "ontology-calibrated reasoning: OWL 어휘 통제 + JSON 룰 + 5차원 정규화 "
        "+ explanation mitigation (변경 사유 충실 시 부가 페널티 부분 완화)"
    )

    st.write("**GitHub 친화 구조**")
    st.code("코드 + OWL + JSON만 저장\n대용량 기사/임베딩은 repo 제외", language="text")

    st.divider()
    st.subheader("LLM (GPT/sLLM) 추출기")
    use_sllm = st.toggle("기사 지표를 LLM(GPT/sLLM)으로 추출", value=False)
    sllm_model = st.text_input("모델명 (OpenAI GPT 또는 Hugging Face)", value=DEFAULT_MODEL, help="Hugging Face 모델명(예: Qwen/Qwen2.5-0.5B-Instruct) 또는 OpenAI GPT 모델명(예: gpt-4o)을 입력할 수 있습니다.")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="GPT 모델 이용 시 필요합니다. 환경변수(OPENAI_API_KEY)에 설정되어 있는 경우 비워두셔도 됩니다.")
    sllm_max_tokens = st.slider("LLM max_new_tokens", 80, 500, 220)
    st.caption("CPU 및 로컬 환경에서는 느릴 수 있습니다. GPT 모델 이용 시 API Key 설정이 필요합니다. 실패 시 휴리스틱 모드로 자동 전환됩니다.")

    st.divider()
    use_dense_rag = st.toggle("🚀 [Beta] 밀집 벡터(Dense) RAG 및 시각화", value=False)
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


def _article_date_value(article: dict):
    """날짜 문자열을 정렬 가능한 datetime으로 변환한다."""
    return pd.to_datetime(article.get("date") or article.get("published_at") or "", errors="coerce")


def _article_text(article: dict) -> str:
    """audit_temporal_pair에 넣을 기사 텍스트를 합성한다."""
    return " ".join(str(article.get(k, "")) for k in ["title", "summary", "body", "text", "content"] if article.get(k))


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
    st.dataframe(dim_pivot, use_container_width=True)

    st.markdown("**Schema type × rule source**")
    schema_pivot = hm_df.pivot_table(index="schema_type", columns="source", values="count", aggfunc="sum", fill_value=0)
    st.dataframe(schema_pivot, use_container_width=True)

    st.markdown("**Penalty sum by dimension**")
    penalty_df = hm_df.groupby("dimension", as_index=False)["penalty_abs"].sum().sort_values("penalty_abs", ascending=False)
    st.bar_chart(penalty_df.set_index("dimension")["penalty_abs"])


tab1, tab2, tab3, tab4, tab5 = st.tabs(["기사 분석", "수동 시뮬레이터", "규칙/온톨로지 탐색", "LLM 설정", "GitHub 배포 구조"])

with tab1:
    st.subheader("1. 기사 묶음 입력")
    st.write("같은 언론사/같은 사안의 기사들을 날짜순으로 비교하면 시계열 논조 이동을 더 잘 볼 수 있습니다.")

    # ── 입력 방식 선택 ────────────────────────────────────────────────────────
    input_mode = st.radio(
        "입력 방식",
        ["📝 기사 직접 입력 (자동 JSON 변환)", "{ } JSON 직접 입력"],
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
                        if st.button(sc["label"], key=f"sc_{sc['id']}", use_container_width=True):
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
            f_body   = st.text_area("본문 *", placeholder="기사 본문을 여기에 붙여넣으세요. 길이 제한 없음.", height=200)
            submitted = st.form_submit_button("➕ 기사 추가", use_container_width=True, type="primary")

        if submitted:
            if not f_title.strip() and not f_body.strip():
                st.warning("제목 또는 본문을 입력해 주세요.")
            else:
                entry: dict = {"title": f_title.strip(), "body": f_body.strip()}
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
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            col_clr, col_json = st.columns([1, 3])
            if col_clr.button("🗑️ 전체 초기화", use_container_width=True):
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
    # 모드 B: JSON 직접 입력 (기존 방식)
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
                 use_container_width=True):
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
                        st.dataframe(sp_df, use_container_width=True, hide_index=True)
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
                            st.dataframe(de_df, use_container_width=True, hide_index=True)
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

        if result.get("dimension_breakdown"):
            with st.expander("⚙️ 최종 axiom 차원별 페널티 분해 (graph audit 우선)", expanded=True):
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
                st.dataframe(bd_df, use_container_width=True, hide_index=True)
                st.bar_chart(bd_df.set_index("dimension")["기여 distortion"])

        if sllm_meta:
            with st.expander("LLM 추출 근거 / 원문 응답"):
                st.write("**모델**", sllm_meta.get("model"))
                st.write(
                    "**RAG 모드**",
                    "Dense (의미 임베딩)" if sllm_meta.get("rag_mode") == "dense" else "Sparse (Jaccard 어절 겹침)",
                )
                st.write("**근거**", sllm_meta.get("reason", ""))
                rag_rules = sllm_meta.get("rag_selected_rules", "")
                if rag_rules:
                    st.write("**RAG가 선택해 프롬프트에 주입한 규칙**")
                    st.code(rag_rules, language="text")
                st.write("**LLM 원문 응답**")
                st.code(sllm_meta.get("raw_response", ""), language="json")

        st.subheader("2. 지표 분해")
        feature_df = pd.DataFrame(
            [{"dimension": k, "score": v, "weight": result["weights"].get(k, 0)} for k, v in result["features"].items()]
        )
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        st.bar_chart(feature_df.set_index("dimension")["score"])

        if result["series"]:
            st.subheader("3. 기사별 시계열 신호")
            series_df = pd.DataFrame(result["series"]).copy()
            article_rows_df = pd.DataFrame(result["article_rows"])
            st.dataframe(article_rows_df, use_container_width=True, hide_index=True)
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
                    st.plotly_chart(fig, use_container_width=True)

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
                            st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)
                except Exception:
                    st.line_chart(series_df.set_index("date")["sentiment"])

        # ─────────────────────────────────────────────────────────
        # [개선] OWL graph audit 결과 표시 — analyze_pipeline에서 이미 계산한 결과 재사용
        # ─────────────────────────────────────────────────────────
        graph_audits = result.get("graph_audits", [])
        if articles and len(articles) >= 2:
            st.subheader("3.5. OWL 그래프 추론 (main axiom audit)")
            st.caption(
                "이 섹션의 graph audit 결과가 이제 메인 `axiom_distortion`의 1순위 입력입니다. "
                "같은 outlet+topic 그룹의 시간순 첫 기사와 마지막 기사를 비교해 "
                "OWL `conflictsWith` / `reinforces` / `calibratesDimension` 관계와 JSON 룰 발화를 추적합니다."
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
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

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
                        st.dataframe(bd_df, use_container_width=True, hide_index=True)

                        # [v4 보강] OWL Reasoning Trace Table — ontology-calibrated reasoning의 논리 사슬
                        trace = build_reasoning_trace(audit)
                        if trace:
                            st.markdown("**🧬 OWL Reasoning Trace (ontology-calibrated reasoning 논리 사슬)**")
                            st.caption(
                                "audit_temporal_pair의 *각 추론 단계*를 구조화한 표. "
                                "ValueAnchor → OWL relation → calibratesDimension → penalty → fired_rule의 *논리 사슬*이 추적됨."
                            )
                            trace_df = pd.DataFrame(trace)
                            st.dataframe(trace_df, use_container_width=True, hide_index=True)

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
                            st.dataframe(fr_df, use_container_width=True, hide_index=True)

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
            st.dataframe(pd.DataFrame(adj_rows), use_container_width=True, hide_index=True)

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

        st.subheader("4. 규칙 결과: actual axiom fired_rules + feature 후보 규칙")
        fired_df = pd.DataFrame(result.get("axiom_fired_rules", []))
        candidate_df = pd.DataFrame(result.get("candidate_rules", result.get("matched_rules", [])))

        if not fired_df.empty:
            st.markdown("**Actual axiom fired_rules** — 최종 axiom_distortion에 직접 기여한 룰")
            hide_cols = ["llm_instruction_ko", "positive_cues", "negative_indicators",
                         "frame_definition_ko", "schema_description_ko", "expected_evidence_ko", "score_hint"]
            st.dataframe(fired_df.drop(columns=hide_cols, errors="ignore"), use_container_width=True, hide_index=True)
        else:
            st.info("최종 graph audit에서 실제 axiom fired_rules가 없거나 graph audit이 불가능했습니다. 아래 후보 규칙은 보조 참고용입니다.")

        if not candidate_df.empty:
            with st.expander("📋 Feature 기반 후보 규칙 + axiom penalty 메타", expanded=not fired_df.empty):
                visible_df = candidate_df.drop(columns=["llm_instruction_ko", "positive_cues", "negative_indicators",
                                                        "frame_definition_ko", "schema_description_ko", "expected_evidence_ko", "score_hint"], errors="ignore")
                st.dataframe(visible_df, use_container_width=True, hide_index=True)
                for r in result.get("candidate_rules", result.get("matched_rules", []))[:5]:
                    st.markdown(f"**`{r.get('rule_id')}`** — {r.get('schema_id')} / {r.get('target_frame')}")
                    rcols = st.columns(4)
                    rcols[0].metric("is_fired", str(r.get("is_fired", False)))
                    rcols[1].metric("axiom_penalty", r.get("axiom_penalty", 0))
                    rcols[2].metric("rule_strength", r.get("rule_strength", 0))
                    rcols[3].metric("intensity", r.get("intensity", 0))
                    if r.get("frame_definition_ko"):
                        st.caption(f"**프레임 정의**: {r['frame_definition_ko']}")
                    if r.get("schema_description_ko"):
                        st.caption(f"**스키마 설명**: {r['schema_description_ko']}")
                    if r.get("expected_evidence_ko"):
                        st.caption(f"**기대 증거**: {r['expected_evidence_ko']}")
                    if r.get("score_hint"):
                        st.caption(f"**score_hint**: {r['score_hint']}")
                    st.divider()

        with st.expander("🧭 RuleSchema / Dimension 발화 heatmap", expanded=False):
            render_rule_heatmap(
                result.get("axiom_fired_rules", []),
                result.get("candidate_rules", result.get("matched_rules", [])),
            )

        # ── [D] 결과 Export ───────────────────────────────────────────────────
        st.subheader("5. 결과 다운로드")
        exp_c1, exp_c2 = st.columns(2)
        # JSON export
        export_json = json.dumps({
            "final_axiom_distortion": result.get("axiom_distortion"),
            "coherence_score": result.get("coherence_score"),
            "axiom_verdict": result.get("axiom_verdict"),
            "axiom_source": result.get("axiom_source"),
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
            use_container_width=True,
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
            use_container_width=True,
        )

        # ── [E] 의미 공간 시각화 (Dense Embedding) ────────────────────────────────
        if use_dense_rag and result.get("matched_rules"):
            st.subheader("6. 의미 공간(Embedding Space) 2D 시각화")
            dense_model = load_dense_model()
            if dense_model:
                try:
                    import plotly.express as px
                    from sklearn.decomposition import PCA
                    from sllm_extractor import _rule_text

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
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("PCA(주성분 분석)를 통해 차원을 2D로 축소했습니다. 거리가 가까울수록 의미적으로 유사함을 나타냅니다.")
                except Exception as e:
                    st.error(f"시각화 중 오류가 발생했습니다: {e}")

with tab2:
    st.subheader("수동 지표 시뮬레이터")
    st.caption("v1.2.4 baseline 5차원 평가 + 가중치 직접 조작. 기본값은 OWL baseline(0.34/cap40), 시계열 엄격 모드는 대안 프로파일로 시연합니다.")

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
            ["v1.2.4 baseline (OWL 기본값)", "시계열 엄격 (0.40/cap45)", "사실 우선", "프레임 우선", "다원 이성"],
            help="다른 가치 입장으로 본 시스템을 보면 어떻게 바뀌는지 시연. 슬라이더를 직접 만져도 됨.",
        )
        PRESETS = {
            "v1.2.2 (현재 기본값)":   (0.40, 0.22, 0.15, 0.13, 0.10),
            "시계열 엄격":             (0.50, 0.18, 0.14, 0.10, 0.08),
            "사실 우선":               (0.20, 0.18, 0.18, 0.14, 0.30),
            "프레임 우선":             (0.25, 0.40, 0.13, 0.12, 0.10),
            "다원 이성":               (0.25, 0.20, 0.15, 0.30, 0.10),
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
        st.dataframe(wf_df, use_container_width=True, hide_index=True)
        st.bar_chart(wf_df.set_index("dimension")["기여도"])

        # 판정 근거
        if result.get("verdict_reason"):
            st.info(f"📌 판정 핵심 근거: {result['verdict_reason']}")
        # 활성 규칙
        if result["matched_rules"]:
            with st.expander("후보 규칙 목록", expanded=False):
                st.dataframe(pd.DataFrame(result["matched_rules"]), use_container_width=True, hide_index=True)


    st.divider()
    st.subheader("Axiom cap 정규화 시뮬레이터")
    st.caption("raw penalty가 per_dim_cap을 거쳐 weighted_distortion으로 바뀌는 과정을 직접 확인합니다. 예: temporal_shift -30, cap 40 → 정규화 75.0 × weight 0.34 = 25.5점 왜곡.")
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
    st.dataframe(sim_df, use_container_width=True, hide_index=True)
    st.bar_chart(sim_df.set_index("dimension")["contribution"])

with tab3:
    st.subheader("규칙/온톨로지 탐색")
    s1, s2, s3 = st.columns(3)
    s1.metric("전체 규칙", f"{summary['count']:,}")
    s2.metric("차원 수", len(summary["by_dimension"]))
    s3.metric("프레임 Top10 수", len(summary["by_frame_top10"]))

    st.write("**차원별 규칙 수**")
    st.dataframe(pd.DataFrame(summary["by_dimension"].items(), columns=["dimension", "count"]), use_container_width=True, hide_index=True)
    st.write("**Schema type별 규칙 수**")
    st.dataframe(pd.DataFrame(summary["by_schema_type"].items(), columns=["schema_type", "count"]), use_container_width=True, hide_index=True)
    st.write("**상위 프레임**")
    st.dataframe(pd.DataFrame(summary["by_frame_top10"].items(), columns=["target_frame", "count"]), use_container_width=True, hide_index=True)

    query = st.text_input("규칙 검색", placeholder="예: SilentPivot, temporal_shift, Election")
    if query:
        q = query.lower()
        filtered = [r for r in rules if q in json.dumps(r, ensure_ascii=False).lower()]
        st.write(f"검색 결과: {len(filtered)}개")
        st.dataframe(pd.DataFrame(filtered[:100]), use_container_width=True, hide_index=True)

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
    st.subheader("GitHub/Streamlit 배포 구조")
    st.write("이 패키지는 GitHub 저장소에 올리기 쉽게 코드와 소형 기준 파일만 포함합니다.")
    st.code(
        """context-sync-news-analyzer/
├─ app.py
├─ rule_engine.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ .streamlit/config.toml
├─ ontology/
│  ├─ context_sync_app_centered_ontology_1024.owl
│  └─ news_rules_1024.json
└─ sample_data/
   └─ sample_articles.json
""",
        language="text",
    )
    st.write("대용량 기사 원문, 임베딩 DB, 캐시, 가상환경은 `.gitignore`로 제외합니다.")
    st.write("GitHub에는 코드와 가벼운 기준층만 두고, 수십 GB 데이터는 Google Drive API/S3/DB에 두는 구조가 안전합니다.")

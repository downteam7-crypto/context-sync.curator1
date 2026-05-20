from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import rdflib

DIMENSIONS = ["temporal_shift", "frame_effect", "context_omission", "consensus_deviation", "evidence_quality"]
DEFAULT_WEIGHTS = {
    "temporal_shift": 0.34,
    "frame_effect": 0.24,
    "context_omission": 0.18,
    "consensus_deviation": 0.14,
    "evidence_quality": 0.10,
}

# [v1.2.4 baseline-restored] OWL baseline 가중치 복귀 + cap 40.
# 시계열 정합성은 여전히 최상위 차원이지만, frame/context/consensus/evidence와 공동 판단한다.
# 0.40/cap45는 "시계열 엄격" 프로파일로 유지하고, 기본 엔진은 OWL baseline을 따른다.
PER_DIM_CAP = 40.0

SEVERITY_MULTIPLIER = {
    "low": 0.65,
    "medium": 0.8,
    "high": 0.93,
    "critical": 1.0,
}

# [v1.2.2 흡수] axiom의 severity_base — 룰 등급별 기본 최대 감점
SEVERITY_BASE = {
    "critical": 18.0,
    "high": 12.0,
    "medium": 7.0,
    "low": 4.0,
}

POSITIVE_LEXICON = [
    "성과", "개선", "필요", "긍정", "기대", "안정", "회복", "확대", "성장", "합리", "미래", "불가피", "개혁",
    "투명", "책임", "근거", "검증", "균형", "공정", "효과", "지지", "환영", "해결",
]
NEGATIVE_LEXICON = [
    "논란", "비판", "위기", "충격", "졸속", "혼란", "우려", "실패", "의혹", "갈등", "불안", "왜곡",
    "책임론", "파문", "무리", "강행", "부실", "후퇴", "악화", "반발", "위험", "논란",
]
CONTEXT_EXPLANATION_CUES = ["새로운 증거", "조건 변화", "정책 변경", "정정", "사과", "반론", "근거", "배경", "맥락", "설명"]
OMISSION_CUES = ["알려졌다", "관계자", "익명", "일각", "논란", "충격", "위기", "단독", "파문"]
FRAME_CUES = ["위기", "충격", "논란", "갈등", "파문", "졸속", "강행", "반발", "책임론", "의혹"]
CONSENSUS_CUES = ["공정", "책임", "투명", "합의", "공공", "인권", "안전", "검증", "절차", "근거"]

# [v1.2.2] evidence_quality 차원: EvidenceTransparency ValueAnchor 기반
# 증거 부재(점수 올라감 = 왜곡): 익명 관계자, 추정, 미확인 표현
EVIDENCE_LACK_CUES = [
    "익명", "관계자에 따르면", "알려졌다", "전해졌다", "추정", "관측", "보인다",
    "것으로 보인다", "예상된다", "전망된다", "지적된다", "평가된다", "분석된다",
]
# 증거 충실(점수 깎이는 방향): 출처·통계·원문 식별 가능
EVIDENCE_PRESENT_CUES = [
    "출처", "통계청", "데이터", "원문", "공시", "공식 발표", "백서", "보고서",
    "논문", "연구진", "공개", "인용", "보도자료", "발언 전문", "원본",
]


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def parse_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not value:
        return datetime.min
    for fmt in ["%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(str(value)[:19], fmt)
        except ValueError:
            pass
    return datetime.min


def text_of(article: Dict[str, Any]) -> str:
    return " ".join(str(article.get(k, "")) for k in ["title", "summary", "body", "text", "content"] if article.get(k))


def count_hits(text: str, cues: Iterable[str]) -> int:
    return sum(text.count(cue) for cue in cues if cue)


def compute_explanation_mitigation_signal(past_text: str, present_text: str) -> Dict[str, Any]:
    """입장 전환 사유 설명의 충실도를 휴리스틱하게 측정한다.

    원칙:
    - stance polarity의 Δ 자체는 줄이지 않는다.
    - 다만 현재 기사에 전환 사유, 조건 변화, 근거 갱신, 정정/반론이 충분히 있으면
      SilentPivot/RetroactiveReframing류의 추가 graph/rule penalty를 부분 완화한다.

    Returns:
        {
            "present_context_expl_hits": int,
            "past_context_expl_hits": int,
            "present_omission_hits": int,
            "mitigation_factor": float,  # 1.0=no mitigation, 0.65=moderate, 0.50=strong
            "level": "none|moderate|strong",
            "reason": str,
        }
    """
    present_context = count_hits(present_text, CONTEXT_EXPLANATION_CUES)
    past_context = count_hits(past_text, CONTEXT_EXPLANATION_CUES)
    present_omission = count_hits(present_text, OMISSION_CUES)
    # 현재 기사의 설명을 가장 강하게 보고, 과거 기사 배경설명은 보조 신호로만 반영한다.
    explanation_signal = present_context + 0.5 * past_context

    if explanation_signal >= 4:
        factor = 0.50
        level = "strong"
        reason = "전환 사유/조건 변화/근거 갱신 설명이 충분함"
    elif explanation_signal >= 2:
        factor = 0.65
        level = "moderate"
        reason = "전환 사유 설명이 일부 확인됨"
    else:
        factor = 1.0
        level = "none"
        reason = "전환 사유 설명 신호가 약함"

    return {
        "present_context_expl_hits": present_context,
        "past_context_expl_hits": past_context,
        "present_omission_hits": present_omission,
        "explanation_signal": round(explanation_signal, 2),
        "mitigation_factor": factor,
        "level": level,
        "reason": reason,
    }


def sentiment_score(text: str) -> float:
    pos = count_hits(text, POSITIVE_LEXICON)
    neg = count_hits(text, NEGATIVE_LEXICON)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def load_rules(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_ontology(path: Path) -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(str(path), format="xml")
    return g


def extract_owl_frame_cues(g: rdflib.Graph) -> List[str]:
    """OWL 온톨로지에서 Frame 클래스의 label/keyword를 SPARQL로 추출해 반환.

    이 함수가 실행됨으로써 OWL이 단순 장식이 아닌 실제 어휘 기준층으로 작동한다.
    """
    cues: List[str] = []
    # rdfs:label로 정의된 Frame 인스턴스 라벨만 수집한다.
    # 기존처럼 모든 rdfs:label을 가져오면 ValueAnchor/Layer/RuleSchema 라벨까지
    # frame cue로 섞여 frame_effect가 과대 계산될 수 있다.
    q = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
        SELECT ?label WHERE {
            ?f rdf:type cs:Frame .
            ?f rdfs:label ?label .
        }
    """
    try:
        for row in g.query(q):
            label = str(row.label).strip()
            if label:
                cues.append(label)
    except Exception:
        pass
    # 개별 토큰으로 분해(공백/언더스코어 기준)
    tokens: List[str] = []
    for c in cues:
        tokens.extend(t for t in re.split(r"[\s_]+", c) if len(t) >= 2)
    return list(dict.fromkeys(tokens))  # 순서 유지 중복 제거


def extract_formula_weights(formula: str | None) -> Dict[str, float]:
    if not formula:
        return dict(DEFAULT_WEIGHTS)
    weights = {}
    for dim in DIMENSIONS:
        # match 0.38*temporal_shift or 0.38 * temporal_shift
        m = re.search(rf"([0-9]*\.?[0-9]+)\s*\*\s*{re.escape(dim)}", formula)
        if m:
            weights[dim] = float(m.group(1))
    return weights or dict(DEFAULT_WEIGHTS)


def determine_verdict(score: float, thresholds: Dict[str, Any]) -> str:
    if score >= float(thresholds.get("deviated_min", 65)):
        return "기준 이탈"
    if score >= float(thresholds.get("caution_min", 35)):
        return "주의 필요"
    return "기준 부합"


def analyze_articles(
    articles: List[Dict[str, Any]],
    owl_frame_cues: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Heuristic feature extractor for demo use.

    This is intentionally transparent and lightweight. In production, replace or
    augment this function with an LLM extractor while preserving the same output
    schema.

    owl_frame_cues: OWL 온톨로지에서 추출한 Frame 키워드. 전달 시 FRAME_CUES와 병합해
        OWL이 실제 어휘 기준층으로 작동하게 한다.
    """
    # [개선1] OWL Frame cues를 FRAME_CUES에 병합
    effective_frame_cues = list(FRAME_CUES)
    if owl_frame_cues:
        effective_frame_cues = list(dict.fromkeys(effective_frame_cues + owl_frame_cues))

    normalized = []
    for idx, article in enumerate(articles):
        item = dict(article)
        item["_idx"] = idx
        item["_date"] = parse_date(article.get("date") or article.get("published_at"))
        item["_text"] = text_of(article)
        item["_sentiment"] = sentiment_score(item["_text"])
        item["_frame_hits"] = count_hits(item["_text"], effective_frame_cues)
        item["_omission_hits"] = count_hits(item["_text"], OMISSION_CUES)
        item["_context_expl_hits"] = count_hits(item["_text"], CONTEXT_EXPLANATION_CUES)
        item["_consensus_hits"] = count_hits(item["_text"], CONSENSUS_CUES)
        # [v1.2.2] evidence_quality: 증거 부재 cues − 증거 충실 cues
        item["_evidence_lack_hits"] = count_hits(item["_text"], EVIDENCE_LACK_CUES)
        item["_evidence_present_hits"] = count_hits(item["_text"], EVIDENCE_PRESENT_CUES)
        normalized.append(item)

    if not normalized:
        return {"features": {k: 0 for k in DIMENSIONS}, "article_rows": [], "series": []}

    # [개선3] (outlet, topic) 그룹 내부에서만 시계열 이동 계산
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for a in normalized:
        key = (str(a.get("outlet", "")), str(a.get("topic", "")))
        grouped[key].append(a)

    shifts: List[float] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda x: x["_date"])
        sents = [a["_sentiment"] for a in group]
        max_shift = max(abs(sents[i] - sents[i - 1]) for i in range(1, len(sents)))
        first_last = abs(sents[-1] - sents[0])
        shifts.append(0.65 * max_shift + 0.35 * first_last)

    if shifts:
        temporal_shift = clamp(max(shifts) * 100)
    elif len(normalized) >= 2:
        # 그룹핑 불가(outlet/topic 미입력)인 경우 전체 기사 기준 fallback
        all_sorted = sorted(normalized, key=lambda x: x["_date"])
        sents = [a["_sentiment"] for a in all_sorted]
        max_shift = max(abs(sents[i] - sents[i - 1]) for i in range(1, len(sents)))
        first_last = abs(sents[-1] - sents[0])
        temporal_shift = clamp((0.65 * max_shift + 0.35 * first_last) * 100)
    else:
        temporal_shift = 0.0

    frame_effect = clamp((sum(a["_frame_hits"] for a in normalized) / max(1, len(normalized))) * 18)

    # [v1.2.4] explanation mitigation for v1 feature extraction
    # 입장 변화량(temporal_shift) 자체는 줄이지 않되, 기사 내부에 전환 사유/조건 변화/근거 갱신이
    # 충분히 제시되면 context_omission 차원의 자동 가중을 완화한다.
    total_omission_hits = sum(a["_omission_hits"] for a in normalized)
    total_context_expl_hits = sum(a["_context_expl_hits"] for a in normalized)
    avg_context_expl_hits = total_context_expl_hits / max(1, len(normalized))
    temporal_auto_factor = 0.5 if avg_context_expl_hits >= 1.5 else 1.0
    context_omission = clamp(
        (total_omission_hits * 12)
        - (total_context_expl_hits * 12)
        + (temporal_shift * 0.18 * temporal_auto_factor)
    )
    consensus_deviation = clamp(
        (sum(a["_frame_hits"] for a in normalized) * 6)
        + (sum(1 for a in normalized if a["_sentiment"] < -0.25) * 8)
        - (sum(a["_consensus_hits"] for a in normalized) * 2)
    )
    # [v1.2.2] evidence_quality: 증거 부재 − 증거 충실 (양수면 왜곡 = 점수 올라감)
    # 기사당 평균 cue 수 기준으로 정규화, 14배 가중 (다른 차원과 스케일 맞춤)
    n = max(1, len(normalized))
    evidence_lack_avg = sum(a["_evidence_lack_hits"] for a in normalized) / n
    evidence_present_avg = sum(a["_evidence_present_hits"] for a in normalized) / n
    evidence_quality = clamp(
        (evidence_lack_avg * 14)
        - (evidence_present_avg * 8)
    )

    article_rows = []
    for a in normalized:
        article_rows.append(
            {
                "date": a.get("date") or a.get("published_at") or "",
                "outlet": a.get("outlet", ""),
                "topic": a.get("topic", ""),
                "title": a.get("title", ""),
                "sentiment": round(a["_sentiment"], 3),
                "frame_hits": a["_frame_hits"],
                "omission_hits": a["_omission_hits"],
                "context_expl_hits": a["_context_expl_hits"],
                "evidence_lack_hits": a["_evidence_lack_hits"],
                "evidence_present_hits": a["_evidence_present_hits"],
            }
        )

    series = [
        {"date": r["date"], "title": r["title"], "sentiment": r["sentiment"]}
        for r in article_rows
    ]

    return {
        "features": {
            "temporal_shift": round(temporal_shift, 2),
            "frame_effect": round(frame_effect, 2),
            "context_omission": round(context_omission, 2),
            "consensus_deviation": round(consensus_deviation, 2),
            "evidence_quality": round(evidence_quality, 2),
        },
        "article_rows": article_rows,
        "series": series,
    }


def _clamp01(value: float) -> float:
    """0~1 범위 클램프. 활성도 계산용."""
    return max(0.0, min(1.0, float(value)))


# ═══════════════════════════════════════════════════════════════
# [v1.2.2 axiom 흡수] #1 sync_frames_with_rules
# ═══════════════════════════════════════════════════════════════

def get_active_frames(rules_data: Dict[str, Any]) -> set:
    """JSON 룰셋에 실제로 target_frame으로 등장하는 프레임 집합을 반환한다.

    OWL은 풍부한 어휘를 담고, JSON 룰셋은 그 부분집합만 다룬다.
    LLM이 OWL-only 프레임을 고르면 fired_rules가 비게 되므로,
    LLM 선택지를 이 active_frames로 필터링해야 한다.
    NeutralFact는 기준선 프레임으로 항상 포함.
    """
    rules = rules_data.get("rules", []) if rules_data else []
    active = {r.get("target_frame") for r in rules if r.get("target_frame")}
    active.add("NeutralFact")  # OWL-only지만 LLM의 'None' 대안으로 보존
    return active


def sync_frames_with_rules(owl_frames: List[str], rules_data: Dict[str, Any]) -> Tuple[List[str], int]:
    """OWL의 frame 리스트에서 JSON 룰셋에 없는 항목을 제거한다.

    Returns:
        (필터링된 프레임 리스트, 제거된 개수)
    """
    if not owl_frames or not rules_data:
        return list(owl_frames), 0
    active = get_active_frames(rules_data)
    before = len(owl_frames)
    filtered = [f for f in owl_frames if f in active]
    return filtered, before - len(filtered)


# ═══════════════════════════════════════════════════════════════
# [v1.2.2 axiom 흡수] #4 OWL 그래프 추론 4단계 (audit_logic 핵심)
# ═══════════════════════════════════════════════════════════════

OWL_NS = "http://www.context-sync.com/ontology/news-app#"


def check_value_frame_conflict(graph: Optional[rdflib.Graph], value: str, frame: str) -> bool:
    """OWL의 conflictsWith 관계로 ValueAnchor와 Frame이 구조적으로 충돌하는지 검사.

    시계열 정합성 검증의 핵심: 과거 텍스트가 옹호한 Value를
    현재 텍스트의 Frame이 위반하는지를 그래프에서 직접 추론.
    """
    if graph is None or not value or not frame or frame in ("None", ""):
        return False
    q = f"""
    PREFIX cs: <{OWL_NS}>
    ASK {{ cs:val_{value} cs:conflictsWith cs:frame_{frame} . }}
    """
    try:
        return bool(graph.query(q).askAnswer)
    except Exception:
        return False


def check_value_reinforces(graph: Optional[rdflib.Graph], value_a: str, value_b: str) -> bool:
    """OWL의 reinforces 관계로 두 ValueAnchor가 강화 관계인지 검사.

    True면 value_a → value_b 이동은 *강조점 이동*이지 구조 충돌이 아님.
    """
    if graph is None or not value_a or not value_b or value_a == value_b:
        return False
    q = f"""
    PREFIX cs: <{OWL_NS}>
    ASK {{
        {{ cs:val_{value_a} cs:reinforces cs:val_{value_b} . }}
        UNION
        {{ cs:val_{value_b} cs:reinforces cs:val_{value_a} . }}
    }}
    """
    try:
        return bool(graph.query(q).askAnswer)
    except Exception:
        return False


def get_calibrated_dimensions(graph: Optional[rdflib.Graph], value: str) -> List[str]:
    """OWL calibratesDimension 관계로 ValueAnchor가 보정하는 평가 차원 조회.

    *Value 위반이 어느 점수 차원으로 반영되어야 하는가*를 그래프에서 추론.
    """
    if graph is None or not value:
        return []
    q = f"""
    PREFIX cs: <{OWL_NS}>
    SELECT ?dim WHERE {{ cs:val_{value} cs:calibratesDimension ?dim . }}
    """
    try:
        dims = []
        for row in graph.query(q):
            iri = str(row.dim)
            if "#dim_" in iri:
                dims.append(iri.split("#dim_")[-1])
        return dims
    except Exception:
        return []


def compute_graph_penalty(
    base_abs: float,
    polarity_shift: Optional[float] = None,
    floor_factor: float = 1.0,
) -> float:
    """OWL 그래프 추론 페널티. cross-temporal은 polarity로 연속 보정, self-contradiction은 base 그대로."""
    if polarity_shift is None:
        return -round(float(base_abs), 1)
    shift_factor = _clamp01(float(polarity_shift) / 1.25)
    factor = max(_clamp01(floor_factor), shift_factor)
    return -round(float(base_abs) * factor, 1)


# ─────────────────────────────────────
# 휴리스틱 ExtractedSymbol (LLM 없이도 그래프 추론 가능하게)
# ─────────────────────────────────────

def heuristic_extract_symbol(
    text: str,
    graph: Optional[rdflib.Graph] = None,
) -> Dict[str, Any]:
    """텍스트에서 ValueAnchor / Frame / stance_polarity를 휴리스틱 추출.

    LLM 없이 *기존 키워드 cue들 + sentiment_score*로 OWL 어휘 매핑.
    축약 추출이지만 그래프 추론 4단계 시연용으로 충분.

    Returns:
        {
            "promoted_value": str (예: "StanceConsistency"),
            "detected_frame": str (예: "SilentPivot" 또는 "None"),
            "stance_polarity": float (-1.0 ~ +1.0),
            "topic": str (휴리스틱 None),
        }
    """
    if not text:
        return {"promoted_value": "", "detected_frame": "None", "stance_polarity": 0.0, "topic": ""}

    # stance_polarity: sentiment_score를 -1~+1로 변환
    sent = sentiment_score(text)  # -1 ~ +1 범위
    polarity = max(-1.0, min(1.0, sent))

    # promoted_value 휴리스틱:
    #   consensus cues (공정/책임/투명/합의) 많으면 PluralPublicReason
    #   context cues (근거/배경/맥락/설명) 많으면 ContextCompleteness
    #   evidence present cues 많으면 EvidenceTransparency
    #   omission cues 많으면 ResponsibilitySeparation (책임 흐림)
    #   기본: StanceConsistency (시계열 검증 대상 가치)
    consensus_n = count_hits(text, CONSENSUS_CUES)
    context_n = count_hits(text, CONTEXT_EXPLANATION_CUES)
    evidence_n = count_hits(text, EVIDENCE_PRESENT_CUES)
    omission_n = count_hits(text, OMISSION_CUES)
    frame_n = count_hits(text, FRAME_CUES)

    value_scores = {
        "PluralPublicReason": consensus_n,
        "ContextCompleteness": context_n,
        "EvidenceTransparency": evidence_n,
        "ResponsibilitySeparation": omission_n,
        "FrameAccountability": frame_n,
        "StanceConsistency": 1,  # 기본 폴백
    }
    promoted_value = max(value_scores, key=value_scores.get)

    # detected_frame 휴리스틱:
    #   frame cues + omission cues 강하면 SilentPivot (시계열 입장 흐림)
    #   frame cues만 강하면 PartisanBias
    #   둘 다 약하면 None
    if frame_n >= 3 and omission_n >= 2:
        detected_frame = "SilentPivot"
    elif frame_n >= 3:
        detected_frame = "PartisanBias"
    elif omission_n >= 3:
        detected_frame = "ResponsibilityShift"
    else:
        detected_frame = "None"

    return {
        "promoted_value": promoted_value,
        "detected_frame": detected_frame,
        "stance_polarity": round(polarity, 3),
        "topic": "",
    }


# ─────────────────────────────────────
# audit_temporal_pair: axiom audit_logic의 N-묶음 적응판
# ─────────────────────────────────────

def audit_temporal_pair(
    past_text: str,
    present_text: str,
    rules_data: Dict[str, Any],
    graph: Optional[rdflib.Graph] = None,
) -> Dict[str, Any]:
    """과거-현재 두 텍스트를 OWL 그래프 + 1024 룰셋으로 비교 검증.

    axiom audit_logic의 4단계를 휴리스틱 추출 기반으로 재구성:
      Stage 1: 휴리스틱 ExtractedSymbol (LLM 없이도 작동)
      Stage 2: polarity_shift 연속 감점 (compute_temporal_shift_penalty)
      Stage 3: OWL 그래프 추론 (conflictsWith + reinforces + calibratesDimension)
      Stage 4: 1024 룰 매칭 + 연속 페널티 + dimension_breakdown 누적

    Returns:
        audit_logic과 동일한 구조의 report dict
    """
    past = heuristic_extract_symbol(past_text, graph)
    present = heuristic_extract_symbol(present_text, graph)

    # JSON 룰셋에 없는 OWL-only 프레임은 graph audit의 fired_rules를 비우는 원인이 된다.
    # 따라서 런타임에서만 active frame으로 제한하고, OWL 파일 자체는 수정하지 않는다.
    frame_sync_notes: List[str] = []
    active_frames = get_active_frames(rules_data)
    for label, symbol in (("PAST", past), ("PRESENT", present)):
        frame = symbol.get("detected_frame")
        if frame and frame != "None" and frame not in active_frames:
            frame_sync_notes.append(
                f"[frame sync] {label} detected_frame '{frame}' is not in JSON active_frames → coerced to 'None'."
            )
            symbol["detected_frame"] = "None"

    report = {
        "validity_violation": False,
        "logic_conflict": False,
        "validity_score": 0,
        "logic_score": 0,
        "score": 100,
        "reasons": [],
        "fired_rules": [],
        "trace_events": [],
        "dimension_breakdown": {d: 0.0 for d in DIMENSIONS},
        "details": {"past": past, "present": present},
    }

    def _dim_note_to_trace(dim_note: str) -> str:
        """_distribute()가 반환한 설명 문자열을 trace table용 dimension 텍스트로 정리한다."""
        if not dim_note:
            return "—"
        if "calibrated:" in dim_note:
            return dim_note.split("calibrated:", 1)[1].strip()
        if "fallback:" in dim_note:
            return dim_note.split("fallback:", 1)[1].strip() + " (fallback)"
        return dim_note.strip(" →")

    def _append_trace(
        stage: str,
        trigger: str,
        owl_relation: str,
        calibrated_dimension: str,
        penalty: float,
        source: str,
        detail: str,
        rule_ids: Optional[List[str]] = None,
    ) -> None:
        """문자열 로그 파싱 없이 추론 단계를 구조화해 저장한다."""
        report["trace_events"].append({
            "stage": stage,
            "trigger": trigger,
            "owl_relation": owl_relation,
            "calibrated_dimension": calibrated_dimension,
            "penalty": round(float(penalty), 1) if isinstance(penalty, (int, float)) else penalty,
            "source": source,
            "detail": detail,
            "rule_ids": rule_ids or [],
        })

    # [v1.2.4] Explanation mitigation signal
    # Δ 자체는 그대로 유지하되, 설명 충실도는 Stage 3/4의 부가 penalty 완화에 사용한다.
    explanation_mitigation = compute_explanation_mitigation_signal(past_text, present_text)
    report["details"]["explanation_mitigation"] = explanation_mitigation
    if explanation_mitigation["mitigation_factor"] < 1.0:
        msg = (
            f"Explanation mitigation active ({explanation_mitigation['level']}): "
            f"signal={explanation_mitigation['explanation_signal']}, "
            f"factor={explanation_mitigation['mitigation_factor']} — {explanation_mitigation['reason']}"
        )
        report["reasons"].append(msg)
        _append_trace(
            "Stage 1.5: Explanation",
            f"context_explanation_signal={explanation_mitigation['explanation_signal']}",
            "explanation_mitigation",
            "context_omission / rule activation",
            0.0,
            "context_explanation_cues",
            "입장 변화량 Δ는 유지하되, 설명 충실도에 따라 Stage 3/4의 부가 페널티를 부분 완화",
        )

    def _mitigate_penalty(penalty: float, reason: str) -> float:
        """설명 충실도가 있으면 부가 penalty만 완화한다. Stage 2 Δ penalty에는 사용하지 않는다."""
        factor = float(explanation_mitigation.get("mitigation_factor", 1.0))
        if factor >= 1.0 or penalty == 0:
            return penalty
        mitigated = round(float(penalty) * factor, 1)
        report["reasons"].append(
            f"Explanation mitigation applied to {reason}: {penalty:.1f} → {mitigated:.1f} "
            f"(factor={factor})"
        )
        return mitigated

    report["reasons"].extend(frame_sync_notes)
    for note in frame_sync_notes:
        _append_trace(
            "Stage 0: Frame sync",
            note,
            "active_frames filter",
            "—",
            0.0,
            "runtime_frame_sync",
            "JSON 룰셋에 없는 OWL-only frame을 런타임에서 None으로 보정",
        )

    # ─── Stage 2: Polarity shift (연속 감점) ───
    polarity_shift = abs(past["stance_polarity"] - present["stance_polarity"])
    temporal_penalty = compute_temporal_shift_penalty(polarity_shift)
    if temporal_penalty != 0:
        if polarity_shift >= 1.0:
            report["logic_conflict"] = True
        report["logic_score"] += temporal_penalty
        report["dimension_breakdown"]["temporal_shift"] += temporal_penalty
        label = describe_temporal_shift(polarity_shift)
        report["reasons"].append(
            f"{label}: {past['stance_polarity']:+.2f} → {present['stance_polarity']:+.2f} "
            f"(Δ={polarity_shift:.2f}, penalty={temporal_penalty:.1f}) → temporal_shift"
        )
        _append_trace(
            "Stage 2: Polarity",
            f"polarity {past['stance_polarity']:+.2f} → {present['stance_polarity']:+.2f} (Δ={polarity_shift:.2f})",
            "—",
            "temporal_shift",
            temporal_penalty,
            "stance_polarity_shift",
            label,
        )

    # ─── Stage 3: OWL 그래프 추론 ───

    def _distribute(value: str, penalty: float) -> str:
        """ValueAnchor 위반 페널티를 OWL calibratesDimension 그래프로 분배.
        매핑 없으면 frame_effect로 폴백.
        """
        if not value or penalty == 0:
            return ""
        dims = get_calibrated_dimensions(graph, value)
        if not dims:
            if "frame_effect" in report["dimension_breakdown"]:
                report["dimension_breakdown"]["frame_effect"] += penalty
                return " → fallback: frame_effect"
            return ""
        per_dim = penalty / len(dims)
        for d in dims:
            if d in report["dimension_breakdown"]:
                report["dimension_breakdown"][d] += per_dim
        return f" → calibrated: {', '.join(dims)}"

    # 3a. 과거 가치 ↔ 현재 프레임 충돌 (cross-temporal)
    if past["promoted_value"] and present["detected_frame"] and present["detected_frame"] != "None":
        if check_value_frame_conflict(graph, past["promoted_value"], present["detected_frame"]):
            report["logic_conflict"] = True
            penalty = compute_graph_penalty(25.0, polarity_shift, floor_factor=0.65)
            penalty = _mitigate_penalty(penalty, "cross-temporal graph conflict")
            report["logic_score"] += penalty
            dim_note = _distribute(past["promoted_value"], penalty)
            report["reasons"].append(
                f"OWL graph conflict (cross-temporal): PAST value '{past['promoted_value']}' "
                f"↔ PRESENT frame '{present['detected_frame']}' "
                f"(penalty={penalty:.1f}){dim_note}"
            )
            _append_trace(
                "Stage 3a: Graph (cross-temporal)",
                f"PAST value '{past['promoted_value']}' ↔ PRESENT frame '{present['detected_frame']}'",
                "conflictsWith",
                _dim_note_to_trace(dim_note),
                penalty,
                "owl_graph_inference",
                "과거 옹호 가치를 현재 프레임이 구조적으로 위반",
            )

    # 자기 모순: 현재 가치 ↔ 현재 프레임 충돌
    if present["promoted_value"] and present["detected_frame"] and present["detected_frame"] != "None":
        if check_value_frame_conflict(graph, present["promoted_value"], present["detected_frame"]):
            penalty = compute_graph_penalty(15.0)
            report["logic_score"] += penalty
            dim_note = _distribute(present["promoted_value"], penalty)
            report["reasons"].append(
                f"OWL graph conflict (self-contradictory): PRESENT value '{present['promoted_value']}' "
                f"↔ PRESENT frame '{present['detected_frame']}' (penalty={penalty:.1f}){dim_note}"
            )
            _append_trace(
                "Stage 3a-self: Graph (self-contradiction)",
                f"PRESENT value '{present['promoted_value']}' ↔ PRESENT frame '{present['detected_frame']}'",
                "conflictsWith",
                _dim_note_to_trace(dim_note),
                penalty,
                "owl_graph_inference",
                "현재 텍스트가 옹호하는 가치를 자기 프레임이 위반",
            )

    # 3b. 가치 이동의 성격 판별
    if past["promoted_value"] != present["promoted_value"]:
        if check_value_reinforces(graph, past["promoted_value"], present["promoted_value"]):
            penalty = compute_graph_penalty(3.0, polarity_shift, floor_factor=0.50)
            penalty = _mitigate_penalty(penalty, "reinforced value emphasis shift")
            report["logic_score"] += penalty
            _distribute(past["promoted_value"], penalty / 2)
            _distribute(present["promoted_value"], penalty - penalty / 2)
            report["reasons"].append(
                f"Value emphasis shift (within reinforcement): "
                f"{past['promoted_value']} ↔ {present['promoted_value']} (penalty={penalty:.1f})"
            )
            _append_trace(
                "Stage 3b: Graph (value shift)",
                f"{past['promoted_value']} ↔ {present['promoted_value']}",
                "reinforces",
                "distributed via both ValueAnchor calibratesDimension",
                penalty,
                "owl_graph_inference",
                "강화 관계 안에서의 강조점 이동",
            )
        else:
            penalty = compute_graph_penalty(12.0, polarity_shift, floor_factor=0.50)
            penalty = _mitigate_penalty(penalty, "unreinforced value shift")
            report["logic_score"] += penalty
            dim_note = _distribute(past["promoted_value"], penalty)
            report["reasons"].append(
                f"Value shift (no OWL reinforces relation): "
                f"{past['promoted_value']} → {present['promoted_value']} "
                f"(penalty={penalty:.1f}){dim_note}"
            )
            _append_trace(
                "Stage 3b: Graph (value shift)",
                f"{past['promoted_value']} → {present['promoted_value']}",
                "(no reinforces)",
                _dim_note_to_trace(dim_note),
                penalty,
                "owl_graph_inference",
                "강화 관계 밖의 구조적 가치 이동",
            )

    # ─── Stage 4: 1024 룰 매칭 ───
    # frame + topic 기반 매칭 (휴리스틱 features는 0으로 두고 polarity로만 활성)
    pseudo_features = {d: 0.0 for d in DIMENSIONS}
    pseudo_features["temporal_shift"] = polarity_shift * 80  # 활성 임계 통과용 (>=10)

    matched = match_rules(
        rules_data.get("rules", []),
        pseudo_features,
        top_n=10,
        polarity_shift=polarity_shift,
        has_logic_conflict=report["logic_conflict"],
    )
    # 현재 프레임과 매칭되는 룰 우선 필터
    if present["detected_frame"] != "None":
        frame_matched = [r for r in matched if r.get("target_frame") == present["detected_frame"]]
        if frame_matched:
            matched = frame_matched

    explanation_sensitive_frames = {"SilentPivot", "RetroactiveReframing", "SelectiveMemory", "ContextOmission"}
    for r in matched:
        penalty = float(r.get("axiom_penalty", 0))
        if penalty == 0:
            continue
        r = dict(r)
        raw_penalty = penalty
        if (
            r.get("dimension") in {"temporal_shift", "context_omission"}
            or r.get("target_frame") in explanation_sensitive_frames
        ):
            penalty = _mitigate_penalty(penalty, f"rule {r.get('rule_id', '?')}")
            r["raw_axiom_penalty"] = raw_penalty
            r["axiom_penalty"] = penalty
            r["explanation_mitigation_factor"] = explanation_mitigation.get("mitigation_factor", 1.0)
        report["logic_score"] += penalty
        rule_dim = r.get("dimension")
        if rule_dim in report["dimension_breakdown"]:
            report["dimension_breakdown"][rule_dim] += penalty
        else:
            rule_value = r.get("value_anchor", "")
            if rule_value:
                _distribute(rule_value, penalty)
        report["fired_rules"].append(r)

    if report["fired_rules"]:
        rule_ids = [r["rule_id"] for r in report["fired_rules"]]
        report["reasons"].append(
            f"Fired {len(rule_ids)} rules: {', '.join(str(x) for x in rule_ids[:5])}"
            f"{'...' if len(rule_ids) > 5 else ''}"
        )

        # Stage 4 trace는 룰마다 한 줄씩 늘리면 시연 가독성이 떨어지므로
        # (target_frame, schema_id, dimension) 단위로 그룹 요약한다.
        grouped_rules: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for rule in report["fired_rules"]:
            key = (
                str(rule.get("target_frame", "?")),
                str(rule.get("schema_id", "?")),
                str(rule.get("dimension", "?")),
            )
            grouped_rules[key].append(rule)
        for (frame, schema, dim), grp in grouped_rules.items():
            total_penalty = sum(float(rule.get("axiom_penalty", 0) or 0) for rule in grp)
            grp_ids = [str(rule.get("rule_id", "?")) for rule in grp]
            detail = grp[0].get("frame_definition_ko", "") or grp[0].get("schema_description_ko", "")
            _append_trace(
                "Stage 4: Rule",
                f"frame={frame} / schema={schema} ({len(grp)}개 룰)",
                "dimension (JSON 우선)",
                dim,
                total_penalty,
                f"rules:[{', '.join(grp_ids[:5])}{'...' if len(grp_ids) > 5 else ''}]",
                detail[:120] if detail else "1024 JSON 룰셋 발화",
                rule_ids=grp_ids,
            )

    # dimension_breakdown 정수화
    for d in report["dimension_breakdown"]:
        report["dimension_breakdown"][d] = round(report["dimension_breakdown"][d], 1)

    # ─── 최종 점수: compute_weighted_distortion ───
    weights = extract_formula_weights(rules_data.get("aggregation_formula"))
    weighted_distortion = compute_weighted_distortion(
        report["dimension_breakdown"], weights
    )
    report["weighted_distortion"] = weighted_distortion
    report["dimension_weights"] = weights

    # 판정
    thresholds = rules_data.get("verdict_thresholds", {})
    aligned_max = float(thresholds.get("aligned_max", 34))
    caution_max = float(thresholds.get("caution_max", 64))
    if weighted_distortion <= aligned_max:
        report["anchor_verdict"] = "기준 부합"
    elif weighted_distortion <= caution_max:
        report["anchor_verdict"] = "주의 필요"
    else:
        report["anchor_verdict"] = "기준 이탈"

    # 안전망 경고
    dim_total = sum(abs(v) for v in report["dimension_breakdown"].values())
    logic_abs = abs(report["logic_score"])
    if logic_abs > dim_total + 0.5:
        report["reasons"].append(
            f"[warning] logic_score 미반영 {logic_abs - dim_total:.1f}점 발생. "
            "dimension_breakdown 연결 확인 필요."
        )

    report["score"] = max(0, round(100 - weighted_distortion + report["validity_score"], 1))

    if not report["reasons"]:
        report["reasons"].append("No significant temporal coherence violations detected.")

    return report


def audit_article_group(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    graph: Optional[rdflib.Graph] = None,
) -> Optional[Dict[str, Any]]:
    """N개 기사 묶음에서 첫 기사 ↔ 마지막 기사를 audit_temporal_pair로 비교.

    같은 (outlet, topic) 그룹 안에서만 수행. 그룹 분리는 호출자가 담당.
    기사 2개 미만이면 None 반환.
    """
    if len(articles) < 2:
        return None
    # 시간순 정렬 (date 키 기준)
    sorted_articles = sorted(articles, key=lambda a: parse_date(a.get("date") or a.get("published_at")))
    past_text = text_of(sorted_articles[0])
    present_text = text_of(sorted_articles[-1])
    report = audit_temporal_pair(past_text, present_text, rules_data, graph=graph)
    report["past_article"] = {
        "title": sorted_articles[0].get("title", ""),
        "date": sorted_articles[0].get("date", ""),
        "outlet": sorted_articles[0].get("outlet", ""),
    }
    report["present_article"] = {
        "title": sorted_articles[-1].get("title", ""),
        "date": sorted_articles[-1].get("date", ""),
        "outlet": sorted_articles[-1].get("outlet", ""),
    }
    report["article_count"] = len(sorted_articles)
    return report


# ═══════════════════════════════════════════════════════════════
# [v1.2.2 axiom 흡수] #2 compute_weighted_distortion (per_dim_cap=45 정규화)
# ═══════════════════════════════════════════════════════════════

def compute_weighted_distortion(
    dimension_breakdown: Dict[str, float],
    weights: Dict[str, float],
    per_dim_cap: float = PER_DIM_CAP,
) -> float:
    """차원별 페널티를 dimension_weights에 따라 최종 왜곡도로 환산한다.

    JSON normalization_note_ko의 권고를 반영:
      1. 각 차원의 누적 페널티 절댓값을 0~100 스케일로 정규화
         (per_dim_cap=45 상한 클리핑 — 한 차원에 45점 이상 누적 시 100% 왜곡)
      2. 정규화된 차원 점수 × 가중치 → 가중 합산
      3. 최종 0~100 클리핑

    *룰 발화 수에 따른 과대평가 방지*가 핵심. 한 차원 폭주가 다른 차원을 지배 못함.
    """
    if not weights:
        return 0.0

    weighted_sum = 0.0
    for dim, weight in weights.items():
        raw_penalty = abs(float(dimension_breakdown.get(dim, 0.0)))
        normalized = min(100.0, (raw_penalty / per_dim_cap) * 100.0)
        weighted_sum += normalized * float(weight)

    return min(100.0, round(weighted_sum, 1))


# ═══════════════════════════════════════════════════════════════
# [v1.2.2 axiom 흡수] #3 연속 감점 함수
# ═══════════════════════════════════════════════════════════════

def compute_temporal_shift_penalty(
    polarity_shift: float,
    min_shift: float = 0.25,
    full_shift: float = 1.25,
    min_penalty: float = 5.0,
    max_penalty: float = 30.0,
) -> float:
    """입장 극성 변화량을 temporal_shift 연속 감점으로 변환한다.

    계단형(-5/-15/-30) 대신 0.25~1.25를 선형 보간:
      Δ=0.25→-5.0, Δ=0.50→-11.2, Δ=0.75→-17.5,
      Δ=1.00→-23.8, Δ≥1.25→-30.0 (cap)
    임계값 직후 점프(Δ=0.99→1.00에서 -15→-30) 해소.
    """
    delta = float(polarity_shift)
    if delta < min_shift:
        return 0.0
    span = max(full_shift - min_shift, 1e-9)
    ratio = _clamp01((min(delta, full_shift) - min_shift) / span)
    penalty = min_penalty + ratio * (max_penalty - min_penalty)
    return -round(min(max_penalty, penalty), 1)


def describe_temporal_shift(polarity_shift: float) -> str:
    """리포트 reason에 사용할 시계열 변화 강도 라벨."""
    if polarity_shift >= 1.25:
        return "Full-scale stance reversal"
    if polarity_shift >= 1.0:
        return "Major stance reversal"
    if polarity_shift >= 0.5:
        return "Significant stance shift"
    if polarity_shift >= 0.25:
        return "Moderate stance shift"
    return "No significant shift"


def compute_rule_penalty(
    rule: Dict[str, Any],
    polarity_shift: float = 0.0,
    has_logic_conflict: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """JSON 룰의 연속 가중치를 실제 감점에 반영한다.

    기존 severity_band 고정 감점(-3/-6/-10/-15) 대신:
      penalty = -severity_base × rule_strength × activation

      rule_strength = 0.5×risk_w + 0.3×distortion_w + 0.2×(1−consensus_w)
        — 룰셋의 4개 연속 가중치가 *실제 감점에 직접 반영*됨
      activation = max(polarity_shift/1.25, 0.35 if has_logic_conflict else 0)
        — 시계열 변화 없으면 룰 발화도 약화

    Returns:
        (penalty, scoring_meta) — scoring_meta는 fired_rules에 노출됨
    """
    severity = str(rule.get("severity_band", "low")).lower()
    severity_base = SEVERITY_BASE.get(severity, 4.0)

    risk_w = _clamp01(float(rule.get("risk_weight", 0.5)))
    distortion_w = _clamp01(float(rule.get("distortion_weight", 0.5)))
    consensus_w = _clamp01(float(rule.get("consensus_weight", 0.5)))

    rule_strength = _clamp01(
        0.50 * risk_w
        + 0.30 * distortion_w
        + 0.20 * (1.0 - consensus_w)
    )

    shift_activation = _clamp01(float(polarity_shift) / 1.25)
    activation = max(shift_activation, 0.35 if has_logic_conflict else 0.0)
    intensity = _clamp01(activation * risk_w * distortion_w)

    scoring_meta = {
        "severity_base": severity_base,
        "risk_weight": risk_w,
        "distortion_weight": distortion_w,
        "consensus_weight": consensus_w,
        "rule_strength": round(rule_strength, 3),
        "activation": round(activation, 3),
        "intensity": round(intensity, 3),
    }

    if intensity < 0.05 and not has_logic_conflict:
        return 0.0, scoring_meta

    eff_activation = max(activation, 0.35 if has_logic_conflict else 0.0)
    penalty = -round(severity_base * rule_strength * eff_activation, 1)
    return penalty, scoring_meta


# ═══════════════════════════════════════════════════════════════
# 기존 score_features (v1 호환용) — features 기반 점수 산출
# ═══════════════════════════════════════════════════════════════

def score_features(features: Dict[str, float], weights: Dict[str, float]) -> float:
    """v1 호환: features (0~100) × weights 단순 가중합 (clamp 0~100).

    이는 *사용자가 직접 입력한 관측치*를 점수화할 때 사용.
    axiom의 compute_weighted_distortion은 *룰 발화로 누적된 페널티*를
    점수화할 때 사용 (별도 경로).
    """
    total = sum(float(features.get(dim, 0)) * float(weights.get(dim, 0)) for dim in DIMENSIONS)
    return round(clamp(total), 2)


def match_rules(
    rules: List[Dict[str, Any]],
    features: Dict[str, float],
    top_n: int = 12,
    profile: Optional[str] = None,
    context: Optional[str] = None,
    polarity_shift: float = 0.0,
    has_logic_conflict: bool = False,
) -> List[Dict[str, Any]]:
    """룰 매칭 + axiom 연속 감점 + fired_rules 메타 7필드 통합.

    [v1.2.2 axiom 흡수]
      - 기존 features 기반 활성화는 보존 (관측치 임계값 필터)
      - 활성화된 룰에 대해 axiom의 compute_rule_penalty로 연속 페널티 산출
      - fired_rules 메타 7필드 추가:
        frame_definition_ko, schema_description_ko, expected_evidence_ko,
        score_hint, severity_base, rule_strength, intensity
    """
    matched = []
    for r in rules:
        dim = r.get("dimension")
        if dim not in DIMENSIONS:
            continue
        if profile and r.get("profile") != profile:
            continue
        if context and r.get("context") != context:
            continue
        feature_score = float(features.get(dim, 0))
        if feature_score < 10:
            continue

        # 기존 features 기반 활성화 (관측치 임계값)
        severity = SEVERITY_MULTIPLIER.get(str(r.get("severity_band", "medium")).lower(), 0.8)
        feature_activation = feature_score * float(r.get("risk_weight", 0.75)) * float(r.get("distortion_weight", 0.75)) * severity / 100
        if feature_activation <= 0.06:
            continue

        # axiom 연속 감점 + 메타 산출
        axiom_penalty, scoring_meta = compute_rule_penalty(
            r,
            polarity_shift=polarity_shift,
            has_logic_conflict=has_logic_conflict,
        )

        matched.append(
            {
                "rule_id": r.get("rule_id"),
                "schema_type": r.get("schema_type"),
                "schema_id": r.get("schema_id"),
                "dimension": dim,
                "target_frame": r.get("target_frame"),
                "value_anchor": r.get("value_anchor"),
                "context": r.get("context"),
                "profile": r.get("profile"),
                "severity_band": r.get("severity_band"),
                # 기존 features 기반 활성도 (관측치 임계값)
                "activation": round(feature_activation, 3),
                "risk_weight": r.get("risk_weight"),
                "distortion_weight": r.get("distortion_weight"),
                "consensus_weight": r.get("consensus_weight"),
                "positive_cues": r.get("positive_cues", []),
                "negative_indicators": r.get("negative_indicators", []),
                "llm_instruction_ko": r.get("llm_instruction_ko", ""),
                # [v1.2.2 흡수] axiom 연속 감점 메타 (#3)
                "axiom_penalty": axiom_penalty,
                "severity_base": scoring_meta["severity_base"],
                "rule_strength": scoring_meta["rule_strength"],
                "axiom_activation": scoring_meta["activation"],  # polarity 기반 활성도
                "intensity": scoring_meta["intensity"],
                # [v1.2.2 흡수] fired_rules 메타 4필드 (#5)
                "frame_definition_ko": r.get("frame_definition_ko", ""),
                "schema_description_ko": r.get("schema_description_ko", ""),
                "expected_evidence_ko": r.get("expected_evidence_ko", ""),
                "score_hint": r.get("score_hint", {}),
            }
        )
    # 후보 룰과 실제 axiom fired rule을 구분한다.
    # axiom_penalty가 0인 룰은 후보(candidate)일 뿐, 실제 페널티를 낳은 fired rule은 아니다.
    for item in matched:
        item["is_fired"] = abs(float(item.get("axiom_penalty", 0) or 0)) > 0
    matched.sort(
        key=lambda x: (
            1 if x.get("is_fired") else 0,
            abs(float(x.get("axiom_penalty", 0) or 0)),
            float(x.get("intensity", 0) or 0),
            float(x.get("activation", 0) or 0),
        ),
        reverse=True,
    )
    return matched[:top_n]


def compute_rule_dimension_breakdown(
    matched_rules: List[Dict[str, Any]],
) -> Dict[str, float]:
    """fired rules의 axiom_penalty를 dimension별로 누적한다.

    [v1.2.2 axiom 흡수]
      각 룰의 dimension 필드에 axiom_penalty 누적 → dimension_breakdown 생성.
      이 결과를 compute_weighted_distortion에 넣으면 정규화된 5차원 가중 점수 산출.
    """
    breakdown = {d: 0.0 for d in DIMENSIONS}
    for r in matched_rules:
        penalty = float(r.get("axiom_penalty", 0) or 0)
        if penalty == 0:
            continue
        dim = r.get("dimension")
        if dim in breakdown:
            breakdown[dim] += penalty
    return {d: round(v, 1) for d, v in breakdown.items()}


def summarize_rules(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rules),
        "by_dimension": dict(Counter(r.get("dimension", "unknown") for r in rules)),
        "by_schema_type": dict(Counter(r.get("schema_type", "unknown") for r in rules)),
        "by_frame_top10": dict(Counter(r.get("target_frame", "unknown") for r in rules).most_common(10)),
        "profiles": sorted(set(r.get("profile", "") for r in rules if r.get("profile"))),
        "contexts": sorted(set(r.get("context", "") for r in rules if r.get("context"))),
    }


def build_owl_mermaid(g: rdflib.Graph) -> str:
    """OWL 그래프에서 Layer → Value → Frame → RuleSchema 계층을 Mermaid flowchart로 생성한다.

    추가 패키지 없이 st.markdown으로 렌더링할 수 있다.
    """
    NS = "http://www.context-sync.com/ontology/news-app#"

    def short(uri: str) -> str:
        return str(uri).replace(NS, "").replace("http://www.w3.org/2002/07/owl#", "owl:")

    # ── SPARQL로 주요 엔티티와 관계 수집 ───────────────────────────
    layers, values, frames, schemas = [], [], [], []
    edges = []

    # AxiomLayer 인스턴스
    for row in g.query(f"SELECT ?s ?label WHERE {{ ?s a <{NS}AxiomLayer> . OPTIONAL {{ ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label }} }}"):
        sid = short(row.s)
        label = str(row.label or sid).strip()
        layers.append((sid, label))

    # ValueAnchor 인스턴스 + belongsToLayer
    for row in g.query(f"SELECT ?s ?label ?layer WHERE {{ ?s a <{NS}ValueAnchor> . OPTIONAL {{ ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label }} . OPTIONAL {{ ?s <{NS}belongsToLayer> ?layer }} }}"):
        sid = short(row.s)
        label = str(row.label or sid).strip()
        values.append((sid, label))
        if row.layer:
            edges.append((short(row.layer), sid))

    # Frame 인스턴스
    for row in g.query(f"SELECT ?s ?label WHERE {{ ?s a <{NS}Frame> . OPTIONAL {{ ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label }} }}"):
        sid = short(row.s)
        label = str(row.label or sid).strip()
        frames.append((sid, label))

    # RuleSchema 인스턴스 + constrainsFrame + anchoredByValue + belongsToLayer
    q = f"""
    SELECT ?s ?label ?frame ?value ?layer WHERE {{
        ?s a ?type .
        FILTER(?type IN (<{NS}TemporalRuleSchema>, <{NS}StructuralRuleSchema>, <{NS}MetacognitiveRuleSchema>))
        OPTIONAL {{ ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label }}
        OPTIONAL {{ ?s <{NS}constrainsFrame> ?frame }}
        OPTIONAL {{ ?s <{NS}anchoredByValue> ?value }}
        OPTIONAL {{ ?s <{NS}belongsToLayer> ?layer }}
    }}
    """
    for row in g.query(q):
        sid = short(row.s)
        label = str(row.label or sid).strip()
        schemas.append((sid, label))
        if row.frame:
            edges.append((sid, short(row.frame)))
        if row.value:
            edges.append((short(row.value), sid))
        if row.layer:
            edges.append((short(row.layer), sid))

    # ── Mermaid 코드 생성 ─────────────────────────────────────────
    lines = ["flowchart TD"]

    # 스타일 클래스
    lines.append('    classDef layerStyle fill:#4a90d9,color:#fff,stroke:#2c5f8a')
    lines.append('    classDef valueStyle fill:#50c878,color:#fff,stroke:#2d8a4e')
    lines.append('    classDef frameStyle fill:#f5a623,color:#fff,stroke:#c17d12')
    lines.append('    classDef schemaStyle fill:#9b59b6,color:#fff,stroke:#6c3483')

    def safe_id(s: str) -> str:
        return s.replace("-", "_").replace(".", "_")

    # 노드 선언
    for sid, label in layers:
        lines.append(f'    {safe_id(sid)}["{label}"]:::layerStyle')
    for sid, label in values:
        lines.append(f'    {safe_id(sid)}["{label}"]:::valueStyle')
    for sid, label in frames[:15]:  # 프레임 수 제한 (시각성)
        lines.append(f'    {safe_id(sid)}["{label}"]:::frameStyle')
    for sid, label in schemas[:12]:  # 스키마 수 제한
        lines.append(f'    {safe_id(sid)}["{label}"]:::schemaStyle')

    # 에지
    visible_nodes = {safe_id(s) for s, _ in layers + values + frames[:15] + schemas[:12]}
    for src, dst in edges:
        s, d = safe_id(src), safe_id(dst)
        if s in visible_nodes and d in visible_nodes:
            lines.append(f'    {s} --> {d}')

    # 범례
    lines.append('    subgraph legend["범례"]')
    lines.append('        L["AxiomLayer"]:::layerStyle')
    lines.append('        V["ValueAnchor"]:::valueStyle')
    lines.append('        F["Frame"]:::frameStyle')
    lines.append('        S["RuleSchema"]:::schemaStyle')
    lines.append('    end')

    return "\n".join(lines)


def build_verdict_reason(matched: List[Dict[str, Any]]) -> str:
    """트리거된 상위 규칙에서 판정 핵심 근거 1~2문장을 생성한다.

    규칙에 descriptionKo가 있으면 그대로 사용하고, 없으면 target_frame + dimension으로
    자동 생성한다.
    """
    reasons: List[str] = []
    for r in matched[:2]:
        desc = str(r.get("descriptionKo") or "").strip()
        if desc:
            reasons.append(desc)
        else:
            frame = r.get("target_frame") or ""
            dim_map = {
                "temporal_shift": "시계열 논조 이동",
                "frame_effect": "프레임 효과",
                "context_omission": "맥락 누락",
                "consensus_deviation": "기준 이탈",
                "evidence_quality": "증거 품질",
            }
            dim_label = dim_map.get(str(r.get("dimension", "")), str(r.get("dimension", "")))
            severity = r.get("severity_band", "")
            parts = [p for p in [frame, dim_label, severity] if p]
            if parts:
                reasons.append(" · ".join(parts) + " 규칙 발화")
    return "  |  ".join(reasons) if reasons else ""


def build_reasoning_trace(audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """audit_temporal_pair 결과를 *ontology-calibrated reasoning의 논리 사슬*로 추출한다.

    각 행은 *하나의 추론 단계*를 표현:
      stage | trigger | OWL relation | calibrated dimension | penalty | source

    audit["reasons"]에 자연어로 있는 추론 단계들을 *구조화된 표*로 풀어낸다.
    audit["fired_rules"]의 룰별 페널티도 trace의 일부로 포함.

    포트폴리오/시연 가치:
      *"이 판정이 어떻게 도출되었는가"*를 한 표로 추적 가능하게 만듦.
      ontology가 *단순 어휘 사전*이 아니라 *점수 라우팅의 메타 기준층*임을 가시화.

    Returns:
        리스트의 각 dict는:
        {
            "stage": "Stage 2 / 3a / 3b / 4",
            "trigger": 어떤 입력이 단계를 발화시켰는가,
            "owl_relation": OWL 그래프 관계 (conflictsWith / reinforces / calibratesDimension),
            "calibrated_dimension": 페널티가 라우팅된 차원,
            "penalty": 부여된 페널티,
            "source": "graph_inference" 또는 "rule:R0123",
            "detail": 한국어 설명
        }
    """
    if not audit:
        return []

    if audit.get("trace_events"):
        # v4.1+: audit_temporal_pair()가 추론 이벤트를 직접 구조화해 저장한다.
        # 문자열 reasons 파싱은 구버전 audit 결과를 위한 fallback으로만 사용한다.
        return list(audit.get("trace_events", []))

    trace: List[Dict[str, Any]] = []
    past = audit.get("details", {}).get("past", {})
    present = audit.get("details", {}).get("present", {})
    breakdown = audit.get("dimension_breakdown", {}) or {}
    reasons = audit.get("reasons", []) or []
    fired_rules = audit.get("fired_rules", []) or []

    # ─── Stage 1: Validity (Red Card) ───
    if audit.get("validity_violation"):
        trace.append({
            "stage": "Stage 1: Validity",
            "trigger": "is_valid_discourse=False",
            "owl_relation": "—",
            "calibrated_dimension": "validity_score",
            "penalty": audit.get("validity_score", 0),
            "source": "red_card",
            "detail": "사실/윤리 위배 — 시계열 분석 중단",
        })

    # ─── Stage 2: Polarity Shift (시계열 입장 이동) ───
    past_polarity = past.get("stance_polarity", 0)
    present_polarity = present.get("stance_polarity", 0)
    polarity_shift = abs(past_polarity - present_polarity)
    if polarity_shift >= 0.25:
        temporal_penalty = compute_temporal_shift_penalty(polarity_shift)
        trace.append({
            "stage": "Stage 2: Polarity",
            "trigger": f"polarity {past_polarity:+.2f} → {present_polarity:+.2f} (Δ={polarity_shift:.2f})",
            "owl_relation": "—",
            "calibrated_dimension": "temporal_shift",
            "penalty": temporal_penalty,
            "source": "stance_polarity_shift",
            "detail": describe_temporal_shift(polarity_shift),
        })

    # ─── Stage 3a: Cross-temporal Value↔Frame conflict ───
    past_value = past.get("promoted_value", "")
    present_frame = present.get("detected_frame", "")
    for r in reasons:
        if "cross-temporal" in r and past_value and present_frame and present_frame != "None":
            # reasons 문자열에서 penalty 추출
            import re as _re
            m = _re.search(r"penalty=(-?[\d.]+)", r)
            penalty = float(m.group(1)) if m else 0
            dims_m = _re.search(r"calibrated dimensions: ([^)]+)", r)
            calibrated = dims_m.group(1).strip() if dims_m else "frame_effect (fallback)"
            trace.append({
                "stage": "Stage 3a: Graph (cross-temporal)",
                "trigger": f"PAST value '{past_value}' ↔ PRESENT frame '{present_frame}'",
                "owl_relation": "conflictsWith",
                "calibrated_dimension": calibrated,
                "penalty": penalty,
                "source": "owl_graph_inference",
                "detail": "과거 옹호 가치를 현재 프레임이 구조적으로 위반",
            })
            break

    # ─── Stage 3a-self: Self-contradictory ───
    present_value = present.get("promoted_value", "")
    for r in reasons:
        if "self-contradictory" in r:
            import re as _re
            m = _re.search(r"penalty=(-?[\d.]+)", r)
            penalty = float(m.group(1)) if m else 0
            dims_m = _re.search(r"calibrated dimensions: ([^)]+)", r)
            calibrated = dims_m.group(1).strip() if dims_m else "frame_effect (fallback)"
            trace.append({
                "stage": "Stage 3a-self: Graph (self-contradiction)",
                "trigger": f"PRESENT value '{present_value}' ↔ PRESENT frame '{present_frame}'",
                "owl_relation": "conflictsWith",
                "calibrated_dimension": calibrated,
                "penalty": penalty,
                "source": "owl_graph_inference",
                "detail": "현재 텍스트가 옹호하는 가치를 자기 프레임이 위반 (자기 모순)",
            })
            break

    # ─── Stage 3b: Value shift (reinforces or no relation) ───
    for r in reasons:
        if "Value emphasis shift" in r:
            import re as _re
            m = _re.search(r"penalty=(-?[\d.]+)", r)
            penalty = float(m.group(1)) if m else 0
            trace.append({
                "stage": "Stage 3b: Graph (value shift)",
                "trigger": f"{past_value} ↔ {present_value}",
                "owl_relation": "reinforces",
                "calibrated_dimension": "분배 (양쪽 calibratesDimension)",
                "penalty": penalty,
                "source": "owl_graph_inference",
                "detail": "강화 관계 안에서의 강조점 이동 (가벼운 페널티)",
            })
            break
        elif "Value shift" in r and "no OWL reinforces" in r:
            import re as _re
            m = _re.search(r"penalty=(-?[\d.]+)", r)
            penalty = float(m.group(1)) if m else 0
            dims_m = _re.search(r"calibrated dimensions: ([^)]+)", r)
            calibrated = dims_m.group(1).strip() if dims_m else "frame_effect (fallback)"
            trace.append({
                "stage": "Stage 3b: Graph (value shift)",
                "trigger": f"{past_value} → {present_value}",
                "owl_relation": "(no reinforces)",
                "calibrated_dimension": calibrated,
                "penalty": penalty,
                "source": "owl_graph_inference",
                "detail": "강화 관계 밖의 구조적 가치 이동 (검토 필요)",
            })
            break

    # ─── Stage 4: 1024 Rule Matching ───
    # 같은 frame의 룰들이 반복되면 가독성이 떨어지므로 frame별로 그룹 요약
    from collections import defaultdict as _dd
    rules_by_frame: Dict[str, List[Dict[str, Any]]] = _dd(list)
    for rule in fired_rules:
        if float(rule.get("axiom_penalty", 0)) == 0:
            continue
        key = (rule.get("target_frame", "?"), rule.get("schema_id", "?"))
        rules_by_frame[key].append(rule)

    for (frame, schema), grp in rules_by_frame.items():
        total_penalty = sum(float(r.get("axiom_penalty", 0)) for r in grp)
        rule_ids = [str(r.get("rule_id", "?")) for r in grp[:5]]
        ids_str = ", ".join(rule_ids) + ("..." if len(grp) > 5 else "")
        dim = grp[0].get("dimension", "?")
        detail = grp[0].get("frame_definition_ko", "") or grp[0].get("schema_description_ko", "")
        trace.append({
            "stage": "Stage 4: Rule",
            "trigger": f"frame={frame} / schema={schema} ({len(grp)}개 룰)",
            "owl_relation": "dimension (JSON 우선)",
            "calibrated_dimension": dim,
            "penalty": round(total_penalty, 1),
            "source": f"rules:[{ids_str}]",
            "detail": detail[:100] if detail else "",
        })

    return trace


def build_graph_audits(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    graph: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """같은 (outlet, topic) 그룹별 first↔last graph audit 결과를 만든다.

    반환값은 weighted_distortion이 큰 순서로 정렬된다. 즉 첫 번째 항목이
    메인 axiom_distortion으로 승격될 primary graph audit이다.
    """
    if not articles or len(articles) < 2:
        return []

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for art in articles:
        key = (str(art.get("outlet", "?")), str(art.get("topic", "?")))
        groups[key].append(art)

    audits: List[Dict[str, Any]] = []
    for (outlet, topic), group in groups.items():
        if len(group) < 2:
            continue
        audit = audit_article_group(group, rules_data, graph=graph)
        if not audit:
            continue
        audit["_group_key"] = f"{outlet} / {topic}"
        audits.append(audit)

    audits.sort(key=lambda a: float(a.get("weighted_distortion", 0) or 0), reverse=True)
    return audits


def analyze_pipeline(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    top_n: int = 12,
    profile: Optional[str] = None,
    context: Optional[str] = None,
    manual_features: Optional[Dict[str, float]] = None,
    owl_graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """통합 분석 파이프라인.

    v1 feature score는 빠른 보조지표로 유지하고, graph audit이 가능한 경우에는
    audit_article_group()의 weighted_distortion을 메인 axiom_distortion으로 승격한다.
    graph audit이 불가능한 경우에만 기존 feature 기반 axiom-lite 점수로 fallback한다.
    """
    owl_cues: Optional[List[str]] = extract_owl_frame_cues(owl_graph) if owl_graph is not None else None
    extraction = analyze_articles(articles, owl_frame_cues=owl_cues)

    # LLM/sLLM feature가 들어와도 temporal_shift는 기사 묶음의 시계열 계산값으로 보정한다.
    # 단, 기사가 없는 수동 시뮬레이터에서는 사용자 입력을 그대로 유지한다.
    if manual_features is not None:
        features = manual_features.copy()
        if articles:
            features["temporal_shift"] = extraction["features"].get("temporal_shift", 0.0)
    else:
        features = extraction["features"]

    weights = extract_formula_weights(rules_data.get("aggregation_formula"))

    # ─────────────────────────────────────
    # v1 보조 점수: features × weights 단순 왜곡도
    # ─────────────────────────────────────
    v1_distortion = score_features(features, weights)
    v1_verdict = determine_verdict(v1_distortion, rules_data.get("verdict_thresholds", {}))

    # ─────────────────────────────────────
    # axiom-lite fallback: graph audit이 불가능할 때만 메인으로 사용
    # ─────────────────────────────────────
    polarity_shift = float(features.get("temporal_shift", 0)) / 100.0 * 1.25
    candidate_rules = match_rules(
        rules_data.get("rules", []),
        features,
        top_n=top_n,
        profile=profile,
        context=context,
        polarity_shift=polarity_shift,
        has_logic_conflict=(polarity_shift >= 1.0),
    )
    lite_breakdown = compute_rule_dimension_breakdown(candidate_rules)
    temporal_penalty = compute_temporal_shift_penalty(polarity_shift)
    lite_breakdown["temporal_shift"] = round(
        lite_breakdown.get("temporal_shift", 0) + temporal_penalty, 1
    )
    axiom_lite_distortion = compute_weighted_distortion(lite_breakdown, weights)
    axiom_lite_score = max(0, round(100 - axiom_lite_distortion, 1))
    axiom_lite_verdict = determine_verdict_axiom(
        axiom_lite_distortion, rules_data.get("verdict_thresholds", {})
    )

    # ─────────────────────────────────────
    # graph audit: 진짜 axiom 점수의 우선 경로
    # ─────────────────────────────────────
    graph_audits = build_graph_audits(articles, rules_data, graph=owl_graph)
    primary_graph_audit = graph_audits[0] if graph_audits else None

    if primary_graph_audit is not None:
        axiom_source = "graph_audit"
        axiom_distortion = float(primary_graph_audit.get("weighted_distortion", 0) or 0)
        coherence_score = float(primary_graph_audit.get("score", max(0, 100 - axiom_distortion)) or 0)
        axiom_verdict = primary_graph_audit.get("anchor_verdict", determine_verdict_axiom(axiom_distortion, rules_data.get("verdict_thresholds", {})))
        dimension_breakdown = primary_graph_audit.get("dimension_breakdown", {d: 0.0 for d in DIMENSIONS})
        axiom_fired_rules = primary_graph_audit.get("fired_rules", [])
        graph_past = primary_graph_audit.get("details", {}).get("past", {})
        graph_present = primary_graph_audit.get("details", {}).get("present", {})
        graph_polarity_shift = abs(float(graph_past.get("stance_polarity", 0) or 0) - float(graph_present.get("stance_polarity", 0) or 0))
        graph_temporal_penalty = compute_temporal_shift_penalty(graph_polarity_shift)
        polarity_shift_for_display = graph_polarity_shift
        temporal_penalty_for_display = graph_temporal_penalty
        primary_group_key = primary_graph_audit.get("_group_key", "-")
    else:
        axiom_source = "feature_lite_fallback"
        axiom_distortion = axiom_lite_distortion
        coherence_score = axiom_lite_score
        axiom_verdict = axiom_lite_verdict
        dimension_breakdown = lite_breakdown
        axiom_fired_rules = [r for r in candidate_rules if r.get("is_fired")]
        polarity_shift_for_display = polarity_shift
        temporal_penalty_for_display = temporal_penalty
        primary_group_key = "-"

    # 판정 근거는 실제 axiom fired_rules를 우선 사용하고, 없으면 후보 룰로 fallback한다.
    verdict_reason = build_verdict_reason(axiom_fired_rules or candidate_rules)

    return {
        "features": features,
        "weights": weights,

        # v1 compatibility + 명시적 alias
        "score": v1_distortion,
        "verdict": v1_verdict,
        "v1_distortion_score": v1_distortion,
        "v1_verdict": v1_verdict,

        # 후보 룰과 실제 axiom 발화 룰 분리
        "matched_rules": candidate_rules,
        "candidate_rules": candidate_rules,
        "axiom_fired_rules": axiom_fired_rules,
        "verdict_reason": verdict_reason,

        "article_rows": extraction["article_rows"],
        "series": extraction["series"],
        "rule_summary": summarize_rules(rules_data.get("rules", [])),

        # axiom-lite 보조값
        "axiom_lite_distortion": axiom_lite_distortion,
        "axiom_lite_score": axiom_lite_score,
        "axiom_lite_verdict": axiom_lite_verdict,
        "axiom_lite_dimension_breakdown": lite_breakdown,

        # 최종 axiom 대표값: graph audit 우선, lite fallback
        "axiom_source": axiom_source,
        "primary_group_key": primary_group_key,
        "graph_audits": graph_audits,
        "primary_graph_audit": primary_graph_audit,
        "polarity_shift": round(polarity_shift_for_display, 3),
        "polarity_shift_label": describe_temporal_shift(polarity_shift_for_display),
        "temporal_penalty": temporal_penalty_for_display,
        "dimension_breakdown": dimension_breakdown,
        "axiom_distortion": axiom_distortion,
        "coherence_score": round(coherence_score, 1),
        "axiom_score": round(coherence_score, 1),  # backward-compatible alias
        "axiom_verdict": axiom_verdict,
    }

def determine_verdict_axiom(distortion: float, thresholds: Dict[str, Any]) -> str:
    """axiom 정규화 점수용 verdict 라벨.

    distortion: 0~100 (높을수록 왜곡 큼)
    thresholds: {"aligned_max": 34, "caution_max": 64}
    """
    aligned_max = float(thresholds.get("aligned_max", 34))
    caution_max = float(thresholds.get("caution_max", 64))
    if distortion <= aligned_max:
        return "기준 부합 (axiom)"
    if distortion <= caution_max:
        return "주의 필요 (axiom)"
    return "기준 이탈 (axiom)"

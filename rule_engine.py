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

# [v1.2.6] Headline/subtitle weighting
# 제목과 부제는 독자가 가장 먼저 접하는 프레임 장치이므로 본문보다 약간 높은 가중치를 둔다.
# 단, 낚시성 표현 하나가 전체 판정을 과도하게 흔들지 않도록 2배 이상으로 올리지 않는다.
TITLE_CUE_WEIGHT = 1.5
SUBTITLE_CUE_WEIGHT = 1.25
BODY_CUE_WEIGHT = 1.0

# [v2.x baseline] OWL baseline 가중치 + per_dim_cap 45.
# 시계열 정합성은 최상위 차원이되 frame/context/consensus/evidence와 공동 판단한다.
# cap 45는 "입장 변화 자체"(major shift 단독 ~22.7점)와 "설명 없는 조용한 전환"
# (silent pivot 부가 페널티가 cap을 채우며 34점 상한 수렴)을 점수 폭으로 분리하기 위한 값이다.
# 시뮬레이터 프로파일: sensitive=40 / default=45 / conservative=60.
PER_DIM_CAP = 45.0

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

# [v2.1.8] Frame-level soft cap + effective cap mitigation
# 1024개 룰셋은 관찰/설명력을 높이기 위해 유사 프레임의 세부 룰을 많이 가진다.
# 그러나 같은 target_frame 룰이 여러 개 동시에 발화했다고 해서 모두 독립 위반으로
# 단순 누적하면 per_dim_cap(45/60 등)이 즉시 100%에 도달해 cap 프로파일의
# 설명력이 사라진다. 따라서 rule-level evidence는 보존하되, score-level penalty는
# (target_frame, dimension) 단위로 묶어 아래 상한 안에서만 반영한다.
# cap=18의 의미: 동일 프레임 반복 증거는 critical 단일 룰 1개치(SEVERITY_BASE=18)를
# 기본 최대치로 본다. 즉 "프레임이 강하게 감지됨"은 보존하되, 독립 위반 10건처럼
# 누적하지 않는다. 설명 충실도가 검증된 경우에는 effective_frame_cap = base_cap ×
# mitigation_factor로 cap 자체도 낮춰, 설명 있는 전환과 설명 없는 silent pivot을 구분한다.
FRAME_SOFT_CAPS = {
    "SilentPivot": 18.0,
    "RetroactiveReframing": 18.0,
    "SelectiveMemory": 14.0,
    "ContextOmission": 14.0,
    "FalseBalance": 14.0,
    "CrisisInflation": 12.0,
    "VictimBlaming": 18.0,
}
DEFAULT_FRAME_SOFT_CAP = 12.0
EXPLANATION_SENSITIVE_FRAMES = {"SilentPivot", "RetroactiveReframing", "SelectiveMemory", "ContextOmission"}

# Explanation mitigation은 단어 등장만으로 과도하게 완화되지 않도록 보수적 tier를 둔다.
# 0.50(strong)은 과거 입장/기준 인지 + 새 근거/조건 변화 + 변경 이유 설명이 함께 보일 때만 목표로 한다.
EXPLANATION_EVIDENCE_CUES = ["새로운 증거", "새 증거", "자료", "데이터", "통계", "보고서", "원문", "공개", "확인", "조사", "판결", "결정"]
EXPLANATION_CHANGE_CUES = ["조건 변화", "상황 변화", "정책 변경", "제도 변화", "기준 변경", "환경 변화", "바뀌", "달라졌", "변경", "전환"]
EXPLANATION_ACK_CUES = ["과거", "이전", "당시", "기존", "종전", "입장", "판단", "평가", "수정", "재검토"]

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


def article_field_texts(article: Dict[str, Any]) -> Dict[str, str]:
    """기사 필드를 제목/부제/본문으로 분리한다.

    subtitle 필드가 없으면 summary, description, lead를 순서대로 fallback한다.
    """
    title = str(article.get("title", "") or "")
    subtitle = str(
        article.get("subtitle")
        or article.get("sub_title")
        or article.get("summary")
        or article.get("description")
        or article.get("lead")
        or ""
    )
    body = str(article.get("body") or article.get("text") or article.get("content") or "")
    return {"title": title, "subtitle": subtitle, "body": body}


def text_of(article: Dict[str, Any]) -> str:
    """기사 텍스트 합성.

    graph audit처럼 text 문자열만 받는 경로에서도 제목 프레임을 놓치지 않도록
    제목을 한 번 더 포함한다. 정밀 cue 계산은 weighted_count_hits_article에서
    title=1.5, subtitle=1.25, body=1.0으로 처리한다.
    """
    fields = article_field_texts(article)
    parts = [
        fields["title"],
        fields["title"],      # text-only downstream을 위한 headline emphasis
        fields["subtitle"],
        fields["body"],
    ]
    return " ".join(p for p in parts if p)


def count_hits(text: str, cues: Iterable[str]) -> int:
    return sum(text.count(cue) for cue in cues if cue)


def weighted_count_hits_article(article: Dict[str, Any], cues: Iterable[str]) -> float:
    """제목/부제/본문별 cue 가중합.

    제목은 실제 뉴스 소비에서 프레임을 강하게 형성하지만, 과도한 단일 표현으로
    전체 점수가 흔들리지 않도록 mild weighting(1.5)을 적용한다.
    """
    fields = article_field_texts(article)
    return (
        TITLE_CUE_WEIGHT * count_hits(fields["title"], cues)
        + SUBTITLE_CUE_WEIGHT * count_hits(fields["subtitle"], cues)
        + BODY_CUE_WEIGHT * count_hits(fields["body"], cues)
    )


def weighted_sentiment_score_article(article: Dict[str, Any]) -> float:
    """제목/부제/본문 가중치를 반영한 sentiment polarity."""
    pos = weighted_count_hits_article(article, POSITIVE_LEXICON)
    neg = weighted_count_hits_article(article, NEGATIVE_LEXICON)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def compute_explanation_mitigation_signal(past_text: str, present_text: str) -> Dict[str, Any]:
    """입장 전환 사유 설명의 충실도를 휴리스틱하게 측정한다.

    원칙:
    - stance polarity의 Δ 자체는 줄이지 않는다.
    - 설명은 단순 cue 등장 여부가 아니라, 독자가 입장 변화의 이유를 추적할 수 있는지로 본다.
    - 형식적/불충분한 설명은 0.90~0.75 수준으로 제한 완화하고,
      strong(0.50)은 과거 기준 인지 + 새 근거/조건 변화 + 변경 이유가 비교적 함께 드러나는 경우에만 준다.
    - 이 factor는 Stage 3/4의 부가 penalty와 explanation-sensitive frame의 effective_frame_cap에만 적용된다.

    Returns:
        {
            "present_context_expl_hits": int,
            "past_context_expl_hits": int,
            "present_omission_hits": int,
            "explanation_category_hits": int,
            "mitigation_factor": float,  # 1.0=none, 0.90=weak, 0.75=moderate, 0.50=strong
            "level": "none|weak|moderate|strong",
            "reason": str,
        }
    """
    present_context = count_hits(present_text, CONTEXT_EXPLANATION_CUES)
    past_context = count_hits(past_text, CONTEXT_EXPLANATION_CUES)
    present_omission = count_hits(present_text, OMISSION_CUES)

    evidence_hits = count_hits(present_text, EXPLANATION_EVIDENCE_CUES)
    change_hits = count_hits(present_text, EXPLANATION_CHANGE_CUES)
    ack_hits = count_hits(present_text, EXPLANATION_ACK_CUES)
    category_hits = int(evidence_hits > 0) + int(change_hits > 0) + int(ack_hits > 0)

    # 현재 기사의 설명을 가장 강하게 보고, 과거 기사 배경설명은 보조 신호로만 반영한다.
    explanation_signal = present_context + 0.5 * past_context

    # strong은 엄격하게: cue 수가 많고, 적어도 두 종류 이상의 설명 범주가 함께 보여야 한다.
    # 단순히 "상황이 바뀌었다" 수준이면 weak 또는 moderate에 머문다.
    if present_context >= 4 and category_hits >= 2 and explanation_signal >= 4:
        factor = 0.50
        level = "strong"
        reason = "과거 기준/조건 변화/새 근거를 추적할 수 있는 설명이 충분함"
    elif present_context >= 2 and category_hits >= 1:
        factor = 0.75
        level = "moderate"
        reason = "전환 사유 설명이 일부 확인되지만 완전한 설명 책임에는 미달"
    elif present_context >= 1 or explanation_signal >= 1.5:
        factor = 0.90
        level = "weak"
        reason = "형식적 설명 신호가 있으나 입장 변화 정당화에는 제한적"
    else:
        factor = 1.0
        level = "none"
        reason = "전환 사유 설명 신호가 약함"

    return {
        "present_context_expl_hits": present_context,
        "past_context_expl_hits": past_context,
        "present_omission_hits": present_omission,
        "explanation_signal": round(explanation_signal, 2),
        "explanation_category_hits": category_hits,
        "evidence_explanation_hits": evidence_hits,
        "change_explanation_hits": change_hits,
        "acknowledgement_hits": ack_hits,
        "mitigation_factor": factor,
        "level": level,
        "reason": reason,
    }


# ─────────────────────────────────────
# Stage 0: Validity / Ethics Red Card
# ─────────────────────────────────────

_VALIDITY_NEGATION_CUES = [
    "아니다", "아니며", "잘못", "허위", "거짓", "반박", "비판", "금지", "문제", "위험", "혐오표현", "차별금지",
]

_VALIDITY_RED_CARD_RULES = [
    {
        "category": "scientific_consensus_violation",
        "label": "과학적 합의 정면 위배",
        "patterns": [
            r"지구\s*(?:는|가)?\s*평평",
            r"평평한\s*지구",
            r"지구\s*평면",
            r"flat\s*earth",
            r"earth\s*is\s*flat",
        ],
    },
    {
        "category": "protected_class_hate_or_discrimination",
        "label": "보호집단 차별·혐오 또는 열등성 주장",
        "patterns": [
            r"(?:인종|민족|혈통)\s*(?:이|은|는|간)?[^\n.]{0,20}(?:우월|열등)",
            r"(?:여성|장애인|이주민|난민|성소수자|종교|무슬림|유대인|흑인|아시아인|노인|아동)\s*(?:이|은|는)?[^\n.]{0,30}(?:열등|권리\s*없|배제해야|추방해야|제거해야)",
            r"차별\s*(?:을|은|이)?\s*(?:정당화|옹호|찬성)",
        ],
    },
    {
        "category": "human_rights_red_card",
        "label": "보편 인권 침해 옹호",
        "patterns": [
            r"(?:인종청소|제노사이드|대량학살)\s*(?:은|이|을)?[^\n.]{0,20}(?:정당|필요|찬성|옹호)",
            r"(?:노예제|강제노동)\s*(?:는|은|을)?[^\n.]{0,20}(?:정당|필요|찬성|옹호)",
            r"나치\s*(?:를|는|의)?[^\n.]{0,20}(?:옹호|찬양|정당화)",
        ],
    },
]


def _has_negating_context(text: str, start: int, end: int, window: int = 28) -> bool:
    """red-card 후보 표현이 비판·반박·부정 맥락인지 간단히 걸러낸다."""
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    snippet = text[lo:hi]
    return any(cue in snippet for cue in _VALIDITY_NEGATION_CUES)


def detect_validity_red_card(text: str) -> Dict[str, Any]:
    """최소한의 사실/윤리 기준 위반을 Stage 0에서 탐지한다.

    설계 원칙:
    - 시계열 정합성 점수와 별개의 상위 자격 심사다.
    - 명시적 red-card만 잡는 보수적 휴리스틱이다.
    - 비판/반박 문맥은 가능한 한 제외한다.
    """
    if not text or not str(text).strip():
        return {"violation": False}
    body = str(text).lower()
    original = str(text)
    for rule in _VALIDITY_RED_CARD_RULES:
        for pat in rule["patterns"]:
            m = re.search(pat, body, flags=re.IGNORECASE)
            if not m:
                continue
            if _has_negating_context(original, m.start(), m.end()):
                continue
            matched = original[m.start():m.end()]
            return {
                "violation": True,
                "category": rule["category"],
                "label": rule["label"],
                "matched_text": matched,
                "pattern": pat,
                "reason": f"Stage 0 validity red-card: {rule['label']} — '{matched}'",
            }
    return {"violation": False}


def scan_articles_validity_red_card(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """기사 묶음 전체에서 validity red-card를 먼저 검사한다.

    first↔last graph audit만으로는 중간 기사에 포함된 red-card를 놓칠 수 있으므로,
    N개 기사 전체를 독립적으로 스캔한다.
    """
    hits: List[Dict[str, Any]] = []
    for idx, article in enumerate(articles or [], 1):
        text = text_of(article)
        red = detect_validity_red_card(text)
        if red.get("violation"):
            red = dict(red)
            red.update({
                "article_index": idx,
                "title": article.get("title", ""),
                "date": article.get("date") or article.get("published_at") or "",
                "outlet": article.get("outlet", ""),
                "topic": article.get("topic", ""),
            })
            hits.append(red)
    if not hits:
        return {"violation": False, "hits": []}
    first = hits[0]
    return {
        "violation": True,
        "hits": hits,
        "category": first.get("category"),
        "label": first.get("label"),
        "reason": first.get("reason"),
    }


def build_validity_trace(red_card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reasoning Trace용 Stage 0 red-card 이벤트."""
    if not red_card or not red_card.get("violation"):
        return []
    hit_count = len(red_card.get("hits", [])) if isinstance(red_card.get("hits"), list) else 1
    return [{
        "stage": "Stage 0: Validity Red Card",
        "trigger": red_card.get("reason") or red_card.get("label") or "validity violation",
        "owl_relation": "validity_override",
        "calibrated_dimension": "validity_score / final override",
        "penalty": "override → final_distortion=100, coherence_score=0",
        "source": red_card.get("category", "validity_red_card"),
        "detail": f"최소 사실/윤리 기준 위반 {hit_count}건 감지. 5차원 가중합보다 우선하는 상위 자격 심사로 처리.",
        "rule_ids": [],
    }]


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
    """OWL 온톨로지에서 Frame 클래스의 label/keyword를 추출해 반환.

    Streamlit Cloud 환경에서 rdflib SPARQL 파서가 버전 조합에 따라 실패할 수 있어,
    SPARQL 대신 triples/subjects/objects 순회 방식으로 수집한다.
    """
    cues: List[str] = []
    NS = rdflib.Namespace("http://www.context-sync.com/ontology/news-app#")
    RDF = rdflib.RDF
    RDFS = rdflib.RDFS
    try:
        for frame_uri in g.subjects(RDF.type, NS.Frame):
            for label in g.objects(frame_uri, RDFS.label):
                label_s = str(label).strip()
                if label_s:
                    cues.append(label_s)
    except Exception:
        return []
    tokens: List[str] = []
    for c in cues:
        tokens.extend(t for t in re.split(r"[\s_]+", c) if len(t) >= 2)
    return list(dict.fromkeys(tokens))

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
    """Distortion score(0~100, 높을수록 왜곡 큼)를 5단계 verdict로 변환한다.

    v1.2.5부터 3단계 신호등 모델을 5단계 농도 모델로 확장한다.
    내부 계산은 distortion 기준을 사용하고, UI/README에서는
    coherence_score = 100 - distortion 기준으로 직관적으로 설명한다.

    기본 distortion bands:
      0~15   안정적 정합
      16~30  기준 부합
      31~45  주의 필요
      46~60  중점 검토 필요
      61~100 기준 이탈
    """
    stable_max = float(thresholds.get("stable_aligned_max", 15))
    aligned_max = float(thresholds.get("aligned_max", 30))
    caution_max = float(thresholds.get("caution_max", 45))
    high_caution_max = float(thresholds.get("high_caution_max", 60))

    if score <= stable_max:
        return "안정적 정합"
    if score <= aligned_max:
        return "기준 부합"
    if score <= caution_max:
        return "주의 필요"
    if score <= high_caution_max:
        return "중점 검토 필요"
    return "기준 이탈"


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
        # [v1.2.6] 제목/부제/본문 cue 가중치 반영
        item["_sentiment"] = weighted_sentiment_score_article(article)
        item["_frame_hits"] = weighted_count_hits_article(article, effective_frame_cues)
        item["_omission_hits"] = weighted_count_hits_article(article, OMISSION_CUES)
        item["_context_expl_hits"] = weighted_count_hits_article(article, CONTEXT_EXPLANATION_CUES)
        item["_consensus_hits"] = weighted_count_hits_article(article, CONSENSUS_CUES)
        # [v1.2.2] evidence_quality: 증거 부재 cues − 증거 충실 cues
        item["_evidence_lack_hits"] = weighted_count_hits_article(article, EVIDENCE_LACK_CUES)
        item["_evidence_present_hits"] = weighted_count_hits_article(article, EVIDENCE_PRESENT_CUES)
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
                "subtitle": a.get("subtitle") or a.get("summary") or "",
                "sentiment": round(a["_sentiment"], 3),
                "frame_hits": round(a["_frame_hits"], 2),
                "omission_hits": round(a["_omission_hits"], 2),
                "context_expl_hits": round(a["_context_expl_hits"], 2),
                "evidence_lack_hits": round(a["_evidence_lack_hits"], 2),
                "evidence_present_hits": round(a["_evidence_present_hits"], 2),
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

    Cloud 안정성을 위해 SPARQL ASK 대신 직접 triple lookup을 사용한다.
    """
    if graph is None or not value or not frame or frame == "None":
        return False
    NS = rdflib.Namespace("http://www.context-sync.com/ontology/news-app#")
    return (NS[f"val_{value}"], NS.conflictsWith, NS[f"frame_{frame}"]) in graph

def check_value_reinforces(graph: Optional[rdflib.Graph], value_a: str, value_b: str) -> bool:
    """OWL의 reinforces 관계로 두 ValueAnchor가 강화 관계인지 검사. 양방향을 허용한다."""
    if graph is None or not value_a or not value_b or value_a == value_b:
        return False
    NS = rdflib.Namespace("http://www.context-sync.com/ontology/news-app#")
    a = NS[f"val_{value_a}"]
    b = NS[f"val_{value_b}"]
    return ((a, NS.reinforces, b) in graph) or ((b, NS.reinforces, a) in graph)

def get_calibrated_dimensions(graph: Optional[rdflib.Graph], value: str) -> List[str]:
    """OWL calibratesDimension 관계로 ValueAnchor가 보정하는 평가 차원 조회."""
    if graph is None or not value:
        return []
    NS = rdflib.Namespace("http://www.context-sync.com/ontology/news-app#")
    dims: List[str] = []
    for dim_uri in graph.objects(NS[f"val_{value}"], NS.calibratesDimension):
        iri = str(dim_uri)
        dim = iri.split("#dim_", 1)[1] if "#dim_" in iri else iri.rsplit("/", 1)[-1].replace("dim_", "")
        if dim in DIMENSIONS and dim not in dims:
            dims.append(dim)
    return dims

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



def _signed_soft_cap(total_penalty: float, cap: float) -> float:
    """부호를 보존해 frame-level raw sum에 절댓값 상한을 적용한다."""
    total = float(total_penalty or 0.0)
    cap_value = max(0.0, float(cap or 0.0))
    if total < 0:
        return -min(abs(total), cap_value)
    return min(total, cap_value)


def apply_frame_soft_caps(
    fired_rules: List[Dict[str, Any]],
    frame_caps: Optional[Dict[str, float]] = None,
    default_cap: float = DEFAULT_FRAME_SOFT_CAP,
    explanation_sensitive_frames: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """동일 프레임 룰 다발 발화의 점수 폭주를 막는 중간 집계층.

    원칙:
    - fired_rules는 근거 추적용으로 모두 보존한다.
    - 실제 score 반영은 (target_frame, dimension) 단위의 capped_penalty만 사용한다.
    - 같은 프레임의 룰 10개 발화는 "10개 독립 위반"이라기보다
      "해당 프레임이 강하게 감지됨"으로 해석한다.
    - explanation-sensitive frame은 effective_frame_cap = base_frame_cap × mitigation_factor를 적용한다.
      충분한 설명(strong=0.50)은 부가 프레임 페널티의 최대치 자체를 낮춘다.
    """
    caps = frame_caps or FRAME_SOFT_CAPS
    sensitive = explanation_sensitive_frames or EXPLANATION_SENSITIVE_FRAMES
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rule in fired_rules:
        penalty = float(rule.get("axiom_penalty", 0) or 0)
        if penalty == 0:
            continue
        frame = str(rule.get("target_frame") or "Unknown")
        dim = str(rule.get("dimension") or "unknown")
        buckets[(frame, dim)].append(rule)

    groups: List[Dict[str, Any]] = []
    for (frame, dim), rules in buckets.items():
        raw_sum = round(sum(float(r.get("axiom_penalty", 0) or 0) for r in rules), 1)
        base_cap = float(caps.get(frame, default_cap))
        mitigation_factors = [float(r.get("explanation_mitigation_factor", 1.0) or 1.0) for r in rules]
        # 보수적 집계: factor가 낮을수록 강한 완화이므로, 여러 룰이 섞이면 가장 덜 관대한 factor(max)를 사용한다.
        group_factor = max(mitigation_factors) if mitigation_factors else 1.0
        if frame in sensitive:
            effective_cap = round(base_cap * group_factor, 2)
        else:
            effective_cap = base_cap
            group_factor = 1.0
        applied = round(_signed_soft_cap(raw_sum, effective_cap), 1)
        representative = max(
            rules,
            key=lambda r: abs(float(r.get("axiom_penalty", 0) or 0)),
        )
        rule_ids = [str(r.get("rule_id", "?")) for r in rules]
        groups.append({
            "target_frame": frame,
            "dimension": dim,
            "rule_count": len(rules),
            "rule_ids": rule_ids,
            "representative_rule_id": representative.get("rule_id"),
            "representative_schema_id": representative.get("schema_id"),
            "raw_rule_sum": raw_sum,
            "base_frame_soft_cap": base_cap,
            "cap_mitigation_factor": group_factor,
            "effective_frame_cap": effective_cap,
            # Backward-compatible alias for older UI code; in v2.1.8 this means effective cap.
            "frame_soft_cap": effective_cap,
            "capped_penalty": applied,
            "suppressed_penalty": round(raw_sum - applied, 1),
            "frame_definition_ko": representative.get("frame_definition_ko", ""),
            "schema_description_ko": representative.get("schema_description_ko", ""),
        })

    groups.sort(key=lambda g: (abs(float(g.get("capped_penalty", 0) or 0)), g.get("rule_count", 0)), reverse=True)
    return groups


def compute_frame_capped_dimension_breakdown(
    matched_rules: List[Dict[str, Any]],
    base_breakdown: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """matched/fired rule 목록을 frame soft cap 적용 후 dimension_breakdown으로 변환."""
    breakdown = {d: float((base_breakdown or {}).get(d, 0.0)) for d in DIMENSIONS}
    fired = [r for r in matched_rules if abs(float(r.get("axiom_penalty", 0) or 0)) > 0]
    groups = apply_frame_soft_caps(fired)
    for group in groups:
        dim = group.get("dimension")
        penalty = float(group.get("capped_penalty", 0) or 0)
        if dim in breakdown:
            breakdown[dim] += penalty
    return {d: round(v, 1) for d, v in breakdown.items()}, groups

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
        "frame_penalty_groups": [],
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

    # ─── Stage 0: Validity / Ethics Red Card ───
    # 최소 사실/윤리 기준 위반은 시계열 정합성 계산보다 우선한다.
    pair_red_card = detect_validity_red_card("\n".join([past_text or "", present_text or ""]))
    if pair_red_card.get("violation"):
        report["validity_violation"] = True
        report["validity_score"] = -100
        report["logic_conflict"] = True
        report["weighted_distortion"] = 100.0
        report["score"] = 0.0
        report["anchor_verdict"] = "기준 이탈"
        report["details"]["validity_red_card"] = pair_red_card
        report["reasons"].append(pair_red_card.get("reason", "Stage 0 validity red-card"))
        report["trace_events"].extend(build_validity_trace(pair_red_card))
        return report

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

    explanation_sensitive_frames = EXPLANATION_SENSITIVE_FRAMES

    # [v2.1.8] Rule-level evidence는 모두 보존하되, score-level penalty는
    # (target_frame, dimension) frame bucket별 soft cap을 적용한 뒤 dimension_breakdown에 반영한다.
    stage4_rules: List[Dict[str, Any]] = []
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
        stage4_rules.append(r)

    frame_groups = apply_frame_soft_caps(stage4_rules)
    report["frame_penalty_groups"] = frame_groups
    report["fired_rules"] = stage4_rules

    for group in frame_groups:
        penalty = float(group.get("capped_penalty", 0) or 0)
        rule_dim = group.get("dimension")
        report["logic_score"] += penalty
        if rule_dim in report["dimension_breakdown"]:
            report["dimension_breakdown"][rule_dim] += penalty
        else:
            # 현재 match_rules는 DIMENSIONS 안의 dimension만 반환하지만, 혹시 모를 확장을 위해 보존.
            representative_rule = next((r for r in stage4_rules if r.get("rule_id") == group.get("representative_rule_id")), None)
            rule_value = (representative_rule or {}).get("value_anchor", "")
            if rule_value:
                _distribute(rule_value, penalty)

    if report["fired_rules"]:
        rule_ids = [r["rule_id"] for r in report["fired_rules"]]
        report["reasons"].append(
            f"Fired {len(rule_ids)} rules: {', '.join(str(x) for x in rule_ids[:5])}"
            f"{'...' if len(rule_ids) > 5 else ''}"
        )

        # Stage 4 trace는 실제 점수 반영 단위인 frame_penalty_groups 기준으로 요약한다.
        for group in report.get("frame_penalty_groups", []):
            frame = str(group.get("target_frame", "?"))
            dim = str(group.get("dimension", "?"))
            grp_ids = [str(x) for x in group.get("rule_ids", [])]
            detail = group.get("frame_definition_ko", "") or group.get("schema_description_ko", "")
            raw_sum = float(group.get("raw_rule_sum", 0) or 0)
            applied = float(group.get("capped_penalty", 0) or 0)
            cap = float(group.get("frame_soft_cap", 0) or 0)
            _append_trace(
                "Stage 4: Rule frame-cap",
                f"frame={frame} ({group.get('rule_count', 0)}개 룰)",
                "frame_soft_cap → dimension",
                dim,
                applied,
                f"raw_sum={raw_sum:.1f}, cap={cap:.1f}, rules:[{', '.join(grp_ids[:5])}{'...' if len(grp_ids) > 5 else ''}]",
                detail[:120] if detail else "1024 JSON 룰셋 발화 — 동일 프레임 룰 다발은 capped_penalty만 점수 반영",
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

    # 판정 — v1.2.5 5단계 verdict 체계
    thresholds = rules_data.get("verdict_thresholds", {})
    report["anchor_verdict"] = determine_verdict_axiom(weighted_distortion, thresholds)

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
# [v2.1.7] #2 compute_weighted_distortion (per_dim_cap=45 정규화 + frame soft cap 전제)
# ═══════════════════════════════════════════════════════════════

def compute_weighted_distortion(
    dimension_breakdown: Dict[str, float],
    weights: Dict[str, float],
    per_dim_cap: float = PER_DIM_CAP,
) -> float:
    """차원별 페널티를 dimension_weights에 따라 최종 왜곡도로 환산한다.

    JSON normalization_note_ko의 권고를 반영:
      1. 각 차원의 누적 페널티 절댓값을 0~100 스케일로 정규화
         (per_dim_cap=45 baseline 상한 클리핑 — 한 차원에 45점 이상 누적 시 100% 왜곡)
      2. 정규화된 차원 점수 × 가중치 → 가중 합산
      3. 최종 0~100 클리핑

    per_dim_cap은 한 차원이 최종 weighted_distortion을 과도하게 지배하지 않도록
    dimension-level 기여도를 제한한다. 동일 프레임의 다중 룰 발화로 raw_penalty가
    cap을 크게 초과하는 문제는 Stage 4의 frame-level soft cap에서 먼저 보정한다.
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
    """fired rules를 frame-level soft cap 적용 후 dimension별로 누적한다.

    fired_rules 목록은 근거 추적용으로 유지하되, 점수화에서는 동일 프레임 룰 다발을
    모두 독립 위반으로 보지 않는다. 실제 반영값은 apply_frame_soft_caps()의
    capped_penalty를 사용한다.
    """
    breakdown, _groups = compute_frame_capped_dimension_breakdown(matched_rules)
    return breakdown


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

    Streamlit Community Cloud에서 rdflib SPARQL query가 환경에 따라 실패할 수 있으므로
    SPARQL 없이 triples/subjects/objects 순회만 사용한다.
    """
    NS = rdflib.Namespace("http://www.context-sync.com/ontology/news-app#")
    OWL_NS = "http://www.w3.org/2002/07/owl#"
    RDF = rdflib.RDF
    RDFS = rdflib.RDFS

    def short(uri: Any) -> str:
        return str(uri).replace(str(NS), "").replace(OWL_NS, "owl:")

    def label_of(uri: Any) -> str:
        labels = list(g.objects(uri, RDFS.label))
        return str(labels[0]).strip() if labels else short(uri)

    def unique_append(items: List[Tuple[str, str]], item: Tuple[str, str]) -> None:
        if item not in items:
            items.append(item)

    layers: List[Tuple[str, str]] = []
    values: List[Tuple[str, str]] = []
    frames: List[Tuple[str, str]] = []
    schemas: List[Tuple[str, str]] = []
    edges: List[Tuple[str, str]] = []

    for s in g.subjects(RDF.type, NS.AxiomLayer):
        unique_append(layers, (short(s), label_of(s)))
    for s in g.subjects(RDF.type, NS.ValueAnchor):
        sid = short(s)
        unique_append(values, (sid, label_of(s)))
        for layer in g.objects(s, NS.belongsToLayer):
            edges.append((short(layer), sid))
    for s in g.subjects(RDF.type, NS.Frame):
        unique_append(frames, (short(s), label_of(s)))

    schema_types = [NS.TemporalRuleSchema, NS.StructuralRuleSchema, NS.MetacognitiveRuleSchema]
    seen_schema = set()
    for schema_type in schema_types:
        for s in g.subjects(RDF.type, schema_type):
            if s in seen_schema:
                continue
            seen_schema.add(s)
            sid = short(s)
            unique_append(schemas, (sid, label_of(s)))
            for frame in g.objects(s, NS.constrainsFrame):
                edges.append((sid, short(frame)))
            for value in g.objects(s, NS.anchoredByValue):
                edges.append((short(value), sid))
            for layer in g.objects(s, NS.belongsToLayer):
                edges.append((short(layer), sid))

    lines = ["flowchart TD"]
    lines.append('    classDef layerStyle fill:#4a90d9,color:#fff,stroke:#2c5f8a')
    lines.append('    classDef valueStyle fill:#50c878,color:#fff,stroke:#2d8a4e')
    lines.append('    classDef frameStyle fill:#f5a623,color:#fff,stroke:#c17d12')
    lines.append('    classDef schemaStyle fill:#9b59b6,color:#fff,stroke:#6c3483')

    def safe_id(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z_가-힣]", "_", str(s))

    def safe_label(s: str) -> str:
        return str(s).replace('"', "'").replace("\n", " ")[:80]

    for sid, label in layers:
        lines.append(f'    {safe_id(sid)}["{safe_label(label)}"]:::layerStyle')
    for sid, label in values:
        lines.append(f'    {safe_id(sid)}["{safe_label(label)}"]:::valueStyle')
    for sid, label in frames[:15]:
        lines.append(f'    {safe_id(sid)}["{safe_label(label)}"]:::frameStyle')
    for sid, label in schemas[:12]:
        lines.append(f'    {safe_id(sid)}["{safe_label(label)}"]:::schemaStyle')

    visible_nodes = {safe_id(s) for s, _ in layers + values + frames[:15] + schemas[:12]}
    seen_edges = set()
    for src, dst in edges:
        s, d = safe_id(src), safe_id(dst)
        if s in visible_nodes and d in visible_nodes and (s, d) not in seen_edges:
            lines.append(f"    {s} --> {d}")
            seen_edges.add((s, d))

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
    # Stage 0 validity red-card: 5차원 점수보다 우선하는 상위 자격 심사
    # ─────────────────────────────────────
    red_card = scan_articles_validity_red_card(articles)
    if red_card.get("violation"):
        zero_breakdown = {d: 0.0 for d in DIMENSIONS}
        validity_trace = build_validity_trace(red_card)
        verdict_reason = red_card.get("reason", "Stage 0 validity red-card")
        return {
            "features": features,
            "weights": weights,

            # v1 compatibility + 명시적 alias
            "score": v1_distortion,
            "verdict": v1_verdict,
            "v1_distortion_score": v1_distortion,
            "v1_verdict": v1_verdict,

            # red-card override
            "validity_violation": True,
            "validity_red_card": red_card,
            "validity_trace": validity_trace,
            "validity_score": -100,

            "matched_rules": [],
            "candidate_rules": [],
            "axiom_fired_rules": [],
            "axiom_frame_penalty_groups": [],
            "verdict_reason": verdict_reason,

            "article_rows": extraction["article_rows"],
            "series": extraction["series"],
            "rule_summary": summarize_rules(rules_data.get("rules", [])),

            "axiom_lite_distortion": 100.0,
            "axiom_lite_score": 0.0,
            "axiom_lite_verdict": "기준 이탈 (validity red-card)",
            "axiom_lite_dimension_breakdown": zero_breakdown,

            "axiom_source": "validity_red_card",
            "primary_group_key": "-",
            "graph_audits": [],
            "primary_graph_audit": None,
            "polarity_shift": 0.0,
            "polarity_shift_label": "Skipped by validity red-card",
            "temporal_penalty": 0.0,
            "dimension_breakdown": zero_breakdown,
            "axiom_distortion": 100.0,
            "coherence_score": 0.0,
            "axiom_score": 0.0,
            "axiom_verdict": "기준 이탈 (validity red-card)",
        }

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
    lite_breakdown, lite_frame_penalty_groups = compute_frame_capped_dimension_breakdown(candidate_rules)
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
        axiom_frame_penalty_groups = primary_graph_audit.get("frame_penalty_groups", [])
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
        axiom_frame_penalty_groups = lite_frame_penalty_groups
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
        "axiom_frame_penalty_groups": axiom_frame_penalty_groups,
        "verdict_reason": verdict_reason,

        "article_rows": extraction["article_rows"],
        "series": extraction["series"],
        "rule_summary": summarize_rules(rules_data.get("rules", [])),

        # axiom-lite 보조값
        "axiom_lite_distortion": axiom_lite_distortion,
        "axiom_lite_score": axiom_lite_score,
        "axiom_lite_verdict": axiom_lite_verdict,
        "axiom_lite_dimension_breakdown": lite_breakdown,
        "axiom_lite_frame_penalty_groups": lite_frame_penalty_groups,

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
    """axiom 정규화 점수용 5단계 verdict 라벨.

    app.py의 메트릭 제목이 이미 "최종 판정 (axiom)"을 표시하므로,
    라벨 자체에는 (axiom) 접미사를 붙이지 않는다.
    """
    return determine_verdict(distortion, thresholds)

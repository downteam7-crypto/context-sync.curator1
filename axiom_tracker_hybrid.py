"""
Axiom Tracker — Hybrid Edition v2.1.1 (OpenAI GPT-4 + OWL Vocabulary + JSON Rules)

GPT-4의 맥락 판독 능력 위에 OWL 어휘 기준층과 JSON 룰셋을 결합한 시계열 논조 판독 앱.

3축 구조:
    1. 추출 (LLM): GPT-4가 OWL의 ValueAnchor/Frame/Topic 어휘로 텍스트를 구조화
    2. 검증 (Python + OWL/JSON): rdflib로 OWL의 conflictsWith 관계 검사 + 1024 OWL-expanded 룰셋 매칭
    3. 리포트 (LLM): 발화된 룰을 증거로 GPT-4가 리치 리포트 생성

설계 원칙:
    - OWL은 어휘 기준층 (rdflib만, owlready2 미사용 → 가벼움 유지)
    - JSON 룰은 실무 검출 규칙 (스키마-룰셋 분리)
    - LLM은 어휘 안에서만 답하도록 강제 (Pydantic + literal options)

실행 전 준비:
    pip install gradio openai pydantic python-dotenv rdflib
    OWL/JSON 파일을 ./ontology/ 폴더에 배치
    .env에 OPENAI_API_KEY 또는 UI에서 직접 입력
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import rdflib
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# ==========================================
# 0. Paths & Constants
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_DIR = BASE_DIR / "ontology"
OWL_PATH = ONTOLOGY_DIR / "context_sync_app_centered_ontology_1024.owl"
RULES_PATH = ONTOLOGY_DIR / "news_rules_1024.json"

NS = "http://www.context-sync.com/ontology/news-app#"

# ==========================================
# 1. Ontology Loader (rdflib)
# ==========================================

class OntologyVocabulary:
    """OWL 어휘를 추출해 LLM 프롬프트와 검증 단계에 공급한다.

    rdflib만 사용 (owlready2 미사용) → 가벼운 어휘 참조 전용.
    """

    def __init__(self, owl_path: Path):
        self.graph = rdflib.Graph()
        self.loaded = False
        self.value_anchors: List[str] = []
        self.frames: List[str] = []
        self.topics: List[str] = []
        self.layers: List[str] = []
        self.value_descriptions: Dict[str, str] = {}
        self.frame_descriptions: Dict[str, str] = {}
        self.conflicts: Dict[str, List[str]] = {}

        if not owl_path.exists():
            print(f"⚠️  OWL not found: {owl_path}")
            return

        try:
            self.graph.parse(str(owl_path), format="xml")
            self.loaded = True
            self._extract_vocabulary()
            print(f"✅ Ontology loaded: {len(self.graph)} triples, "
                  f"{len(self.value_anchors)} values, {len(self.frames)} frames")
        except Exception as e:
            print(f"⚠️  OWL load error: {e}")

    def _extract_vocabulary(self) -> None:
        """rdflib SPARQL로 어휘 추출."""
        # ValueAnchor 추출
        q_values = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
        SELECT ?label ?desc WHERE {
            ?v rdf:type cs:ValueAnchor .
            ?v rdfs:label ?label .
            OPTIONAL { ?v cs:descriptionKo ?desc . }
        }
        """
        for row in self.graph.query(q_values):
            label = str(row.label)
            self.value_anchors.append(label)
            if row.desc:
                self.value_descriptions[label] = str(row.desc)

        # Frame 추출
        q_frames = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
        SELECT ?label ?desc WHERE {
            ?f rdf:type cs:Frame .
            ?f rdfs:label ?label .
            OPTIONAL { ?f rdfs:comment ?desc . }
        }
        """
        for row in self.graph.query(q_frames):
            label = str(row.label)
            self.frames.append(label)
            if row.desc:
                self.frame_descriptions[label] = str(row.desc)

        # Topic 추출 (ctx_*)
        q_topics = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
        SELECT ?label WHERE {
            ?t rdf:type cs:Topic .
            ?t rdfs:label ?label .
        }
        """
        self.topics = [str(row.label) for row in self.graph.query(q_topics)]

        # AxiomLayer 추출
        q_layers = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
        SELECT ?label WHERE {
            ?l rdf:type cs:AxiomLayer .
            ?l rdfs:label ?label .
        }
        """
        self.layers = [str(row.label) for row in self.graph.query(q_layers)]

    def get_vocabulary_prompt(self) -> str:
        """LLM 추출 프롬프트에 주입할 어휘 설명."""
        lines = ["[OWL ValueAnchor — 텍스트가 옹호하는 기준 가치 (택1)]"]
        for v in self.value_anchors:
            desc = self.value_descriptions.get(v, "")
            lines.append(f"  - {v}: {desc}" if desc else f"  - {v}")

        lines.append("\n[OWL Frame — 텍스트가 작동시키는 프레임 (해당하면 택1, 없으면 'None')]")
        for f in sorted(self.frames):
            desc = self.frame_descriptions.get(f, "")
            lines.append(f"  - {f}: {desc}" if desc else f"  - {f}")

        lines.append("\n[OWL Topic — 텍스트의 주제 영역 (택1)]")
        for t in sorted(self.topics):
            lines.append(f"  - {t}")

        return "\n".join(lines)


# ==========================================
# 2. Rule Loader (JSON 1024)
# ==========================================

class RuleSet:
    """1024개 OWL-expanded 외부 룰셋을 로드하고 매칭 가능한 형태로 보관."""

    def __init__(self, rules_path: Path):
        self.loaded = False
        self.rules: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, Any] = {}
        self.formula: str = ""
        self.dimension_weights: Dict[str, float] = {}

        if not rules_path.exists():
            print(f"⚠️  Rules not found: {rules_path}")
            return

        try:
            with rules_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.rules = data.get("rules", [])
            self.thresholds = data.get("verdict_thresholds", {})
            self.formula = data.get("aggregation_formula", "")
            self.dimension_weights = data.get("dimension_weights", {
                "temporal_shift": 0.40,
                "frame_effect": 0.22,
                "context_omission": 0.15,
                "consensus_deviation": 0.13,
                "evidence_quality": 0.10,
            })
            self.loaded = True
            print(f"✅ Rules loaded: {len(self.rules)} rules")
        except Exception as e:
            print(f"⚠️  Rules load error: {e}")

    def match_rules(
        self,
        target_frame: Optional[str],
        topic: Optional[str],
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """추출된 frame과 topic에 해당하는 룰을 필터.

        target_frame이 'None'이거나 매치 없으면 빈 리스트.
        topic 미매치면 frame만으로 폴백.
        """
        if not target_frame or target_frame == "None":
            return []

        # 1순위: frame + topic 둘 다 매치
        primary = [
            r for r in self.rules
            if r.get("target_frame") == target_frame and r.get("context") == topic
        ]
        if primary:
            return primary[:max_results]

        # 2순위: frame만 매치
        fallback = [r for r in self.rules if r.get("target_frame") == target_frame]
        return fallback[:max_results]


def compute_weighted_distortion(
    dimension_breakdown: Dict[str, float],
    weights: Dict[str, float],
    per_dim_cap: float = 45.0,
) -> float:
    """차원별 페널티를 JSON dimension_weights에 따라 최종 왜곡도로 환산한다.

    JSON normalization_note_ko의 권고:
      "최종 점수는 dimension별 0~100 정규화 후 dimension_weights로 가중합산한다.
       기사 길이 또는 매칭 룰 수에 따른 과대평가를 피하기 위해
       단순 합산보다 평균/상한 클리핑을 권장한다."

    구현 (v2.1 기준):
      1. 각 차원의 누적 페널티 절댓값을 0~100 스케일로 정규화 (per_dim_cap=45 상한 클리핑)
         - per_dim_cap=45는 "한 차원에 45점 이상 누적되면 100% 왜곡"으로 해석
         - Major polarity shift (-30) 단독 시 정합 영역의 가장 낮은 지대(score 70대 초중반)에 안착
         - 룰이 많이 발화되어도 한 차원이 다른 차원을 지배하지 못하게 함
      2. 정규화된 차원 점수 × 가중치 → 가중 합산
         (v2.1 가중치: temporal_shift 0.40, frame_effect 0.22, context_omission 0.15,
          consensus_deviation 0.13, evidence_quality 0.10)
      3. 최종 0~100 클리핑

    per_dim_cap은 튜닝 파라미터다. README의 *임시 파라미터 명시* 사상에 따라
    실측 보도 코퍼스로 검증 후 보정될 예정.
    """
    if not weights:
        return 0.0

    weighted_sum = 0.0
    for dim, weight in weights.items():
        raw_penalty = abs(float(dimension_breakdown.get(dim, 0.0)))
        # 정규화: 0 ~ 100 (per_dim_cap이 100% 기준점)
        normalized = min(100.0, (raw_penalty / per_dim_cap) * 100.0)
        weighted_sum += normalized * float(weight)

    return min(100.0, round(weighted_sum, 1))

def verdict_from_distortion(distortion: float, thresholds: Dict[str, Any]) -> str:
    """JSON verdict_thresholds 기준으로 판정 라벨을 반환한다."""
    aligned_max = float(thresholds.get("aligned_max", 34))
    caution_max = float(thresholds.get("caution_max", 64))
    if distortion <= aligned_max:
        return "기준 부합"
    if distortion <= caution_max:
        return "주의 필요"
    return "기준 이탈"


# ==========================================
# 3. Pydantic Schemas
# ==========================================

class ExtractedSymbol(BaseModel):
    """단일 텍스트에서 OWL 어휘로 구조화된 메타데이터."""
    title_subject: str = Field(description="제목이 다루는 주제 (1~3 단어)")
    body_subject: str = Field(description="본문이 실제로 다루는 주제 (1~3 단어)")
    promoted_value: str = Field(
        description="텍스트가 옹호하는 OWL ValueAnchor 중 하나 (StanceConsistency, FrameAccountability, ContextCompleteness, ResponsibilitySeparation, PluralPublicReason, EvidenceTransparency 등 OWL에 정의된 것만)"
    )
    detected_frame: str = Field(
        description="텍스트가 작동시키는 OWL Frame 중 하나, 또는 'None'. (SilentPivot, RetroactiveReframing, SelectiveMemory, EmotionalAppeal, FalseBalance 등)"
    )
    topic: str = Field(
        description="OWL Topic 중 하나 (Election, PolicyDebate, Judiciary, CorporateScandal 등)"
    )
    stance_polarity: float = Field(
        description="주제에 대한 입장 극성. -1.0(강한 반대) ~ +1.0(강한 지지)",
        ge=-1.0, le=1.0,
    )
    is_valid_discourse: bool = Field(
        description="과학적 사실 및 민주적 윤리에 부합하면 True, 위배되면 False"
    )
    violation_reason: str = Field(
        description="위배 이유 (예: 'Flat Earth', 'Hate Speech'). 위배 없으면 'None'"
    )
    key_claims: List[str] = Field(description="텍스트의 핵심 주장 1~3개")


class RichReport(BaseModel):
    """최종 리치 리포트."""
    verdict: str = Field(description="최종 판정 (예: '정합', '주의', '시계열 정합성 위반', '실격')")
    summary: str = Field(description="2~4문장 요약. 발화된 룰을 근거로 인용")
    self_attack: str = Field(description="이 판정 자체에 대한 자기 비판")
    steelman: str = Field(description="저자 입장에서의 가장 강한 변호 논리")
    evidence_rules: List[str] = Field(description="이 판정의 근거가 된 룰 ID 리스트")


# ==========================================
# 4. Extractor
# ==========================================

def build_extraction_prompt(vocab: OntologyVocabulary) -> str:
    """OWL 어휘를 system prompt에 주입."""
    return f"""You are a discourse analyst. Extract structured metadata from a single text using ONLY the controlled vocabulary below.

[Safety Criteria — Red Card Check]
Set is_valid_discourse=False ONLY if the text:
1. Contradicts well-established scientific consensus (flat earth, climate denial of mechanism, etc.)
2. Advocates totalitarianism, ethnic cleansing, or universal human rights violations
3. Promotes hate speech against protected groups

Controversial political stance is NOT a violation. Disagreement with mainstream is NOT violation.

[Controlled Vocabulary]
{vocab.get_vocabulary_prompt()}

[Extraction Rules]
- promoted_value: pick the closest ValueAnchor from the list above. Required.
- detected_frame: pick the most fitting Frame, or 'None' if no clear frame is at work.
- topic: pick the closest Topic from the list above.
- stance_polarity: how strongly does the author support/oppose the main subject?
- key_claims: 1-3 most important claims.

If you must pick a value not in the vocabulary, pick the closest one and note in key_claims why.

Output strictly in the requested JSON schema."""


def extract_symbol(client: OpenAI, text: str, vocab: OntologyVocabulary, model: str) -> ExtractedSymbol:
    """단일 텍스트에서 OWL 어휘로 구조화 추출."""
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": build_extraction_prompt(vocab)},
            {"role": "user", "content": f"Analyze this text:\n\n{text}"},
        ],
        response_format=ExtractedSymbol,
        temperature=0.1,
    )
    return response.choices[0].message.parsed


# ==========================================
# 5. OWL-based Conflict Check + Rule Matching
# ==========================================

def check_value_frame_conflict(vocab: OntologyVocabulary, value: str, frame: str) -> bool:
    """OWL의 conflictsWith 관계로 ValueAnchor와 Frame이 구조적으로 충돌하는지 검사.

    이것이 시계열 정합성 검증의 핵심: 과거 텍스트가 옹호한 Value를
    현재 텍스트의 Frame이 위반하는지를 그래프에서 직접 추론한다.
    """
    if not vocab.loaded or not value or not frame or frame == "None":
        return False

    q = f"""
    PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
    ASK {{
        cs:val_{value} cs:conflictsWith cs:frame_{frame} .
    }}
    """
    try:
        result = vocab.graph.query(q)
        return bool(result.askAnswer)
    except Exception:
        return False


def check_value_reinforces(vocab: OntologyVocabulary, value_a: str, value_b: str) -> bool:
    """OWL의 reinforces 관계로 두 ValueAnchor가 같은 메타 가치의
    다른 측면인지(서로 강화하는 관계인지) 검사.

    이게 True면 value_a → value_b 이동은 *강조점 이동*이지 충돌이 아님.
    """
    if not vocab.loaded or value_a == value_b:
        return False

    q = f"""
    PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
    ASK {{
        {{ cs:val_{value_a} cs:reinforces cs:val_{value_b} . }}
        UNION
        {{ cs:val_{value_b} cs:reinforces cs:val_{value_a} . }}
    }}
    """
    try:
        result = vocab.graph.query(q)
        return bool(result.askAnswer)
    except Exception:
        return False


def get_calibrated_dimensions(vocab: OntologyVocabulary, value: str) -> List[str]:
    """OWL의 calibratesDimension 관계로 ValueAnchor가 보정하는 평가 차원 조회.

    각 ValueAnchor는 어떤 EvaluationDimension을 보정하는지 OWL에 명시되어 있다.
    이를 통해 *Value 위반*이 *어느 점수 차원으로 반영되어야 하는가*를 그래프에서
    직접 추론할 수 있다.
    """
    if not vocab.loaded or not value:
        return []

    q = f"""
    PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
    SELECT ?dim WHERE {{
        cs:val_{value} cs:calibratesDimension ?dim .
    }}
    """
    try:
        results = vocab.graph.query(q)
        # IRI에서 dim_ prefix 제거하고 차원 이름만 추출
        dims = []
        for row in results:
            iri = str(row.dim)
            if "#dim_" in iri:
                dims.append(iri.split("#dim_")[-1])
        return dims
    except Exception:
        return []


def audit_logic(
    past: ExtractedSymbol,
    present: ExtractedSymbol,
    vocab: OntologyVocabulary,
    ruleset: RuleSet,
) -> Dict[str, Any]:
    """추출된 두 메타데이터를 OWL 관계 + 1024 OWL-expanded 룰셋으로 비교 검증.

    점수 분리:
        - validity_score: 사실/윤리 위배 (Red Card, 시계열 무관)
        - logic_score: 시계열 입장 변경, OWL 충돌, JSON 룰 발화의 원점수
        - dimension_breakdown: logic_score 페널티가 5개 EvaluationDimension에 누적된 값
        - weighted_distortion: dimension_breakdown을 per_dim_cap=45와 dimension_weights로 정규화한 최종 왜곡도
        - total = max(0, 100 - weighted_distortion + validity_score)
    """
    report = {
        "validity_violation": False,
        "logic_conflict": False,
        "validity_score": 0,
        "logic_score": 0,
        "score": 100,
        "reasons": [],
        "fired_rules": [],
        # OWL의 calibratesDimension 관계로 추론된 차원별 점수 분해.
        # 각 ValueAnchor 위반이 어느 EvaluationDimension에 반영되었는지를 기록.
        "dimension_breakdown": {
            "temporal_shift": 0,
            "frame_effect": 0,
            "context_omission": 0,
            "consensus_deviation": 0,
            "evidence_quality": 0,
        },
        "details": {
            "past": past.model_dump(),
            "present": present.model_dump(),
        },
    }

    # ─────────────────────────────────────
    # Stage 1: Validity (Red Card)
    # ─────────────────────────────────────
    if not past.is_valid_discourse:
        report["validity_violation"] = True
        report["validity_score"] -= 100
        report["reasons"].append(f"PAST validity violation: {past.violation_reason}")

    if not present.is_valid_discourse:
        report["validity_violation"] = True
        report["validity_score"] -= 100
        report["reasons"].append(f"PRESENT validity violation: {present.violation_reason}")

    if report["validity_violation"]:
        report["reasons"].append("⛔ AUDIT STOPPED: validity violation supersedes temporal analysis.")
        report["weighted_distortion"] = 100.0
        report["anchor_verdict"] = "기준 이탈"
        report["score"] = max(0, 100 + report["validity_score"])
        return report

    # ─────────────────────────────────────
    # Stage 2: Stance Polarity Shift (정량)
    # 입장 극성 이동은 본질적으로 temporal_shift 차원의 왜곡이므로,
    # logic_score뿐 아니라 dimension_breakdown["temporal_shift"]에도 페널티를 직접 누적한다.
    # 이렇게 해야 weighted_distortion 가중합에 실제로 반영된다.
    # ─────────────────────────────────────
    polarity_shift = abs(past.stance_polarity - present.stance_polarity)
    if polarity_shift >= 1.0:
        report["logic_conflict"] = True
        report["logic_score"] -= 30
        report["dimension_breakdown"]["temporal_shift"] -= 30
        report["reasons"].append(
            f"Major stance reversal: {past.stance_polarity:+.2f} → {present.stance_polarity:+.2f} (Δ={polarity_shift:.2f}) → temporal_shift"
        )
    elif polarity_shift >= 0.5:
        report["logic_score"] -= 15
        report["dimension_breakdown"]["temporal_shift"] -= 15
        report["reasons"].append(
            f"Significant stance shift: {past.stance_polarity:+.2f} → {present.stance_polarity:+.2f} (Δ={polarity_shift:.2f}) → temporal_shift"
        )
    elif polarity_shift >= 0.25:
        report["logic_score"] -= 5
        report["dimension_breakdown"]["temporal_shift"] -= 5
        report["reasons"].append(
            f"Moderate stance shift: Δ={polarity_shift:.2f} → temporal_shift"
        )

    # ─────────────────────────────────────
    # Stage 3: OWL Graph Reasoning
    # 두 종류의 그래프 추론을 수행:
    #   3a. Value ↔ Frame conflictsWith — 과거 옹호 가치를 현재 프레임이
    #       구조적으로 위반하는가 (시계열 정합성의 핵심 추론)
    #   3b. Value ↔ Value reinforces — Value 이동이 강조점 이동인가
    #       구조적 위반인가
    # ─────────────────────────────────────

    # 헬퍼: penalty를 calibrate된 차원에 분배
    # 한 Value가 여러 차원을 calibrate하면 페널티를 균등 분배
    def _distribute_to_dimensions(value: str, penalty: int) -> str:
        """ValueAnchor의 위반 페널티를 OWL calibratesDimension 그래프로
        추론한 차원들에 분배 기록한다. calibrated dim이 없으면 frame_effect로 폴백한다
        (그래프 추론의 default 차원). 이렇게 해야 logic_score 페널티가 weighted_distortion에
        반드시 반영된다.

        반환: 분배된 차원들의 한국어 요약 (reasons에 추가하기 위해).
        """
        if not value or penalty == 0:
            return ""
        dims = get_calibrated_dimensions(vocab, value)
        if not dims:
            # 폴백: 그래프 추론 페널티가 어디에도 매핑되지 않으면 frame_effect로
            if "frame_effect" in report["dimension_breakdown"]:
                report["dimension_breakdown"]["frame_effect"] += penalty
                return f" → fallback dimension: frame_effect"
            return ""
        per_dim_penalty = penalty / len(dims)  # 음수 페널티
        for d in dims:
            if d in report["dimension_breakdown"]:
                report["dimension_breakdown"][d] += per_dim_penalty
        return f" → calibrated dimensions: {', '.join(dims)}"

    # 3a. Value ↔ Frame 충돌 검사 (양방향 + 교차)
    # 가장 의미 있는 케이스: 과거 옹호 가치를 현재 프레임이 위반
    if past.promoted_value and present.detected_frame and present.detected_frame != "None":
        if check_value_frame_conflict(vocab, past.promoted_value, present.detected_frame):
            report["logic_conflict"] = True
            penalty = -25
            report["logic_score"] += penalty
            dim_note = _distribute_to_dimensions(past.promoted_value, penalty)
            report["reasons"].append(
                f"OWL graph conflict (cross-temporal): PAST value '{past.promoted_value}' "
                f"↔ PRESENT frame '{present.detected_frame}'{dim_note}"
            )

    # 보조: 현재 옹호 가치를 현재 프레임이 위반 (자기 모순)
    if present.promoted_value and present.detected_frame and present.detected_frame != "None":
        if check_value_frame_conflict(vocab, present.promoted_value, present.detected_frame):
            penalty = -15
            report["logic_score"] += penalty
            dim_note = _distribute_to_dimensions(present.promoted_value, penalty)
            report["reasons"].append(
                f"OWL graph conflict (self-contradictory): PRESENT value '{present.promoted_value}' "
                f"↔ PRESENT frame '{present.detected_frame}' (text claims a value its frame undermines){dim_note}"
            )

    # 3b. 가치 이동의 성격 판별
    if past.promoted_value != present.promoted_value:
        if check_value_reinforces(vocab, past.promoted_value, present.promoted_value):
            # 강화 관계 → 강조점 이동, 가벼운 페널티
            penalty = -3
            report["logic_score"] += penalty
            # 강조점 이동은 두 가치 모두에 영향 → 두 가치의 calibrated 차원에 분배
            _distribute_to_dimensions(past.promoted_value, penalty // 2)
            _distribute_to_dimensions(present.promoted_value, penalty - penalty // 2)
            report["reasons"].append(
                f"Value emphasis shift (within reinforcement relation): "
                f"{past.promoted_value} ↔ {present.promoted_value}"
            )
        else:
            # 강화 관계도 아님 → 더 큰 이동, 검토 필요
            penalty = -12
            report["logic_score"] += penalty
            dim_note = _distribute_to_dimensions(past.promoted_value, penalty)
            report["reasons"].append(
                f"Value shift (no OWL reinforces relation): "
                f"{past.promoted_value} → {present.promoted_value}{dim_note}"
            )

    # ─────────────────────────────────────
    # Stage 4: 1024 Rule Matching (실무 검출)
    # ─────────────────────────────────────
    # 현재 텍스트의 frame이 룰에 매치되는지 확인
    matched_rules = ruleset.match_rules(
        target_frame=present.detected_frame,
        topic=present.topic,
        max_results=10,
    )

    for rule in matched_rules:
        # 룰 발화 조건: stance shift가 있거나 frame이 검출된 경우
        # severity_band에 따라 점수 차감
        severity = rule.get("severity_band", "low")
        risk_w = float(rule.get("risk_weight", 0.5))
        distortion_w = float(rule.get("distortion_weight", 0.5))

        # 규칙 발화 강도: stance shift × risk × distortion
        # (stance shift가 작으면 룰이 약하게 발화)
        intensity = polarity_shift * risk_w * distortion_w
        if intensity < 0.05 and not report["logic_conflict"]:
            continue  # 약한 발화는 무시

        severity_penalty = {"critical": -15, "high": -10, "medium": -6, "low": -3}.get(severity, -3)
        report["logic_score"] += severity_penalty

        # JSON 개별 룰의 dimension을 우선 반영한다.
        # dimension이 없거나 앱 breakdown에 없는 경우에만 OWL value_anchor 매핑으로 폴백한다.
        rule_dim = rule.get("dimension")
        if rule_dim in report["dimension_breakdown"]:
            report["dimension_breakdown"][rule_dim] += severity_penalty
        else:
            rule_value = rule.get("value_anchor", "")
            if rule_value:
                _distribute_to_dimensions(rule_value, severity_penalty)

        report["fired_rules"].append({
            "rule_id": rule.get("rule_id"),
            "schema_id": rule.get("schema_id"),
            "schema_type": rule.get("schema_type"),
            "target_frame": rule.get("target_frame"),
            "context": rule.get("context"),
            "profile": rule.get("profile"),
            "value_anchor": rule.get("value_anchor"),
            "dimension": rule.get("dimension"),
            "severity_band": severity,
            "penalty": severity_penalty,
            "instruction": rule.get("llm_instruction_ko", ""),
            "frame_definition_ko": rule.get("frame_definition_ko", ""),
            "schema_description_ko": rule.get("schema_description_ko", ""),
            "expected_evidence_ko": rule.get("expected_evidence_ko", ""),
            "score_hint": rule.get("score_hint", {}),
            "positive_cues": rule.get("positive_cues", []),
            "negative_indicators": rule.get("negative_indicators", []),
        })

    if report["fired_rules"]:
        rule_ids = [r["rule_id"] for r in report["fired_rules"]]
        report["reasons"].append(f"Fired {len(rule_ids)} rules: {', '.join(rule_ids[:5])}{'...' if len(rule_ids) > 5 else ''}")

    # dimension_breakdown 정수화 (분배 시 소수가 들어갈 수 있음)
    for d in report["dimension_breakdown"]:
        report["dimension_breakdown"][d] = round(report["dimension_breakdown"][d], 1)

    # ─────────────────────────────────────
    # 최종 점수: JSON dimension_weights 기반 weighted_distortion 반영
    #
    # 설계 원칙:
    # - 정상 경로: logic_score 페널티가 발생할 때마다 dimension_breakdown에도 같이 누적됨
    #   → weighted_distortion이 그 누적값을 5차원 가중합으로 환산 → 최종 점수에 반영
    # - 안전망: 혹시 어떤 페널티가 dimension에 못 들어간 채 logic_score에만 남으면
    #   (예: 신규 페널티 추가 시 dimension 분배를 깜빡한 경우) 점수를 임의 보정하지 않고
    #   warning만 남긴다. 점수 차원 왜곡을 막기 위해 frame_effect 강제 흡수는 사용하지 않는다.
    # ─────────────────────────────────────
    weighted_distortion = compute_weighted_distortion(
        report["dimension_breakdown"],
        ruleset.dimension_weights,
    )
    report["weighted_distortion"] = weighted_distortion
    report["dimension_weights"] = ruleset.dimension_weights
    report["anchor_verdict"] = verdict_from_distortion(weighted_distortion, ruleset.thresholds)

    # 안전망: weighted_distortion이 logic_score 감점을 충분히 반영하지 못한 듯하면 경고만 남긴다.
    # 점수를 임의 보정하지 않는 이유: 미반영분이 temporal/context/evidence 중 어느 축인지
    # 확정할 수 없는데 frame_effect로 강제 흡수하면 리포트 해석이 왜곡될 수 있다.
    dim_total_penalty = sum(abs(v) for v in report["dimension_breakdown"].values())
    logic_abs = abs(report["logic_score"])
    if logic_abs > dim_total_penalty + 0.5:  # 부동소수점 오차 허용
        unreflected = logic_abs - dim_total_penalty
        report["reasons"].append(
            f"[warning] logic_score 미반영 가능성: {unreflected:.1f}점. "
            "신규 페널티가 dimension_breakdown에 연결되었는지 확인 필요."
        )

    report["score"] = max(0, round(100 - weighted_distortion + report["validity_score"], 1))

    if not report["reasons"]:
        report["reasons"].append("No significant temporal coherence violations detected.")

    return report


# ==========================================
# 6. Reporter
# ==========================================

REPORTER_SYSTEM_PROMPT = """You are a discourse audit reporter. Generate a rich report in Korean.

You will receive:
- Audit summary with score breakdown
- List of fired rules from the 1024-rule ontology-backed ruleset

Your job:
- verdict: 한 줄 최종 판정
- summary: 2~4문장. *발화된 룰의 instruction을 인용*하여 구체적으로 무엇이 어떻게 변했는지 설명
- self_attack: 이 판정이 틀렸을 가능성. 가장 그럴듯한 반론
- steelman: 저자 입장에서의 가장 강한 변호 논리
- evidence_rules: 판정 근거가 된 룰 ID 리스트 (fired_rules에서 발췌)

If validity_violation is True, prioritize warning about false information / unethical claims.
Otherwise, ground your analysis in the fired rules and the OWL-derived conflicts.

Output strictly in the requested JSON schema. Write in Korean."""


def generate_report(client: OpenAI, audit: Dict[str, Any], model: str) -> RichReport:
    """audit 결과를 받아 룰 기반 리치 리포트 생성."""
    audit_summary = json.dumps(
        {
            "validity_violation": audit["validity_violation"],
            "logic_conflict": audit["logic_conflict"],
            "validity_score": audit["validity_score"],
            "logic_score": audit["logic_score"],
            "score": audit["score"],
            "weighted_distortion": audit.get("weighted_distortion"),
            "anchor_verdict": audit.get("anchor_verdict"),
            "dimension_breakdown": audit.get("dimension_breakdown", {}),
            "dimension_weights": audit.get("dimension_weights", {}),
            "reasons": audit["reasons"],
            "fired_rules": audit["fired_rules"],
            "past": {
                "subject": audit["details"]["past"]["body_subject"],
                "value": audit["details"]["past"]["promoted_value"],
                "frame": audit["details"]["past"]["detected_frame"],
                "topic": audit["details"]["past"]["topic"],
                "polarity": audit["details"]["past"]["stance_polarity"],
                "claims": audit["details"]["past"]["key_claims"],
            },
            "present": {
                "subject": audit["details"]["present"]["body_subject"],
                "value": audit["details"]["present"]["promoted_value"],
                "frame": audit["details"]["present"]["detected_frame"],
                "topic": audit["details"]["present"]["topic"],
                "polarity": audit["details"]["present"]["stance_polarity"],
                "claims": audit["details"]["present"]["key_claims"],
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Audit result:\n\n{audit_summary}"},
        ],
        response_format=RichReport,
        temperature=0.3,
    )
    return response.choices[0].message.parsed


# ==========================================
# 7. Pipeline (Gradio entrypoint)
# ==========================================

# Singleton 로드 (앱 기동 시 1회)
_VOCAB = OntologyVocabulary(OWL_PATH)
_RULES = RuleSet(RULES_PATH)


def sync_frames_with_rules(vocab: "OntologyVocabulary", ruleset: "RuleSet") -> int:
    """LLM 선택지에서 OWL-only 프레임을 제거한다.

    OWL은 천천히 변하는 풍부한 어휘를 담고, JSON 룰셋은 실무 검출 규칙을 담는다.
    OWL이 더 많은 프레임을 가질 수 있으나, LLM이 JSON 룰셋에 없는 프레임을
    고르면 fired_rules가 비게 된다. 이를 막기 위해 *런타임에만* 선택지를 필터링.
    OWL 파일 자체는 손대지 않는다.

    Returns:
        제거된 프레임 수
    """
    if not vocab.loaded or not ruleset.loaded:
        return 0

    valid_frames = {r.get("target_frame") for r in ruleset.rules if r.get("target_frame")}
    # NeutralFact는 기준선 프레임으로 보존 (룰셋에 없어도 LLM이 'None' 대안으로 선택 가능)
    valid_frames.add("NeutralFact")

    before = len(vocab.frames)
    vocab.frames = [f for f in vocab.frames if f in valid_frames]
    vocab.frame_descriptions = {
        k: v for k, v in vocab.frame_descriptions.items() if k in valid_frames
    }
    removed = before - len(vocab.frames)
    if removed > 0:
        print(f"✅ sync_frames: OWL-only 프레임 {removed}개 제거 (LLM 선택지 정합화)")
    return removed


# OWL 풍부한 어휘 vs JSON 룰셋의 부분집합 — 런타임 동기화
sync_frames_with_rules(_VOCAB, _RULES)


def run_pipeline(
    past_text: str,
    present_text: str,
    api_key: str,
    model: str,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """전체 파이프라인 실행."""
    if not api_key or not api_key.strip():
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"error": "OPENAI_API_KEY가 입력되지 않았습니다."}, {}, {}, {}

    if not past_text or not present_text:
        return {"error": "Past와 Present 텍스트를 모두 입력해주세요."}, {}, {}, {}

    if not _VOCAB.loaded:
        return {"error": f"OWL 파일을 로드하지 못했습니다: {OWL_PATH}"}, {}, {}, {}
    if not _RULES.loaded:
        return {"error": f"룰셋 파일을 로드하지 못했습니다: {RULES_PATH}"}, {}, {}, {}

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return {"error": f"OpenAI 클라이언트 초기화 실패: {e}"}, {}, {}, {}

    # Step 1: 추출
    try:
        past_data = extract_symbol(client, past_text, _VOCAB, model=model)
        present_data = extract_symbol(client, present_text, _VOCAB, model=model)
    except Exception as e:
        return {"error": f"추출 단계 실패: {e}"}, {}, {}, {}

    # Step 2: 검증 (OWL 관계 + 1024 룰)
    audit_result = audit_logic(past_data, present_data, _VOCAB, _RULES)

    # Step 3: 리포트
    try:
        report = generate_report(client, audit_result, model=model)
    except Exception as e:
        return (
            {"error": f"리포트 생성 실패: {e}", "partial_audit": audit_result},
            past_data.model_dump(),
            present_data.model_dump(),
            audit_result,
        )

    return (
        report.model_dump(),
        past_data.model_dump(),
        present_data.model_dump(),
        audit_result,
    )


# ==========================================
# 8. Gradio UI
# ==========================================

def _count_graph_relations(vocab: OntologyVocabulary) -> Tuple[int, int, int]:
    """그래프에서 conflictsWith / reinforces / calibratesDimension 관계 인스턴스 개수 카운트."""
    if not vocab.loaded:
        return 0, 0, 0
    q_conflict = """
    PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
    SELECT (COUNT(*) AS ?n) WHERE { ?v cs:conflictsWith ?f . }
    """
    q_reinforce = """
    PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
    SELECT (COUNT(*) AS ?n) WHERE { ?a cs:reinforces ?b . }
    """
    q_calibrate = """
    PREFIX cs: <http://www.context-sync.com/ontology/news-app#>
    SELECT (COUNT(*) AS ?n) WHERE { ?v cs:calibratesDimension ?d . }
    """
    try:
        c = next(iter(vocab.graph.query(q_conflict)))
        r = next(iter(vocab.graph.query(q_reinforce)))
        cd = next(iter(vocab.graph.query(q_calibrate)))
        return int(c.n), int(r.n), int(cd.n)
    except Exception:
        return 0, 0, 0


def build_ontology_status() -> str:
    """OWL/룰 로드 상태를 사이드 패널에 표시."""
    lines = ["### 시스템 상태"]
    if _VOCAB.loaded:
        n_conflict, n_reinforce, n_calibrate = _count_graph_relations(_VOCAB)
        lines.append(f"- ✅ OWL: {len(_VOCAB.graph)} triples")
        lines.append(f"  - {len(_VOCAB.value_anchors)} ValueAnchor: `{', '.join(_VOCAB.value_anchors)}`")
        lines.append(f"  - {len(_VOCAB.frames)} Frame")
        lines.append(f"  - {len(_VOCAB.topics)} Topic")
        lines.append(f"  - **{n_conflict} conflictsWith** relations (Value ↔ Frame)")
        lines.append(f"  - **{n_reinforce} reinforces** relations (Value ↔ Value)")
        lines.append(f"  - **{n_calibrate} calibratesDimension** relations (Value → EvaluationDimension)")
    else:
        lines.append(f"- ❌ OWL not loaded ({OWL_PATH})")

    if _RULES.loaded:
        lines.append(f"- ✅ Rules: {len(_RULES.rules)}개")
        if getattr(_RULES, "dimension_weights", None):
            lines.append(f"- ✅ Dimension weights: {_RULES.dimension_weights}")
    else:
        lines.append(f"- ❌ Rules not loaded ({RULES_PATH})")

    return "\n".join(lines)


with gr.Blocks(title="Axiom Tracker — Hybrid Edition") as demo:
    gr.Markdown("## 🛡️ Axiom Tracker — Hybrid Edition (LLM + OWL + 1024 Rules)")
    gr.Markdown(
        "GPT-4 + OWL 어휘 기준층 + 1024 OWL-expanded 룰셋 결합 시계열 논조 판독기.\n\n"
        "**3축 구조**: GPT-4가 OWL 어휘로 추출 → rdflib로 OWL 관계 검증 + 1024 룰 매칭 → GPT-4가 룰 근거 리포트 생성"
    )
    gr.Markdown(build_ontology_status())

    with gr.Row():
        api_input = gr.Textbox(
            label="OpenAI API Key",
            type="password",
            placeholder="sk-... (또는 .env의 OPENAI_API_KEY 자동 로드)",
        )
        model_input = gr.Dropdown(
            label="Model",
            choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            value="gpt-4o",
        )

    with gr.Row():
        p_in = gr.Textbox(
            label="Past Text (과거 논조)",
            value="A정책은 시민의 기본권을 침해할 우려가 있다. 충분한 사회적 합의 없이 강행해서는 안 된다.",
            lines=5,
        )
        n_in = gr.Textbox(
            label="Present Text (현재 논조)",
            value="A정책은 사회 전체의 안전을 위한 불가피한 조치다. 일부 권리 제한은 감수할 만하다.",
            lines=5,
        )

    btn = gr.Button("🔍 Run Hybrid Audit", variant="primary")

    with gr.Row():
        report_out = gr.JSON(label="📋 Final Report (with evidence_rules)")
        audit_out = gr.JSON(label="🧮 Audit + Fired Rules")

    with gr.Row():
        p_data_out = gr.JSON(label="🔍 Past Extraction (OWL vocabulary)")
        n_data_out = gr.JSON(label="🔍 Present Extraction (OWL vocabulary)")

    btn.click(
        run_pipeline,
        inputs=[p_in, n_in, api_input, model_input],
        outputs=[report_out, p_data_out, n_data_out, audit_out],
    )

    gr.Markdown(
        "---\n"
        "**점수 분리**\n"
        "- `validity_score`: 사실/윤리 위배. Red Card 사유이며 시계열 평가와 분리된다.\n"
        "- `logic_score`: 시계열 입장 변경, OWL 충돌, JSON 룰 발화의 원점수다.\n"
        "- `dimension_breakdown`: logic_score 페널티가 5개 평가 차원에 누적된 값이다.\n"
        "- `weighted_distortion`: dimension_breakdown을 `per_dim_cap=45`와 `dimension_weights`로 정규화한 최종 왜곡도다.\n"
        "- `score = max(0, 100 - weighted_distortion + validity_score)`\n\n"
        "**차원별 분해 (`dimension_breakdown`)**: 각 ValueAnchor 위반 페널티가 OWL의 "
        "`calibratesDimension` 관계 또는 JSON rule dimension에 따라 5개 EvaluationDimension에 분배된다.\n"
        "- `temporal_shift` ← StanceConsistency\n"
        "- `frame_effect` ← FrameAccountability, ResponsibilitySeparation\n"
        "- `context_omission` ← ContextCompleteness\n"
        "- `consensus_deviation` ← PluralPublicReason, ResponsibilitySeparation\n"
        "- `evidence_quality` ← EvidenceTransparency\n\n"
        "**발화된 룰**: `audit_out.fired_rules`에서 어떤 룰이 왜 발화했는지 확인 가능. "
        "리포트의 `evidence_rules`는 그중 핵심을 발췌."
    )


if __name__ == "__main__":
    demo.launch()

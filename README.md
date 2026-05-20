# Context-Sync Curator

> *LLM이 OWL 통제 어휘 안에서 추출한 담론 구조를, 1024개 JSON 룰로 검증하고, axiom 5차원 정규화 점수로 환산하는 **ontology-calibrated reasoning** 시스템.*

뉴스 기사 묶음의 시계열 논조 변화가 *맥락 정합성* 안에서 일관되는지, 아니면 *기준 이탈*인지를 다층적으로 평가하는 프로토타입.

## 프로젝트 개요

본 프로젝트는 단일 기사 내부의 사실관계 검증(전형적 팩트체크)을 넘어, **기사 묶음이 시계열과 논조 속에서 형성하는 맥락의 정합성**을 검토하는 데 초점을 둔다.

핵심 사상:

- 본 시스템은 *시계열 변화 자체를 단죄하는 도구*가 아니라 ***맥락 정합성 평가기***다
- 입장 변화는 *양적으로 측정*하되, *변경의 정당성*은 별도 메커니즘에서 평가한다
- *5차원이 공동으로 작동*하며, 시계열은 가장 무거운 차원이되 *단일 차원이 종합 판정을 지배하지 않는다*
- *기준층은 절대적 진실이 아니라 합의의 표상*이며, 시뮬레이터에서 *손으로 만져볼 수 있다*

핵심 구조:

- **LLM**: 기사 묶음의 맥락과 논조 이동을 판독 (유동성)
- **OWL 온톨로지**: 판독이 사회적으로 합의된 보편 기준에서 이탈하지 않도록 정합성에 개입 (부동성)
- **JSON 룰셋 + Python 엔진**: 두 층위를 연결하고 점수를 계산하는 매개 아키텍처
- **Streamlit**: N개 기사 묶음을 시계열로 펼쳐 분석하는 인터페이스

본 프로젝트의 부동성 기준층은 궁극적으로는 *잘 정의된 집단지성* — 다양한 위치에 놓인 사람들의 감각이 가중치 과다 없이 녹아드는 구조 — 위에 자리잡아야 한다고 본다. 프로토타입에서는 그 기준층을 임의 파라미터로 대체하되, **수동 시뮬레이터에서 사용자가 직접 가중치를 조작**하여 *기준층이 바뀌면 결과가 어떻게 달라지는지*를 시연한다.

## 온톨로지 아키텍처

본 프로젝트의 부동성 기준층은 **3층 분리 구조**로 설계되어 있다.

- **OWL 온톨로지** (`ontology/context_sync_app_centered_ontology_1024.owl`): 17개의 추상 스키마, 6개의 ValueAnchor, 4개의 AxiomLayer, 5개의 EvaluationDimension, 20개의 Frame, 18개의 Topic 등 **느리게 변하는 보편 어휘**와, 그 어휘 사이의 **그래프 관계**(`conflictsWith`, `reinforces`, `calibratesDimension`)를 정의한다.
- **외부 룰셋** (`ontology/news_rules_1024.json`): 위 스키마에 속하는 **1024개의 실무 검출 규칙**을 별도 파일로 관리한다 (v2.1.4-baseline-restored-with-explanation-mitigation).
- **Python 계산 엔진** (`rule_engine.py`): OWL의 어휘 선언과 JSON의 규칙을 결합해 *실제 점수를 계산*한다.

각 층의 책임이 다르다 — OWL은 *어휘와 관계*, JSON은 *실무 규칙*, Python은 *계산 실행*. 이 분리가 *"공리를 80개에서 800개로, 다시 1024개로 늘리면 정교해질까"*에 대한 응답이다.

### OWL의 역할: 의미 기준층 vs 계산 엔진

본 OWL은 **점수를 계산하는 엔진이 아니라 의미 기준층**이다.

- **OWL이 하는 일** — 어휘(클래스, ValueAnchor, Frame, Topic)와 그 관계(`conflictsWith`, `reinforces`, `calibratesDimension`)를 선언. 평가 공식의 *메타데이터*도 문자열로 보관.
- **OWL이 하지 않는 일** — 점수 계산 자체. OWL은 *어떤 차원으로 어떻게 가중되어야 하는지*를 *선언*하지만, 실제 산수는 Python의 `audit_temporal_pair`와 `compute_weighted_distortion`이 수행한다.
- **외부 JSON 룰셋** — 1024개의 실무 검출 규칙. OWL의 RuleSchema가 추상 어휘를 제공하고, JSON이 그 어휘에 속하는 구체적 룰들을 공급한다.

### 6개 ValueAnchor와 calibratesDimension

본 시스템의 핵심 ValueAnchor는 페르소나 자료의 사고 결을 클래스 어휘 수준에 직접 반영한 것이다.

| ValueAnchor | 의미 | calibrates → EvaluationDimension |
|---|---|---|
| `StanceConsistency` | 같은 매체가 같은 사안에 대해 논조를 바꿀 때 *설명 책임*을 요구 | `temporal_shift` |
| `FrameAccountability` | 표면 명제가 작동시키는 *실제 사회적 프레임 효과*까지 평가 | `frame_effect` |
| `ContextCompleteness` | 표면 사실이 전체 맥락을 과도하게 생략하지 않아야 함 | `context_omission` |
| `ResponsibilitySeparation` | 잘못의 *원인 귀속*과 *대응 책임*을 분리해 평가 | `frame_effect` + `consensus_deviation` |
| `EvidenceTransparency` | 주장의 근거와 출처가 식별 가능해야 함 | `evidence_quality` |
| `PluralPublicReason` | 단순 다수결이 아니라 다양한 의견이 *구조적*으로 반영되는 기준 | `consensus_deviation` |

각 ValueAnchor의 `calibratesDimension` 매핑이 *Value 위반*이 *어느 점수 차원으로 반영되어야 하는가*를 그래프 추론으로 답한다.

### 5차원 평가와 dimension_weights (v1.2.4 baseline)

```
final_distortion =
    0.34 × temporal_shift         ← 시계열 정합성 (5차원 중 최고 가중치)
  + 0.24 × frame_effect
  + 0.18 × context_omission
  + 0.14 × consensus_deviation
  + 0.10 × evidence_quality

per_dim_cap = 40 (정규화 상한)
```

본 시스템은 *시계열 변화를 가장 무겁게* 보되, *0.34라는 가중치*가 *나머지 4차원의 합 0.66*과 균형을 이루도록 한다. 이게 *"맥락 정합성 평가기"* 정체성의 코드적 표현이다.

**시계열 엄격 모드** (0.40 / 0.22 / 0.15 / 0.13 / 0.10, cap=45)는 수동 시뮬레이터의 *프리셋*으로 유지되어, 슬라이더로 즉시 전환 가능하다.

### Explanation Mitigation (v1.2.4)

Major polarity shift(완전 입장 반전)가 *설명 충실한 정당한 반전*인지 *silent pivot*인지를 구분하는 메커니즘.

- **Δ값 자체는 보존** — *"입장이 얼마나 바뀌었는가"*는 양적 측정으로 그대로
- **시계열 입장 변경 관련 페널티만 부분 완화** — `SilentPivot`, `RetroactiveReframing`, `ResponsibilityShift`, `FalseBalance` 등 관련 frame의 graph/rule penalty가 *설명 충실 시 50% 또는 65%로 축소*
- **자기 모순 페널티는 무관** — *과거 가치 ↔ 현재 프레임 충돌* 같은 self-contradiction은 *어떤 설명에도 정당화되지 않음*

설명 cue 카운트 (`새로운 증거`, `정책 변경`, `근거`, `배경`, `맥락`, `정정`, `반론`, `설명` 등)에 따라 3단계 분류:
- 4+ cues → `strong` (factor 0.50)
- 2~4 cues → `moderate` (factor 0.65)
- <2 cues → `none` (factor 1.0, 완화 없음)

### OWL과 JSON의 어휘 비대칭 — sync_frames_with_rules

OWL은 *어휘로 풍부*(20개 Frame)하지만, JSON 룰셋은 그 *부분집합*(15개 target_frame)만 다룬다. OWL-only 프레임(`VictimBlaming`, `CrisisInflation`, `ConflictAmplification`, `FairnessDiscourse`, `NeutralFact`)은 *어휘 자산*으로 보존되며 향후 룰 추가의 거점이 된다.

`sync_frames_with_rules` 함수가 *런타임 LLM 선택지에서만* OWL-only 프레임을 필터링하여 이 간극을 메운다. OWL 파일 자체는 손대지 않는다.

## 페르소나 / 컨텍스트 자료

본 프로젝트의 LLM은 명시적인 **페르소나와 컨텍스트**를 부여받아 동작한다. 이 페르소나의 사고 결을 형성한 자료는 `docs/` 폴더에 정리되어 있다.

- [`docs/01_problem_framing.md`](./docs/01_problem_framing.md) — 문제 정의: 유동성 vs 부동성
- [`docs/02_cognition_and_metacognition.md`](./docs/02_cognition_and_metacognition.md) — 사고 구조: 일반성·특수성·메타인지
- [`docs/03_hybrid_ontology.md`](./docs/03_hybrid_ontology.md) — 확장된 비전: 하이브리드 온톨로지와 의미 인프라
- [`docs/04_background.md`](./docs/04_background.md) — 개인 배경: 비선형 궤적이 이 문제의식에 닿기까지

이 자료들은 프로젝트의 페르소나가 *왜 이런 방식으로 사안을 보는가*를 설명하기 위한 컨텍스트이며, 동시에 모델링 단계에서 LLM에게 실제로 주입한 시스템 프롬프트의 근거이기도 하다.

---

## Streamlit 분석 도구 (4단계 — 현재 메인)

`app.py`는 본 프로젝트의 **4단계 Streamlit 분석 환경**이다. 단발 비교(과거/현재 두 텍스트)를 넘어, **N개 기사 묶음을 시계열로 펼쳐 분석**하는 도구.

### 핵심 기능 4가지

#### ① 5차원 시계열 그래프 + 1024 룰 발화

- Plotly 기반 sentiment 시계열 그래프 + *변곡점 자동 마커*
- RuleSchema × 시간 발화 히트맵
- 5차원 dimension_breakdown 분해 표 + 막대그래프
- 같은 (outlet, topic) 그룹 인지 — *교차 매체 섞임 없음*

#### ② Dense/Sparse 이중 RAG (Evidence Panel)

- 같은 기사에 대해 *Sparse(Jaccard 키워드) + Dense(ko-sroberta 의미 임베딩)*를 **동시에 실행**
- 두 검색 결과를 *나란히 표시* + 공통/Sparse-only/Dense-only 대비 분석
- 최소 관련도 필터 (Sparse ≥ 0.001, Dense ≥ 0.15) — 노이즈 룰 제거

#### ③ 수동 시뮬레이터 (의미값 파라미터 직접 조작)

- 5개 차원 슬라이더 (features 조작 모드)
- *5개 가중치 슬라이더 + 합 검증* (weights 조작 모드)
- 5개 프리셋: **v1.2.4 baseline (OWL 기본값)**, **시계열 엄격 (v1.2.2 모드)**, 사실 우선, 프레임 우선, 다원 이성
- 차원별 기여도 분해 표시

*"기준층은 절대 진실이 아니라 합의의 표상이다"*가 *손에 잡히는 슬라이더*로 시연된다.

#### ④ OWL 계층 시각화 + Reasoning Trace

- Mermaid 기반 OWL 4단 레이어 (Layer → ValueAnchor → Frame → RuleSchema) 자동 추출
- **OWL Reasoning Trace Table** — audit 결과를 *"ValueAnchor → conflictsWith → calibrated dimension → penalty → fired rule"*의 논리 사슬로 표화
- ontology가 *단순 어휘 사전*이 아니라 *점수 라우팅의 메타 기준층*임을 *추적 가능한 단계*로 가시화

### 처리 단계 (axiom audit_logic)

```
N개 기사 묶음
   ↓
group by (outlet, topic) → 같은 매체-같은 사안 그룹화
   ↓
첫 기사 ↔ 마지막 기사 audit_temporal_pair
   ↓
Stage 1: 휴리스틱 또는 LLM ExtractedSymbol (ValueAnchor + Frame + stance_polarity)
Stage 2: polarity_shift 연속 감점 (compute_temporal_shift_penalty)
Stage 3: OWL 그래프 추론
  3a. 과거 Value ↔ 현재 Frame conflictsWith (cross-temporal, explanation_factor 적용)
  3b. 가치 이동 reinforces 여부 (강조점 이동 vs 구조 이동)
  자기 모순: 현재 Value ↔ 현재 Frame
Stage 4: 1024 룰 매칭 (continuous scoring, TEMPORAL_PIVOT_FRAMES만 explanation 완화)
   ↓
dimension_breakdown 누적
   ↓
compute_weighted_distortion (per_dim_cap=40 정규화)
   ↓
final_distortion + verdict + reasoning_trace
```

---

## 설치 및 실행

### 1. 클론

```bash
git clone https://github.com/downteam7-crypto/context-sync.curator1.git
cd context-sync.curator1
```

### 2. 가상환경 + 패키지 설치

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. (선택) OpenAI API 키 설정

LLM 추출기를 사용하려면:

```bash
cp .env.example .env
# 편집기로 .env 열어서 OPENAI_API_KEY 입력
```

또는 앱 사이드바에서 직접 입력 가능. LLM 없이도 *휴리스틱 추출기*로 모든 기능 작동.

### 4. Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 열리거나 `http://localhost:8501` 접속.

### 5. (선택) Dense RAG 활성화

사이드바에서 *"[Beta] 밀집 벡터(Dense) RAG"* 토글. ko-sroberta 모델이 자동 다운로드(약 500MB)된다.

 ***Streamlit Cloud 배포 참고***  

> Dense RAG를 활성화하면 `sentence-transformers` 기반 ko-sroberta 모델이 최초 실행 시 자동 다운로드되므로 초기 로딩 시간이 길어질 수 있습니다.  
> 배포 안정성을 우선할 경우 Dense RAG 토글을 끄고 Sparse RAG만으로도 앱의 핵심 분석 기능을 사용할 수 있습니다.

---

## 사용 방법

### 입력

**N개 기사 묶음** (JSON 형식 또는 사이드바 *기사 입력기*로 1개씩 누적):

```json
[
  {
    "title": "기사 제목",
    "body": "본문",
    "date": "2024-01-15",
    "outlet": "○○일보",
    "topic": "A정책"
  },
  ...
]
```

같은 `outlet` + `topic` 조합이 *시계열 그룹*으로 묶여 비교된다.

### 출력 패널

- **요약 메트릭**: 최종 왜곡도, 정합성 점수, v1 보조 왜곡도, 트리거 규칙 수
- **차원별 페널티 분해**: 5차원 dimension_breakdown 표 + 막대그래프
- **시계열 그래프**: sentiment + 변곡점 마커
- **OWL Reasoning Trace**: 추론 논리 사슬 표
- **RAG Evidence Panel**: Sparse/Dense 병렬 비교
- **트리거된 규칙**: axiom fired_rules + 메타 7필드 (frame_definition, schema_description, expected_evidence, score_hint 등)
- **OWL 계층 시각화**: Mermaid 다이어그램
- **수동 시뮬레이터**: 차원/가중치 슬라이더

---

## 폴더 구조

```
context-sync.curator1/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
├── app.py                                       ← Streamlit 메인 앱
├── rule_engine.py                               ← axiom audit 엔진 (3층 분리의 계산 층)
├── sllm_extractor.py                            ← LLM/sLLM 추출기 + Dense/Sparse RAG
├── ontology/
│   ├── context_sync_app_centered_ontology_1024.owl
│   └── news_rules_1024.json
├── docs/                                        ← 페르소나/컨텍스트 자료
│   ├── 01_problem_framing.md
│   ├── 02_cognition_and_metacognition.md
│   ├── 03_hybrid_ontology.md
│   └── 04_background.md
├── legacy/
│   └── v1.0-3.5stage/                           ← v1.0 ~ v1.2.2 Gradio Hybrid (보존)
│       ├── README.md
│       ├── axiom_tracker_hybrid.py
│       ├── requirements.txt
│       └── ontology/
└── comparison/                                  ← Pure LLM 베이스라인 (비교용)
    ├── README.md
    └── axiom_tracker_pure_llm.py
```

---

## 변경 이력

### v1.2.4 (현재) — 4단계 도구 완성 + baseline 복귀

본 시스템이 *맥락 정합성 평가기*임을 가중치 차원에서 명시한 버전. 직전 v1.2.2까지의 *3.5단계 Gradio Hybrid*를 *4단계 Streamlit 분석 환경*으로 진화.

**4단계 도구 신설**:
- N개 기사 묶음 시계열 분석 + 변곡점 마커 + RuleSchema 히트맵
- Sparse/Dense 이중 RAG Evidence Panel (병렬 비교)
- 수동 시뮬레이터 (5차원 features + 5가중치 + 5프리셋)
- OWL Reasoning Trace Table (논리 사슬 추적)

**axiom 엔진 흡수**:
- v1.2.2의 audit_temporal_pair (4단계 검증 체인) 흡수
- 연속 감점 (compute_temporal_shift_penalty + compute_rule_penalty)
- fired_rules 메타 7필드 (frame_definition_ko, schema_description_ko 등)
- sync_frames_with_rules (OWL↔JSON 어휘 동기화)
- compute_weighted_distortion (per_dim_cap 정규화)

**baseline 복귀 + explanation mitigation**:
- OWL baseline 가중치 복귀: 0.34/0.24/0.18/0.14/0.10 (5차원 공동 판단)
- per_dim_cap 40 (Major reversal 단독 시 *주의 직전 영역* 유지)
- explanation mitigation 통합: SilentPivot/RetroactiveReframing 등 시계열 입장 변경 관련 frame의 graph/rule penalty를 *변경 사유 충실 시 부분 완화* (Δ값 자체는 보존)
- 시계열 엄격 모드 (v1.2.2의 0.40/cap 45)는 시뮬레이터 프리셋으로 유지

**구조 정리**:
- `legacy/v1.0-3.5stage/`에 v1.0~v1.2.2 Gradio Hybrid 보존
- `build_reasoning_trace` 등 *3층 분리 원칙* 정합 리팩토링

### v1.0 ~ v1.2.2 (3.5단계, legacy) — Gradio Hybrid

OWL + JSON + Python 3축 구조의 초기 구현. 과거/현재 두 텍스트의 단발 비교 + GPT-4o 추출기 + OWL 그래프 추론 4단계.

자세한 변경 이력은 [`legacy/로드맵_3단계/v1.1 ~ v1.2.2(3.5단계)/README.md`](./legacy/로드맵_3단계/v1.1 ~ v1.2.2(3.5단계)/README.md) 참조.

---

## 한계와 다음 단계

본 버전은 **시연용 프로토타입**이다. 명시적 한계:

- **`conflictsWith` 관계의 임시성** — 21개 Value↔Frame 충돌 관계는 합리적 가정으로 정의된 것이지, *집단지성으로 검증된* 것이 아니다.
- **`layer_social_meaning_proxy`는 임의 파라미터** — 본래 비전은 *집단지성의 주기적 측정*에 기반한 가중치이지만, 본 프로토타입에서는 임시 대체하고 있다. *수동 시뮬레이터로 사용자가 직접 조작 가능*하다.
- **`dimension_weights` (0.34/0.24/0.18/0.14/0.10)와 `per_dim_cap=40`은 튜닝 파라미터** — v1.2.4까지 baseline 복귀 + explanation mitigation 통합을 거쳐 정립된 잠정값이지만, 실측 보도 코퍼스로 검증된 값이 아니다.
- **M04 VictimBlaming 부재** — OWL 어휘로는 존재하나 룰셋에는 미포함. 민감 도메인 가이드라인 정립 후 추가 예정.
- **휴리스틱 ExtractedSymbol** — `audit_temporal_pair`의 기본 추출기는 *키워드 cue 기반*이라 정확도 제한적. LLM 추출 활성화 시 향상.
- **OpenAI API 의존** (선택) — 휴리스틱으로도 작동하지만, LLM 추출 시 클라우드 API 의존. 향후 로컬 sLLM 옵션 강화 검토.

### 로드맵

- [x] **1단계 — 문제의식과 페르소나 정립**
  - 페르소나/컨텍스트 자료 4종 정리 (`docs/`)
  - 부동성 기준층의 비전 정의

- [x] **2단계 — 온톨로지 아키텍처 공개**
  - OWL 온톨로지 (17 스키마, 6 ValueAnchor, 4 AxiomLayer, 20 Frame 등)
  - 1024개 외부 룰셋 JSON
  - Value↔Frame conflictsWith 관계 + Value↔Value reinforces 관계

- [x] **3단계 — Hybrid 데모 앱 공개 (legacy)**
  - GPT-4o + OWL 그래프 추론 + 룰 매칭의 3축 구조
  - 4단계 검증 체인 (Validity / Polarity / Graph / Rule)
  - Gradio 인터페이스

- [x] **3.5단계 — 5차원 평가 + 시계열 중심 보정 + 연속 감점 + dimension 정합화 (v1.1 ~ v1.2.2, legacy)**
  - `evidence_quality` 차원 신설 및 `dimension_weights` 도입
  - 1024개 룰셋 (M03·M05·M06 확장)
  - `weighted_distortion` 정규화 + 연속 감점
  - `sync_frames_with_rules` 런타임 동기화
  - JSON + OWL + Python 3층 동시 갱신 + ValueAnchor↔dimension 매핑 정합성 정리

- [x] **4단계 — 기사 묶음 일괄 분석 (Streamlit, v1.2.4)**
  - 5차원 지표 + 1024 룰 + Plotly 시계열 sentiment 그래프 + 변곡점 마커 + RuleSchema 히트맵
  - Dense/Sparse 이중 RAG (병렬 Evidence Panel + 교집합/차집합 대비)
  - 수동 시뮬레이터 (5features + 5weights + 5프리셋)
  - OWL 계층 시각화 (Mermaid) + Reasoning Trace Table
  - baseline 복귀 (0.34/cap 40) + explanation mitigation 통합

- [ ] **5단계 — 집단지성 측정층 보강 방향**
  - `layer_social_meaning_proxy`의 임의 파라미터 → 실측 데이터 점진 대체 방향 제안
  - `dimension_weights`와 `per_dim_cap`의 실측 보정
  - 단일 매체 시계열 → 교차 매체 비교 확장
  - M04 VictimBlaming 등 보류 프레임의 룰셋 추가
  - Interactive OWL graph (streamlit-agraph) — 고도화 옵션

---

## 라이선스

본 프로젝트는 [MIT License](./LICENSE)로 배포된다.

OWL 온톨로지 및 1024 룰셋은 본 프로젝트가 처음 공개하는 자산이며, 동일 라이선스 하에 자유롭게 활용 가능하다. 학술·시민사회·언론 영역에서의 활용을 환영한다.

## 기여 / 문의

이슈 또는 토론은 GitHub Issues로. 페르소나 자료에 대한 피드백, OWL 관계의 추가/수정 제안, 룰셋 보강 등은 모두 환영한다.

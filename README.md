# Context-Sync Curator

> *LLM이 OWL 통제 어휘 안에서 추출한 담론 구조를, 1024개 JSON 룰로 검증하고, axiom 5차원 정규화 점수로 환산하는 **ontology-calibrated reasoning** 시스템.*

**Live Demo**: <https://context-sync-curator1.streamlit.app/>  
**GitHub Repository**: <https://github.com/downteam7-crypto/context-sync.curator1>

**현재 버전: v2.2.1** (4단계 Streamlit — 기사 묶음 일괄 분석 + Cloud Dense-OpenAI RAG + Readability Patch) · **Streamlit Community Cloud 2차 배포는 경량 모드** · 이전 3.5단계 Gradio Hybrid는 [`legacy/로드맵_3단계`](./legacy/로드맵_3단계/)에 보존

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
- **외부 룰셋** (`ontology/news_rules_1024.json`): 위 스키마에 속하는 **1024개의 실무 검출 규칙**을 별도 파일로 관리한다 (데이터셋 버전 `v2.1.6-five-level-verdict-consistent`).
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

### 5차원 평가와 dimension_weights (v2.0 baseline)

```
final_distortion =
    0.34 × temporal_shift         ← 시계열 정합성 (5차원 중 최고 가중치)
  + 0.24 × frame_effect
  + 0.18 × context_omission
  + 0.14 × consensus_deviation
  + 0.10 × evidence_quality

per_dim_cap = 45 (정규화 상한)
```

본 시스템은 *시계열 변화를 가장 무겁게* 보되, *0.34라는 가중치*가 *나머지 4차원의 합 0.66*과 균형을 이루도록 한다. 이게 *"맥락 정합성 평가기"* 정체성의 코드적 표현이다.

**시계열 엄격 모드** (0.40 / 0.22 / 0.15 / 0.13 / 0.10, cap=45)는 수동 시뮬레이터의 *프리셋*으로 유지되어, 슬라이더로 즉시 전환 가능하다.

### Frame-level soft cap + effective cap mitigation (v2.1)

`per_dim_cap=45`는 차원 단위의 최종 정규화 상한이다. 그러나 동일 프레임에 속한 유사 룰이 여러 개 함께 발화하면, `dimension_breakdown`에 들어가기 전 raw penalty가 이미 -100~-190처럼 폭주해 cap 40/45/60의 차이가 모두 사라질 수 있다.

이를 막기 위해 v2.1의 Stage 4 룰 집계는 다음 2층 구조를 따른다.

```text
rule-level evidence: fired_rules에 모두 보존
score-level penalty: (target_frame, dimension) 단위 frame_penalty_groups로 묶어 capped_penalty만 반영
```

여기에 더해 `effective_frame_cap = base_frame_cap × mitigation_factor`를 적용한다. 즉 설명 충실도가 엄격 기준을 통과한 경우, 룰 페널티만 줄이는 것이 아니라 해당 explanation-sensitive frame의 최대 부가 페널티 자체도 낮춘다. 이로써 *설명 없는 major shift*와 *설명 충실한 major shift*가 점수상 분리된다.

예를 들어 `RetroactiveReframing` critical 룰 10개가 각각 -16점대 페널티를 내더라도, 이를 10개의 독립 위반으로 보지 않고 *RetroactiveReframing 프레임이 강하게 감지됨*으로 해석한다. 따라서 `raw_rule_sum`은 설명 테이블에 남기되, 실제 점수 계산에는 `effective_frame_cap`을 거친 `capped_penalty`만 반영한다.

기본 soft cap:

| frame | base cap | 해석 |
|---|---:|---|
| `SilentPivot` | 18 | 설명 없는 입장 전환 |
| `RetroactiveReframing` | 18 | 과거 의미의 소급 재구성 |
| `SelectiveMemory` | 14 | 유리한 과거만 선택 기억 |
| `ContextOmission` | 14 | 필수 맥락 생략 |
| `FalseBalance` | 14 | 근거 수준이 다른 주장의 기계적 병렬 |
| `CrisisInflation` | 12 | 위기·충격 프레임 과잉 |
| `VictimBlaming` | 18 | 피해자 책임 전가 프레임 |
| 기타 | 12 | 기본 제한값 |

`18`은 임의의 큰 값이 아니라 `critical` 단일 룰의 기본 최대 감점(`SEVERITY_BASE=18`)과 같은 값이다. 즉 동일 프레임 반복 증거는 *최강 단일 위반 이상으로 무한 누적하지 않는다*는 뜻이다.

### Explanation Mitigation (v2.1)

Major polarity shift(완전 입장 반전)가 *설명 충실한 정당한 반전*인지 *silent pivot*인지를 구분하는 메커니즘.

- **Δ값 자체는 보존** — *"입장이 얼마나 바뀌었는가"*는 양적 측정으로 그대로 둔다.
- **부가 페널티와 cap만 완화** — `SilentPivot`, `RetroactiveReframing`, `SelectiveMemory`, `ContextOmission` 등 explanation-sensitive frame의 rule penalty와 `effective_frame_cap`만 조정한다.
- **충분한 설명 기준은 엄격하게 적용** — 단순히 "상황이 바뀌었다" 수준의 표현은 weak 또는 moderate에 머물며, strong은 과거 기준 인지와 새 근거/조건 변화가 함께 드러날 때만 적용한다.

설명 수준별 factor:

| level | factor | 의미 | temporal 예시: major shift -30 + frame cap 18 |
|---|---:|---|---:|
| `none` | 1.00 | 설명 거의 없음 | 48 → temporal 상한 34점 |
| `weak` | 0.90 | 형식적 설명 신호 | 46.2 → temporal 상한 34점 |
| `moderate` | 0.75 | 일부 사유 설명 | 43.5 → 약 32.9점 |
| `strong` | 0.50 | 과거 기준/조건 변화/새 근거를 추적 가능 | 39 → 약 29.5점 |

따라서 입장이 선명히 바뀌었는데 설명이 부족하면 정합선 아래로 내려가고, 설명이 충분한 경우에만 정합 판정 하단으로 복귀할 수 있다.

### 5단계 판정 농도 모델 (v2.1)

본 시스템은 *통과/주의/이탈*의 3단계 신호등을 넘어, **정합성의 농도**를 5단계로 보여준다. 이분법적 판정이 아니라 *경계의 농도*를 본다는 페르소나 사상의 UI적 구현이다.

| 정합성 점수 (사용자) | 왜곡도 (내부) | 표시 라벨 | 의미 |
|---|---|---|---|
| 85~100 | 0~15 | **안정적 정합** | 시계열·맥락·증거 구조가 대체로 안정적 |
| 70~84 | 16~30 | **기준 부합** | 기준 안에 있으나 일부 경계 신호 가능 |
| 55~69 | 31~45 | **주의 필요** | 복수 차원에서 왜곡 신호 감지 |
| 40~54 | 46~60 | **중점 검토 필요** | 기준 이탈 직전의 고위험 구간 — 자동 판정만으로 넘기지 말고 사람이 근거를 집중 검토 |
| 0~39 | 61~100 | **기준 이탈** | 기준층에서 명백한 이탈 또는 red-card |

*"중점 검토 필요"* 라벨은 *경고문*이 아니라 *분석 도구의 겸손한 위임* — 시스템이 단정하지 않고 사람에게 검토를 넘긴다. 본 시스템의 *자기 한계 인정* 사상과 정합한다.

판정 농도가 살아나는 흐름:
- *Major polarity shift 단독* → 주의 필요 (제목 프레임 + 룰 발화 반영 시)
- *추가 맥락 생략/룰 발화* → 중점 검토 필요로 하락
- *심각한 복합 왜곡 또는 red-card* → 기준 이탈

내부 계산은 *왜곡도(distortion)* 기준, 사용자 표시는 *정합성 점수(coherence = 100 - distortion)* 기준 — 표현 층과 계산 층의 분리.

### 제목·부제 가중치 (v2.1)

본문은 무난하게 쓰되 *제목에서 프레임을 세게 거는* 보도 관행을 반영한다. 제목/부제/본문의 cue를 차등 가중한다.

```
TITLE_CUE_WEIGHT    = 1.5   ← 제목: 독자 프레임 인식에 강하게 작동
SUBTITLE_CUE_WEIGHT = 1.25  ← 부제/요약: 제목보다 약하나 본문보다 강함
BODY_CUE_WEIGHT     = 1.0   ← 본문: 기준값
```

2배 이상은 의도적으로 주지 않았다 — *제목 낚시성 표현을 잡되, 표현 하나가 전체 판정을 과도하게 흔들지 않도록* 한 절충값. 제목에 `위기`, `충격`, `논란`, `졸속`, `강행` 같은 프레임 cue가 들어가면 본문보다 강하게 반영된다.

부제는 JSON의 `subtitle` 필드(없으면 `summary`/`description`/`lead` fallback)로 입력하거나, URL 자동 추출 시 `og:description`·`twitter:description` 메타태그에서 인식한다.

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

#### ② RAG Evidence Panel — Dense 엔진 이원화 (Cloud: OpenAI / Local: ko-sroberta)

같은 기사에 **Sparse(Jaccard 키워드)**와 **Dense(의미 임베딩)**를 동시에 실행해 어떤 룰을 가져오는지 *나란히* 비교한다. 두 검색의 교집합/차집합 차이 *자체가 정보*다.

Dense 엔진은 실행 환경에 따라 둘로 나뉜다.

- **Local — ko-sroberta**: `requirements-sllm.txt` + `STREAMLIT_CLOUD=0`으로 실행하면 `sentence-transformers` 기반 한국어 임베딩(`jhgan/ko-sroberta-multitask`)으로 Dense RAG를 수행한다.
- **Cloud — Dense-OpenAI**: Streamlit Community Cloud 배포판에서는 `sentence-transformers`/`torch`를 포함하지 않지만, **OpenAI API Key가 있으면 `text-embedding-3-small` 기반 Dense-OpenAI 비교가 가능하다.** 키가 없으면 자동으로 *Sparse(Jaccard) 단독*으로 폴백한다.
- **출처 표시**: 반환값 `dense_engine`(`ko-sroberta` / `text-embedding-3-small`)과 `rag_mode`(`dense_ko_sroberta` / `dense_openai` / `sparse_jaccard`)로 *어느 엔진이 쓰였는지* UI에 명시한다. 두 엔진은 *의미 공간이 다르므로* cosine 분포·임계값을 같은 의미로 해석하지 않는다.
- **cutoff 슬라이더**: 사이드바의 *Dense RAG min similarity cutoff* 슬라이더로 `min_dense_score`를 직접 조작한다. 기본값 0.15는 ko-sroberta 기준 잠정값이며, OpenAI 임베딩에서는 *실측 보정 대상*이다 (→ `tools/calibrate_min_dense_score.py`).

#### ③ 수동 시뮬레이터 (의미값 파라미터 직접 조작)

- 5개 차원 슬라이더 (features 조작 모드)
- *5개 가중치 슬라이더 + 합 검증* (weights 조작 모드)
- 5개 프리셋: **v2.0 baseline (OWL 기본값)**, **시계열 엄격 (legacy 3.5단계 모드)**, 사실 우선, 프레임 우선, 다원 이성
- 차원별 기여도 분해 표시
- **Dense RAG cutoff 슬라이더** (사이드바 전역 설정): `min_dense_score`를 0.00~0.90 범위에서 직접 조작. cutoff를 올리면 Evidence Panel의 Dense 후보가 줄어드는 것을 *실시간으로* 확인할 수 있다. (단 LLM 추출용 RAG는 후보 부족 시 top_k로 폴백 — 아래 처리 단계 참조)

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
  · LLM 추출 시 RAG로 관련 룰을 프롬프트에 주입. Dense cutoff가 높아 통과 후보가
    top_k 미만이면 unfiltered top_k로 폴백 → LLM이 룰 맥락 없이 추출하는 일을 방지
    (Evidence Panel 표시용 Dense는 cutoff를 엄격 적용 / LLM 입력용은 맥락 보존, 두 경로 분리)
Stage 2: polarity_shift 연속 감점 (compute_temporal_shift_penalty)
Stage 3: OWL 그래프 추론
  3a. 과거 Value ↔ 현재 Frame conflictsWith (cross-temporal, explanation_factor 적용)
  3b. 가치 이동 reinforces 여부 (강조점 이동 vs 구조 이동)
  자기 모순: 현재 Value ↔ 현재 Frame
Stage 4: 1024 룰 매칭 (continuous scoring, explanation 완화)
   ↓
frame_penalty_groups 집계 (target_frame × dimension soft cap)
   ↓
dimension_breakdown 누적
   ↓
compute_weighted_distortion (per_dim_cap=45 정규화)
   ↓
final_distortion + verdict + reasoning_trace
```

---

## Streamlit Community Cloud 배포 안내 (2차 배포 기준)

본 저장소는 Streamlit Community Cloud에서 바로 실행 가능한 **경량 웹 배포판**을 기준으로 정리되어 있다.

- Live Demo: <https://context-sync-curator1.streamlit.app/>
- GitHub 저장소: <https://github.com/downteam7-crypto/context-sync.curator1>
- Cloud main file path: `app.py`
- 기본 의존성: `requirements.txt`
- 테마/업로드 설정: `.streamlit/config.toml`
- OpenAI API Key: GitHub에 올리지 않고 Streamlit Cloud의 **Secrets**에 등록
- Dense RAG: Cloud에서는 **OpenAI 임베딩(`text-embedding-3-small`) 기반 Dense-OpenAI**를 API Key가 있을 때 제공. `sentence-transformers`/`torch`/로컬 ko-sroberta Dense·2D 시각화·로컬 sLLM은 Cloud 기본 배포에서 제외하고 로컬 확장 옵션으로 분리

### Cloud 배포 범위

Cloud 배포판은 다음 기능을 우선 제공한다.

- OWL 온톨로지 로드
- 1024개 JSON 룰셋 로드
- `rule_engine.py` 기반 5차원 정규화 점수 계산
- frame-level soft cap + effective frame cap mitigation
- Streamlit UI
- URL/JSON/직접 입력 기반 기사 분석
- OpenAI API Key 입력 또는 Secrets 기반 LLM 추출기
- Sparse/Jaccard 기반 룰 근거 탐색
- **Dense-OpenAI RAG** (`text-embedding-3-small`) — OpenAI API Key가 있을 때 활성화. Sparse vs Dense-OpenAI 병렬 비교 + cutoff 슬라이더 제공

Cloud 배포판에서 의도적으로 제외하는 기능 (로컬 확장 옵션):

- 로컬 ko-sroberta Dense (`sentence-transformers` / `torch`)
- 의미 공간(Embedding Space) 2D 시각화
- 로컬 sLLM inference
- 한국어 임베딩 모델 자동 다운로드

이 기능들은 `sentence-transformers`·`torch`·임베딩 모델 다운로드를 요구해 Cloud 초기 배포를 무겁게 만들므로 제외한다. **단 Dense RAG 자체는 Cloud에서 사라진 것이 아니라, ko-sroberta(로컬) 대신 OpenAI 임베딩(Cloud)으로 엔진이 바뀌는 것이다.** 두 엔진은 의미 공간이 다르므로 `min_dense_score`를 동일하게 해석하지 않으며, OpenAI Dense의 cutoff는 `tools/calibrate_min_dense_score.py`로 실측 보정하는 것을 권장한다.

> **재현성 주의**: LLM 추출기 ON + Cloud + OpenAI Key 조합에서는, RAG 후보 룰 선택에 Dense-OpenAI가 우선 사용될 수 있다. 단, 키가 없거나 Dense 후보가 충분하지 않으면 Sparse 또는 unfiltered `top_k` 폴백이 작동한다. 따라서 동일 기사라도 환경(로컬 ko-sroberta / Cloud OpenAI / Sparse)과 cutoff 설정에 따라 feature 추출 결과가 달라질 수 있다. 이는 의도된 개선이지만, 결과 비교 시 실행 환경을 함께 기록해야 한다.

### Streamlit Cloud Secrets 예시

Streamlit Cloud의 앱 설정 → **Secrets**에 다음처럼 입력한다.

```toml
OPENAI_API_KEY = "sk-..."
```

API Key를 GitHub 코드, README, `.env` 파일에 직접 올리지 않는다.

### 배포 절차 요약

1. GitHub 저장소에 파일을 push한다.
2. Streamlit Community Cloud에서 **New app**을 선택한다.
3. Repository: `downteam7-crypto/context-sync.curator1`
4. Branch: `main`
5. Main file path: `app.py`
6. Advanced settings에서 필요한 경우 Python 버전을 선택한다.
7. Secrets에 `OPENAI_API_KEY`를 등록한다.
8. Deploy를 실행한다.

Cloud에서 문제가 생기면 우선 다음을 확인한다.

- `ontology/context_sync_app_centered_ontology_1024.owl` 존재 여부
- `ontology/news_rules_1024.json` 존재 여부
- `requirements.txt`에 필요한 경량 의존성이 들어 있는지
- `.streamlit/config.toml` 경로가 정확한지
- API Key를 코드가 아니라 Secrets에 넣었는지


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

`requirements.txt`는 Cloud 배포를 고려한 경량 의존성이다. Dense RAG, `sentence-transformers`, `torch`, 로컬 sLLM은 포함하지 않는다.

### 3. (선택) OpenAI API 키 설정

LLM 추출기를 사용하려면:

```bash
cp .env.example .env
# 편집기로 .env 열어서 OPENAI_API_KEY 입력
```

또는 앱 사이드바에서 직접 입력 가능. LLM 없이도 *휴리스틱 추출기*로 모든 기능 작동. 이 키는 LLM 추출뿐 아니라 **Cloud 환경의 Dense-OpenAI RAG**(`text-embedding-3-small`)에도 사용된다 — 키가 없으면 Dense는 Sparse로 자동 폴백.

### 4. Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 열리거나 `http://localhost:8501` 접속.

### 5. (선택) 로컬 고급 모드: ko-sroberta Dense RAG / sLLM / 2D 시각화

Cloud 배포판에서는 Dense RAG가 OpenAI 임베딩으로 제공되지만(API Key 필요), **로컬 ko-sroberta Dense**와 의미 공간 2D 시각화·로컬 sLLM은 `sentence-transformers`/`torch`를 요구해 기본 포함하지 않는다. 이들을 로컬에서 실험하려면 별도 의존성을 설치한다.

```bash
pip install -r requirements-sllm.txt
```

그 뒤 로컬 환경에서 실행한다.

```bash
STREAMLIT_CLOUD=0 streamlit run app.py
```

Windows PowerShell에서는 다음처럼 실행할 수 있다.

```powershell
$env:STREAMLIT_CLOUD="0"
streamlit run app.py
```

Dense RAG를 켜면 한국어 임베딩 모델이 자동 다운로드될 수 있으며, 환경에 따라 수백 MB 이상의 추가 용량과 긴 초기 로딩 시간이 필요하다.

---

## 사용 방법

### 입력

**N개 기사 묶음** — 세 가지 입력 방식:

1. **JSON 직접 입력**:

```json
[
  {
    "title": "기사 제목",
    "subtitle": "부제 또는 요약 (선택)",
    "body": "본문",
    "date": "2024-01-15",
    "outlet": "○○일보",
    "topic": "A정책"
  },
  ...
]
```

`subtitle`이 없으면 `summary`/`description`/`lead`를 순서대로 fallback한다.

2. **직접 입력 폼**: 사이드바에서 제목/부제/본문/매체/날짜/사안을 1개씩 누적
3. **URL 자동 추출**: 기사 URL 입력 → title/body/outlet/date + `og:description` 부제를 휴리스틱 추출 (BeautifulSoup 필요)

같은 `outlet` + `topic` 조합이 *시계열 그룹*으로 묶여 비교된다. 제목·부제의 프레임 cue는 본문보다 강하게 가중된다 (title 1.5 / subtitle 1.25 / body 1.0).

### 출력 패널

- **Executive Summary**: 정합성 점수와 최종 왜곡도뿐 아니라, 대표 변곡 구간, 주요 감점 차원, 주요 발화 프레임, 설명 완화 여부를 상단에서 먼저 요약
- **차원별 기여도 분해**: 5차원 dimension_breakdown을 기여 distortion 큰 순서로 정렬해 표시. 고왜곡 케이스에서 최종 점수는 100으로 클리핑될 수 있으므로, 표의 기여도 합은 *원인 비중 파악용*으로 해석
- **시계열 그래프**: sentiment + 변곡점 마커
- **OWL Reasoning Trace**: 긴 trace 원본 표를 바로 펼치기보다, 시계열 변화 / OWL 가치구조 / 룰·페널티 반영 단계로 요약한 뒤 상세 표는 접힘 처리
- **Axiom Graph Audit Reasons**: 동일한 설명 완화 로그는 factor와 전후 penalty 값이 같은 룰끼리 묶어 표시. rule_id만 다른 반복 로그는 상세 목록으로 접음
- **actual fired_rules 요약**: rule_id 단위 반복표 대신 `target_frame × dimension` 기준으로 묶고, raw penalty / capped penalty / suppressed penalty를 분리 표시. signed penalty와 양수 왜곡 반영량을 함께 보여 혼선을 줄임
- **RAG Evidence Panel**: Sparse + Dense 병렬 비교. Dense 엔진은 Cloud=OpenAI(`text-embedding-3-small`, 키 필요) / 로컬 고급 모드=ko-sroberta. 키 없으면 Sparse 단독
- **candidate_rules 보조 표**: Feature/RAG 기반 후보 규칙은 최종 점수의 직접 근거인 actual fired_rules와 분리하고, 기본 접힘 상태에서 `schema_id × target_frame × dimension` 묶음 요약으로 표시
- **OWL 계층 시각화**: Mermaid 다이어그램
- **수동 시뮬레이터**: 차원/가중치 슬라이더

---

## 폴더 구조

```
context-sync.curator1/
├── README.md
├── LICENSE
├── requirements.txt                              ← Streamlit Cloud 경량 의존성
├── requirements-sllm.txt                         ← 로컬 Dense RAG/sLLM 선택 의존성
├── .streamlit/
│   └── config.toml                               ← Cloud 테마/업로드 설정
├── .gitignore
├── .env.example
├── app.py                                       ← Streamlit 메인 앱
├── rule_engine.py                               ← axiom audit 엔진 (3층 분리의 계산 층)
├── sllm_extractor.py                            ← LLM/sLLM 추출기 + Dense(ko-sroberta/OpenAI)/Sparse RAG
├── tools/
│   └── calibrate_min_dense_score.py             ← OpenAI Dense cutoff 진단·보정 도구 (측정용, 앱 자동반영 없음)
├── ontology/
│   ├── context_sync_app_centered_ontology_1024.owl
│   └── news_rules_1024.json
├── docs/                                        ← 페르소나/컨텍스트 자료
│   ├── 01_problem_framing.md
│   ├── 02_cognition_and_metacognition.md
│   ├── 03_hybrid_ontology.md
│   └── 04_background.md
├── legacy/로드맵_3단계
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

> **버전 체계 안내**: 4단계 Streamlit 분석 도구는 *3.5단계까지의 Gradio Hybrid(legacy v1.x)와 별개의 메이저 라인*으로, **v2.0부터 새로 시작**한다. 즉 *4단계 진입 = v2.0*. legacy의 v1.0~v1.2.2는 `legacy/로드맵_3단계/`에 보존된다. (룰셋/OWL 데이터셋 내부 버전인 `v2.1.6-...`은 *데이터셋 자체의 일련 번호*이며 앱 버전과 별개 네임스페이스다.)

### v2.2.1 (현재) — Readability Patch: Executive Summary + compact audit UI

v2.2.1은 v2.2의 Cloud Dense-OpenAI RAG 구조와 v2.1의 점수·판정 구조를 유지한 상태에서, **분석 결과를 사람이 더 빨리 읽을 수 있도록 정리한 UI/가독성 패치**다.

> **변경 범위 (3층 분리 관점)**: OWL 어휘, 1024개 JSON 룰셋, 5차원 점수 계산식, `per_dim_cap=45`, frame-level soft cap 로직은 변경하지 않는다. 변경은 `app.py`의 렌더링 계층에 집중되어 있으며, 같은 계산 결과를 더 적은 중복과 더 명확한 위계로 보여주는 데 목적이 있다.

- **Executive Summary 추가**: 대표 변곡 구간, 주요 감점 차원, 주요 발화 프레임, 설명 완화 여부를 상단에서 요약한다.
- **대표 audit 선택 안정화**: `primary_graph_audit`을 우선 사용하고, 없을 때만 `graph_audits[0]`로 fallback한다.
- **차원별 기여도 정렬**: dimension breakdown을 기여 distortion 큰 순서로 표시하고, 고왜곡 케이스에서 최종 왜곡도 100 클리핑과 기여도 합이 다를 수 있음을 안내한다.
- **OWL Reasoning Trace 요약화**: 긴 trace 표를 바로 노출하지 않고, 시계열 변화 / OWL 가치구조 / 룰·페널티 반영 단계로 먼저 요약한다. 원본 stage 라벨은 렌더링 계층에서 한국어로 표시한다.
- **Audit reasons compact rendering**: `factor=0.9`, `-12.3 → -11.1`처럼 같은 형식의 설명 완화 로그를 하나로 묶고, rule_id 목록은 상세 접힘으로 이동한다.
- **actual fired_rules 프레임 단위 집계**: rule_id만 다른 반복 행을 `target_frame × dimension` 단위로 묶고, `raw_rule_sum`, `capped_penalty`, `suppressed_penalty`를 분리 표시한다.
- **signed penalty / positive contribution 분리**: 엔진 내부의 음수 penalty와 사용자 해석용 양수 왜곡 반영량을 함께 표시해 UI 혼선을 줄인다.
- **candidate_rules 기본 접힘 처리**: 최종 점수 직접 근거인 actual fired_rules와 Feature/RAG 기반 후보 규칙을 분리하고, 후보 규칙은 `schema_id × target_frame × dimension` 요약을 먼저 보여준다.

### v2.2 — Cloud Dense-OpenAI RAG + cutoff calibration

v2.2는 v2.1의 판정 구조를 유지한 상태에서, **Cloud 배포 환경의 RAG 근거 탐색을 보강한 버전**이다. v2.1이 *점수·판정 구조 정교화*라면, v2.2는 *Cloud/Local Dense 엔진 분리와 OpenAI 임베딩 기반 RAG 확장*에 초점을 둔다.

> **변경 범위 (3층 분리 관점)**: v2.2는 OWL 어휘·1024 룰셋·5차원 점수 계산식(OWL/JSON 점수층)을 *변경하지 않는다*. 변경은 RAG 근거 탐색과 Cloud 배포 인프라(앱/추출기 층, `app.py`·`sllm_extractor.py`)에 한정된다. 즉 *판정 사상은 불변, 근거 탐색 인프라만 보강*한 버전이다. 동일 입력의 5차원 점수·verdict는 RAG 엔진과 무관하게 유지되며, 다만 LLM 추출 ON 시 프롬프트에 주입되는 RAG 후보가 엔진(Sparse / ko-sroberta / OpenAI)에 따라 달라져 feature 추출 결과가 달라질 수 있다.

- **Cloud Dense-OpenAI RAG 도입**: Cloud 배포판에서 OpenAI API Key가 있으면 `text-embedding-3-small` 임베딩으로 Dense RAG를 수행한다. 기존 Cloud의 Sparse-only 한계를 줄이고, Sparse + Dense-OpenAI 병렬 비교를 제공한다.
- **`compare_rag_results`·`search_relevant_rules` 양쪽에 OpenAI Dense 경로 추가**: Evidence Panel의 Dense 비교뿐 아니라, LLM feature 추출 프롬프트에 주입되는 RAG 후보 규칙도 Cloud에서는 OpenAI Dense를 사용할 수 있다.
- **Dense 엔진 출처 명시**: `dense_engine`(`ko-sroberta` / `text-embedding-3-small`)과 `rag_mode`(`dense_ko_sroberta` / `dense_openai` / `sparse_jaccard`)로 *어느 엔진이 쓰였는지* UI에 표시한다. ko-sroberta와 OpenAI 임베딩은 의미 공간이 다르므로 cosine 값을 같은 의미로 해석하지 않는다.
- **Dense cutoff 슬라이더 추가**: 사이드바 전역 설정에서 `min_dense_score`를 0.00~0.90 범위로 직접 조작한다. 기본값 0.15는 ko-sroberta 기준 잠정값이며, OpenAI Dense에서는 실측 보정 대상이다.
- **Evidence Panel 표시용 cutoff와 LLM 입력용 RAG 경로 분리**: Evidence Panel은 cutoff를 엄격 적용하지만, LLM feature 추출용 RAG는 cutoff 통과 후보가 `top_k` 미만이면 unfiltered `top_k`로 폴백한다. 이로써 슬라이더를 높였을 때 LLM이 룰 맥락 없이 추출하는 문제를 막는다.
- **`dense_path_used` 플래그로 cutoff 오적용 방지**: 실제 Dense 경로가 성공했을 때만 Dense cutoff를 적용하고, Sparse Jaccard 점수에는 Dense용 cutoff를 적용하지 않는다.
- **`tools/calibrate_min_dense_score.py` 추가**: OpenAI Dense cutoff 진단·보정 도구. 기사-룰 유사도 분포의 percentile과, 라벨이 있을 경우 F1 최적 cutoff를 탐색한다. 이 스크립트는 앱 값을 자동으로 바꾸지 않는 *측정 도구*이며, 산출된 후보값을 사용자가 사이드바 슬라이더에 수동 반영하거나 코드 기본값으로 채택할지 판단하는 데 사용한다.
- **Streamlit Community Cloud 2차 배포 정리**: Cloud는 경량 `requirements.txt` 기준으로 배포하고, 로컬 ko-sroberta Dense/sentence-transformers/sLLM은 `requirements-sllm.txt` 기반 로컬 확장 옵션으로 분리한다.

### v2.1 — 5단계 판정 농도 + 제목 가중치 + frame-level soft cap

v2.1은 **점수·판정 구조를 정교화한 버전**이다. 5단계 verdict 농도, 제목·부제 가중치, `per_dim_cap=45`, frame-level soft cap, effective frame cap mitigation을 통해 *major shift 단독*과 *설명 없는 silent pivot*을 점수상 구분한다.

- **5단계 verdict 농도 모델**: 3단계 신호등 → 5단계 농도 (안정적 정합 / 기준 부합 / 주의 필요 / 중점 검토 필요 / 기준 이탈). 정합성 농도를 UI에 반영
- **제목·부제 가중치**: title 1.5 / subtitle 1.25 / body 1.0 — 제목 프레임 효과 반영, 과잉 반응 방지 (2배 미만)
- **URL 자동 추출**: 기사 URL에서 title/body/outlet/date + `og:description` 부제 인식 (BeautifulSoup)
- **부제/요약 필드**: 직접 입력 폼 + JSON `subtitle`/`summary` 필드 + fallback 체인
- **LLM 추출기 반영**: 프롬프트가 TITLE/SUBTITLE/BODY 구조로 입력받고 *"제목·부제는 본문보다 강한 프레임 신호로 고려하되 단일 표현으로 과잉 판정 말라"* 지시
- **per_dim_cap 40 → 45 조정**: major shift 단독(~22.7점)과 설명 없는 silent pivot(부가 페널티가 cap을 채우며 34점 상한 수렴)을 *점수 폭으로 분리*. cap 40은 sensitive 프로파일로 격하 (default 45 / sensitive 40 / conservative 60)
- **effective frame cap mitigation 도입**: 동일 `target_frame` 룰 다발 발화는 `frame_penalty_groups`로 묶어 점수 폭주를 막고, 설명 충실 시 `effective_frame_cap = base_frame_cap × mitigation_factor`로 부가 프레임 페널티의 최대치 자체를 낮춤
- 3층 동시 갱신 (OWL evaluationFormula 5단계 + JSON verdict_thresholds 5단계 + per_dim_cap 45 + Python)

### v2.0 (이전) — 4단계 도구 출범 + baseline 정립

본 시스템이 *맥락 정합성 평가기*임을 가중치 차원에서 명시한 버전. 3.5단계까지의 *Gradio Hybrid*를 *4단계 Streamlit 분석 환경*으로 진화시키며 메이저 라인을 새로 출범.

**4단계 도구 신설**:
- N개 기사 묶음 시계열 분석 + 변곡점 마커 + RuleSchema 히트맵
- Sparse/Dense 이중 RAG Evidence Panel (병렬 비교)
- 수동 시뮬레이터 (5차원 features + 5가중치 + 5프리셋)
- OWL Reasoning Trace Table (논리 사슬 추적)

**axiom 엔진 흡수** (legacy 3.5단계에서):
- audit_temporal_pair (4단계 검증 체인) 흡수
- 연속 감점 (compute_temporal_shift_penalty + compute_rule_penalty)
- fired_rules 메타 7필드 (frame_definition_ko, schema_description_ko 등)
- sync_frames_with_rules (OWL↔JSON 어휘 동기화)
- compute_weighted_distortion (per_dim_cap 정규화)

**baseline 정립 + explanation mitigation**:
- OWL baseline 가중치: 0.34/0.24/0.18/0.14/0.10 (5차원 공동 판단)
- per_dim_cap 40 (Major reversal 단독 시 *주의 직전 영역* 유지)
- explanation mitigation 통합: SilentPivot/RetroactiveReframing 등 시계열 입장 변경 관련 frame의 graph/rule penalty를 *변경 사유 충실 시 부분 완화* (Δ값 자체는 보존)
- 시계열 엄격 모드 (legacy 3.5단계의 0.40/cap 45)는 시뮬레이터 프리셋으로 유지

**구조 정리**:
- `legacy/로드맵_3단계/`에 v1.0~v1.2.2 Gradio Hybrid 보존
- `build_reasoning_trace` 등 *3층 분리 원칙* 정합 리팩토링

### legacy v1.0 ~ v1.2.2 (3.5단계) — Gradio Hybrid

OWL + JSON + Python 3축 구조의 초기 구현. 과거/현재 두 텍스트의 단발 비교 + GPT-4o 추출기 + OWL 그래프 추론 4단계. v1.1~v1.2.2를 거치며 evidence_quality 차원 신설, 1024 룰셋 확장, 연속 감점화, dimension 정합화가 이루어졌다.

자세한 변경 이력은 [`legacy/로드맵_3단계/v1.1 ~ v1.2.2(3.5단계)/README.md`](./legacy/로드맵_3단계/v1.1 ~ v1.2.2(3.5단계)/README.md) 참조.

---

## 한계와 다음 단계

본 버전은 **시연용 프로토타입**이다. 명시적 한계:

- **`conflictsWith` 관계의 임시성** — 21개 Value↔Frame 충돌 관계는 합리적 가정으로 정의된 것이지, *집단지성으로 검증된* 것이 아니다.
- **`layer_social_meaning_proxy`는 임의 파라미터** — 본래 비전은 *집단지성의 주기적 측정*에 기반한 가중치이지만, 본 프로토타입에서는 임시 대체하고 있다. *수동 시뮬레이터로 사용자가 직접 조작 가능*하다.
- **`dimension_weights` (0.34/0.24/0.18/0.14/0.10)와 `per_dim_cap=45`는 튜닝 파라미터** — v2.0~v2.1을 거쳐 정립된 잠정값이지만, 실측 보도 코퍼스로 검증된 값이 아니다.
- **`min_dense_score=0.15`는 ko-sroberta 기준 잠정 cutoff** — OpenAI 임베딩(`text-embedding-3-small`)은 cosine 분포가 달라 0.15가 같은 의미를 갖지 않는다 (무관한 텍스트끼리도 상대적으로 높은 유사도가 나오는 경향). Dense-OpenAI에서는 *실측 보정 대상*이며, `tools/calibrate_min_dense_score.py`로 분포 진단(percentile) 및 라벨 기반 F1 최적 cutoff를 탐색할 수 있다. 현재는 사이드바 슬라이더로 시연 중 직접 조정한다.
- **M04 VictimBlaming 부재** — OWL 어휘로는 존재하나 룰셋에는 미포함. 민감 도메인 가이드라인 정립 후 추가 예정.
- **휴리스틱 ExtractedSymbol** — `audit_temporal_pair`의 기본 추출기는 *키워드 cue 기반*이라 정확도 제한적. LLM 추출 활성화 시 향상.
- **OpenAI API 의존** (선택) — 휴리스틱으로도 작동하지만, LLM 추출 및 Cloud Dense-OpenAI RAG는 클라우드 API에 의존. 향후 로컬 sLLM 옵션 강화 검토.
- **웹 배포판의 Dense 엔진 차이** — Streamlit Community Cloud에서는 로컬 ko-sroberta Dense·2D 시각화·torch 기반 sLLM을 기본 포함하지 않는다. 단 Dense RAG 자체는 OpenAI 임베딩으로 대체 제공되며(키 필요), 로컬 ko-sroberta 확장은 GitHub 저장소를 내려받아 `requirements-sllm.txt`로 별도 설치·실행한다. 로컬과 Cloud의 Dense 엔진은 의미 공간이 달라 결과가 달라질 수 있다.

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

- [x] **4단계 — 기사 묶음 일괄 분석 (Streamlit, v2.0 ~ v2.2.1)**
  - 5차원 지표 + 1024 룰 + Plotly 시계열 sentiment 그래프 + 변곡점 마커 + RuleSchema 히트맵
  - RAG Evidence Panel (Sparse + Dense 병렬 비교). Dense 엔진은 환경별 이원화 — Cloud는 OpenAI 임베딩(`text-embedding-3-small`, 키 필요), 로컬 고급 모드는 ko-sroberta. cutoff 슬라이더 + `tools/calibrate_min_dense_score.py` 보정 도구
  - 수동 시뮬레이터 (5features + 5weights + 5프리셋)
  - OWL 계층 시각화 (Mermaid) + Reasoning Trace Table
  - Executive Summary + compact audit UI: 대표 변곡 구간, frame-level fired_rules 요약, trace 접힘 처리, candidate_rules 분리 표시
  - baseline 복귀 (0.34/cap 45) + explanation mitigation 통합
  - 5단계 verdict 농도 모델 (안정적 정합 → 기준 이탈)
  - 제목·부제 가중치 (title 1.5 / subtitle 1.25 / body 1.0) + URL 자동 추출

- [ ] **5단계 — 사회적 의미 프록시의 실측 보정 (방향 초안)**

  > 5단계는 *완성형*이 아니라 *실측 기반 보정 단계*다. **이 시스템은 5천만 국민 전체의 집단지성을 직접 대표한다고 주장하지 않는다.** 대신 공개 기사 데이터, 매체 간 비교, 사용자 라벨링, 댓글·반응 신호, 언론윤리 기준, 심의 사례 등을 활용하여 *사회적 의미 기준을 근사하는 프록시(proxy)*를 점진적으로 보정한다. 즉 *"집단지성 구현"*이 아니라 *"집단지성의 관찰 가능한 흔적을 체계적으로 반영"*하는 단계다.

  개발 우선순위 (현실적 순서):
  - **(1) 실험 데이터셋 구축** — 같은 사건에 대한 기사 20~50개 수집 + `issue_id`로 묶기 + 사람 라벨링 (frame_effect / context_omission / victim_blaming / overall_verdict)
  - **(2) M04 VictimBlaming 룰셋 추가** — 사회적 위해성이 크고 윤리 기준이 비교적 분명. *단, 문자열 패턴만으로 판정 금지* — `"피해자도 책임이라는 주장은 부적절하다"`처럼 *비판* 문장을 *동조*로 오판할 수 있으므로 LLM/문맥 판별 조건 필수
  - **(3) 교차 매체 비교** — 단일 매체 시계열 → 같은 이슈를 여러 매체가 어떻게 다르게 다루는지 `frame_vector`로 수치화. 기사 하나는 *주관적 판단*처럼 보이지만 *여러 보도 비교*는 프레임 이탈을 분명히 드러냄
  - **(4) Scoring profile 실험 모드** — `dimension_weights`와 `per_dim_cap`을 default(cap 45) / sensitive(cap 40) / conservative(cap 60) / 실측 보정 모드로 제공. *"정답 하나"를 주장하지 않고 평가 목적에 따라 조절*. Dense RAG의 `min_dense_score`도 같은 맥락의 실측 보정 대상이며 `tools/calibrate_min_dense_score.py`의 분포 진단·F1 탐색 결과로 엔진별(ko-sroberta / OpenAI) cutoff를 정한다
  - **(5) `layer_social_meaning_proxy` 외부 신호 점진 반영** — 언론윤리 강령 → 심의·판례 사례 → 댓글·반응 신호 순. *댓글·반응은 "의미 기준"이 아니라 "주의 환기 신호"로만 사용* — 실시간 여론을 그대로 반영하지 않는다는 사상(조작·잡음·선동 취약성)을 보호
  - **(6) Interactive OWL graph** — streamlit-agraph로 *분석에 실제 작동한 ValueAnchor/Rule/Dimension만* 부분 그래프로 표시 (전체 그래프 X). 설명가능성 UI. 단 4단계 Mermaid Reasoning Trace로 상당 부분 충족되므로 *후순위*

  **v2.1에서 반영한 정합성 보정**: 동일 프레임(예: RetroactiveReframing)에 속한 복수 룰이 함께 발화할 때 dimension raw penalty가 과도하게 누적되어 `per_dim_cap` 차이를 무력화하던 문제를 줄이기 위해 `frame-level soft cap`을 도입했고, 설명 충실도가 확인된 경우 `effective_frame_cap`을 낮춰 설명 있는 전환과 설명 없는 silent pivot을 구분한다. 5단계에서는 이 cap 값과 mitigation tier를 실측 데이터셋과 사람 라벨링으로 재보정한다.

---

## 라이선스

본 프로젝트는 [MIT License](./LICENSE)로 배포된다.

OWL 온톨로지 및 1024 룰셋은 본 프로젝트가 처음 공개하는 자산이며, 동일 라이선스 하에 자유롭게 활용 가능하다. 학술·시민사회·언론 영역에서의 활용을 환영한다.

## 기여 / 문의

이슈 또는 토론은 GitHub Issues로. 페르소나 자료에 대한 피드백, OWL 관계의 추가/수정 제안, 룰셋 보강 등은 모두 환영한다.

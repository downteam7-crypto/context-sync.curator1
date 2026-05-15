# Context-Sync Curator

LLM의 맥락 판단 능력 위에 온톨로지의 고정된 기준층을 결합하여, 기사 이면의 시계열 정합성과 논조 이동을 측정하는 프로토타입 프로젝트.

## 프로젝트 개요

본 프로젝트는 단일 기사 내부의 사실관계 검증(전형적 팩트체크)을 넘어, **기사가 시계열과 논조 속에서 형성하는 맥락의 정합성**을 검토하는 데 초점을 둔다.

핵심 구조는 다음과 같다.

- **LLM** : 기사의 맥락과 논조 이동을 판독 (유동성)
- **온톨로지** : 판독이 사회적으로 합의된 보편 기준에서 이탈하지 않도록 정합성에 개입 (부동성)
- **RAG / 그래프 추론** : 두 층위를 실시간으로 연결하는 매개 아키텍처

여기서 **부동성**의 기준층은 단순히 사전적·전문가적 정의에 머무르지 않는다. 본 프로젝트는 그 기준층이 궁극적으로는 잘 정의된 집단지성 — 즉 다수결이라는 기능적 절차에 그치지 않고, 다양한 위치에 놓인 사람들의 감각이 가중치 과다 없이 녹아드는 구조 — 위에 자리잡아야 한다고 본다. 본 프로토타입에서는 이 기준층을 임의 파라미터로 대체하고 있으나, 그 방향성을 시연하는 데에 의의가 있다.

## 온톨로지 아키텍처

본 프로젝트의 부동성 기준층은 **스키마-룰셋 분리 구조**로 설계되어 있다.

- **OWL 온톨로지** (`ontology/context_sync_app_centered_ontology.owl`): 17개의 추상 스키마, 6개의 ValueAnchor, 4개의 AxiomLayer, 5개의 EvaluationDimension, 20개의 Frame, 18개의 Topic 등 **느리게 변하는 보편 어휘**와, 그 어휘 사이의 **그래프 관계**(21개 conflictsWith, 8개 reinforces, 7개 calibratesDimension)를 정의한다.
- **외부 룰셋** (`ontology/news_rules_1024.json`): 위 스키마에 속하는 **1024개의 실무 검출 규칙**을 별도 파일로 관리한다. (v2.1.1-dimension-aligned, 시계열 중심 가중치 보정 및 OWL↔JSON dimension 정합성 반영)

이 분리는 페르소나 자료에서 제기한 *"공리를 80개가 아니라 800개, 8000개로 늘리면 정교해질까"*라는 질문에 대한 공학적 응답이다. 어휘를 무겁게 늘리는 대신, 추상 어휘는 OWL에 안정적으로 두고 실무 규칙은 외부 데이터셋으로 분리해 운용한다.

특히 6개의 ValueAnchor는 페르소나 자료의 핵심 사고 결을 클래스 어휘 수준에 직접 반영한 것이다.

| ValueAnchor | 의미 |
|---|---|
| `StanceConsistency` | 같은 매체가 같은 사안에 대해 논조를 바꿀 때 *설명 책임*을 요구 |
| `FrameAccountability` | 표면 명제가 작동시키는 *실제 사회적 프레임 효과*까지 평가 |
| `ContextCompleteness` | 표면 사실이 전체 맥락을 과도하게 생략하지 않아야 함 |
| `ResponsibilitySeparation` | 잘못의 *원인 귀속*과 *대응 책임*을 분리해 평가 |
| `EvidenceTransparency` | 주장의 근거와 출처가 식별 가능해야 함 |
| `PluralPublicReason` | 단순 다수결이 아니라 다양한 의견이 *구조적*으로 반영되는 기준 |

각 ValueAnchor는 OWL의 `calibratesDimension` 관계로 5개의 EvaluationDimension에 연결된다. 이 매핑이 *Value 위반*이 *어느 점수 차원으로 반영되어야 하는가*를 그래프 추론으로 답한다.

| ValueAnchor | calibrates → EvaluationDimension |
|---|---|
| `StanceConsistency` | `temporal_shift` |
| `FrameAccountability` | `frame_effect` |
| `ContextCompleteness` | `context_omission` |
| `ResponsibilitySeparation` | `frame_effect` + `consensus_deviation` (이중 매핑) |
| `EvidenceTransparency` | `evidence_quality` |
| `PluralPublicReason` | `consensus_deviation` |

이 연결을 통해 앱은 *"이 기사는 ContextCompleteness 기준을 위반했기 때문에 context_omission 차원에서 점수가 깎였다"*는 식의 **그래프 추론으로 도출된 설명 가능한 판정**을 만들 수 있다.

또한 4개의 AxiomLayer 중 `layer_social_meaning_proxy`는, 페르소나 자료에서 비전으로 제시한 **국가 의미 인프라**를 본 프로토타입에서 임시 대체하는 기준층으로 명시되어 있다. 즉 본 프로젝트는 *"지금 무엇을 임시로 대체하고 있고, 궁극적으로 어디로 가야 하는가"*가 어휘 자체에 박혀 있다.

### OWL의 역할: 의미 기준층 vs 계산 엔진

본 OWL은 **점수를 계산하는 엔진이 아니라 의미 기준층**이다. 자주 오해가 발생하는 지점을 명확히 해두면:

- **OWL이 하는 일** — 어휘(클래스, ValueAnchor, Frame, Topic)와 그 관계(`conflictsWith`, `reinforces`, `calibratesDimension`)를 선언한다. 또한 `evaluationFormula`(5차원 가중 공식)와 `dimension_weights` 같은 *수식 메타데이터*를 `RuleEngine` 클래스 위에 *문자열로 보관*한다.
- **OWL이 하지 않는 일** — 점수 계산 자체. OWL은 *어떤 차원으로 어떻게 가중되어야 하는지*를 *선언*하지만, 실제 산수는 Python 측 `audit_logic` 함수와 `compute_weighted_distortion`이 수행한다.
- **외부 JSON 룰셋** — 1024개의 실무 검출 규칙을 보관한다. OWL의 RuleSchema가 추상 어휘를 제공하고, JSON이 그 어휘에 속하는 구체적 룰들을 공급하는 분리.

따라서 본 프로젝트의 부동성 기준층은 단일 파일이 아니라 **OWL(어휘·관계 선언) + JSON(실무 규칙·가중치) + Python(계산 실행)의 3층 분리 구조**다. 이 분리가 *"공리를 단순히 80개에서 800개로, 다시 8000개로 늘리기만 한다면 정교해질까"*에 대한 응답이며, 동시에 *각 층이 무엇을 책임지는지*를 명확히 한다.

### 5차원 평가와 dimension_weights

본 룰셋 v2.1.1 기준으로 평가는 5개 EvaluationDimension의 가중합으로 정의되며, **시계열 정합성을 본 시스템의 정체성으로 명시하는 가중치 분포**가 적용된다. JSON 최상위에 명시된 `dimension_weights`가 Python의 점수 계산에 직접 반영된다.

```
final_distortion =
    0.40 × temporal_shift        ← 시계열 정합성 (본 시스템 정체성)
  + 0.22 × frame_effect
  + 0.15 × context_omission
  + 0.13 × consensus_deviation
  + 0.10 × evidence_quality
```

5번째 차원 `evidence_quality`는 *증거 부재로 인한 왜곡*을 *시간축 변화로 인한 왜곡*과 같은 무게로 다루지 않아야 한다는 자연스러운 분화의 결과다. `EvidenceTransparency` ValueAnchor가 이 차원으로 calibrate된다.

`temporal_shift`의 가중치를 0.40으로 설정한 것은 본 시스템이 *시계열 정합성 검토 도구*임을 가중치 차원에서 선언한 것이다. 다만 0.50을 넘어가면 *5차원 종합 평가*가 *시계열 단독 평가*로 변질될 위험이 있어, *5차원 균형을 보존하는 상한*으로 0.40을 채택했다. 다른 4차원의 합 0.60이 *시계열 단독 결정을 방지*하는 안전장치 역할을 한다.

### JSON 룰셋과 OWL 매핑의 관계

`news_rules_1024.json`의 1024개 룰은 OWL의 어휘를 사용하지만, 두 자원의 *역할이 다르다*는 점을 명시해둔다.

**ValueAnchor 분포** — 1024개 룰의 ValueAnchor 분포는 시계열·구조·메타 갈래에 걸쳐 균형 잡혀 있다.

**dimension 매핑의 정합성** — 각 룰은 명시적 `dimension` 필드를 갖지만, 이 값은 OWL의 `calibratesDimension` 기준층과 충돌하지 않아야 한다. v2.1.1에서는 1024개 룰 전체를 점검해 `ContextCompleteness → context_omission`, `PluralPublicReason → consensus_deviation` 등 ValueAnchor별 기본 매핑과 JSON 룰의 실행 차원을 정렬했다. 본 코드의 `audit_logic`은 JSON 룰의 `dimension` 필드를 우선 사용하고, 비어 있거나 유효하지 않은 경우에만 OWL의 `calibratesDimension`으로 폴백한다. 즉 JSON은 실행용 명시값이고, OWL은 그 명시값의 기준층이다.

**룰셋의 생성 방식** — 1024개 룰은 16개 RuleSchema(T01~T05, S01~S06, M01~M03·M05·M06)와 15개 target_frame, severity 4단계의 조합으로 *체계적으로 생성*되었다. severity는 4단계(low/medium/high/critical)에 256개씩 균등 분포되어 있어, *현실 빈도 기반 가중치가 아닌 시연용 대표 분포*임을 밝혀둔다. 향후 실제 보도 코퍼스 분석을 통한 빈도 보정이 보강 방향이다.

**M04 VictimBlaming의 부재** — OWL에는 정의되어 있으나 1024 룰셋에는 룰이 포함되지 않았다. 피해자 책임 전가 프레임은 *민감 콘텐츠 위험*과 *검출 단서의 미묘함* 때문에 현 단계에서 룰 작성을 보류한 상태이며, M05 ResponsibilityShift가 책임 이전의 일부 케이스를 흡수한다. OWL 어휘로는 *남겨두어* 향후 별도 가이드라인 정립 후 룰 추가가 가능하도록 설계되어 있다.

### OWL과 JSON의 어휘 비대칭 — sync_frames_with_rules

OWL은 *어휘로 풍부*(20개 Frame)하지만, JSON 룰셋은 그 *부분집합*(15개 target_frame)만 다룬다. 본 시스템은 *OWL = 보편 어휘, JSON = 실무 룰*이라는 사상을 따르므로 이 비대칭 자체는 정상이다. OWL-only 프레임(`VictimBlaming`, `CrisisInflation`, `ConflictAmplification`, `FairnessDiscourse`, `NeutralFact`)은 *어휘 자산*으로 보존되며 향후 룰 추가의 거점이 된다.

다만 LLM이 OWL의 모든 Frame을 선택지로 받으면 *JSON에 없는 프레임을 고를 경우 룰이 발화하지 않는* 문제가 생긴다. 본 앱은 기동 시 `sync_frames_with_rules` 함수가 *런타임 LLM 선택지에서만* OWL-only 프레임을 필터링하여 이 간극을 메운다. OWL 파일 자체는 손대지 않는다.

## 페르소나 / 컨텍스트 자료

본 프로젝트의 LLM은 단순한 범용 어시스턴트로 작동하지 않으며, 명시적인 **페르소나와 컨텍스트**를 부여받아 동작한다. 이 페르소나의 사고 결을 형성한 자료는 `docs/` 폴더에 정리되어 있다.

- [`docs/01_problem_framing.md`](./docs/01_problem_framing.md) — 문제 정의: 유동성 vs 부동성
- [`docs/02_cognition_and_metacognition.md`](./docs/02_cognition_and_metacognition.md) — 사고 구조: 일반성·특수성·메타인지
- [`docs/03_hybrid_ontology.md`](./docs/03_hybrid_ontology.md) — 확장된 비전: 하이브리드 온톨로지와 국가 의미 인프라
- [`docs/04_background.md`](./docs/04_background.md) — 개인 배경: 비선형 궤적이 이 문제의식에 닿기까지

이 자료들은 프로젝트의 페르소나가 *왜 이런 방식으로 사안을 보는가*를 설명하기 위한 컨텍스트이며, 동시에 모델링 단계에서 LLM에게 실제로 주입한 시스템 프롬프트의 근거이기도 하다.

---

## Hybrid 데모 앱

`axiom_tracker_hybrid.py`는 본 프로젝트의 시계열 논조 판독 데모 앱이다. **3축 구조**로 작동한다.

```
┌─────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│  유동적 판독     │ ←→ │   매개 (그래프 추론) │ ←→ │  부동성 기준층    │
│                 │    │                     │    │                   │
│  GPT-4o         │    │  rdflib SPARQL +    │    │  OWL Ontology +   │
│  (OpenAI API)   │    │  Rule Matching      │    │  1024 JSON Rules  │
└─────────────────┘    └────────────────────┘    └──────────────────┘
        ↓                       ↓                          ↑
  OWL 어휘로 추출       그래프 관계 + 룰 발화          어휘·관계·룰 공급
```

### 처리 단계

1. **추출 (LLM)** — GPT-4o가 두 텍스트(과거/현재)에서 OWL 어휘로 메타데이터 추출
   - `promoted_value`: 6개 ValueAnchor 중 하나
   - `detected_frame`: 15개 활성 Frame 중 하나 또는 `None` (sync_frames로 정합화)
   - `topic`: 18개 Topic 중 하나
   - `stance_polarity`: -1.0 ~ +1.0 정량 극성
   - `is_valid_discourse`: 사실/윤리 위배 여부 (Red Card)

2. **검증 (Python + OWL/JSON)** — 4단계 검증 체인
   - **Stage 1 — Validity (Red Card)**: 평평한 지구설, 인종 청소 옹호 등 *시계열과 무관한* 즉시 실격
   - **Stage 2 — Stance Polarity Shift**: 입장 극성의 정량적 이동 측정
   - **Stage 3 — OWL Graph Reasoning**:
     - *Cross-temporal*: 과거 옹호 가치를 현재 프레임이 위반하는가 (`conflictsWith` 그래프 쿼리)
     - *Self-contradictory*: 텍스트가 옹호한다고 *주장*하는 가치를 그 텍스트의 *프레임*이 스스로 위반
     - *Reinforces*: 가치 이동이 *강화 관계 안*인지(강조점 이동) *밖*인지(구조적 이동) 판별
   - **Stage 4 — 1024 Rule Matching**: 발화된 frame과 topic에 매치되는 룰을 활성화하고, 각 룰의 `dimension` 필드에 따라 5개 차원에 페널티 분배

3. **리포트 (LLM)** — 발화된 룰을 *증거*로 GPT-4o가 한국어 리치 리포트 생성
   - `verdict`, `summary`, `self_attack`, `steelman`, `evidence_rules`
   - 각 fired_rule은 `frame_definition_ko`, `schema_description_ko`, `expected_evidence_ko`, `score_hint`(increase_when/decrease_when) 등 *프레임별 분화된 텍스트*를 함께 노출하여 *"왜 이 룰이 발화했는가"*를 추적 가능하게 만든다.

### 점수 계산: weighted_distortion (5차원 정규화)

본 버전부터 최종 점수는 *단순 합산*이 아니라 *dimension별 정규화 후 가중합산*으로 계산된다.

```
1. 각 차원의 누적 페널티 절댓값을 0~100 스케일로 정규화 (per_dim_cap=45 상한 클리핑)
2. dimension_weights로 가중합산
3. 0~100 범위로 클리핑

final_score = max(0, 100 - weighted_distortion + validity_score)
```

JSON `normalization_note_ko`의 권고를 반영한 설계다 — *기사 길이 또는 매칭 룰 수에 따른 과대평가*를 피하기 위함이다. 한 차원에 페널티가 폭주해도 다른 차원을 지배하지 못하도록 차원별 상한 클리핑이 적용된다.

### 점수 분리

- `validity_score`: 사실/윤리 위배에 의한 감점 (시계열 무관, Red Card)
- `logic_score`: Stage 2 polarity shift, OWL 충돌, JSON 룰 발화의 원점수
- `dimension_breakdown`: `logic_score` 페널티가 5개 평가 차원에 누적된 값
- `weighted_distortion`: `dimension_breakdown`을 `per_dim_cap=45`와 `dimension_weights`로 정규화한 최종 왜곡도
- `score`: `max(0, 100 - weighted_distortion + validity_score)`로 계산되는 최종 정합성 점수

이 분리는 *"왜 점수가 깎였는가"*가 한눈에 보이도록 한 의도적 설계다. 또한 신규 페널티가 `logic_score`에만 남고 `dimension_breakdown`에 연결되지 않는 경우에는 점수를 임의 보정하지 않고, 리포트의 `reasons`에 warning을 남기도록 했다.

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

### 3. OpenAI API 키 설정

두 가지 방법 중 택1:

**방법 A — `.env` 파일**

```bash
cp .env.example .env
# 편집기로 .env 열어서 OPENAI_API_KEY 입력
```

**방법 B — 앱 실행 후 UI에 직접 입력**

웹 화면 상단 *OpenAI API Key* 필드에 입력. 새 세션마다 다시 입력해야 함.

> API 키는 [OpenAI Platform](https://platform.openai.com/api-keys)에서 발급. 사용량에 따라 과금되며, 본 앱 1회 시연(추출 2회 + 리포트 1회)당 약 $0.01 ~ $0.03 수준.

### 4. 실행

```bash
python axiom_tracker_hybrid.py
```

성공 시 콘솔에:

```
✅ Ontology loaded: XXX triples, 6 values, 20 frames
✅ Rules loaded: 1024 rules
✅ Dimension weights: {'temporal_shift': 0.40, 'frame_effect': 0.22, 'context_omission': 0.15, 'consensus_deviation': 0.13, 'evidence_quality': 0.10}
✅ sync_frames: OWL-only 프레임 5개 제거 (LLM 선택지 정합화)
* Running on local URL: http://127.0.0.1:7860
```

브라우저에서 `http://127.0.0.1:7860` 접속.

---

## 사용 방법

### 입력

- **Past Text**: 과거 시점의 텍스트 (같은 매체의 같은 사안에 대한 이전 보도/주장)
- **Present Text**: 현재 시점의 텍스트 (같은 매체의 같은 사안에 대한 현재 보도/주장)

> 단일 텍스트가 아니라 **두 텍스트의 비교**가 핵심이다. 시계열 정합성은 본질적으로 *"같은 주체가 시점에 따라 어떻게 변했는가"*의 문제이기 때문.

### 출력 패널

- **📋 Final Report** — 최종 판정·요약·자기 비판·변호 논리·근거 룰 ID
- **🧮 Audit + Fired Rules** — 점수 분해 + weighted_distortion + 5차원 dimension_breakdown + 발화된 룰 전체 목록 (프레임별 분화 텍스트 포함)
- **🔍 Past Extraction** — 과거 텍스트의 OWL 어휘 메타데이터
- **🔍 Present Extraction** — 현재 텍스트의 OWL 어휘 메타데이터

### 시연 시나리오 예시

**입력**:
- Past: *"A정책은 시민의 기본권을 침해할 우려가 있다. 충분한 사회적 합의 없이 강행해서는 안 된다."*
- Present: *"A정책은 사회 전체의 안전을 위한 불가피한 조치다. 일부 권리 제한은 감수할 만하다."*

**예상 발화**:
- Stage 2: Stance polarity 큰 이동 감지 → `temporal_shift` 차원에 직접 -30 누적
- Stage 3a: 과거의 `StanceConsistency` 가치를 현재의 `SilentPivot` 또는 유사 프레임이 위반
- Stage 4: T-계열 시계열 룰 다수 발화, `temporal_shift` 차원에 페널티 집중
- weighted_distortion: 0.40 × normalized(temporal_shift 페널티)가 지배적 기여

**예상 판정 (시계열 단독 발화 시)**: *score 70대 초중반 — 정합 영역의 가장 낮은 지대. "분명한 시계열 경고지만 다른 차원이 멀쩡하면 종합 붕괴는 아님"*

**예상 판정 (복합 발화 시)**: *score 60 이하 — caution 영역 진입. 추가 차원이 함께 발화되면 deviated 영역까지 도달*

이 분포는 *"단일 차원 위반은 주의 깊은 정합, 복합 차원 위반에서만 왜곡 확정"*이라는 본 시스템의 5차원 종합 평가 사상을 점수에 직접 반영한 결과다.

---

## 폴더 구조

```
context-sync.curator1/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
├── axiom_tracker_hybrid.py             ← 메인 데모 앱
├── ontology/
│   ├── context_sync_app_centered_ontology.owl
│   └── news_rules_1024.json
├── docs/                                ← 페르소나/컨텍스트 자료
│   ├── 01_problem_framing.md
│   ├── 02_cognition_and_metacognition.md
│   ├── 03_hybrid_ontology.md
│   └── 04_background.md
└── comparison/                          ← 비교용 보조 버전
    ├── README.md
    └── axiom_tracker_pure_llm.py
```

---

## 변경 이력

### v1.2.1 (현재) — dimension 정합성 정리 + 안전망 경고화

v1.2의 시계열 중심 가중치 보정은 유지하되, 방금 점검한 Python↔JSON↔OWL 3층 정합성을 문서와 룰셋 수준까지 정리한 버전이다. 핵심은 *점수 공식의 동기화*와 *ValueAnchor↔dimension 매핑의 정렬*이다.

- **JSON↔OWL dimension 정합성 정리**: 1024개 룰을 점검해 OWL의 `calibratesDimension` 기준과 어긋나던 192개 룰을 정렬
  - `ContextCompleteness → temporal_shift` 케이스를 `context_omission`으로 수정
  - `PluralPublicReason → frame_effect` 케이스를 `consensus_deviation`으로 수정
- **Stage 2 점수 반영 명확화**: stance polarity shift 페널티가 `logic_score`에만 남지 않고 `dimension_breakdown["temporal_shift"]`에 직접 누적되도록 정리
- **safety net 경고화**: 미반영 logic 페널티가 의심될 때 `frame_effect`로 강제 흡수하지 않고, `reasons`에 warning을 남기도록 변경
- **문서 공식 정리**: UI/README의 예전 공식 `100 + validity + logic`을 제거하고, 실제 공식 `100 - weighted_distortion + validity_score`로 통일
- **버전 표기 갱신**: JSON 룰셋과 OWL 수식 메타데이터를 `v2.1.1-dimension-aligned` 계열로 정리

### v1.2 — 시계열 중심 가중치 보정

본 시스템이 *시계열 정합성 검토 도구*임을 가중치 차원에서 선언한 버전. v1.1의 점수 계산 구조는 유지하되 가중치와 정규화 상한을 보정해, Major polarity shift 단독 케이스가 *정합 영역의 가장 낮은 지대*(score 70대 초중반)에 자연스럽게 안착하도록 조정했다.

- **temporal_shift 가중치 강화**: 0.34 → 0.40. 본 시스템의 정체성을 가중치로 명시
- **`dimension_weights` 재배분**: 0.40 / 0.22 / 0.15 / 0.13 / 0.10 (합 1.00). 5차원 종합 평가 사상을 보존하면서 시계열 우위 표명
- **`per_dim_cap` 보정**: 60 → 45. 한 차원의 단독 위반이 *주의 영역*까지만 도달하고 *왜곡 확정*은 복합 발화에서만 가능하도록 조정
- **3층 동시 갱신**: JSON `aggregation_formula` + OWL `evaluationFormula` + Python `compute_weighted_distortion`이 모두 v2.1 가중치로 일관 갱신됨 (본 시스템의 3층 분리 원칙 유지)
- **점수 분포 변화** (Major polarity shift -30 단독 케이스 기준):
  - v1.1: score 83 (정합 영역) — *시계열 위반 케이스가 지나치게 가볍게 잡힘*
  - v1.2: score 73 (정합 끝, 주의 직전) — *분명한 경고지만 다른 차원이 멀쩡하면 종합 붕괴 아님*

### v1.1 (이전) — 5차원 평가 + 룰셋 확장

- **룰셋 확장**: 800 → 1024 (M03 InstitutionalDistrust, M05 ResponsibilityShift, M06 PreemptiveDiscrediting 신규 64개씩 추가)
- **5차원 평가 도입**: `evidence_quality` 차원 신설, `dimension_weights` 명시 (0.34 / 0.24 / 0.18 / 0.14 / 0.10)
- **점수 계산 재설계**: `compute_weighted_distortion`이 dimension별 정규화 + 가중합산 수행. 룰 발화 수에 따른 과대평가 방지
- **JSON dimension 우선 분배**: `audit_logic`이 룰의 `dimension` 필드를 우선 사용, OWL `calibratesDimension`은 폴백
- **fired_rules 메타 확장**: `frame_definition_ko`, `schema_description_ko`, `expected_evidence_ko`, `score_hint` 등 7개 신규 필드를 리포트에 노출
- **OWL ↔ JSON 동기화**: `sync_frames_with_rules` 함수가 런타임에 OWL-only 프레임을 LLM 선택지에서 필터링. OWL 파일은 손대지 않음

### v1.0 (초기) — 초기 공개

- OWL 온톨로지 + 800개 외부 룰셋
- GPT-4o + OWL 그래프 추론 + 룰 매칭의 3축 구조
- 4단계 검증 체인

---

## 한계와 다음 단계

본 버전은 **시연용 프로토타입**이다. 명시적 한계:

- **`conflictsWith` 관계의 임시성** — 21개 Value↔Frame 충돌 관계는 합리적 가정으로 정의된 것이지, *집단지성으로 검증된* 것이 아니다. 페르소나 자료의 *"잘 정의된 집단지성 위에 자리잡아야 한다"*는 비전 기준에서 보면, 이 관계 자체가 *임시 파라미터*다.
- **`layer_social_meaning_proxy`의 가중치는 임의 파라미터** — 본래 비전은 *집단지성의 주기적 측정*에 기반한 가중치이지만, 본 프로토타입에서는 임시 대체하고 있다.
- **`dimension_weights`(0.40/0.22/0.15/0.13/0.10)와 `per_dim_cap=45`는 튜닝 파라미터** — v1.2.1에서 시계열 중심 보정과 dimension 정합성 점검을 거친 합리적 기본값이지만, 실측 보도 코퍼스로 검증된 값이 아니다. *시계열 단독 위반 → 정합 끝*, *복합 위반 → 주의/왜곡* 분포를 의도한 잠정값이다.
- **M04 VictimBlaming 부재** — OWL 어휘로는 존재하나 룰셋에는 미포함. 민감 도메인 가이드라인 정립 후 추가 예정.
- **2-텍스트 비교** — 현재는 과거/현재 두 텍스트의 직접 비교만 지원. *동일 매체-동일 사안의 N개 기사 묶음*에 대한 시계열 분석은 다음 단계.
- **OpenAI API 의존** — 클라우드 LLM 의존. 향후 로컬 sLLM 옵션 추가 검토.

### 로드맵

- [x] **1단계 — 문제의식과 페르소나 정립**
  - 페르소나/컨텍스트 자료 4종 정리 (`docs/`)
  - 부동성 기준층의 비전 정의 (잘 정의된 집단지성, 국가 의미 인프라)

- [x] **2단계 — 온톨로지 아키텍처 공개**
  - OWL 온톨로지 (17 스키마, 6 ValueAnchor, 4 AxiomLayer, 20 Frame 등)
  - 800개 외부 룰셋 JSON (v1.0)
  - Value↔Frame conflictsWith 관계 21개, Value↔Value reinforces 관계 8개

- [x] **3단계 — Hybrid 데모 앱 공개**
  - GPT-4o + OWL 그래프 추론 + 800 룰 매칭의 3축 구조
  - 4단계 검증 체인 (Validity / Polarity / Graph / Rule)
  - Pure LLM 비교 버전 (`comparison/`)

- [x] **3.5단계 — 5차원 평가 + 룰셋 확장 + 시계열 중심 보정 (v1.1~v1.2.1)**
  - `evidence_quality` 차원 신설 및 `dimension_weights` 도입
  - 1024개 룰셋 (M03·M05·M06 확장)
  - `weighted_distortion` 정규화 점수 계산
  - `sync_frames_with_rules` 런타임 동기화
  - temporal_shift 가중치 0.34 → 0.40 (시스템 정체성 선언)
  - per_dim_cap 60 → 45 (단일 차원 위반의 영역 보정)
  - Stage 2 polarity shift 페널티를 `temporal_shift` 차원에 직접 반영
  - JSON + OWL + Python 3층 동시 갱신 및 ValueAnchor↔dimension 매핑 정합성 정리

- [ ] **4단계 — 기사 묶음 일괄 분석 (Streamlit)** — 작업 진행 중
  - 5개 차원 지표 + 1024 룰 + 시계열 sentiment 그래프
  - Dense/Sparse 이중 RAG
  - 수동 시뮬레이터 (의미값 파라미터 직접 조작)
  - OWL 계층 시각화

- [ ] **5단계 — 집단지성 측정층 보강 방향**
  - `layer_social_meaning_proxy`의 임의 파라미터 → 실측 데이터 점진 대체 방향 제안
  - `dimension_weights`와 `per_dim_cap`의 실측 보정
  - 단일 매체 시계열 → 교차 매체 비교 확장
  - M04 VictimBlaming 등 보류 프레임의 룰셋 추가

---

## 라이선스

본 프로젝트는 [MIT License](./LICENSE)로 배포된다.

OWL 온톨로지 및 1024 룰셋은 본 프로젝트가 처음 공개하는 자산이며, 동일 라이선스 하에 자유롭게 활용 가능하다. 학술·시민사회·언론 영역에서의 활용을 환영한다.

## 기여 / 문의

이슈 또는 토론은 GitHub Issues로. 페르소나 자료에 대한 피드백, OWL 관계의 추가/수정 제안, 룰셋 보강 등은 모두 환영한다.

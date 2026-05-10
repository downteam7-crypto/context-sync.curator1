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

- **OWL 온톨로지** (`ontology/context_sync_app_centered_ontology.owl`): 17개의 추상 스키마, 6개의 ValueAnchor, 4개의 AxiomLayer, 5개의 EvaluationDimension, 17개의 Frame, 16개의 Topic 등 **느리게 변하는 보편 어휘**와, 그 어휘 사이의 **그래프 관계**(21개 conflictsWith, 8개 reinforces)를 정의한다.
- **외부 룰셋** (`ontology/news_rules_800.json`): 위 스키마에 속하는 **800개의 실무 검출 규칙**을 별도 파일로 관리한다.

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

- **OWL이 하는 일** — 어휘(클래스, ValueAnchor, Frame, Topic)와 그 관계(`conflictsWith`, `reinforces`, `calibratesDimension`)를 선언한다. 또한 `evaluationFormula` 같은 *수식 메타데이터*를 `RuleEngine` 클래스 위에 *문자열로 보관*한다.
- **OWL이 하지 않는 일** — 점수 계산 자체. OWL은 *어떤 차원으로 어떻게 가중되어야 하는지*를 *선언*하지만, 실제 산수는 Python 측 `audit_logic` 함수가 수행한다.
- **외부 JSON 룰셋** — 800개의 실무 검출 규칙을 보관한다. OWL의 RuleSchema가 추상 어휘를 제공하고, JSON이 그 어휘에 속하는 구체적 룰들을 공급하는 분리.

따라서 본 프로젝트의 부동성 기준층은 단일 파일이 아니라 **OWL(어휘·관계 선언) + JSON(실무 규칙) + Python(계산 실행)의 3층 분리 구조**다. 이 분리가 *"공리를 80개에서 800개로 늘리면 정교해질까"*에 대한 응답이며, 동시에 *각 층이 무엇을 책임지는지*를 명확히 한다.

### JSON 룰셋과 OWL 매핑의 관계

`news_rules_800.json`의 800개 룰은 OWL의 어휘를 사용하지만, 두 자원의 *역할이 다르다*는 점을 명시해둔다.

**ValueAnchor 분포** — 800개 룰은 5개 ValueAnchor 기반으로 분포되어 있다.

| ValueAnchor | 룰 수 |
|---|---|
| `ContextCompleteness` | 256 |
| `StanceConsistency` | 192 |
| `FrameAccountability` | 160 |
| `PluralPublicReason` | 128 |
| `EvidenceTransparency` | 64 |
| `ResponsibilitySeparation` | 0 |

6번째인 `ResponsibilitySeparation`은 본 룰셋에서는 사용되지 않으며, **OWL 그래프 추론(Stage 3) 단에서만 작동**한다. 이는 *책임 분리*가 단일 룰 단위가 아니라 *전체 텍스트의 책임 위치 평가*에서만 의미가 있다는 설계 선택이다. 즉 룰 매칭으로는 잡기 어렵고, *과거 옹호 가치 ↔ 현재 프레임* 같은 텍스트 간 관계 추론으로만 잡힌다.

**dimension 매핑의 두 층위** — 일부 룰(약 24%)은 OWL의 `calibratesDimension` 기본 매핑과 다른 dimension 필드를 가진다. 예를 들어 `ContextCompleteness`는 OWL에서 `context_omission`을 보정한다고 선언되어 있지만, 일부 룰은 `temporal_shift` 차원에서 측정되도록 정의되어 있다.

이는 *"같은 가치를 다른 차원으로 측정하는 개별 케이스"*로 해석할 수 있다. **OWL은 *기본(prior) 매핑*을 제공하고, JSON은 *개별 룰의 측정 차원*을 명시하는 구조**다. 본 코드의 `audit_logic`은 두 층위를 모두 활용한다 — Stage 3에서는 OWL의 기본 매핑으로 그래프 추론을, Stage 4에서는 JSON 룰의 개별 dimension으로 룰 매칭을 수행한다.

**룰셋의 생성 방식** — 800개 룰은 17개 RuleSchema(T01~T05, S01~S06, M01~M06)와 12개 target_frame, 16개 context, 4개 severity 단계의 조합으로 *체계적으로 생성*되었다. `severity_band`는 4단계(low/medium/high/critical)에 200개씩 균등 분포되어 있어, *현실 빈도 기반 가중치가 아닌 시연용 대표 분포*임을 밝혀둔다. 향후 실제 보도 코퍼스 분석을 통한 빈도 보정이 보강 방향이다.

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
│  (OpenAI API)   │    │  Rule Matching      │    │  800 JSON Rules   │
└─────────────────┘    └────────────────────┘    └──────────────────┘
        ↓                       ↓                          ↑
  OWL 어휘로 추출       그래프 관계 + 룰 발화          어휘·관계·룰 공급
```

### 처리 단계

1. **추출 (LLM)** — GPT-4o가 두 텍스트(과거/현재)에서 OWL 어휘로 메타데이터 추출
   - `promoted_value`: 6개 ValueAnchor 중 하나
   - `detected_frame`: 17개 Frame 중 하나 또는 `None`
   - `topic`: 16개 Topic 중 하나
   - `stance_polarity`: -1.0 ~ +1.0 정량 극성
   - `is_valid_discourse`: 사실/윤리 위배 여부 (Red Card)

2. **검증 (Python + OWL/JSON)** — 4단계 검증 체인
   - **Stage 1 — Validity (Red Card)**: 평평한 지구설, 인종 청소 옹호 등 *시계열과 무관한* 즉시 실격
   - **Stage 2 — Stance Polarity Shift**: 입장 극성의 정량적 이동 측정
   - **Stage 3 — OWL Graph Reasoning**:
     - *Cross-temporal*: 과거 옹호 가치를 현재 프레임이 위반하는가 (`conflictsWith` 그래프 쿼리)
     - *Self-contradictory*: 텍스트가 옹호한다고 *주장*하는 가치를 그 텍스트의 *프레임*이 스스로 위반
     - *Reinforces*: 가치 이동이 *강화 관계 안*인지(강조점 이동) *밖*인지(구조적 이동) 판별
   - **Stage 4 — 800 Rule Matching**: 발화된 frame과 topic에 매치되는 룰을 활성화

3. **리포트 (LLM)** — 발화된 룰을 *증거*로 GPT-4o가 한국어 리치 리포트 생성
   - `verdict`, `summary`, `self_attack`, `steelman`, `evidence_rules`

### 점수 분리

```
score = max(0, 100 + validity_score + logic_score)
```

- `validity_score`: 사실/윤리 위배에 의한 감점 (시계열 무관)
- `logic_score`: 시계열 입장 변경에 의한 감점 (룰 발화 + 그래프 추론 합산)

이 분리는 *"왜 점수가 깎였는가"*가 한눈에 보이도록 한 의도적 설계다.

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
✅ Ontology loaded: XXX triples, 6 values, 17 frames
✅ Rules loaded: 800 rules
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
- **🧮 Audit + Fired Rules** — 점수 분해 + 발화된 룰 전체 목록
- **🔍 Past Extraction** — 과거 텍스트의 OWL 어휘 메타데이터
- **🔍 Present Extraction** — 현재 텍스트의 OWL 어휘 메타데이터

### 시연 시나리오 예시

**입력**:
- Past: *"A정책은 시민의 기본권을 침해할 우려가 있다. 충분한 사회적 합의 없이 강행해서는 안 된다."*
- Present: *"A정책은 사회 전체의 안전을 위한 불가피한 조치다. 일부 권리 제한은 감수할 만하다."*

**예상 발화**:
- Stage 2: Stance polarity 큰 이동 감지
- Stage 3a: 과거의 `StanceConsistency` 가치를 현재의 `SilentPivot` 또는 유사 프레임이 위반
- Stage 4: T-계열 시계열 룰 다수 발화

**예상 판정**: *"시계열 정합성 위반 — 입장 변경에 대한 명시적 사유 누락"*

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
│   └── news_rules_800.json
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

## 한계와 다음 단계

본 버전은 **시연용 프로토타입**이다. 명시적 한계:

- **`conflictsWith` 관계의 임시성** — 21개 Value↔Frame 충돌 관계는 합리적 가정으로 정의된 것이지, *집단지성으로 검증된* 것이 아니다. 페르소나 자료의 *"잘 정의된 집단지성 위에 자리잡아야 한다"*는 비전 기준에서 보면, 이 관계 자체가 *임시 파라미터*다.
- **`layer_social_meaning_proxy`의 가중치는 임의 파라미터** — 본래 비전은 *집단지성의 주기적 측정*에 기반한 가중치이지만, 본 프로토타입에서는 임시 대체하고 있다.
- **2-텍스트 비교** — 현재는 과거/현재 두 텍스트의 직접 비교만 지원. *동일 매체-동일 사안의 N개 기사 묶음*에 대한 시계열 분석은 다음 단계.
- **OpenAI API 의존** — 클라우드 LLM 의존. 향후 로컬 sLLM 옵션 추가 검토.

### 로드맵

- [x] **1단계 — 문제의식과 페르소나 정립**
  - 페르소나/컨텍스트 자료 4종 정리 (`docs/`)
  - 부동성 기준층의 비전 정의 (잘 정의된 집단지성, 국가 의미 인프라)

- [x] **2단계 — 온톨로지 아키텍처 공개**
  - OWL 온톨로지 (17 스키마, 6 ValueAnchor, 4 AxiomLayer 등)
  - 800개 외부 룰셋 JSON
  - Value↔Frame conflictsWith 관계 21개, Value↔Value reinforces 관계 8개

- [x] **3단계 — Hybrid 데모 앱 공개**
  - GPT-4o + OWL 그래프 추론 + 800 룰 매칭의 3축 구조
  - 4단계 검증 체인 (Validity / Polarity / Graph / Rule)
  - Pure LLM 비교 버전 (`comparison/`)

- [ ] **4단계 — 기사 묶음 일괄 분석 (Streamlit)** — 작업 진행 중
  - 4개 차원 지표 + 800 룰 + 시계열 sentiment 그래프
  - Dense/Sparse 이중 RAG
  - 수동 시뮬레이터 (의미값 파라미터 직접 조작)
  - OWL 계층 시각화

- [ ] **5단계 — 집단지성 측정층 보강 방향**
  - `layer_social_meaning_proxy`의 임의 파라미터 → 실측 데이터 점진 대체 방향 제안
  - 단일 매체 시계열 → 교차 매체 비교 확장

---

## 라이선스

본 프로젝트는 [MIT License](./LICENSE)로 배포된다.

OWL 온톨로지 및 800 룰셋은 본 프로젝트가 처음 공개하는 자산이며, 동일 라이선스 하에 자유롭게 활용 가능하다. 학술·시민사회·언론 영역에서의 활용을 환영한다.

## 기여 / 문의

이슈 또는 토론은 GitHub Issues로. 페르소나 자료에 대한 피드백, OWL 관계의 추가/수정 제안, 룰셋 보강 등은 모두 환영한다.

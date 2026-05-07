# Comparison: Pure LLM Edition

본 폴더는 메인 Hybrid 앱과 *비교*하기 위한 **순수 LLM 버전**을 보관한다.

## 두 버전의 차이

| 항목 | Pure LLM (이 폴더) | Hybrid (메인) |
|---|---|---|
| 의존성 | OpenAI + Pydantic만 | OpenAI + Pydantic + rdflib + OWL/JSON |
| 어휘 | LLM 자유 어휘 | OWL의 6 ValueAnchor + 17 Frame + 16 Topic 강제 |
| 추론 | LLM 단독 | LLM 추출 + OWL 그래프 추론 + 800 룰 매칭 |
| 해석 가능성 | LLM 응답에 의존 | 발화된 룰을 근거로 명시 (`evidence_rules`) |
| 시연 가치 | 빠른 데모 | 프로젝트 비전 시연 |

## 무엇을 비교할 수 있는가

같은 텍스트 쌍을 두 버전에 입력하면:

- **Pure LLM**: GPT-4가 *자기 어휘로 자유롭게* 판정
- **Hybrid**: GPT-4가 *프로젝트 어휘 안에서* 추출 → 그래프 추론 → 룰 발화 → 룰 근거 리포트

이 비교가 보여주는 것은, **온톨로지 기준층이 LLM의 판정을 어떻게 *어휘적으로 제약*하고 *해석 가능하게* 만드는가**이다.

## 실행

```bash
# 메인 폴더의 requirements.txt 그대로 사용 (rdflib는 사용하지 않지만 설치되어 있어도 무관)
pip install gradio openai pydantic python-dotenv

python axiom_tracker_pure_llm.py
```

`.env`에 `OPENAI_API_KEY` 입력 또는 UI에서 직접 입력.

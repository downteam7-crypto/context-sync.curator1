"""
Axiom Tracker — Pure LLM Edition (OpenAI GPT-4)

순수 LLM 기반 시계열 논조 판독 앱.
온톨로지/룰 기반 추론 없이, GPT-4가 추출-비교-판정을 모두 수행한다.

구조:
    1. 추출: 두 텍스트에서 각각 구조화된 메타데이터를 추출 (Pydantic schema)
    2. Safety check: 과학적 사실/민주적 윤리 위배 여부 사전 체크 (Red Card)
    3. 비교: Python에서 추출 결과를 비교하여 시계열 정합성 점수 계산
    4. 리포트: GPT-4가 비교 결과를 바탕으로 리치 리포트 생성

실행 전 준비:
    pip install gradio openai pydantic python-dotenv
    .env 파일에 OPENAI_API_KEY=... 또는 UI에서 직접 입력
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# ==========================================
# 1. Pydantic Schemas
# ==========================================

class ExtractedSymbol(BaseModel):
    """단일 텍스트에서 추출되는 구조화 메타데이터."""
    title_subject: str = Field(description="제목이 다루는 주제 (1~3 단어)")
    body_subject: str = Field(description="본문이 실제로 다루는 주제 (1~3 단어)")
    promoted_value: str = Field(description="텍스트가 옹호하는 가치 (예: PublicSafety, IndividualLiberty, ProceduralJustice 등)")
    stance_polarity: float = Field(description="주제에 대한 입장 극성. -1.0(강한 반대) ~ +1.0(강한 지지)", ge=-1.0, le=1.0)
    context_type: str = Field(description="텍스트가 놓인 맥락 유형 (예: NormalContext, CrisisContext, ElectionContext, JudicialContext)")
    is_valid_discourse: bool = Field(description="과학적 사실 및 민주적 윤리에 부합하면 True, 위배되면 False")
    violation_reason: str = Field(description="위배되는 이유 (예: 'Flat Earth', 'Hate Speech'). 위배 없으면 'None'")
    key_claims: List[str] = Field(description="텍스트의 핵심 주장 1~3개")


class RichReport(BaseModel):
    """최종 리치 리포트."""
    verdict: str = Field(description="최종 판정 (예: '정합', '주의', '시계열 정합성 위반', '실격')")
    summary: str = Field(description="2~4문장 요약")
    self_attack: str = Field(description="이 판정 자체에 대한 자기 비판 (이 판정이 틀렸을 가능성)")
    steelman: str = Field(description="저자 입장에서의 가장 강한 변호 논리")


# ==========================================
# 2. Extractor (GPT-4 + Pydantic)
# ==========================================

EXTRACTION_SYSTEM_PROMPT = """You are a discourse analyst specializing in news articles.

Your job is to extract structured metadata from a single text.

[Safety Criteria - Red Card Check]
Set is_valid_discourse=False ONLY if the text:
1. Contradicts well-established scientific consensus (e.g., flat earth, climate denial of basic mechanism)
2. Advocates totalitarianism, ethnic cleansing, or violations of universal human rights
3. Promotes hate speech against protected groups

If the text merely takes a controversial political stance, this is NOT a violation.
Disagreement with mainstream opinion is not the same as violation of universal norms.

[Extraction Guidelines]
- promoted_value: identify the core value the text champions, even implicitly
- stance_polarity: how strongly does the author support/oppose the main subject?
- context_type: what situational frame is the text operating in?
- key_claims: extract 1-3 most important claims the author makes

Output strictly in the requested JSON schema."""


def extract_symbol(client: OpenAI, text: str, model: str = "gpt-4o") -> ExtractedSymbol:
    """단일 텍스트에서 구조화 메타데이터 추출."""
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this text:\n\n{text}"},
        ],
        response_format=ExtractedSymbol,
        temperature=0.1,
    )
    return response.choices[0].message.parsed


# ==========================================
# 3. Comparison Logic (Pure Python)
# ==========================================

def audit_logic(past: ExtractedSymbol, present: ExtractedSymbol) -> Dict[str, Any]:
    """추출된 두 메타데이터를 비교하여 시계열 정합성을 검토.

    점수 분리:
        - validity_score: 사실/윤리 위배에 의한 감점 (시계열 무관)
        - logic_score: 시계열 입장 변경에 의한 감점
        - total = 100 + validity_score + logic_score
    """
    report = {
        "validity_violation": False,
        "logic_conflict": False,
        "validity_score": 0,
        "logic_score": 0,
        "score": 100,
        "reasons": [],
        "details": {
            "past": past.model_dump(),
            "present": present.model_dump(),
        },
    }

    # [Rule 0] Validity Check (Red Card) — 시계열과 무관
    if not past.is_valid_discourse:
        report["validity_violation"] = True
        report["validity_score"] -= 100
        report["reasons"].append(f"PAST text validity violation: {past.violation_reason}")

    if not present.is_valid_discourse:
        report["validity_violation"] = True
        report["validity_score"] -= 100
        report["reasons"].append(f"PRESENT text validity violation: {present.violation_reason}")

    # 위배 시에는 시계열 분석을 하지 않고 조기 반환
    if report["validity_violation"]:
        report["reasons"].append("⛔ AUDIT STOPPED: validity violation supersedes temporal analysis.")
        report["score"] = max(0, 100 + report["validity_score"])
        return report

    # [Rule 1] Clickbait — 제목과 본문의 주제 불일치
    if past.title_subject.strip().lower() != past.body_subject.strip().lower():
        report["logic_score"] -= 5
        report["reasons"].append(f"PAST clickbait: title='{past.title_subject}' vs body='{past.body_subject}'")
    if present.title_subject.strip().lower() != present.body_subject.strip().lower():
        report["logic_score"] -= 5
        report["reasons"].append(f"PRESENT clickbait: title='{present.title_subject}' vs body='{present.body_subject}'")

    # [Rule 2] Stance Polarity Shift — 입장 극성의 변동
    polarity_shift = abs(past.stance_polarity - present.stance_polarity)
    if polarity_shift >= 1.0:
        # 강한 반대 → 강한 지지 같은 극단적 반전
        report["logic_conflict"] = True
        report["logic_score"] -= 50
        report["reasons"].append(
            f"Major stance reversal: {past.stance_polarity:+.2f} → {present.stance_polarity:+.2f} (Δ={polarity_shift:.2f})"
        )
    elif polarity_shift >= 0.5:
        report["logic_score"] -= 25
        report["reasons"].append(
            f"Significant stance shift: {past.stance_polarity:+.2f} → {present.stance_polarity:+.2f} (Δ={polarity_shift:.2f})"
        )
    elif polarity_shift >= 0.25:
        report["logic_score"] -= 10
        report["reasons"].append(
            f"Moderate stance shift: {past.stance_polarity:+.2f} → {present.stance_polarity:+.2f} (Δ={polarity_shift:.2f})"
        )

    # [Rule 3] Promoted Value Change
    if past.promoted_value.strip().lower() != present.promoted_value.strip().lower():
        report["logic_score"] -= 15
        report["reasons"].append(f"Promoted value changed: '{past.promoted_value}' → '{present.promoted_value}'")

    # [Rule 4] Crisis Context Mitigation
    # 다만 단순 라벨 의존이 아니라, 실제 polarity_shift가 있을 때만 mitigate
    if (
        present.context_type == "CrisisContext"
        and past.context_type == "NormalContext"
        and report["logic_conflict"]
    ):
        # 위기 맥락 변경은 입장 변경의 *부분적* 정당화 사유
        mitigation = 20
        report["logic_score"] += mitigation
        report["reasons"].append(
            f"Crisis context mitigation applied (+{mitigation}): NormalContext → CrisisContext explains part of the stance shift."
        )

    # 최종 점수 (음수 방지)
    report["score"] = max(0, 100 + report["validity_score"] + report["logic_score"])

    if not report["reasons"]:
        report["reasons"].append("No significant temporal coherence violations detected.")

    return report


# ==========================================
# 4. Reporter (GPT-4 + Pydantic)
# ==========================================

REPORTER_SYSTEM_PROMPT = """You are a discourse audit reporter.

Given a structured audit result, generate a rich human-readable report in Korean.

[Required fields]
- verdict: 한 줄 최종 판정. validity 위배 시에는 강한 경고, 정합 시에는 정합 명시
- summary: 2~4문장 요약. 무엇이 어떻게 변했는지 구체적으로
- self_attack: 이 판정 자체가 틀렸을 가능성. 가장 그럴듯한 반론
- steelman: 저자 입장에서의 가장 강한 변호 논리

If validity_violation is True, prioritize warning about false information or unethical claims over temporal analysis.
Otherwise, focus on the temporal coherence analysis.

Output strictly in the requested JSON schema. Write in Korean."""


def generate_report(client: OpenAI, audit: Dict[str, Any], model: str = "gpt-4o") -> RichReport:
    """audit 결과를 받아 리치 리포트 생성."""
    audit_summary = json.dumps(
        {
            "validity_violation": audit["validity_violation"],
            "logic_conflict": audit["logic_conflict"],
            "validity_score": audit["validity_score"],
            "logic_score": audit["logic_score"],
            "score": audit["score"],
            "reasons": audit["reasons"],
            "past_summary": {
                "subject": audit["details"]["past"]["body_subject"],
                "value": audit["details"]["past"]["promoted_value"],
                "polarity": audit["details"]["past"]["stance_polarity"],
                "claims": audit["details"]["past"]["key_claims"],
            },
            "present_summary": {
                "subject": audit["details"]["present"]["body_subject"],
                "value": audit["details"]["present"]["promoted_value"],
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
# 5. Pipeline
# ==========================================

def run_pipeline(
    past_text: str,
    present_text: str,
    api_key: str,
    model: str,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """전체 파이프라인 실행. Gradio 출력용 4개 dict 반환."""
    if not api_key or not api_key.strip():
        # 환경변수 폴백
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not api_key:
        err = {"error": "OPENAI_API_KEY가 입력되지 않았고 환경변수에도 없습니다."}
        return err, {}, {}, {}

    if not past_text or not present_text:
        err = {"error": "Past와 Present 텍스트를 모두 입력해주세요."}
        return err, {}, {}, {}

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return {"error": f"OpenAI 클라이언트 초기화 실패: {e}"}, {}, {}, {}

    # Step 1: 추출
    try:
        past_data = extract_symbol(client, past_text, model=model)
        present_data = extract_symbol(client, present_text, model=model)
    except Exception as e:
        return {"error": f"추출 단계 실패: {e}"}, {}, {}, {}

    # Step 2: 비교 (Pure Python)
    audit_result = audit_logic(past_data, present_data)

    # Step 3: 리포트 생성
    try:
        report = generate_report(client, audit_result, model=model)
    except Exception as e:
        # 리포트 실패해도 audit 결과는 보여주기
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
# 6. Gradio UI
# ==========================================

with gr.Blocks(title="Axiom Tracker — Pure LLM Edition") as demo:
    gr.Markdown("## 🛡️ Axiom Tracker — Pure LLM Edition")
    gr.Markdown(
        "OpenAI GPT-4 기반 순수 LLM 시계열 논조 판독기. "
        "온톨로지/룰 없이 LLM이 추출-비교-판정을 모두 수행합니다.\n\n"
        "**Red Card**: 과학적 사실 위배(예: 지구 평평설)나 비윤리적 주장은 시계열과 무관하게 즉시 실격 처리됩니다."
    )

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
            value="A정책은 명백히 시민의 자유를 침해한다. 정부는 이런 정책을 즉시 철회해야 한다.",
            lines=5,
        )
        n_in = gr.Textbox(
            label="Present Text (현재 논조)",
            value="A정책은 사회 전체의 안전을 위해 불가피한 조치다. 일부 자유의 제한은 감수할 만하다.",
            lines=5,
        )

    btn = gr.Button("🔍 Run Audit", variant="primary")

    with gr.Row():
        report_out = gr.JSON(label="📋 Final Report")
        audit_out = gr.JSON(label="🧮 Audit Calculation (Score Detail)")

    with gr.Row():
        p_data_out = gr.JSON(label="🔍 Past Extraction")
        n_data_out = gr.JSON(label="🔍 Present Extraction")

    btn.click(
        run_pipeline,
        inputs=[p_in, n_in, api_input, model_input],
        outputs=[report_out, p_data_out, n_data_out, audit_out],
    )

    gr.Markdown(
        "---\n"
        "**점수 해석**\n"
        "- 100점: 시계열 정합성 양호\n"
        "- 75~99점: 경미한 입장 이동\n"
        "- 50~74점: 중대한 입장 이동\n"
        "- 25~49점: 극단적 반전 또는 가치 충돌\n"
        "- 0점: 사실/윤리 위배 (Red Card)\n\n"
        "**점수 분리**: validity_score(사실/윤리) + logic_score(시계열) 별도 표시"
    )


if __name__ == "__main__":
    demo.launch()

#!/usr/bin/env python3
"""test_stance.py — target-aware stance 후보 기능 스모크 테스트.

실행 위치: repo 루트
    python tools/test_stance.py

검증 범위:
1) LLM off: 기본 휴리스틱 폴백 + 필수 키 보존
2) 강화 휴리스틱: 외부 LLM 없이 보조 cue를 켜는 선택 모드
3) LLM mock: 우려→수혜 부호 반전과 target_shift 진단
4) 확장 domain: 교육·에너지·문화 등 새 domain이 other로 뭉개지지 않는지 확인
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import rule_engine as E


def J(**k):
    return json.dumps(k, ensure_ascii=False)


def mock_llm(prompt: str) -> str:
    if "수혜로 역주행" in prompt:
        return J(surface_topic="고환율", evaluation_target="수출주", target_scope="sector", domain="finance",
                 stance_polarity=0.6, stance_label="benefit_positive", risk_frame=False, benefit_frame=True,
                 promoted_value="FrameAccountability", detected_frame="None", topic="환율", polarity_confidence=0.85,
                 evidence_basis="수출주 환차익 수혜·강세")
    if "위기 우려" in prompt:
        return J(surface_topic="고환율", evaluation_target="거시경제", target_scope="macro", domain="economy",
                 stance_polarity=-0.45, stance_label="risk_negative", risk_frame=True, benefit_frame=False,
                 promoted_value="StanceConsistency", detected_frame="None", topic="환율", polarity_confidence=0.8,
                 evidence_basis="위기 우려·책임론")
    return J(stance_polarity=0.0, stance_label="neutral")


PAST = "원/달러 환율 1200원 돌파, 경제 위기 우려 확산."
PRES = "환율 1400원, 오히려 좋아… 수출주는 환차익 수혜로 역주행 강세."

# 1) 백워드 호환: llm 미설정 → 기본 휴리스틱, 필수 키 보존
off = E.extract_symbol(PRES, llm_call=None)
assert off["extraction_source"] == "heuristic"
assert all(k in off for k in ["promoted_value", "detected_frame", "stance_polarity", "topic"])
print("백워드 호환 ✓ (기본 휴리스틱 폴백 + 필수 키)")

# 2) 강화 휴리스틱: 명시적으로 켰을 때만 작동
old_market = getattr(E, "FALLBACK_USE_MARKET_SUPPLEMENT", False)
old_stance = getattr(E, "FALLBACK_USE_STANCE_SUPPLEMENT", False)
E.FALLBACK_USE_MARKET_SUPPLEMENT = True
if hasattr(E, "FALLBACK_USE_STANCE_SUPPLEMENT"):
    E.FALLBACK_USE_STANCE_SUPPLEMENT = True
enh = E.extract_symbol(PRES, llm_call=None)
assert enh["extraction_source"] == "heuristic_enhanced"
print(f"강화 휴리스틱 ✓ (source={enh['extraction_source']}, polarity={enh['stance_polarity']:+.2f})")
E.FALLBACK_USE_MARKET_SUPPLEMENT = old_market
if hasattr(E, "FALLBACK_USE_STANCE_SUPPLEMENT"):
    E.FALLBACK_USE_STANCE_SUPPLEMENT = old_stance

# 3) LLM 경로: 부호 있는 target-aware stance
p = E.extract_symbol(PAST, llm_call=mock_llm)
c = E.extract_symbol(PRES, llm_call=mock_llm)
ts = E.compute_target_shift(p, c)
print(f"LLM stance: past {p['stance_polarity']:+.2f}({p['domain']}/{p['evaluation_target']}) → "
      f"present {c['stance_polarity']:+.2f}({c['domain']}/{c['evaluation_target']})")
print(f"target_shift: 반전={ts['frame_reversal']}, {ts['note']}")
assert p["stance_polarity"] < 0 < c["stance_polarity"], "부호 반전 미검출"
assert c["domain"] == "finance"
print("부호 반전 검출 ✓")

# 4) 확장 domain이 other로 뭉개지지 않는지 확인
expanded_domains = ["education", "energy", "environment", "culture", "housing", "welfare", "finance", "diplomacy"]
for dom in expanded_domains:
    sym = E.extract_symbol("도메인 보존 테스트", llm_call=lambda prompt, d=dom: J(domain=d, stance_polarity=0.2, stance_label="benefit_positive"))
    assert sym["domain"] == dom, f"domain {dom} 이 보존되지 않음: {sym['domain']}"
print("확장 domain 보존 ✓", ", ".join(expanded_domains))

invalid = E.extract_symbol("도메인 검증 테스트", llm_call=lambda prompt: J(domain="religion", stance_polarity=0.0, stance_label="neutral"))
assert invalid["domain"] == "other"
print("off-menu domain → other coerce ✓")

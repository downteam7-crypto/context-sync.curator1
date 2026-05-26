"""calibrate_min_dense_score.py — OpenAI 임베딩용 Dense 컷오프 진단/보정

문제:
  min_dense_score=0.15는 ko-sroberta 분포 기준 값이다. OpenAI 임베딩은
  유사도 분포가 다르므로 0.15가 같은 의미를 갖지 않는다.

이 스크립트가 하는 일(측정 도구, 결정 도구 아님):
  A. 분포 진단 (라벨 불필요): 기사-룰 쌍 유사도 분포를 보고
     percentile 기반 후보 컷오프를 제시. "현재 0.15가 어느 분위인지" 진단.
  B. 라벨 기반 보정 (소수 라벨 있을 때): relevant/irrelevant 라벨에서
     F1이 최대가 되는 컷오프를 탐색.

사용:
  - 실측: embed_fn에 sllm_extractor.openai_embed_texts(api_key=...)를 주입
  - 검증/오프라인: embed_fn 미지정 시 합성 임베딩으로 로직만 시연
"""

from __future__ import annotations
import math
import statistics as stats
from typing import Callable, List, Dict, Optional, Tuple


def _cosine(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


# ── A. 분포 진단 (라벨 불필요) ──────────────────────────────────────
def diagnose_distribution(
    article_texts: List[str],
    rule_texts: List[str],
    embed_fn: Callable[[List[str]], Optional[List[List[float]]]],
    current_cutoff: float = 0.15,
) -> Dict:
    """모든 기사-룰 쌍의 유사도 분포를 보고 현재 컷오프의 위치를 진단한다."""
    a_vecs = embed_fn(article_texts)
    r_vecs = embed_fn(rule_texts)
    if not a_vecs or not r_vecs:
        return {"ok": False, "reason": "임베딩 실패(키 없음 또는 호출 실패)"}

    sims = []
    for av in a_vecs:
        for rv in r_vecs:
            sims.append(_cosine(av, rv))
    sims.sort()

    n = len(sims)
    pct_below = sum(1 for s in sims if s < current_cutoff) / n
    report = {
        "ok": True,
        "n_pairs": n,
        "min": round(sims[0], 4),
        "p10": round(_percentile(sims, 0.10), 4),
        "median": round(_percentile(sims, 0.50), 4),
        "mean": round(stats.fmean(sims), 4),
        "p90": round(_percentile(sims, 0.90), 4),
        "max": round(sims[-1], 4),
        "current_cutoff": current_cutoff,
        "current_cutoff_filters_out": round(pct_below, 3),  # 현재 컷오프가 걸러내는 비율
        # 후보: "상위 ~10%만 통과"가 되도록 하는 컷오프
        "suggested_p90_cutoff": round(_percentile(sims, 0.90), 4),
        "suggested_p75_cutoff": round(_percentile(sims, 0.75), 4),
    }
    return report


# ── B. 라벨 기반 보정 (relevant/irrelevant 라벨) ────────────────────
def calibrate_with_labels(
    labeled: List[Tuple[str, str, int]],  # (article_text, rule_text, is_relevant 0/1)
    embed_fn: Callable[[List[str]], Optional[List[List[float]]]],
    grid: Optional[List[float]] = None,
) -> Dict:
    """라벨된 쌍에서 F1이 최대가 되는 컷오프를 탐색한다."""
    if grid is None:
        grid = [round(0.05 * i, 2) for i in range(1, 17)]  # 0.05~0.80

    arts = [x[0] for x in labeled]
    rules = [x[1] for x in labeled]
    labels = [x[2] for x in labeled]
    av = embed_fn(arts)
    rv = embed_fn(rules)
    if not av or not rv:
        return {"ok": False, "reason": "임베딩 실패"}

    sims = [_cosine(av[i], rv[i]) for i in range(len(labeled))]

    best = None
    rows = []
    for c in grid:
        tp = sum(1 for s, y in zip(sims, labels) if s >= c and y == 1)
        fp = sum(1 for s, y in zip(sims, labels) if s >= c and y == 0)
        fn = sum(1 for s, y in zip(sims, labels) if s < c and y == 1)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"cutoff": c, "precision": round(prec, 3),
                     "recall": round(rec, 3), "f1": round(f1, 3)})
        if best is None or f1 > best["f1"]:
            best = {"cutoff": c, "f1": round(f1, 3),
                    "precision": round(prec, 3), "recall": round(rec, 3)}
    return {"ok": True, "best": best, "grid": rows}


# ── 시연 (오프라인: 합성 임베딩으로 로직 검증) ──────────────────────
if __name__ == "__main__":
    import random
    rng = random.Random(0)

    # 합성 임베딩: 'topic:X' 토큰을 공유하면 비슷한 방향, 아니면 랜덤
    def synthetic_embed(texts):
        vecs = []
        for t in texts:
            base = [0.0] * 8
            if "pivot" in t:
                base[0] = 1.0
            if "evidence" in t:
                base[1] = 1.0
            if "frame" in t:
                base[2] = 1.0
            # 약한 노이즈
            v = [b + rng.uniform(-0.15, 0.15) for b in base]
            # 모든 벡터에 공통 성분(OpenAI처럼 무관해도 양의 유사도가 나오는 효과)
            v = [x + 0.5 for x in v]
            vecs.append(v)
        return vecs

    arts = ["pivot 입장 번복 기사", "evidence 증거 부재 기사"]
    rules = ["pivot 룰", "evidence 룰", "frame 룰", "무관한 일반 룰"]

    print("=== A. 분포 진단 (현재 0.15) ===")
    rep = diagnose_distribution(arts, rules, synthetic_embed, current_cutoff=0.15)
    for k, v in rep.items():
        print(f"  {k}: {v}")

    print("\n=== B. 라벨 기반 보정 ===")
    labeled = [
        ("pivot 입장 번복 기사", "pivot 룰", 1),
        ("pivot 입장 번복 기사", "무관한 일반 룰", 0),
        ("evidence 증거 부재 기사", "evidence 룰", 1),
        ("evidence 증거 부재 기사", "frame 룰", 0),
        ("pivot 입장 번복 기사", "evidence 룰", 0),
        ("evidence 증거 부재 기사", "pivot 룰", 0),
    ]
    cal = calibrate_with_labels(labeled, synthetic_embed)
    print(f"  best: {cal['best']}")
    print("  grid(일부):")
    for row in cal["grid"][1:7]:
        print(f"    {row}")

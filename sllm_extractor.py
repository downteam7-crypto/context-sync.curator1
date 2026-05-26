from __future__ import annotations

import json
import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple

DIMENSIONS = ["temporal_shift", "frame_effect", "context_omission", "consensus_deviation", "evidence_quality"]

# ── 클라우드용 Dense RAG: OpenAI 임베딩 ──────────────────────────────
# 로컬 ko-sroberta(dense_model)가 없을 때, OpenAI /v1/embeddings로 Dense를 대체한다.
# 주의: 이 임베딩은 ko-sroberta와 *다른 의미공간*이므로, 반환 dict의 dense_engine으로
#       어느 엔진인지 구분 표시한다(출처 정직성). 두 엔진은 cosine 분포·임계값이 다르다.
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
_OPENAI_EMBED_BATCH = 128


def openai_embed_texts(
    texts: List[str],
    api_key: str = "",
    model: str = DEFAULT_OPENAI_EMBED_MODEL,
    timeout: int = 60,
) -> Optional[List[List[float]]]:
    """텍스트 리스트 → 임베딩 벡터 리스트. 키 없거나 실패 시 None(→ Sparse fallback)."""
    active_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not active_key or not texts:
        return None if not active_key else []
    try:
        import requests
    except Exception:
        return None
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {active_key}"}
    safe = [t if (t and t.strip()) else " " for t in texts]
    vectors: List[List[float]] = []
    for start in range(0, len(safe), _OPENAI_EMBED_BATCH):
        batch = safe[start:start + _OPENAI_EMBED_BATCH]
        try:
            resp = requests.post(_OPENAI_EMBED_URL, headers=headers,
                                 json={"model": model, "input": batch}, timeout=timeout)
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        try:
            data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            vectors.extend([d["embedding"] for d in data])
        except Exception:
            return None
    return vectors


def _cosine_py(a: List[float], b: List[float]) -> float:
    """순수 파이썬 코사인 유사도(-1~1). sklearn 불필요."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _dense_threshold_note(engine: Optional[str], cutoff: float) -> str:
    """Dense 엔진별 cutoff 해석 안내. UI가 출처/임계값을 정직하게 표시할 때 사용한다."""
    if not engine:
        return "Dense 비활성: Sparse-only 결과입니다."
    if engine == "ko-sroberta":
        return f"ko-sroberta Dense cutoff={cutoff:.2f}. 로컬 sentence-transformers 의미공간 기준입니다."
    return (
        f"Dense-OpenAI({engine}) cutoff={cutoff:.2f}. "
        "OpenAI 임베딩은 ko-sroberta와 cosine 분포가 다르므로 이 값은 임시 기준이며 "
        "tools/calibrate_min_dense_score.py로 실측 보정하는 것을 권장합니다."
    )


try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

if HAS_TRANSFORMERS and os.getenv("STREAMLIT_CLOUD", "1") != "1":
    DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
else:
    DEFAULT_MODEL = "gpt-4o-mini"


def is_cloud_mode() -> bool:
    """Return True when running in the Streamlit Community Cloud profile.

    The default is intentionally Cloud-like ("1") so that public deployments do
    not accidentally select local Hugging Face/sentence-transformers paths.
    Local advanced mode should be launched with STREAMLIT_CLOUD=0.
    """
    return os.getenv("STREAMLIT_CLOUD", "1") == "1"


def _compact_articles(articles: List[Dict[str, Any]], max_chars: int = 6500) -> str:
    chunks = []
    for i, a in enumerate(articles, 1):
        title = str(a.get("title", ""))
        date = str(a.get("date") or a.get("published_at") or "")
        outlet = str(a.get("outlet", ""))
        topic = str(a.get("topic", ""))
        subtitle = str(a.get("subtitle") or a.get("summary") or a.get("description") or a.get("lead") or "")
        body = str(a.get("body") or a.get("text") or a.get("content") or "")
        body = body[:1200]
        chunks.append(f"[{i}] date={date} outlet={outlet} topic={topic}\nTITLE: {title}\nSUBTITLE: {subtitle}\nBODY: {body}")
    text = "\n\n".join(chunks)
    return text[:max_chars]


def build_prompt(articles: List[Dict[str, Any]], rules_context: str = "") -> str:
    article_text = _compact_articles(articles)
    return f"""당신은 뉴스 시계열 논조 분석기의 sLLM 추출기입니다.
아래 기사 묶음을 읽고 다섯 가지 수치 지표를 0~100 사이 정수로 산출하세요. 제목과 부제는 본문보다 강한 프레임 신호로 고려하되, 단일 표현만으로 과잉 판정하지 마세요.
반드시 JSON 하나만 출력하세요. 설명 문장은 reason 안에만 넣으세요.

평가 지표:
- temporal_shift: 같은 사안에 대한 과거-현재 논조 이동 정도
- frame_effect: 표면 명제를 넘어 실제로 작동하는 프레임 효과의 강도
- context_omission: 배경/조건/전환 사유/반론이 누락된 정도
- consensus_deviation: 공정성, 책임성, 투명성 같은 사회적 기준에서 이탈한 정도
- evidence_quality: 출처의 불분명함, 익명 관계자 인용, 검증되지 않은 사실 주장 등 증거 품질의 취약 정도

참조 규칙 맥락:
{rules_context[:1200]}

기사 묶음:
{article_text}

출력 형식:
{{
  "temporal_shift": 0,
  "frame_effect": 0,
  "context_omission": 0,
  "consensus_deviation": 0,
  "evidence_quality": 0,
  "reason": "1~3문장 근거"
}}
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """sLLM 응답 텍스트에서 유효한 JSON 객체를 추출한다.

    0.5B 소형 모델은 답변 앞뒤에 불필요한 설명을 붙이거나 JSON 블록을 여러 개 뱉는
    경우가 많다. rfind 방식 대신 중첩 없는 가장 작은 {...} 블록들을 순서대로 시도해
    첫 번째 파싱 성공 블록을 반환한다.
    """
    # 1순위: 코드펜스 안의 JSON
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 2순위: 중첩 없는 {...} 블록을 작은 것부터 순차 시도 (개선2)
    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    # 3순위: 첫 { ~ 마지막 } (최후 수단)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("LLM 응답에서 JSON 객체를 찾지 못했습니다.")


def _normalize_features(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in DIMENSIONS:
        value = data.get(key, 0)
        try:
            num = float(value)
        except Exception:
            num = 0.0
        out[key] = max(0.0, min(100.0, num))
    return out


def _tokenize_ko(text: str) -> set:
    """한글 어절 토큰화 (경량 RAG용). 2글자 이상 어절만 유지."""
    return {w for w in re.split(r"\s+", text) if len(w) >= 2}


def _rule_text(r: Dict[str, Any]) -> str:
    """룰의 검색 대상 텍스트를 합성한다."""
    parts = [
        r.get("llm_instruction_ko", ""),
        " ".join(r.get("positive_cues", [])),
        " ".join(r.get("negative_indicators", [])),
        r.get("target_frame", ""),
        r.get("context", ""),
        r.get("dimension", ""),
    ]
    return " ".join(p for p in parts if p)


def search_relevant_rules(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    top_k: int = 8,
    dense_model: Optional[Any] = None,
    rule_embeddings: Optional[Any] = None,
    openai_api_key: str = "",
    openai_rule_embeddings: Optional[List[List[float]]] = None,
    openai_embed_model: str = DEFAULT_OPENAI_EMBED_MODEL,
    min_dense_score: Optional[float] = None,
) -> str:
    """기사 텍스트와 룰 간의 연관성을 계산하여 top_k개를 선택해 반환한다.

    Dense 우선순위:
      1) dense_model+rule_embeddings(ko-sroberta) → 기존 의미 검색 (불변)
      2) 없고 openai_api_key 있음 → OpenAI 임베딩 Dense (클라우드; ko-sroberta와 다른 의미공간)
      3) 둘 다 없음 → Jaccard(어절 겹침) Sparse 검색 (불변)
    """
    # 기사 텍스트 합성
    article_text = " ".join(
        " ".join([
            str(a.get("title", "")),
            str(a.get("title", "")),
            str(a.get("subtitle") or a.get("summary") or a.get("description") or a.get("lead") or ""),
            str(a.get("body") or a.get("text") or a.get("content") or ""),
        ])
        for a in articles
    )
    if not article_text.strip():
        # fallback: 기사가 없으면 기존 방식으로 상위 k개 반환
        return _format_rules(rules_data.get("rules", [])[:top_k])

    rules_list = rules_data.get("rules", [])
    scored: List[Tuple[float, Dict[str, Any]]] = []
    dense_path_used = False

    if dense_model is not None and rule_embeddings is not None:
        # ── 1) ko-sroberta Dense (기존, 불변) ──
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query_emb = dense_model.encode([article_text])
            sims = cosine_similarity(query_emb, rule_embeddings)[0]
            scored = [(sim, r) for sim, r in zip(sims, rules_list)]
            dense_path_used = True
        except Exception:
            scored = []
            dense_path_used = False
    elif openai_api_key:
        # ── 2) OpenAI Dense (신규, 클라우드) ──
        try:
            q = openai_embed_texts([article_text], api_key=openai_api_key, model=openai_embed_model)
            if q:
                rv = openai_rule_embeddings or openai_embed_texts(
                    [_rule_text(r) for r in rules_list], api_key=openai_api_key, model=openai_embed_model
                )
                if rv:
                    scored = [(_cosine_py(q[0], vec), r) for vec, r in zip(rv, rules_list)]
                    dense_path_used = True
        except Exception:
            scored = []
            dense_path_used = False
    else:
        # ── 3) Sparse (Jaccard, 기존 불변) ──
        query_tokens = _tokenize_ko(article_text)
        if not query_tokens:
            return _format_rules(rules_list[:top_k])
        for r in rules_list:
            rule_tokens = _tokenize_ko(_rule_text(r))
            if not rule_tokens:
                continue
            intersection = len(query_tokens & rule_tokens)
            union = len(query_tokens | rule_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            scored.append((jaccard, r))

    if not scored:
        return _format_rules(rules_list[:top_k])

    scored.sort(key=lambda x: x[0], reverse=True)

    # LLM 프롬프트 주입용 RAG는 항상 top_k 후보를 확보하는 것이 우선이다.
    # Evidence Panel(compare_rag_results)은 cutoff를 엄격히 적용해 화면 표시용 필터링을 수행하지만,
    # 여기서는 cutoff가 너무 높아 후보가 0~소수로 줄어 LLM이 룰 맥락 없이 feature를 추출하는 일을 막는다.
    # 따라서 Dense 경로에서 cutoff 통과 후보가 top_k개 이상일 때만 필터 결과를 쓰고,
    # 부족하면 unfiltered top_k로 폴백한다. Sparse 점수에는 Dense cutoff를 적용하지 않는다.
    if min_dense_score is not None and dense_path_used:
        filtered_scored = [(float(s), r) for s, r in scored if float(s) >= float(min_dense_score)]
        if len(filtered_scored) >= min(top_k, len(scored)):
            scored = filtered_scored

    top_rules = [r for _, r in scored[:top_k]]
    return _format_rules(top_rules)


def _format_rules(rules: List[Dict[str, Any]]) -> str:
    """룰 리스트를 프롬프트에 넣을 문자열로 포매팅한다."""
    lines = []
    for r in rules:
        lines.append(
            f"- {r.get('rule_id')}: dimension={r.get('dimension')}, frame={r.get('target_frame')}, "
            f"context={r.get('context')}, cue={', '.join(r.get('positive_cues', [])[:2])}"
        )
    return "\n".join(lines)


def compare_rag_results(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    top_k: int = 8,
    dense_model: Optional[Any] = None,
    rule_embeddings: Optional[Any] = None,
    min_sparse_score: float = 0.001,
    min_dense_score: float = 0.15,
    openai_api_key: str = "",
    openai_rule_embeddings: Optional[List[List[float]]] = None,
    openai_embed_model: str = DEFAULT_OPENAI_EMBED_MODEL,
) -> Dict[str, Any]:
    """같은 기사에 Sparse(Jaccard)와 Dense(ko-sroberta 또는 OpenAI)를 동시에 실행해 비교한다.

    Dense 엔진 우선순위:
      1) ko-sroberta(dense_model+rule_embeddings) → dense_engine="ko-sroberta"
      2) 없고 openai_api_key → OpenAI 임베딩 → dense_engine=<openai 모델명>
      3) 둘 다 없음 → Dense 비활성(Sparse-only)
    두 엔진은 의미공간이 달라 cosine 분포·임계값(min_dense_score)이 서로 다르다. 특히 OpenAI Dense에서 0.15는 확정 기준이 아니라 실측 보정 전 임시 cutoff다.
    """
    # 1. Sparse RAG (Jaccard Similarity)
    article_text = " ".join(
        " ".join(str(a.get(k, "")) for k in ["title", "subtitle", "summary", "body", "text", "content"] if a.get(k))
        for a in articles
    )
    article_tokens = set(re.findall(r"[0-9A-Za-z가-힣_]+", article_text.lower()))
    sparse_scores = []
    
    rules = rules_data.get("rules", [])
    for rule in rules:
        rt = _rule_text(rule).lower()
        rule_tokens = set(re.findall(r"[0-9A-Za-z가-힣_]+", rt))
        if not rule_tokens or not article_tokens:
            score = 0.0
        else:
            score = len(article_tokens & rule_tokens) / max(1, len(article_tokens | rule_tokens))
        if score >= min_sparse_score:
            sparse_scores.append((score, rule))
            
    sparse_scores.sort(key=lambda x: x[0], reverse=True)
    sparse_top = sparse_scores[:top_k]
    
    # 2. Dense RAG
    dense_available = False
    dense_top = []
    dense_filtered_count = 0
    dense_engine = None
    
    if dense_model is not None and rule_embeddings is not None:
        # ── 1) ko-sroberta Dense (기존, 불변) ──
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query_emb = dense_model.encode([article_text])
            sims = cosine_similarity(query_emb, rule_embeddings)[0]
            
            dense_scores = []
            for sim, r in zip(sims, rules):
                sim_val = float(sim)
                if sim_val >= min_dense_score:
                    dense_scores.append((sim_val, r))
            dense_scores.sort(key=lambda x: x[0], reverse=True)
            dense_top = dense_scores[:top_k]
            dense_filtered_count = len(dense_scores)
            dense_available = True
            dense_engine = "ko-sroberta"
        except Exception:
            dense_available = False
            dense_top = []
            dense_filtered_count = 0
    elif openai_api_key:
        # ── 2) OpenAI Dense (신규, 클라우드) ──
        try:
            q = openai_embed_texts([article_text], api_key=openai_api_key, model=openai_embed_model)
            if q:
                rv = openai_rule_embeddings or openai_embed_texts(
                    [_rule_text(r) for r in rules], api_key=openai_api_key, model=openai_embed_model
                )
                if rv:
                    dense_scores = []
                    for vec, r in zip(rv, rules):
                        sim_val = _cosine_py(q[0], vec)
                        if sim_val >= min_dense_score:
                            dense_scores.append((sim_val, r))
                    dense_scores.sort(key=lambda x: x[0], reverse=True)
                    dense_top = dense_scores[:top_k]
                    dense_filtered_count = len(dense_scores)
                    dense_available = True
                    dense_engine = openai_embed_model
        except Exception:
            dense_available = False
            dense_top = []
            dense_filtered_count = 0

    # 3. Intersect / Difference analysis
    sparse_ids = [r.get("rule_id") for _, r in sparse_top]
    dense_ids = [r.get("rule_id") for _, r in dense_top] if dense_available else []
    
    common_ids = list(set(sparse_ids) & set(dense_ids))
    sparse_only_ids = list(set(sparse_ids) - set(dense_ids))
    dense_only_ids = list(set(dense_ids) - set(sparse_ids))
    
    return {
        "sparse_available": bool(sparse_top),
        "dense_available": dense_available,
        "dense_engine": dense_engine,
        "dense_threshold_note": _dense_threshold_note(dense_engine, min_dense_score),
        "sparse_top": sparse_top,
        "dense_top": dense_top,
        "common_ids": common_ids,
        "sparse_only_ids": sparse_only_ids,
        "dense_only_ids": dense_only_ids,
        "min_sparse_score": min_sparse_score,
        "min_dense_score": min_dense_score,
        "sparse_filtered_count": len(sparse_scores),
        "dense_filtered_count": dense_filtered_count,
    }


def extract_features_with_sllm(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 220,
    temperature: float = 0.1,
    dense_model: Optional[Any] = None,
    rule_embeddings: Optional[Any] = None,
    api_key: Optional[str] = None,
    min_dense_score: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run extraction using OpenAI GPT or a local Hugging Face model.

    If OpenAI GPT model is specified, it runs API call without needing transformers/torch.
    """
    rag_selected_rules = search_relevant_rules(
        articles, rules_data, dense_model=dense_model, rule_embeddings=rule_embeddings,
        openai_api_key=(api_key or os.environ.get("OPENAI_API_KEY", "")),
        min_dense_score=min_dense_score,
    )
    prompt = build_prompt(articles, rag_selected_rules)

    is_openai = model_name.lower().startswith("gpt-") or "gpt" in model_name.lower()

    if is_openai:
        import os
        active_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not active_api_key:
            raise ValueError(
                "OpenAI API Key가 설정되지 않았습니다. 사이드바에 입력하거나 "
                "환경변수(OPENAI_API_KEY)에 설정해 주세요."
            )
        
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {active_api_key}"
        }
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only extraction engine. "
                        "Return exactly one valid JSON object and no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API 호출 실패 (상태 코드 {response.status_code}): {response.text}")
        
        res_json = response.json()
        raw = res_json["choices"][0]["message"]["content"]
    else:
        try:
            from transformers import pipeline
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "로컬 sLLM 모드를 사용하려면 transformers/torch가 필요합니다. "
                "OpenAI GPT를 사용하시려면 모델명을 'gpt-4o' 등으로 지정하고 API 키를 입력하세요. "
                "또는 로컬 sLLM을 쓰시려면 pip install -r requirements-sllm.txt 를 실행하세요."
            ) from e

        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=True,
        )
        outputs = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            return_full_text=False,
        )
        raw = outputs[0].get("generated_text", "") if outputs else ""

    try:
        parsed = _extract_json(raw)
    except ValueError as e:
        raw_preview = (raw or "")[:500].replace("\n", " ")
        if is_openai and is_cloud_mode():
            raise ValueError(
                "Cloud OpenAI 응답에서 JSON 객체를 찾지 못했습니다. "
                "모델이 JSON 형식을 지키지 않았거나 출력이 중간에 잘렸을 수 있습니다. "
                "모델명을 gpt-4o-mini 또는 gpt-4o로 확인하고, max_new_tokens를 늘려보세요. "
                f"응답 미리보기: {raw_preview}"
            ) from e
        if is_openai:
            raise ValueError(
                "로컬 실행 중 OpenAI 응답에서 JSON 객체를 찾지 못했습니다. "
                "response_format 설정, 모델명, max_new_tokens를 확인하세요. "
                f"응답 미리보기: {raw_preview}"
            ) from e
        raise ValueError(
            "로컬 sLLM 응답에서 JSON 객체를 찾지 못했습니다. "
            "소형 Hugging Face 모델이 JSON 형식을 지키지 않았을 가능성이 큽니다. "
            "max_new_tokens를 늘리거나 GPT 계열 모델을 사용해 보세요. "
            f"응답 미리보기: {raw_preview}"
        ) from e

    features = _normalize_features(parsed)
    meta = {
        "model": model_name,
        "raw_response": raw,
        "reason": parsed.get("reason", ""),
        "prompt_preview": prompt[:1600],
        "rag_selected_rules": rag_selected_rules,
        "rag_mode": (
            "dense_ko_sroberta" if (dense_model is not None and rule_embeddings is not None)
            else ("dense_openai" if (api_key or os.environ.get("OPENAI_API_KEY", "")) else "sparse_jaccard")
        ),
        "rag_min_dense_score": min_dense_score,
    }
    return features, meta

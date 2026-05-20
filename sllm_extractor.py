from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

DIMENSIONS = ["temporal_shift", "frame_effect", "context_omission", "consensus_deviation", "evidence_quality"]

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _compact_articles(articles: List[Dict[str, Any]], max_chars: int = 6500) -> str:
    chunks = []
    for i, a in enumerate(articles, 1):
        title = str(a.get("title", ""))
        date = str(a.get("date") or a.get("published_at") or "")
        outlet = str(a.get("outlet", ""))
        topic = str(a.get("topic", ""))
        body = str(a.get("body") or a.get("text") or a.get("content") or a.get("summary") or "")
        body = body[:1200]
        chunks.append(f"[{i}] date={date} outlet={outlet} topic={topic}\nTITLE: {title}\nBODY: {body}")
    text = "\n\n".join(chunks)
    return text[:max_chars]


def build_prompt(articles: List[Dict[str, Any]], rules_context: str = "") -> str:
    article_text = _compact_articles(articles)
    return f"""당신은 뉴스 시계열 논조 분석기의 sLLM 추출기입니다.
아래 기사 묶음을 읽고 다섯 가지 수치 지표를 0~100 사이 정수로 산출하세요.
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

    raise ValueError("sLLM 응답에서 JSON 객체를 찾지 못했습니다.")


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


def _score_sparse_rules(article_text: str, rules: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    """Sparse Jaccard 기반 룰 점수화. Dense 실패 시에도 같은 fallback을 사용한다."""
    query_tokens = _tokenize_ko(article_text)
    if not query_tokens:
        return []
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in rules:
        rule_tokens = _tokenize_ko(_rule_text(r))
        if not rule_tokens:
            continue
        intersection = len(query_tokens & rule_tokens)
        union = len(query_tokens | rule_tokens)
        jaccard = intersection / union if union > 0 else 0.0
        scored.append((jaccard, r))
    return scored


def _score_dense_rules(
    article_text: str,
    rules: List[Dict[str, Any]],
    dense_model: Any,
    rule_embeddings: Any,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Dense 임베딩 기반 코사인 유사도 점수화. 실패 시 빈 리스트."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = dense_model.encode([article_text])
        sims = cosine_similarity(query_emb, rule_embeddings)[0]
        return [(float(sim), r) for sim, r in zip(sims, rules)]
    except Exception:
        return []


def compare_rag_results(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    top_k: int = 8,
    dense_model: Optional[Any] = None,
    rule_embeddings: Optional[Any] = None,
    min_sparse_score: float = 0.001,
    min_dense_score: float = 0.15,
) -> Dict[str, Any]:
    """[v4 보강] Sparse와 Dense 검색을 *동시에* 실행해 결과를 나란히 반환한다.

    포트폴리오/시연 가치:
      *"이중 RAG"*가 정말로 *두 검색기를 동시에* 작동시키는 모습 시연.
      Sparse는 *키워드 매칭* (고유명사·정확 일치에 강함),
      Dense는 *의미 임베딩* (구조적 유사성에 강함).
      두 결과의 *교집합/차집합*이 그 자체로 정보가 됨.

    Returns:
        {
            "sparse_top": [(score, rule_dict), ...],  # 상위 top_k
            "dense_top": [(score, rule_dict), ...],
            "sparse_available": bool,
            "dense_available": bool,
            "common_ids": [rule_id, ...],           # Sparse ∩ Dense
            "sparse_only_ids": [rule_id, ...],      # Sparse - Dense
            "dense_only_ids": [rule_id, ...],       # Dense - Sparse
            "article_text_length": int,
            "min_sparse_score": float,
            "min_dense_score": float,
            "sparse_filtered_count": int,
            "dense_filtered_count": int,
        }
    """
    article_text = " ".join(
        " ".join(str(a.get(k, "")) for k in ["title", "body", "summary", "text", "content"] if a.get(k))
        for a in articles
    )
    rules = rules_data.get("rules", [])
    result = {
        "sparse_top": [],
        "dense_top": [],
        "sparse_available": False,
        "dense_available": False,
        "common_ids": [],
        "sparse_only_ids": [],
        "dense_only_ids": [],
        "article_text_length": len(article_text),
        "min_sparse_score": float(min_sparse_score),
        "min_dense_score": float(min_dense_score),
        "sparse_filtered_count": 0,
        "dense_filtered_count": 0,
    }

    if not article_text.strip() or not rules:
        return result

    # Sparse는 항상 시도
    sparse_scored = _score_sparse_rules(article_text, rules)
    if sparse_scored:
        sparse_scored.sort(key=lambda x: x[0], reverse=True)
        sparse_filtered = [(score, rule) for score, rule in sparse_scored if float(score) >= min_sparse_score]
        result["sparse_filtered_count"] = max(0, len(sparse_scored) - len(sparse_filtered))
        result["sparse_top"] = sparse_filtered[:top_k]
        result["sparse_available"] = bool(result["sparse_top"])

    # Dense는 모델이 있을 때만
    if dense_model is not None and rule_embeddings is not None:
        dense_scored = _score_dense_rules(article_text, rules, dense_model, rule_embeddings)
        if dense_scored:
            dense_scored.sort(key=lambda x: x[0], reverse=True)
            dense_filtered = [(score, rule) for score, rule in dense_scored if float(score) >= min_dense_score]
            result["dense_filtered_count"] = max(0, len(dense_scored) - len(dense_filtered))
            result["dense_top"] = dense_filtered[:top_k]
            result["dense_available"] = bool(result["dense_top"])

    # 교집합/차집합 (rule_id 기준)
    sparse_ids = {r.get("rule_id") for _, r in result["sparse_top"]}
    dense_ids = {r.get("rule_id") for _, r in result["dense_top"]}
    result["common_ids"] = sorted(sparse_ids & dense_ids)
    result["sparse_only_ids"] = sorted(sparse_ids - dense_ids)
    result["dense_only_ids"] = sorted(dense_ids - sparse_ids)

    return result


def search_relevant_rules(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    top_k: int = 8,
    dense_model: Optional[Any] = None,
    rule_embeddings: Optional[Any] = None,
    return_mode: bool = False,
) -> Any:
    """기사 텍스트와 룰 간의 연관성을 계산하여 top_k개를 선택해 반환한다.

    dense_model과 rule_embeddings가 주어지면 Sentence-Transformers 기반의
    Dense(의미적) 검색을 수행하고, 없으면 기존의 Jaccard(어절 겹침) 검색을 수행한다.
    """
    # 기사 텍스트 합성
    article_text = " ".join(
        " ".join(str(a.get(k, "")) for k in ["title", "body", "summary", "text", "content"] if a.get(k))
        for a in articles
    )
    mode = "sparse_jaccard"
    rules = rules_data.get("rules", [])

    if not article_text.strip():
        formatted = _format_rules(rules[:top_k])
        return (formatted, "empty_article_fallback") if return_mode else formatted

    if dense_model is not None and rule_embeddings is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query_emb = dense_model.encode([article_text])
            sims = cosine_similarity(query_emb, rule_embeddings)[0]
            scored = [(float(sim), r) for sim, r in zip(sims, rules)]
            mode = "dense"
        except Exception:
            # Dense 검색 실패 시 룰셋 앞부분으로 떨어지지 말고 Sparse Jaccard로 재시도한다.
            scored = _score_sparse_rules(article_text, rules)
            mode = "dense_failed_sparse_fallback"
    else:
        scored = _score_sparse_rules(article_text, rules)
        mode = "sparse_jaccard"

    if not scored:
        formatted = _format_rules(rules[:top_k])
        return (formatted, f"{mode}_empty_scored_fallback") if return_mode else formatted

    scored.sort(key=lambda x: x[0], reverse=True)
    top_rules = [r for _, r in scored[:top_k]]
    formatted = _format_rules(top_rules)
    return (formatted, mode) if return_mode else formatted


def _format_rules(rules: List[Dict[str, Any]]) -> str:
    """룰 리스트를 프롬프트에 넣을 문자열로 포매팅한다."""
    lines = []
    for r in rules:
        lines.append(
            f"- {r.get('rule_id')}: dimension={r.get('dimension')}, frame={r.get('target_frame')}, "
            f"context={r.get('context')}, cue={', '.join(r.get('positive_cues', [])[:2])}"
        )
    return "\n".join(lines)


def extract_features_with_sllm(
    articles: List[Dict[str, Any]],
    rules_data: Dict[str, Any],
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 220,
    temperature: float = 0.1,
    dense_model: Optional[Any] = None,
    rule_embeddings: Optional[Any] = None,
    api_key: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run extraction using OpenAI GPT or a local Hugging Face model.

    If OpenAI GPT model is specified, it runs API call without needing transformers/torch.
    """
    rag_selected_rules, actual_rag_mode = search_relevant_rules(
        articles,
        rules_data,
        dense_model=dense_model,
        rule_embeddings=rule_embeddings,
        return_mode=True,
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
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_new_tokens
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

    parsed = _extract_json(raw)
    features = _normalize_features(parsed)
    meta = {
        "model": model_name,
        "raw_response": raw,
        "reason": parsed.get("reason", ""),
        "prompt_preview": prompt[:1600],
        "rag_selected_rules": rag_selected_rules,
        "rag_mode": actual_rag_mode,
    }
    return features, meta

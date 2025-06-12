"""
은행,상품명 다중 추출 + LLM 보정 모듈
"""
from __future__ import annotations
import json, re
from pathlib import Path
from functools import lru_cache
from difflib import SequenceMatcher
from openai import OpenAI                       # openai-python ≥1.3

FINDATA_DIR = Path(__file__).resolve().parents[1] / "findata"

@lru_cache
def _load_sets() -> tuple[set[str], set[str]]:
    banks, prods = set(), set()

    def _walk(item: dict):
        """단일 엔트리(dict)에서 은행·상품명 추출"""
        banks.add(item.get("bank", ""))
        prods.add(item.get("product_name", ""))

    for fp in FINDATA_DIR.glob("*.json"):
        data = json.load(open(fp, encoding="utf-8"))

        # ① 리스트 형태  -----------------------
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict):
                    _walk(it)

        # ② dict(wrapper) 형태  ---------------
        elif isinstance(data, dict):
            for v in data.values():          # products, items 등
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            _walk(it)
                elif isinstance(v, dict):
                    _walk(v)
    # 빈 문자열 제거
    return banks - {""}, prods - {""}


def _fuzzy(cand: str, text: str, thr=0.8) -> bool:
    a = "".join(cand.lower().split())
    b = "".join(text.lower().split())
    return SequenceMatcher(None, a, b).ratio() >= thr or a in b

def _rough_pick(q: str, pool: set[str]) -> list[str]:
    return [p for p in pool if _fuzzy(p, q)]

def _llm_refine(q: str, rough: list[str], etype: str) -> list[str]:
    if len(rough) >= 2:                # 2개 이상이면 LLM 생략
        return []
    prompt = f"""문장: {q}
추출해야 할 항목: {('은행명' if etype=='bank' else '상품명')}
결과를 쉼표로 구분된 한 줄로만 출력."""
    res = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0
    ).choices[0].message.content
    pools = _load_sets()[0 if etype=="bank" else 1]
    return [p for p in pools if _fuzzy(p, res)]

def extract_banks(q: str) -> list[str]:
    banks, _ = _load_sets()
    rough = _rough_pick(q, banks)
    return list(dict.fromkeys(rough + _llm_refine(q, rough, "bank")))

def extract_products(q: str) -> list[str]:
    _, prods = _load_sets()
    rough = _rough_pick(q, prods)
    return list(dict.fromkeys(rough + _llm_refine(q, rough, "prod")))

def extract_entity_pairs(q: str) -> list[dict]:
    """
    LLM으로 “은행명”과 “상품명”을 짝지어 JSON array로 추출.
    예시 출력: [{"bank":"광주은행","product":"The플러스예금"}, …]
    """
    prompt = f"""
    You are a JSON-only output machine. Given an input sentence, extract all (bank, product) pairs.
    Output **only** a JSON array of objects, each with exactly two keys: "bank" and "product".
    Do **not** output any extra text, markdown, or explanation.

    Input sentence:
    {q}

    Output format example (no extra whitespace or comments):
    [
    {{"bank": "A은행", "product": "상품1"}},
    {{"bank": "B은행", "product": "상품2"}}
    ]
    """
    # 1) LLM 호출
    resp = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0
    ).choices[0].message.content

    # 2) JSON 응답 블록만 추출 → 파싱
    m = re.search(r'\[.*\]', resp, flags=re.DOTALL)
    if not m:
        return []
    json_text = m.group(0)
    try:
        pairs = json.loads(json_text)
    except json.JSONDecodeError:
        # 트레일링 콤마 제거 후 재시도
        cleaned = re.sub(r',\s*([\]\}])', r'\1', json_text)
        try:
            pairs = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
    
    # 3) 유효성 검증: our DB에 있는 값인지 fuzzy 체크
    banks_set, prods_set = _load_sets()
    valid = []
    for pair in pairs:
        bank = pair.get("bank","").strip()
        prod = pair.get("product","").strip()
        if any(_fuzzy(bank, b) for b in banks_set) and any(_fuzzy(prod, p) for p in prods_set):
            valid.append({"bank": bank, "product": prod})
    return valid
# qd_mapping_multi.py
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from openai import AsyncOpenAI, OpenAIError

# ────────── 사용 예시 ───────────
# 기본값(샘플 100, 콤보당 질문 3, 동시 10요청)
# python qd_mapping_multi_async.py

# 콤보 60개, 질문 3개씩, 동시 8요청, 결과 파일 지정
# python qd_mapping_multi_async.py --sample 60 -n 3 --concurrency 8 --output results/multi.csv
# ──────────────────────────

# ──────────────────────────
# 설정 / 상수
# ──────────────────────────
DEFAULT_JSON_PATHS: tuple[str, ...] = (
    "./../findata/fixed_deposit.json",
    "./../findata/demand_deposit.json",
    "./../findata/loan.json",
    "./../findata/savings.json",
)
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
CATEGORIES = ["정기예금", "입출금자유예금", "적금", "대출"]

# ──────────────────────────
# OpenAI 비동기 클라이언트
# ──────────────────────────
openai = AsyncOpenAI()


# ──────────────────────────
# 1) 데이터 적재
# ──────────────────────────
def load_docs(paths: Sequence[str] = DEFAULT_JSON_PATHS) -> List[Dict]:
    docs: list[Dict] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fp:
            data = json.load(fp)
            docs.extend(data.get("documents", []))
    print(f"[INFO] Loaded {len(docs)} documents from {len(paths)} files")
    return docs

# ──────────────────────────
# 2) 콤보(문서 묶음) 생성
# ──────────────────────────
def build_combos(
    docs: Sequence[Dict],
    sample_size: int,
    random_seed: int = 42,
) -> List[Tuple[Dict, ...]]:
    """
    동일 type 내에서만 2개 또는 3개 조합을 만들고 샘플링.
    """
    # 1) 타입별로 문서 그룹화
    by_type: Dict[str, List[Dict]] = {}
    for d in docs:
        by_type.setdefault(d["type"], []).append(d)

    # 2) 각 타입별로 2개 조합, 3개 조합 생성
    same_pairs = [
        combo
        for same_docs in by_type.values()
        for combo in combinations(same_docs, 2)
    ]
    # same_triples = [
    #     combo
    #     for same_docs in by_type.values()
    #     for combo in combinations(same_docs, 3)
    # ]

    # 3) 최종 후보는 동일 타입 페어 + 동일 타입 트리플
    # all_combos = same_pairs + same_triples
    all_combos = same_pairs
    # print(
    #     f"[INFO] combos → same_pairs={len(same_pairs)}, "
    #     f"same_triples={len(same_triples)}, total={len(all_combos)}"
    # )

    print(
        f"[INFO] combos → same_pairs={len(same_pairs)}"
        )

    # 4) 샘플링
    random.seed(random_seed)
    sampled = random.sample(all_combos, k=min(sample_size, len(all_combos)))
    print(f"[INFO] Sampled {len(sampled)} combos (seed={random_seed})")
    return sampled


# ──────────────────────────
# 3) 비동기 질문 생성
# ──────────────────────────
def _build_prompt(combos: Sequence[Dict], n: int) -> str:
    items = "\n".join(
        f"- [{d['id']}] {d['bank']}의 {d['product_name']} ({d['type']})" for d in combos
    )
    return f"""
아래 상품들을 모두 참고해야 답할 수 있는 질문을
서로 다른 관점으로 {n}개 만들어 주세요.

상품 목록:
{items}

조건:
- 반드시 모든 문서를 비교하거나 조합해서 물어봐야 합니다.
- 가능한 실제 사용자가 챗봇에게 물어볼 법한 질문 형태로 자연스럽게 작성합니다.
- 일반 사용자가 챗봇에게 던질 법한 모호하고 자연스러운 질문 형태의 질문이 포함될 수 있음
- 모든 상품의 차이점·비교·조합 등의 맥락을 포함해야 합니다.
- 자산관리 금리·우대조건·가입방법·만기 후 이자 등을 다양하게 골고루 물어보세요.
- 실사용 시나리오(고객 유형, 투자 목적 등)를 반영하면 좋습니다.
- 고객 상황(연령대, 상황, 목적, 감정 등)을 반영하면 좋습습니다.


출력:
1. 질문1
2. 질문2
...
""".strip()

async def query_variants_multi_async(
    combo: Sequence[Dict],
    n: int,
    semaphore: asyncio.Semaphore,
    retry: int = 3,
    backoff: float = 2.0,
) -> List[str]:
    """콤보(2~3개 문서)에 대해 n개의 비교/조합 질문 생성."""
    prompt = _build_prompt(combo, n)

    async with semaphore:
        for attempt in range(1, retry + 1):
            try:
                resp = await openai.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                )
                lines = (l.strip() for l in resp.choices[0].message.content.splitlines())
                return [
                    l.split(".", 1)[1].strip()
                    for l in lines
                    if l and l[0].isdigit() and "." in l
                ][:n]
            except OpenAIError as e:
                if attempt == retry:
                    raise
                wait = backoff * (2 ** (attempt - 1)) + random.random()
                print(
                    f"[WARN] OpenAI error ({e}). retry {attempt}/{retry} in {wait:.1f}s"
                )
                await asyncio.sleep(wait)

    return []
# ──────────────────────────
# 4) CSV 저장
# ──────────────────────────
def save_csv(rows: List[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["doc_ids", "question"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] Wrote {len(rows)} rows to {path}")


# ──────────────────────────
# 5) 메인 비동기 파이프라인
# ──────────────────────────
async def build_mapping_async(
    json_paths: Sequence[str],
    output_csv: str,
    sample_size: int,
    n_variants_per_combo: int,
    concurrency: int,
):
    docs = load_docs(json_paths)
    combos = build_combos(docs, sample_size)

    semaphore = asyncio.Semaphore(concurrency)
    rows: list[Dict[str, str]] = []

    async def process_combo(combo: Tuple[Dict, ...]):
        doc_ids = [d["id"] for d in combo]
        qs = await query_variants_multi_async(
            combo, n=n_variants_per_combo, semaphore=semaphore
        )
        rows.extend({"doc_ids": ",".join(doc_ids), "question": q} for q in qs)
        print(f"  • combo {doc_ids} → {len(qs)} questions")

    await asyncio.gather(*(process_combo(c) for c in combos))
    save_csv(rows, output_csv)


# ──────────────────────────
# CLI 실행
# ──────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build multi-doc Q-D mapping CSV (async version)"
    )
    parser.add_argument("--output", default="qd_mapping_multi.csv", help="출력 CSV 경로")
    parser.add_argument(
        "--sample", type=int, default=50, help="콤보 샘플 수 (조합 개수)"
    )
    parser.add_argument(
        "-n",
        "--n_variants",
        type=int,
        default=2,
        help="콤보당 생성할 질문 개수",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="동시 OpenAI 요청 수(세마포어 한도)",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    random.seed(args.seed)

    asyncio.run(
        build_mapping_async(
            json_paths=DEFAULT_JSON_PATHS,
            output_csv=args.output,
            sample_size=args.sample,
            n_variants_per_combo=args.n_variants,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
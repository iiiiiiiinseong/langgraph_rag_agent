# qd_mapping.py
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence
from openai import AsyncOpenAI, OpenAIError


# ────────── 사용 예시 ───────────
# 기본 옵션 (출력 파일: qd_mapping.csv, 문서당 5개 질문, 동시 10요청)
# python qd_mapping_single_async.py

# 동시 요청 8개, 질문 3개씩, 다른 파일로 저장
# python qd_mapping_single_async.py --concurrency 8 -n 3 --output results/qd_map.csv
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

# — JSON 파일 경로 리스트 —
JSON_PATHS = [
    "./../findata/fixed_deposit.json",
    "./../findata/demand_deposit.json",
    "./../findata/loan.json",
    "./../findata/savings.json"
]

META_FIELDS: tuple[str, ...] = (
                #예금
                "기본금리(단리이자 %)",
                "최고금리(우대금리포함, 단리이자 %)",
                "가입방법",
                "만기 후 금리",
                "우대조건",
                #적금
                "적립방식",
                "세전이자율",
                "세후이자율",
                "최고우대금리",
                "가입대상",
                "이자계산방식",
                #대출출
                "대출종류",
                "주택종류",
                "금리방식",
                "상환방식",
                "최저금리",
                "최고금리",
                "가입방법",
                "대출 부대비용",
                "중도상환 수수료",
                "연체 이자율"
)

CATEGORIES = ["정기예금", "입출금자유예금", "적금", "대출"]

# ──────────────────────────
# OpenAI 비동기 클라이언트 (병렬 호출 위함)
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
# 2) 비동기 질문 생성
# ──────────────────────────
def _build_prompt(meta: Dict, n: int) -> str:
    meta_lines = "\n".join(f"- {k}: {v}" for k, v in meta.items())
    return f"""
다음은 금융상품 문서 메타정보입니다:
{meta_lines}

이 문서에 대해 사용자가 실제로 묻고 싶어할 만한,
서로 다른 {n}가지 질문을 생성해 주세요.
- 가능한 실제 사용자가 챗봇에게 물어볼 법한 질문 형태로 자연스럽게 작성
- 일반 사용자가 챗봇에게 던질 법한 모호하고 자연스러운 질문 형태의 질문이 포함될 수 있음
- 금융상품 카테고리: {', '.join(CATEGORIES)}
- 질문 목적(자산관리, 금리, 우대조건, 가입방법, 만기 후 이자 등)을 다양화
- 고객 상황(연령대, 상황, 목적, 감정 등)을 반영
- 문서의 'bank'와 'product_name'을 반드시 언급

포맷:
1. 질문1
2. 질문2
...
""".strip()

async def query_variants_async(
    meta: Dict,
    n: int = 5,
    retry: int = 3,
    backoff: float = 2.0,
    semaphore: asyncio.Semaphore | None = None,
) -> List[str]:
    """OpenAI 비동기 호출로 질문 n개 생성 (간단 재시도 포함)."""
    prompt = _build_prompt(meta, n)

    # 요청 동시 수 제한
    if semaphore is None:
        semaphore = asyncio.Semaphore(5)

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
                sleep = backoff * (2 ** (attempt - 1)) + random.random()
                print(f"[WARN] OpenAI error ({e}). retry {attempt}/{retry} in {sleep:.1f}s")
                await asyncio.sleep(sleep)
    print("실패 하여 빈 리스트를 반환합니다.")
    return []  # 실패 시 빈 리스트

# ──────────────────────────
# 3) CSV 저장
# ──────────────────────────
def save_csv(rows: List[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["doc_id", "question"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] Wrote {len(rows)} rows to {path}")

# ──────────────────────────
# 4) 메인 비동기 파이프라인
# ──────────────────────────
async def build_mapping_async(
    json_paths: Sequence[str],
    output_csv: str,
    n_variants_per_doc: int,
    concurrency: int,
):
    docs = load_docs(json_paths)
    semaphore = asyncio.Semaphore(concurrency)

    rows: list[Dict[str, str]] = []

    async def process_doc(doc: Dict):
        doc_id = doc.get("id", "")
        base_meta = {
            "bank": doc.get("bank", ""),
            "product_name": doc.get("product_name", ""),
            "type": doc.get("type", ""),
        }
        extra_meta = {
            k: str(v) for k, v in doc.get("metadata", {}).items() if k in META_FIELDS
        }
        meta = {**base_meta, **extra_meta}

        qs = await query_variants_async(
            meta,
            n=n_variants_per_doc,
            semaphore=semaphore,
        )
        rows.extend({"doc_id": doc_id, "question": q} for q in qs)
        print(f"  • {doc_id} → {len(qs)} questions")

    # 모든 문서를 병렬 처리
    await asyncio.gather(*(process_doc(d) for d in docs))
    save_csv(rows, output_csv)

# ──────────────────────────
# CLI 실행
# ──────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build single-doc Q-D mapping CSV (async version)"
    )
    parser.add_argument("--output", default="qd_mapping.csv", help="출력 CSV 경로")
    parser.add_argument(
        "-n", "--n_variants", type=int, default=5, help="문서당 생성할 질문 개수, 기본값은 5"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="동시 OpenAI 요청 수(세마포어 한도), 기본값은 10",
    )
    args = parser.parse_args()

    asyncio.run(
        build_mapping_async(
            json_paths=DEFAULT_JSON_PATHS,
            output_csv=args.output,
            n_variants_per_doc=args.n_variants,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
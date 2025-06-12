# evaluate_retrieval.py
# Adaptive-Self RAG 프로젝트의 **검색·라우팅 통합 성능**을 자동으로 측정한다.

# ※ 검색 알고리즘(2종) x 임베딩 모델(4종) = 8개 조합을
#    단일 / 다중 Q-D 매핑에 대해 한 번에 평가한다.

#  임베딩 모델  
#    • `bge`  →  *juampahc/bge-m3-m2v*  
#    • `mini` →  *all-MiniLM-L6-v2*  
#    • `ko`   →  *jhgan/ko-sroberta-multitask*
#    • `kf`   →  *upskyy/kf-deberta-multitask*  

#  검색 알고리즘  
#    • `vector`  → Chroma 유사도 검색만 사용  
#    • `hybrid`  → `adaptive_self_rag.py`의 BM25 + Vector 하이브리드 사용


# 각 조합별 JSON·PNG와 전체 Heatmap 이미지를 `assets/` 폴더에 저장합니다.

# CLI Example
# ───────────
# $ python evaluate_retrieval.py --search_algo hybrid --embed_model ko --workers 4

from __future__ import annotations
import os
os.environ["CHROMA_DOWNLOAD_PARALLELISM"] = "1"   # Chroma가 한번만 내려받음

import sys
import logging
import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple
from transformers import AutoModel, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

import asyncio
from async_lru import alru_cache

from adaptive_self_rag import (
    hybrid_core_search,
    retrieval_grader_binary,
    extract_banks, 
    extract_products,
    get_banks_in_docs,
)


# ────────────────────────────────────────────────────────────────────────────────
# 전역변수·경로·상수 정의
# ────────────────────────────────────────────────────────────────────────────────

router_raw: any = None
rewrite_raw: any = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # langgraph_rag_agent/
SRC_ROOT = Path(__file__).resolve().parent                   # langgraph_rag_agent/src/
FINDATA_DIR = PROJECT_ROOT / "findata"
RESULTS_DIR = PROJECT_ROOT / "results"

MAPPING_FILES = {
    "single": SRC_ROOT / "qd_mapping.csv",
    "multi":  SRC_ROOT / "qd_mapping_multi.csv",
}

EMBED_MODELS = {
    "bge": "juampahc/bge-m3-m2v", # 256차원
    "mini": "all-MiniLM-L6-v2",   # 384차원
    "ko":   "jhgan/ko-sroberta-multitask", # 카카오뱅크에서 만든 금융 도메인 임베딩 모델을 파인튜닝 768차원
    "kf":   "upskyy/kf-deberta-multitask", # jhgan/ko-sroberta-multitask 모델보다 더 좋게 튜닝한 모델 (768차원) https://github.com/upskyy/kf-deberta-multitask
}

COLLECTION_FMT = "combined_products_{embed}"

DEFAULT_TOP_K = 3        # 검색 결과 수
MAX_CYCLES = 1           # 1: 재작성 없이 1회 탐색
SEARCH_ALGOS = ["vector","hybrid"]
EMBED_KEYS   = ["bge", "mini", "ko", "kf"]
PRODUCT_CATEGORIES = ["정기예금", "입출금자유예금", "적금", "대출"]

# ─────────────────────────────────────────────────────────────────
# 전역 토큰버킷(초당 8요청 ≈ 480RPM)  
# ─────────────────────────────────────────────────────────────────

# 비동기용 rate-limit
TOKENS = 8
router_sema = asyncio.Semaphore(TOKENS)
rewrite_sema = asyncio.Semaphore(TOKENS)

# ────────────────────────────────────────────────────────────────────────────────
# 로깅
# ────────────────────────────────────────────────────────────────────────────────

# 전체 로그 수준은 INFO (또는 DEBUG)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# 1. OpenAI 요청 관련 로깅 줄이기
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# 2. Gradio 요청 관련 로깅 줄이기
logging.getLogger("gradio").setLevel(logging.WARNING)


# ────────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class MappingItem:
    """질문-정답(문서 ID) 매핑 구조."""
    question: str
    gold_docs: List[str]

def parse_mapping_csv(path: Path) -> List[MappingItem]:
    """CSV → MappingItem 리스트 변환."""
    df = pd.read_csv(path)
    # 1) 단일 매핑: doc_id 컬럼
    if "doc_id" in df.columns:
        return [
            MappingItem(q, [str(d)]) 
            for q, d in zip(df["question"], df["doc_id"])
        ]

    # 2) 다중 매핑: doc_ids (콤마 구분) 컬럼
    if "doc_ids" in df.columns:
        return [
            MappingItem(q, [s.strip() for s in d.split(",")]) 
            for q, d in zip(df["question"], df["doc_ids"])
        ]

# ────────────────────────────────────────────────────────────────────────────────
# Chroma 컬렉션 로드·캐시
# ────────────────────────────────────────────────────────────────────────────────

def load_or_ingest_collection(embed_key: str):
    """임베딩 키 기반 컬렉션 로드(없으면 인제스트)."""
    from chromadb import Settings
    from sentence_transformers import SentenceTransformer
    from chromadb import PersistentClient
    from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE

    embed_name = EMBED_MODELS[embed_key]

    embedder = SentenceTransformer(embed_name, device="cpu")

    persist_dir = FINDATA_DIR / f"chroma_db_{embed_key}" # persist 디렉토리 embed_key별로 분리 (임베딩 모델에 따라 차원수가 달라 따로 만들어서 평가 진행)
    persist_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(anonymized_telemetry=False)

    client = PersistentClient(
        path=str(persist_dir),
        settings=settings,
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )

    # 컬렉션 가져오거나 생성
    col = client.get_or_create_collection(
        name=COLLECTION_FMT.format(embed=embed_key)
    )

    # 이미 한 번 인제스트된 컬렉션이면 건너뛰기
    if col.count() > 0:
        return col, embedder

    # 인제스트
    ids, texts, metas = [], [], []
    for fp in FINDATA_DIR.glob("*.json"):
        data = json.load(open(fp, encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("documents", [])
        for it in items:
                prod_id = it.get("id")  # '정기예금_023'
                if not prod_id:
                    continue
                ids.append(prod_id)
                texts.append(it.get("description", json.dumps(it, ensure_ascii=False)))
                metas.append({
                    "file": fp.name,
                    "product_name": it.get("product_name"),
                    "category": fp.stem})
                
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64).tolist()
    col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)  # type: ignore[arg-type]
    return col, embedder

# ────────────────────────────────────────────────────────────────────────────────
# 검색 함수 생성
# ────────────────────────────────────────────────────────────────────────────────

def make_vector_search(collection, embedder):
    def _search(q: str, k: int = DEFAULT_TOP_K):
        q_emb = embedder.encode([q], show_progress_bar=False).tolist() # 쿼리 임베딩 (리스트 단위로 호출)
        out = collection.query(query_embeddings=q_emb, n_results=k)
        return list(out["ids"][0])                                     # type: ignore[index]
    return _search

def rag_search_with_filter(question: str,
                           initial_docs: list,
                           top_k: int) -> list:
    """
    1) initial_docs: hybrid_core_search 로 뽑힌 Document 리스트
    2) retrieval_grader_binary 로 binary relevance 평가
    3) 은행명 누락 시 재검색 (최대 1회)
    4) 최종 filtered Document 리스트 반환
    """
    # (1) 관련성 평가
    filtered = []
    for d in initial_docs:
        score = retrieval_grader_binary.invoke({
            "question": question,
            "document": d.page_content
        })
        if getattr(score, "binary_score", None) == "yes":
            filtered.append(d)

    # (2) 은행 엔티티 추출 & 문서 내 등장 은행
    requested = extract_banks(question)
    found = get_banks_in_docs(filtered)

    state = {"missing_bank_retry": 0}
    # (3) 누락 은행 보완 재검색 1회
    missing = [b for b in requested if b not in found]
    if missing and state["missing_bank_retry"] < 1:
        covered = {(d.metadata.get("bank"), d.metadata.get("type")) for d in filtered}
        for bank in missing:
            for category in ["정기예금","입출금자유예금","적금","대출"]:
                if (bank, category) in covered:
                    continue
                more = hybrid_core_search(question, category=category,
                                          bank=bank, top_k=top_k)
                for d in more:
                    key = (d.metadata.get("bank"), d.metadata.get("type"))
                    if key not in covered:
                        score = retrieval_grader_binary.invoke({
                            "question": question,
                            "document": d.page_content
                        })
                        if getattr(score, "binary_score",None)=="yes":
                            filtered.append(d)
                            covered.add(key)
        state["missing_bank_retry"] += 1

    # (4) 최종 k개만
    return filtered[:top_k]

def make_hybrid_search(collection, map_dict):
    """
    RAG 시스템 전체 탐색 로직을 흉내 낸 hybrid search wrapper.
    - initial hybrid_core_search → rag_search_with_filter 적용 → ID 리스트 반환
    """
    def _search(q: str, k: int = DEFAULT_TOP_K):
        # 1) 질문에서 은행·상품명 복수 추출 (service logic 그대로)
        banks    = extract_banks(q)
        products = extract_products(q)

        # 2) gold 매핑에서 카테고리 집합 결정
        gold_list = map_dict.get(q)
        if gold_list:
            cats = {doc_id.split("_",1)[0] for doc_id in gold_list}
        else:
            cats = PRODUCT_CATEGORIES

        initial = []
        # 3) 각 카테고리별 하이브리드 서치 호출 (은행·상품 파라미터 포함)

        # 각 카테고리별 hybrid_core_search
        initial = []
        for cat in cats:
            docs = hybrid_core_search(
                q,
                category=cat,
                bank=banks    if banks    else None,
                product_name=products if products else None,
                top_k=k
            )            
            initial.extend(docs)

        # RAG 필터링 로직 (생략) → 최종 docs
        final = rag_search_with_filter(q, initial, k)
        return [d.metadata["id"] for d in final]

    return _search
# ────────────────────────────────────────────────────────────────────────────────
# 라우터·재작성 로드
# ────────────────────────────────────────────────────────────────────────────────

def load_router_and_rewriter():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("adaptive_self_rag",
                                                  SRC_ROOT / "adaptive_self_rag.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module   # type: ignore[arg-type]
    spec.loader.exec_module(module)                                                   # type: ignore[call-arg]
    return module.question_router, module.rewrite_question                            # type: ignore[attr-defined]

# router_raw: load_router_and_rewriter()로 얻은 sync 객체
@alru_cache(maxsize=4096)
async def async_route(q: str) -> str:
    # rate-limit
    async with router_sema:
        # sync invoke()를 스레드풀에서 실행
        result = await asyncio.to_thread(router_raw.invoke, {"question": q})
    # route 속성 또는 직접 문자열 반환
    return getattr(result, "route", result)

@alru_cache(maxsize=4096)
async def async_rewrite(q: str) -> str:
    async with rewrite_sema:
        # sync rewrite_raw(q)를 스레드풀에서 실행
        return await asyncio.to_thread(rewrite_raw, q)

# ────────────────────────────────────────────────────────────────────────────────
# 메트릭 계산
# ────────────────────────────────────────────────────────────────────────────────

def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

async def eval_one_async(
    question: str,
    gold: Sequence[str],
    router: Callable[[str], str],
    search: Callable[[str, int], Sequence[str]],
    rewrite: Callable[[str], str],
) -> Tuple[float, float, float, int, str, List[str]]:
    """
    비동기 평가: 
    1) router(question) 으로 분기  
    2) search/rewrite 로 검색 반복  
    3) precision/recall/f1 계산  
    """
    route_label = await async_route(question)
    route_ok = int(route_label == "search_data")

    preds: List[str] = []
    if route_ok:
        q = question
        for _ in range(MAX_CYCLES):
            # search는 sync → to_thread
            preds = await asyncio.to_thread(search, q, DEFAULT_TOP_K)
            if preds:
                break
            # 재작성
            q = await async_rewrite(q)

    # 2) precision/recall/F1 계산 (동기)
    tp = len(set(preds) & set(gold))
    precision = tp / len(preds) if preds else 0.0
    recall    = tp / len(gold)  if gold   else 0.0
    f1_val    = f1(precision, recall)
    return precision, recall, f1_val, route_ok, route_label, preds


# ────────────────────────────────────────────────────────────────────────────────
# 데이터셋 단위 평가
# ────────────────────────────────────────────────────────────────────────────────

async def evaluate_async(
    mapping: List[MappingItem],
    router,
    search: Callable[[str,int], List[str]],
    rewrite,
    workers: int,
    description: str = "Evaluating Q-D Pairs",
) -> Tuple[dict, List[dict]]:
    sem = asyncio.Semaphore(workers)
    agg = defaultdict(list)
    details: List[dict] = []
    
    async def worker(m: MappingItem):
        async with sem:
            p, r, f, ok, lbl, preds = await eval_one_async(m.question, m.gold_docs, router, search, rewrite)
            agg["p"].append(p); agg["r"].append(r); agg["f"].append(f); agg["ok"].append(ok)
            agg["lbl"].append(lbl)
            details.append({
                "question": m.question,
                "gold":     m.gold_docs,
                "predicted":preds,
                "precision":p, "recall":r, "f1":f,
                "note":lbl
            })

    tasks = [asyncio.create_task(worker(m)) for m in mapping]  # 모든 worker 태스크 생성

    for coro in tqdm.as_completed(tasks,
                                total=len(tasks),
                                desc=description,
                                dynamic_ncols=True):
        await coro

    summary = {
        "precision": float(np.mean(agg["p"] or [0])),
        "recall":    float(np.mean(agg["r"] or [0])),
        "f1":        float(np.mean(agg["f"] or [0])),
        "router_accuracy": float(np.mean(agg["ok"] or [0])),
        "route_stats": {
            "search_data":  agg["lbl"].count("search_data"),
            "llm_fallback": agg["lbl"].count("llm_fallback"),
            "error":        agg["lbl"].count("error"),
        },
        "total": len(mapping),
    }
    return summary, details

# ────────────────────────────────────────────────────────────────────────────────
# 시각화
# ────────────────────────────────────────────────────────────────────────────────

def bar_chart(res: dict, title: str, path: Path):
    labels, vals = ["Precision", "Recall", "F1"], [res["precision"], res["recall"], res["f1"]]
    plt.figure(figsize=(6,4)); plt.bar(labels, vals); plt.ylim(0,1); plt.title(title)
    for i, v in enumerate(vals): plt.text(i, v+0.02, f"{v:.2f}", ha="center")
    plt.tight_layout(); plt.savefig(path); plt.close()

def router_chart(acc: float, title: str, path: Path):
    plt.figure(figsize=(3,4)); plt.bar(["Routing"], [acc]); plt.ylim(0,1); plt.title(title)
    plt.text(0, acc+0.02, f"{acc:.2f}", ha="center"); plt.tight_layout(); plt.savefig(path); plt.close()

def heatmap(overview: dict):
    algos, embeds = SEARCH_ALGOS, EMBED_KEYS
    singles = np.array([[overview[a][e]["single"]["f1"] for e in embeds] for a in algos])
    multis  = np.array([[overview[a][e]["multi"]["f1"]  for e in embeds] for a in algos])
    for name, data in {"single": singles, "multi": multis}.items():
        plt.figure(figsize=(6,4)); plt.imshow(data, cmap="Blues", vmin=0, vmax=1)
        plt.xticks(range(len(embeds)), embeds); plt.yticks(range(len(algos)), algos)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center")
        plt.colorbar(label="F1"); plt.title(f"F1 Heatmap ({name})")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"overview_f1_heatmap_{name}.png"); plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# 결과 저장
# ────────────────────────────────────────────────────────────────────────────────
async def run_combo_async(algo: str, embed: str, workers: int) -> dict:
    """
    벡터/하이브리드 + 임베딩 모델 조합으로 single·multi 평가.
    """
    # 컬렉션 로드·검색 함수 생성 (sync)
    logging.info(f"[{algo}/{embed}] 임베딩 컬렉션 로드")
    collection, embedder = load_or_ingest_collection(embed)

    combo_res = {}
    for mode, csv in MAPPING_FILES.items():
        # mapping 파싱 & map_dict 생성
        logging.info(f"[{algo}/{embed}] - {mode} 매핑 파일 파싱: {csv.name}")
        mapping = parse_mapping_csv(csv)
        logging.info(f"[{algo}/{embed}] - {mode} 질문 개수: {len(mapping)}")
        map_dict = {m.question: m.gold_docs for m in mapping}

        # 검색 함수 정의
        if algo == "vector":
            search = make_vector_search(collection, embedder)
        else:  # hybrid
            search = make_hybrid_search(collection, map_dict)

        # 평가 실행
        desc = f"Evaluating {algo}-{embed}-{mode} combo"
        summary, details = await evaluate_async(mapping, router_raw, search, rewrite_raw, workers, description=desc)
        combo_res[mode] = summary
        logging.info(f"[{algo}/{embed}] - {mode} 평가 완료 (F1={summary['f1']:.2f})")


        # 파일 저장 (RESULTS_DIR 경로 파일 존재해야 함)
        with open(RESULTS_DIR/f"results_{algo}_{embed}_{mode}.json","w",encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(RESULTS_DIR/f"results_{algo}_{embed}_{mode}_details.json",
                  "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

        # 시각화
        bar_chart(summary, f"{algo.upper()} + {embed.upper()} ({mode})",
                  RESULTS_DIR / f"{algo}_{embed}_{mode}_metrics.png")
        router_chart(summary["router_accuracy"],
                     f"{algo.upper()} + {embed.upper()} ({mode}) router",
                     RESULTS_DIR / f"{algo}_{embed}_{mode}_router.png")
        logging.info(f"[{algo}/{embed}] - {mode} 시각화 결과 저장 완료")
    return combo_res

# ────────────────────────────────────────────────────────────────────────────────
# 메인 루틴
# ────────────────────────────────────────────────────────────────────────────────

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="스레드 수(기본 4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stderr)   # 로그를 stderr로 분리)
    global router_raw, rewrite_raw
    router_raw, rewrite_raw = load_router_and_rewriter()

    overview = {}
    for algo in SEARCH_ALGOS:
        overview[algo] = {}
        for embed in EMBED_KEYS:
            print(f"\n>>> Evaluating: {algo}/{embed}")
            overview[algo][embed] = await run_combo_async(algo, embed, args.workers)

    heatmap(overview)
    logging.info("모든 조합 평가 완료. 결과는 results/ 폴더에 저장되었습니다.")

    # ── 1) 요약 데이터 수집
    rows = []
    for algo in SEARCH_ALGOS:
        for embed in EMBED_KEYS:
            res = overview[algo][embed]
            s = res["single"]
            m = res["multi"]
            rows.append({
                "algo": algo,
                "embed": embed,
                "single_router_acc": s["router_accuracy"],
                "multi_router_acc":  m["router_accuracy"],
                "single_p": s["precision"],
                "single_r": s["recall"],
                "single_f1": s["f1"],
                "multi_p":  m["precision"],
                "multi_r":  m["recall"],
                "multi_f1": m["f1"],
            })

    df = pd.DataFrame(rows)
    print("\n=== 전체 조합 성능 요약 ===")
    print(df.to_markdown(index=False, floatfmt=".3f"))

    df.to_csv(RESULTS_DIR / "summary.csv", index=False, float_format="%.3f")
    df.to_json(RESULTS_DIR / "summary.json", orient="records", force_ascii=False, indent=2)

    # 통합 그래프 그리기
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    x_labels = [f"{a}/{e}" for a in SEARCH_ALGOS for e in EMBED_KEYS]

    # (1) 라우터 정확도
    router_vals_single = df["single_router_acc"]
    router_vals_multi  = df["multi_router_acc"]
    ax = axes[0]
    idx = range(len(df))
    ax.bar([i-0.2 for i in idx], router_vals_single, width=0.4, label="Single Mapping")
    ax.bar([i+0.2 for i in idx], router_vals_multi,  width=0.4, label="Multi Mapping")
    ax.set_xticks(idx); ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylim(0,1); ax.set_title("Routing Accuracy by Combo")
    ax.legend()

    # (2) F1 점수
    f1_vals_single = df["single_f1"]
    f1_vals_multi  = df["multi_f1"]
    ax = axes[1]
    ax.bar([i-0.2 for i in idx], f1_vals_single, width=0.4, label="Single Mapping")
    ax.bar([i+0.2 for i in idx], f1_vals_multi,  width=0.4, label="Multi Mapping")
    ax.set_xticks(idx); ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylim(0,1); ax.set_title("Retrieval F1 by Combo")
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "summary_plot.png")
    plt.close()

if __name__ == "__main__":
    asyncio.run(main_async())

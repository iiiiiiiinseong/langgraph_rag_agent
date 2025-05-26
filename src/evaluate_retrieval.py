# evaluate_system_retrieval.py
import json, csv
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

from adaptive_self_rag import (
    question_router,      # 금융 vs 일반
    search_fixed_deposit,
    search_demand_deposit,
    rewrite_question
)

# 2) 설정
SINGLE_CSV = "qd_mapping.csv"
MULTI_CSV  = "qd_mapping_multi.csv"
METRICS    = ["precision", "recall", "f1"]
MAX_CYCLES = 2

# 3) JSON → content → id
def load_docs(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["documents"]

fixed_json  = load_docs("./../findata/fixed_deposit_20250212.json")
demand_json = load_docs("./../findata/demand_deposit_20250213.json")
all_json    = fixed_json + demand_json

# id → type 매핑 gold_ids → 어떤 래퍼 함수를 쓸지 결정용
id_to_type = { e["id"]: e["type"] for e in all_json } 

# 4) 매핑 로드
def load_map(path: str, multi=False) -> List[Dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gold = r["doc_ids"].split(",") if multi else [r["doc_id"]]
            out.append({"question": r["question"], "gold": gold})
    return out

single_map = load_map(SINGLE_CSV, multi=False)
multi_map  = load_map(MULTI_CSV,  multi=True)

# 5) 한 질의 평가
def eval_question(q: str, gold_ids: List[str]) -> Dict:
    # 1) 라우팅 판정
    decision = question_router.invoke({"question": q})
    routed = decision.route
    is_finance = (routed == "search_data")
    note = "search_data" if is_finance else "llm_fallback"

    if not is_finance:
        return {
            "question": q, "gold": gold_ids, "predicted": [],
            "precision": None, "recall": None, "f1": None,
            "routed": routed, "note": note,
            "cycles": 0
        }

    # 2) 카테고리별 래퍼 검색 (multi-cycle 재시도)
    question = q
    filtered = []
    for cycle in range(1, MAX_CYCLES+1):
         # gold_ids의 첫 문서로부터 카테고리 결정
         cat = id_to_type[gold_ids[0]]
         if cat == "정기예금":
             filtered = search_fixed_deposit.invoke(question)
         elif cat == "입출금자유예금":
             filtered = search_demand_deposit.invoke(question)
         else:
             pass
         print(f"    - cycle {cycle}: filtered {len(filtered)} docs")
         if filtered:
             break
         question = rewrite_question(question)

    # 3) 최종 filtered → ID 매핑 → metric 계산
    pred_ids = { d.metadata["id"] for d in filtered }

    gold_set = set(gold_ids)
    tp = len(pred_ids & gold_set)
    p  = tp/len(pred_ids) if pred_ids else 0.0
    r  = tp/len(gold_set)  if gold_set else 0.0
    f1 = (2*p*r/(p+r))    if (p+r)>0 else 0.0

    return {
        "question": q, "gold": gold_ids,
        "predicted": list(pred_ids),
        "precision": p, "recall": r, "f1": f1,
        "routed": routed, "note": note,
        "cycles": cycle
    }

# 6) 전체 평가
def evaluate_set(name: str, mappings):
    print(f"\n==== [{name}] {len(mappings)} queries ====")
    results = []
    for i, m in enumerate(mappings, 1):
        print(f"{i:3d}/{len(mappings)} Q:", m["question"])
        res = eval_question(m["question"], m["gold"])
        if res["note"]=="llm_fallback":
            print("   → llm_fallback 처리")
        else:
            print(f"   → Predicted {res['predicted']}")
            print(f"     P={res['precision']:.2f}, R={res['recall']:.2f}, F1={res['f1']:.2f}, cycles={res['cycles']}")
        results.append(res)
    return results

res_single = evaluate_set("단일 매핑", single_map)
res_multi  = evaluate_set("다중 매핑",  multi_map)

# 7) 분류 정확도 계산
def class_accuracy(results):
    total = len(results)
    correct = sum(1 for r in results if r["routed"]=="search_data")
    return correct/total if total else 0.0

cls_acc_single = class_accuracy(res_single)
cls_acc_multi  = class_accuracy(res_multi)

# Retrieval 메트릭 평균 계산
def avg(rs, key):
    vals = [r[key] for r in rs if r[key] is not None]
    return sum(vals)/len(vals) if vals else 0.0

avg_single = {m: avg(res_single, m) for m in METRICS}
avg_multi  = {m: avg(res_multi,  m) for m in METRICS}

print("\n================== 결과 ==================")

print("\n>>> 라우팅 정확도 (Classification Accuracy)")
print(f" 단일 매핑: {cls_acc_single:.3f}")
print(f" 다중 매핑: {cls_acc_multi:.3f}")

print("\n>>> Retrieval 요약")
print(f" 단일: P={avg_single['precision']:.3f}, R={avg_single['recall']:.3f}, F1={avg_single['f1']:.3f}")
print(f" 다중: P={avg_multi['precision']:.3f}, R={avg_multi['recall']:.3f}, F1={avg_multi['f1']:.3f}")

# 8) 시각화
labels = ["Single","Multi"]

# Classification Accuracy
fig, ax = plt.subplots()
ax.bar(labels, [cls_acc_single, cls_acc_multi])
ax.set_ylim(0,1); ax.set_title("Routing Accuracy")
for i,v in enumerate([cls_acc_single, cls_acc_multi]):
    ax.text(i, v+0.02, f"{v:.2f}", ha="center")
fig.savefig("routing_accuracy.png")

# Retrieval 메트릭
for metric in METRICS:
    fig, ax = plt.subplots()
    ax.bar(labels, [avg_single[metric], avg_multi[metric]])
    ax.set_ylim(0,1); ax.set_title(f"{metric.capitalize()}@filtered")
    for i,v in enumerate([avg_single[metric], avg_multi[metric]]):
        ax.text(i, v+0.02, f"{v:.2f}", ha="center")
    fig.savefig(f"{metric}_filtered_comparison.png")

# 9) JSON으로 결과 저장
def save_results_to_json(path: str, results: List[Dict]):
    """
    question, gold, predicted, precision, recall, f1, note 만 뽑아서 JSON으로 저장
    """
    slim = []
    for r in results:
        slim.append({
            "question":  r["question"],
            "gold":      r["gold"],
            "predicted": r["predicted"],
            "precision": r["precision"],
            "recall":    r["recall"],
            "f1":        r["f1"],
            "note":      r["note"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)

# 단일/다중 매핑 결과를 각각 JSON으로 저장
save_results_to_json("results_single.json", res_single)
save_results_to_json("results_multi.json",  res_multi)

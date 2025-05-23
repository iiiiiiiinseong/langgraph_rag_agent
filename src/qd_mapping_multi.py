# Updated qd_mapping_multi.py

import json, csv, random
from pathlib import Path
from openai import OpenAI
from itertools import combinations
from typing import List, Dict

# — 설정 경로 —
FIXED_JSON_PATH  = "./../findata/fixed_deposit_20250212.json"
DEMAND_JSON_PATH = "./../findata/demand_deposit_20250213.json"

# — OpenAI SDK 초기화 —
openai = OpenAI()

# — JSON 로더 —
def load_docs(path: str) -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("documents", [])

all_docs = load_docs(FIXED_JSON_PATH) + load_docs(DEMAND_JSON_PATH)

# — 다중 문서 질문 생성 함수 —
def generate_multi_doc_queries(docs: List[Dict], n_variants=3) -> List[str]:
    meta_snippets = []
    for d in docs:
        meta_snippets.append(f"- [{d['id']}] {d['bank']}의 {d['product_name']} ({d['type']})")
    snippet = "\n".join(meta_snippets)

    prompt = f"""
아래 문서 정보들을 모두 참고해야만 제대로 답변할 수 있는 질문을
서로 다른 관점으로 {n_variants}개 생성해 주세요.

상품 목록:
{snippet}

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
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    lines = resp.choices[0].message.content.splitlines()
    qs = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            qs.append(line.split(".",1)[1].strip())
            if len(qs) >= n_variants:
                break
    return qs

# — 매핑 생성 로직 (랜덤 40개 조합) —
def build_multi_mapping(output_csv="qd_mapping_multi.csv", sample_size=40):
    # Prepare combinations
    by_type = {}
    for d in all_docs:
        by_type.setdefault(d["type"], []).append(d)

    # 1) Same-type pairs
    same_pairs = [combo for docs in by_type.values() for combo in combinations(docs, 2)]

    # 2) Cross-type pairs
    cross_pairs = [combo for combo in combinations(all_docs, 2)
                   if combo[0]['type'] != combo[1]['type']]

    # 3) Multi-type triples (at least two types)
    triples = [combo for combo in combinations(all_docs, 3)
               if len({d["type"] for d in combo}) > 1]

    print(f"[INFO] same-type pairs: {len(same_pairs)}, cross-type pairs: {len(cross_pairs)}, mixed triples: {len(triples)}")
    all_combos = same_pairs + cross_pairs + triples
    print(f"[INFO] total possible combos: {len(all_combos)}")

    # Random sample
    random.seed(42)
    sampled_combos = random.sample(all_combos, min(sample_size, len(all_combos)))
    print(f"[INFO] sampled {len(sampled_combos)} combos for Q-D mapping")

    # Generate questions
    rows = []
    total_qs = 0
    for combo in sampled_combos:
        doc_ids = [d["id"] for d in combo]
        qs = generate_multi_doc_queries(list(combo), n_variants=2)
        for q in qs:
            rows.append({
                "doc_ids": ",".join(doc_ids),
                "question": q
            })
        print(f"  • {doc_ids} → {len(qs)}개 질문 생성")
        total_qs += len(qs)

    # Save CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_ids","question"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] 다중 문서 Q-D 매핑 생성 완료: 총 {total_qs}개 질문 → {output_csv}")

if __name__ == "__main__":
    build_multi_mapping()

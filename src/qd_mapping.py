# qd_mapping.py
import json
import csv
from pathlib import Path
from openai import OpenAI
from typing import List, Dict

# — JSON 파일 경로 설정 —
FIXED_JSON_PATH  = "./../findata/fixed_deposit_20250212.json"
DEMAND_JSON_PATH = "./../findata/demand_deposit_20250213.json"

# — OpenAI 초기화 —
openai = OpenAI()

# — 질의 확장 함수 (메타 반영) — 
CATEGORIES = [
    "정기예금", "입출금자유예금"
#    "정기적금", "청년적금",
#   "주택담보대출상품", "신용대출상품"
]

def expand_query_for_doc(meta: Dict, seed_q: str, n: int = 5) -> List[str]:
    """
    해당 문서의 메타정보를 반영하여 '이 문서를 묻고 싶게 만드는' n개의 질문 시나리오 생성
    """
    meta_lines = "\n".join(f"- {k}: {v}" for k, v in meta.items())
    prompt = f"""
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
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    lines = resp.choices[0].message.content.splitlines()
    variants = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            q = line.split(".",1)[1].strip()
            variants.append(q)
            if len(variants) >= n:
                break
    return variants

# — JSON에서 문서 로드 —
def load_doc_entries(json_path: str) -> List[Dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return data.get("documents", [])

# — 매핑 생성 및 저장 —
def build_qd_mapping(output_csv: str = "qd_mapping.csv"):
    # 1) 모든 문서 불러오기
    docs = load_doc_entries(FIXED_JSON_PATH) + load_doc_entries(DEMAND_JSON_PATH)
    print(f"[INFO] 총 {len(docs)}개 문서 로드")

    # 2) 매핑 리스트
    mapping_rows = []

    # 3) 각 문서별로 질문 생성
    for entry in docs:
        doc_id   = entry.get("id", "")
        meta     = {
            "bank": entry.get("bank",""),
            "product_name": entry.get("product_name",""),
            "type": entry.get("type",""),
            **{ k:str(v) for k,v in entry.get("metadata",{}).items() if k in ["기본금리(단리이자 %)","최고금리(우대금리포함, 단리이자 %)","가입방법","만기 후 금리","우대조건"] }
        }
        # seed 질문 템플릿
        seed_q = f"{meta['bank']}의 {meta['product_name']}에 대해 알려주세요."
        # 시나리오 질의 n개 생성
        variants = expand_query_for_doc(meta, seed_q, n=5)

        for q in variants:
            mapping_rows.append({"doc_id": doc_id, "question": q})
        print(f"  • {doc_id} → {len(variants)}질의 생성")

    # 4) CSV로 저장
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id","question"])
        writer.writeheader()
        writer.writerows(mapping_rows)

    print(f"[DONE] Q-D 매핑 저장: {output_csv}")

if __name__ == "__main__":
    build_qd_mapping()

#type: 입출금자유예금 demand deposit products

import os
import json
import pandas as pd
import numpy as np

def generate_key_summary(row: dict, keys: list) -> str:
    """
    row 데이터(dict)에서 지정한 키에 해당하는 값들을 "키: 값" 형식으로 결합합니다.
    값이 없거나 "정보 없음"이면 제외합니다.
    """
    summary_items = []
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value and value != "정보 없음":
            summary_items.append(f"{key}: {value}")
    return ", ".join(summary_items) if summary_items else "정보 없음"

def process_data(raw_csv_path: str, processed_json_dir: str) -> None:
    """
    예금금리_입출금자유예금 CSV 파일을 읽어 각 행을 내러티브 문서로 변환한 후,
    새로운 JSON 구조(최상위에 "documents" 키)를 생성하여 저장합니다.
    
    최종 JSON 문서의 구조:
    {
      "documents": [
        {
          "id": "입출금자유예금_product_000",
          "bank": "은행명",
          "product_name": "상품명",
          "type": "입출금자유예금",
          "content": "은행: ...\n상품명: ...\n기본금리( %): ...\n최고금리(우대금리포함,  %): ...\n...",
          "key_summary": "은행: ..., 상품명: ..., 기본금리( %): ..., 최고금리(우대금리포함,  %): ...",
          "metadata": { ... }  // 원본 row 데이터 + key_summary 추가
        },
        ...
      ]
    }
    """
    try:
        df = pd.read_csv(raw_csv_path, encoding="utf-8")
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return

    # 불필요한 "Unnamed:" 컬럼 제거 및 결측치 "정보 없음" 처리
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.replace({np.nan: "정보 없음"}, inplace=True)

    documents = []
    # 입출금자유예금에서 핵심 필드: 은행, 상품명, 기본금리( %), 최고금리(우대금리포함,  %)
    key_fields = [
        "은행",
        "상품명",
        "기본금리( %)",
        "최고금리(우대금리포함,  %)"
    ]
    
    for idx, row in df.iterrows():
        bank = row.get("은행", "정보 없음")
        product_name = row.get("상품명", "정보 없음")
        product_type = "입출금자유예금"
        
        # 모든 컬럼 정보를 줄바꿈으로 연결하여 내러티브 생성
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        
        # 핵심 필드 기반 key_summary 생성
        key_summary = generate_key_summary(row.to_dict(), key_fields)
        
        # 원본 메타데이터에 key_summary 추가
        metadata = row.to_dict()
        metadata["key_summary"] = key_summary

        document = {
            "id": f"입출금자유예금_product_{idx:03d}",
            "bank": bank,
            "product_name": product_name,
            "type": product_type,
            "content": content,
            "key_summary": key_summary,
            "metadata": metadata
        }
        documents.append(document)

    output = {"documents": documents}
    json_filename = os.path.splitext(os.path.basename(raw_csv_path))[0] + ".json"
    processed_json_path = os.path.join(processed_json_dir, json_filename)
    os.makedirs(processed_json_dir, exist_ok=True)

    with open(processed_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"{len(documents)}개의 문서를 {processed_json_path}에 저장했습니다.")

if __name__ == "__main__":
    raw_csv = "/home/inseong/langgraph_rag_agent/db_preprocessing/rawdata/예금금리_입출금자유예금_20250213.csv"
    processed_json_dir = "C:/Users/sj/Documents/CAPD/langgraph_rag_agent/findata"
    process_data(raw_csv, processed_json_dir)
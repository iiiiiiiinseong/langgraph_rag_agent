# Adaptive Self-RAG 기반 금융상품 상담 챗봇

## 1. 프로젝트 개요

이 프로젝트는 금융 상품 관련 질문(정기예금, 입출금자유예금 등)에 대해 정확하고 신속한 상담을 제공하는 것을 목표로 합니다. 특히, Adaptive Self-RAG 기법을 활용하여, 사용자의 질문을 분석하여 적절한 문서를 검색하고, 질문의 특성과 문서의 품질에 따라 질문을 재구성하거나 직접 답변을 생성하여 정확도를 높이는 AI 챗봇 시스템입니다.

---

## 2. 사용 모델 및 프레임워크

### **LLM 모델 (gpt-4o-mini)**

* OpenAI의 GPT-4o-mini 모델 사용
* 금융 및 일반 질의응답에서 빠르고 정확한 응답 생성에 적합

### **주요 프레임워크 및 모듈**

* **LangChain & LangGraph**: 복잡한 RAG(Retrieval-Augmented Generation) 파이프라인 구축
* **HuggingFace Embeddings**: 문서 및 질의 임베딩 생성
* **ChromaDB**: 문서 데이터 저장 및 벡터 유사성 기반 검색
* **Gradio**: 사용자 친화적인 챗봇 웹 인터페이스 제공

---

## 3. 시스템 동작 구조

### 질문 라우팅

* 사용자 질문을 분석하여 금융상품 관련 질문(search\_data)인지 일반 질문(llm\_fallback)인지 구분
* 금융 질문은 문서 검색을 통해, 일반 질문은 LLM에서 바로 처리

### 병렬 서브 그래프 문서 검색

* 정기예금 및 입출금자유예금 데이터를 ChromaDB에서 병렬로 검색
* 검색 결과의 관련성 평가 후 최적의 문서만을 필터링

### 답변 생성 및 평가

* 관련 문서를 기반으로 GPT-4o-mini 모델이 질문에 대한 답변을 생성
* 생성된 답변의 품질을 환각(hallucination)과 관련성 측면에서 평가
* 필요시 질문 재작성(transform\_query) 후 재검색 및 답변 재생성

### 랭그래프 구조

다음 다이어그램은 구축된 Adaptive Self-RAG 체인의 전체 프로세스를 나타냅니다:

![Adaptive Self-RAG Graph](./assets/AdaptiveSelfRAG_ProcessGraph.png)

---

## 4. 사용된 알고리즘 및 기술 설명

### Retrieval Grader

* 검색된 문서가 질문과 관련성이 있는지 판단하는 알고리즘

### Hallucination Grader

* 생성된 답변이 문서 내용에 기반한 사실만으로 이루어졌는지 평가

### Answer Grader

* 생성된 답변이 질문을 정확히 해결했는지 평가

### Question Re-writer

* 검색 효율성을 높이기 위해 질문을 최적화된 형태로 재구성

---

## 5. 데이터 처리 및 관리

### 🔹 ChromaDB 구축

* JSON 형식의 금융상품 데이터를 HuggingFace의 임베딩 모델로 변환

### 🔹 데이터 인제스천(Ingestion)

* ChromaDB가 비어 있으면 자동으로 JSON 데이터를 로드하여 벡터 데이터베이스에 인제스천

---

## 6. 사용자 인터페이스 (Gradio)

### 챗봇 사용법

* Gradio 기반의 웹 인터페이스 제공
![챗봇 사용 예시](./assets/AdaptiveSelfRAG_example_screenshot.png)

### 예시 질문

* "정기예금 상품 중 금리가 가장 높은 것은?"

---

## 7. 설치 및 실행 방법

```bash
# 프로젝트 복제
git clone https://github.com/iiiiiiiinseong/adaptive_self_rag.git
cd project_name

# 환경설정 및 실행
pip install -r requirements.txt
python src/adaptive_self_rag.py
```

---

## 8. 향후 개발 계획

* 대출 상품(주택담보대출, 신용대출), 정기적금상품, 청년우대적금상품 데이터베이스 추가 예정
* 웹서치(Web Search) Tool calling 기능 추가 예정

---

## 9. 기대 효과

* 금융상품 상담 업무 효율성 향상 및 고객 만족도 증가

---

**문의 및 기여**

* GitHub에서 Issue 또는 PR로 참여해주세요!

"""
Adaptive_self_rag
금융상품(예: 정기예금, 입출금자유예금) 관련 질의에 대해:
1. 질문 라우팅 → (금융상품 관련이면) 문서 검색 (병렬 서브 그래프) → 문서 평가 → (조건부) 질문 재작성 → 답변 생성
   / (금융상품과 무관하면) LLM fallback을 통해 바로 답변 생성
그리고 생성된 답변의 품질(환각, 관련성) 평가 후 필요시 재생성 또는 재작성하는 Adaptive Self-RAG 체인.
"""


#############################
# 1. 기본 환경 및 라이브러리
#############################

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

# 기타 유틸
import json
import uuid
import re
from textwrap import dedent
from operator import add
from typing import List, Literal, Sequence, TypedDict, Annotated, Tuple

# 서치 알고리즘
from rank_bm25 import BM25Okapi
from entity_extraction import extract_banks, extract_entity_pairs

# LangChain, Chroma, LLM 관련
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool 
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import TavilySearchResults

# Grader 평가지표용
from pydantic import BaseModel, Field

# 그래프 관련
from langgraph.graph import StateGraph, START, END

# Gradio 관련
import gradio as gr

#############################
# 2. 임베딩, DB 설정, 서치 인덱스 생성
#############################
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import torch 

class LangChainSentenceTransformer(Embeddings):
    def __init__(self, model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()

embeddings_model_koMultitask = LangChainSentenceTransformer("jhgan/ko-sroberta-multitask") # 768차원 임배딩 모델로 변경

# Chroma DB 경로
CHROMA_DIR = "./../findata/chroma_db"

# JSON 데이터 경로
FIXED_JSON_PATH = "./../findata/fixed_deposit.json"
DEMAND_JSON_PATH = "./../findata/demand_deposit.json"
LOAN_JSON_PATH = "./../findata/loan.json"
SAVINGS_JSON_PATH = "./../findata/savings.json"


# DB 이름
FIXED_COLLECTION = "fixed_deposit"
DEMAND_COLLECTION = "demand_deposit"
LOAN_COLLECTION = "loan"
SAVINGS_COLLECTION = "savings"


def load_and_prepare_all_documents(json_paths: list[str]) -> list[Document]:
    docs: list[Document] = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data["documents"]:
            # 본문
            content = entry["content"]
            # metadata 필드 영어 키로 통일
            md = entry.get("metadata", {})
            bank         = entry.get("bank",         md.get("은행"))
            product_name = entry.get("product_name", md.get("상품명"))
            category     = entry.get("type")
            pdf_link     = entry.get("pdf_link",     md.get("pdf_link"))
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "id":           entry.get("id"),
                        "type":         category,
                        "bank":         bank,
                        "product_name": product_name,
                        "pdf_link":     pdf_link
                    }
                )
            )
    return docs

# JSON 파일 경로 리스트
ALL_JSON = [
    FIXED_JSON_PATH,
    DEMAND_JSON_PATH,
    LOAN_JSON_PATH,
    SAVINGS_JSON_PATH,
]

all_documents = load_and_prepare_all_documents(ALL_JSON)
corpus = [doc.page_content.split() for doc in all_documents]
bm25_index = BM25Okapi(corpus)

# 인제스천
vector_db = Chroma(
    embedding_function=embeddings_model_koMultitask,
    collection_name="combined_products_koMultitask",  # 임배딩 모델에 따른 인제스천 이름 변경
    persist_directory=CHROMA_DIR,
)

# 한 번만 인제스천
if not vector_db._collection.count():
    print("DB에 문서 없음으로 인제스천 진행")
    vector_db.add_documents(all_documents)

#인제스천 확인
print("총 문서 수:", len(all_documents))
from collections import Counter
print(Counter(doc.metadata['type'] for doc in all_documents))

#############################
# 3. 서치알고리즘 및 도구(검색 함수) 정의
#############################
from difflib import SequenceMatcher
def _match_meta(v, t, thr=0.8):
    v = "".join(v.lower().split())
    t = "".join(t.lower().split())
    return SequenceMatcher(None, v, t).ratio()>=thr or t in v

def _document_matches(doc: Document, filters: dict | None) -> bool:
    if not filters:                      # None 또는 빈 dict → 통과
        return True
    for k, v in filters.items():
        if v is None:                    # 필터 값이 None → 조건 없음
            continue

        doc_val = doc.metadata.get(k, "")
        # ---------- 리스트 값 처리 ----------
        if isinstance(v, list):
            if not any(_match_meta(doc_val, x) for x in v):
                return False
            continue                     # 다음 필터로
        # ---------- 단일 값 처리 ----------
        if k in ("bank", "product_name"):
            if not _match_meta(doc_val, v):
                return False
        else:
            if doc_val != v:
                return False
    return True

def _search_with_filters(query: str, filters: dict, top_k: int) -> list[Document]:
    # 1) BM25: 전체 코퍼스에서 score 계산 → metadata 필터 적용해 top_k
    tokenized = query.split()
    scores = bm25_index.get_scores(tokenized)
    idxs   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_docs = []
    for i in idxs:
        d = all_documents[i]
        if _document_matches(d, filters):
            bm25_docs.append(d)
            if len(bm25_docs)>=top_k: break

    # 2) 벡터 검색용 filter 준비: 리스트 → $in, 스칼라 → 그대로
    raw_meta = {k: v for k, v in filters.items() if v is not None}
    prepared = {}
    for k, v in raw_meta.items():
        if isinstance(v, list):
            # 여러 값을 허용하려면 $in 사용
            prepared[k] = {"$in": v}
        else:
            prepared[k] = v

    if not prepared:
        # 필터 없으면 전체
        vec_docs = vector_db.similarity_search(query, k=top_k)
    elif len(prepared) == 1:
        # 단일 필터
        vec_docs = vector_db.similarity_search(
            query, k=top_k, filter=prepared
        )
    else:
        # 다중 필터: $and 로 묶어 전달
        and_clauses = []
        for k, v in prepared.items():
            and_clauses.append({k: v})
        vec_docs = vector_db.similarity_search(
            query,
            k=top_k,
            filter={"$and": and_clauses}
        )
    # 3) 중복 제거 및 PDF 링크 추가 처리
    seen, merged = set(), []
    for d in bm25_docs + vec_docs:
        uid = d.metadata["id"]
        if uid not in seen:
            seen.add(uid)
            merged.append(d)
    return merged


# 하이브리드 서치 알고리즘
def hybrid_core_search(query: str, category: str, bank: str=None, product_name: str=None, top_k: int=3) -> List[Document]:
    # 1) 메타 필터 준비 (category 필수 포함)
    filters = {"type": category}
    if bank: 
        filters["bank"] = bank
    if product_name: 
        filters["product_name"] = product_name

    # 2) 필터 레벨별로 점진적 검색
    filter_levels = [
        filters,
        {**filters, **{"product_name": None}},  # 상품명 제외
        {**filters, **{"bank": None}},          # 은행 제외
        { "type": category }            # 카테고리만
    ]

    # 3) 순서대로 BM25+벡터 병렬 검색 → 결과 반환
    for flt in filter_levels:
        docs = _search_with_filters(query, flt, top_k=top_k)
        if docs:
            return docs
    return []

def get_banks_in_docs(documents: list[Document]) -> set[str]:
    banks = set()
    for doc in documents:
        bank = doc.metadata.get("bank", "")
        # IBK기업은행, IBK, 기업은행 모두 매칭할 수 있도록 처리 필요시 normalize
        if bank:
            banks.add(bank)
    return banks

@tool
def search_fixed_deposit(query: str) -> List[Document]:
    """
    Search for relevant fixed deposit (정기예금) product information using semantic similarity.
    This tool retrieves products matching the user's query, such as interest rates or terms.
    """
    pairs = extract_entity_pairs(query)
    docs, seen = [], set()
    if pairs:
        # 1:1 매핑된 (bank, product) 조합만 순회
        for pair in pairs:
            b, p = pair["bank"], pair["product"]
            for d in hybrid_core_search(
                query, category="정기예금", bank=b, product_name=p
            ):
                if d.metadata["id"] not in seen:
                    seen.add(d.metadata["id"])
                    docs.append(d)
    else:
        # 페어링 실패 시: category-only fallback
        for d in hybrid_core_search(query, category="정기예금"):
            if d.metadata["id"] not in seen:
                seen.add(d.metadata["id"])
                docs.append(d)
    return docs

@tool
def search_demand_deposit(query: str) -> List[Document]:
    """
    Search for demand deposit (입출금자유예금) product information using semantic similarity.
    This tool retrieves products matching the user's query, such as flexible withdrawal or interest features.
    """
    pairs = extract_entity_pairs(query)
    docs, seen = [], set()
    if pairs:
        # 1:1 매핑된 (bank, product) 조합만 순회
        for pair in pairs:
            b, p = pair["bank"], pair["product"]
            for d in hybrid_core_search(
                query, category="입출금자유예금", bank=b, product_name=p
            ):
                if d.metadata["id"] not in seen:
                    seen.add(d.metadata["id"])
                    docs.append(d)
    else:
        # 페어링 실패 시: category-only fallback
        for d in hybrid_core_search(query, category="입출금자유예금"):
            if d.metadata["id"] not in seen:
                seen.add(d.metadata["id"])
                docs.append(d)
    return docs

@tool
def search_loan(query: str) -> List[Document]:
    """
    Search for loan (대출) product information using semantic similarity.
    This tool retrieves products matching the user's query, such as flexible withdrawal or interest features.
    """
    pairs = extract_entity_pairs(query)
    docs, seen = [], set()
    if pairs:
        # 1:1 매핑된 (bank, product) 조합만 순회
        for pair in pairs:
            b, p = pair["bank"], pair["product"]
            for d in hybrid_core_search(
                query, category="대출", bank=b, product_name=p
            ):
                if d.metadata["id"] not in seen:
                    seen.add(d.metadata["id"])
                    docs.append(d)
    else:
        # 페어링 실패 시: category-only fallback
        for d in hybrid_core_search(query, category="대출"):
            if d.metadata["id"] not in seen:
                seen.add(d.metadata["id"])
                docs.append(d)
    return docs


@tool
def search_savings(query:str) -> List[Document]:
    """
    Search for savings (적금) product information using semantic similarity.
    This tool retrieves products matching the user's query, such as flexible withdrawal or interest features.
    """
    pairs = extract_entity_pairs(query)
    docs, seen = [], set()
    if pairs:
        # 1:1 매핑된 (bank, product) 조합만 순회
        for pair in pairs:
            b, p = pair["bank"], pair["product"]
            for d in hybrid_core_search(
                query, category="적금", bank=b, product_name=p
            ):
                if d.metadata["id"] not in seen:
                    seen.add(d.metadata["id"])
                    docs.append(d)
    else:
        # 페어링 실패 시: category-only fallback
        for d in hybrid_core_search(query, category="적금"):
            if d.metadata["id"] not in seen:
                seen.add(d.metadata["id"])
                docs.append(d)
    return docs

@tool
def web_search(query: str) -> List[str]:
    """
    This tool serves as a supplementary utility for the financial product recommendation model.
    It retrieves up-to-date external information via web search using the Tavily API, 
    especially when relevant data is not available in the local vector databases

    Unlike the RAG-based tools that query embedded product databases,
    this tool is designed to handle broader or real-time questions—such as current interest rates, financial trends,
    or general queries outside the scope of structured deposit data.

    It returns the top 2 semantically relevant documents from the web.
    """
    tavily_search = TavilySearchResults(max_results=2)
    docs = tavily_search.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content= f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>',
                metadata={"source": "web search", "url": doc["url"]}
                )
        )

    if len(formatted_docs) > 0:
        return formatted_docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]


#############################
# 4. LLM 초기화 & 도구 바인딩
#############################

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

#############################
# 5. LLM 체인 (Retrieval Grader / Hallucination / Answer Graders / Question Re-writer)
#############################

# (1) Retrieval Grader (검색평가)
class BinaryGradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_BinaryGradeDocuments = llm.with_structured_output(BinaryGradeDocuments)

system_prompt = """You are an expert in evaluating the relevance of search results to user queries.

[Evaluation criteria]
1. 키워드 관련성: 문서가 질문의 주요 단어나 유사어를 포함하는지 확인
2. 의미적 관련성: 문서의 전반적인 주제가 질문의 의도와 일치하는지 평가
3. 부분 관련성: 질문의 일부를 다루거나 맥락 정보를 제공하는 문서도 고려
4. 답변 가능성: 직접적인 답이 아니더라도 답변 형성에 도움될 정보 포함 여부 평가

[Scoring]
- Rate 'yes' if relevant, 'no' if not

[Key points]
- Consider the full context of the query, not just word matching
- Rate as relevant if useful information is present, even if not a complete answer
"""
# 채점 프롬프트 템플릿릿
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "[Retrieved document]\n{document}\n\n[User question]\n{question}")
])

retrieval_grader_binary = grade_prompt | structured_llm_BinaryGradeDocuments

# (3) Hallucination Grader
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_HradeHallucinations = llm.with_structured_output(GradeHallucinations)

# 환각 평가를 위한 시스템 프롬프트 정의
halluci_system_prompt = """
You are an expert evaluator assessing whether an LLM-generated answer is grounded in and supported by a given set of facts.

[Your task]
    - Review the LLM-generated answer.
    - Determine if the answer is fully supported by the given facts.

[Evaluation criteria]
    - 답변에 주어진 사실이나 명확히 추론할 수 있는 정보 외의 내용이 없어야 합니다.
    - 답변의 모든 핵심 내용이 주어진 사실에서 비롯되어야 합니다.
    - 사실적 정확성에 집중하고, 글쓰기 스타일이나 완전성은 평가하지 않습니다.

[Scoring]
    - 'yes': The answer is factually grounded and fully supported.
    - 'no': The answer includes information or claims not based on the given facts.

Your evaluation is crucial in ensuring the reliability and factual accuracy of AI-generated responses. Be thorough and critical in your assessment.
"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", halluci_system_prompt),
        ("human", "[Set of facts]\n{documents}\n\n[LLM generation]\n{generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_HradeHallucinations

# (4) Answer Grader 
class BinaryGradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_BinaryGradeAnswer = llm.with_structured_output(BinaryGradeAnswer)
grade_system_prompt = """
You are an expert evaluator tasked with assessing whether an LLM-generated answer effectively addresses and resolves a user's question.

[Your task]
    - Carefully analyze the user's question to understand its core intent and requirements.
    - Determine if the LLM-generated answer sufficiently resolves the question.

[Evaluation criteria]
    - 관련성: 답변이 질문과 직접적으로 관련되어야 합니다.
    - 완전성: 질문의 모든 측면이 다뤄져야 합니다.
    - 정확성: 제공된 정보가 정확하고 최신이어야 합니다.
    - 명확성: 답변이 명확하고 이해하기 쉬워야 합니다.
    - 구체성: 질문의 요구 사항에 맞는 상세한 답변이어야 합니다.

[Scoring]
    - 'yes': The answer effectively resolves the question.
    - 'no': The answer fails to sufficiently resolve the question or lacks crucial elements.

Your evaluation plays a critical role in ensuring the quality and effectiveness of AI-generated responses. Strive for balanced and thoughtful assessments.
"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grade_system_prompt),
        ("human", "[User question]\n{question}\n\n[LLM generation]\n{generation}"),
    ]
)

answer_grader_binary = answer_prompt | structured_llm_BinaryGradeAnswer

# (5) Question Re-writer
def rewrite_question(question: str) -> str:
    """
    입력 질문을 벡터 검색에 최적화된 형태로 재작성한다.
    """
    local_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = """
    You are an expert question re-writer. Your task is to convert input questions into optimized versions 
    for vectorstore retrieval. Analyze the input carefully and focus on capturing the underlying semantic 
    intent and meaning. Your goal is to create a question that will lead to more effective and relevant 
    document retrieval.

    [Guidelines]
        1. Identify and emphasize core concepts and key subjects.
        2. Expand abbreviations or ambiguous terms.
        3. Include synonyms or related terms that might appear in relevant documents.
        4. Maintain the original intent and scope.
        5. For complex questions, break them down into simpler, focused sub-questions.
    """
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "[Initial question]\n{question}\n\n[Improved question]\n")
    ])
    question_rewriter = re_write_prompt | local_llm | StrOutputParser()
    rewritten_question = question_rewriter.invoke({"question": question})
    return rewritten_question

# (6) Generation Evaluation & Decision Nodes
def grade_generation_self(state: "SelfRagOverallState") -> str:
    print("--- 답변 평가 (생성) ---")
    print(f"--- 생성된 답변: {state['generation']} ---")
    if state['num_generations'] > 1:
        print("--- 생성 횟수 초과, 종료 -> end ---")
        return "end"
    # 평가를 위한 문서 텍스트 구성
    print("--- 답변 할루시네이션 평가 ---")
    docs_text = "\n\n".join([d.page_content for d in state['documents']])
    hallucination_grade = hallucination_grader.invoke({
        "documents": docs_text,
        "generation": state['generation']
    })
    if hallucination_grade.binary_score == "yes":
        relevance_grade = retrieval_grader_binary.invoke({
            "question": state['question'],
            "document": state['filtered_documents'],
            "generation": state['generation']
        })
        print("--- 답변-질문 관련성 평가 ---")
        if relevance_grade.binary_score == "yes":
            print("--- 생성된 답변이 질문을 잘 해결함 ---")
            return "useful"
        else:
            print("--- 답변 관련성이 부족 -> transform_query ---")
            return "not useful"
    else:
        print("--- 생성된 답변의 근거가 부족 -> generate 재시도 ---")
        return "not supported"
    
def decide_to_generate_self(state: "SelfRagOverallState") -> str:
    print("--- 평가된 문서 분석 ---")
    if state['num_generations'] > 1:
        print("--- 생성 횟수 초과, 생성 결정 ---")
        return "generate"
    # 여기서는 필터링된 문서가 존재하는지 확인
    if not state['filtered_documents']:
        print("--- 관련 문서 없음 -> transform_query ---")
        return "transform_query"
    else:
        print("--- 관련 문서 존재 -> generate ---")
        return "generate"


# (7) RoutingDecision 
class RoutingDecision(BaseModel):
    """Determines whether a user question should be routed to document search or LLM fallback."""
    route: Literal["search_data","llm_fallback"] = Field(
        description="Classify the question as 'search_data' (financial) or 'llm_fallback' (general)"
        )

#############################
# 6. 상태 정의 및 노드 함수 (전체 Adaptive 체인)
#############################

# 상태 통합: SelfRagOverallState (질문, 생성, 원본 문서, 필터 문서, 생성 횟수)
# 메인 그래프 상태 정의
class SelfRagOverallState(TypedDict):
    """
    Adaptive Self-RAG 체인의 전체 상태를 관리    
    """
    question: str
    generation: Annotated[List[str], add]
    routing_decision: str
    num_generations: int
    documents: List[Document]
    filtered_documents: List[Document]
    history: List[Tuple[str,str]]     # (user, bot) 메시지 쌍 저장용

def initialize_state() -> SelfRagOverallState:
    """Create a new state with proper initialization of all fields"""
    return {
        "question": "",
        "generation": [],
        "routing_decision": "",
        "num_generations": 0,
        "documents": [],
        "filtered_documents": [],
        "history": []
    }

# 새로운 재작성 전용 LLM 체인 - 히스토리 답변이 있는 경우 이전 대화 맥락에 맞게 질문을 수정하여 문서 서치하기 위함
def contextualize_query(state: SelfRagOverallState) -> dict:
    # 최근 3턴 히스토리 추출
    recent = state['history'][-3:]
    hist_block = "\n".join(f"User: {u}\nAssistant: {a}" for u,a in recent)
    payload = {"history": hist_block, "question": state['question']}
    improved = question_rewriter_chain.invoke(payload)
    return {"question": improved}

rewrite_input = ChatPromptTemplate.from_messages([
    ("system", "당신은 금융 상품 챗봇 AI와 유저의 대화 히스토리를 기반으로 마지막 유저의 질문을 분석하여 금융상품 추천 RAG 시스템이 문서를 잘 찾을 수 있게 질문을 구체적으로 재작성하세요."
    "재작성된 질문은 길이가 너무 길어지지 않게 하며, 유저가 원히는 핵심이 무엇인지 명확하게 들어나는 문자이어야 합니다."
    "적용되는 RAG의 서치알고리즘은 백터유사도를 기반으로 하기 때문에 이를 고려하여 질문을 재작성하세요."
    "만약 [History]에 아무것도 없거나 유저의 마지막 질의가 맥락상 금융상품과 관련된 것이 아니라면 유저의 질문을 그대로 작성하세요."),
    ("system", "[History]\n{history}"),
    ("human", "[Question]\n{question}\n\n[Improved Question]\n"),
])
question_rewriter_chain = rewrite_input | llm | StrOutputParser()


# 질문 재작성 노드 (변경 후 검색 루프)
def transform_query_self(state: SelfRagOverallState) -> dict:
    print("--- 질문 개선 ---")
    new_question = rewrite_question(state['question'])
    print(f"--- 개선된 질문 : \n{new_question} ")
    new_count = state['num_generations'] + 1
    print(f"num_generations : {new_count}")
    return {"question": new_question, "num_generations": new_count}

# 답변 생성 노드 (서브 그래프로부터 받은 필터 문서 우선 사용, 이전 대화를 참고 할 수 있도록 수정)
def format_chat_history(history):
    messages = []
    for user_msg, assistant_msg in history:
        messages.append(("human", user_msg))
        messages.append(("ai", assistant_msg))
    return messages

generate_template = ChatPromptTemplate.from_messages([
    ("system", 
     """
[Your task]
You are a financial product expert and consultant who always responds in Korean.
Analyze the user query and the given financial product data to recommend the most suitable product.
Use the conversation history to maintain context. Rely only on the provided documents and history.

[Instructions]
1. 질문과 관련된 정보를 문맥에서 신중하게 확인합니다.
2. 답변에 질문과 직접 관련된 정보만 사용합니다.
3. 문맥에 명시되지 않은 내용에 대해 추측하지 않습니다.
4. 불필요한 정보를 피하고, 명확하게 작성합니다.
5. 문맥에서 정확한 답변을 생성할 수 없다면 마지막에 "더 구체적인 정보를 알려주시면 더욱 명쾌한 답변을 할 수 있습니다."를 추가합니다.     
""".strip()),
    ("system", "[Context]\n{context}"),
    ("system", "[History]\n{formatted_history}"),
    ("human", "{question}")
])

def generate_self(state: SelfRagOverallState) -> dict:
    print("--- 답변 생성 (히스토리 포함) ---")
    
    # 최근 대화 제한
    recent_history = state["history"][10:] if len(state["history"]) > 5 else state["history"][:5]

    # 대화 히스토리 포맷팅
    formatted_history = ""
    for user_msg, assistant_msg in recent_history:
        formatted_history += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"

    # 2) context 직렬화
    docs = state['filtered_documents'] or state['documents']
    context = "\n\n".join(d.page_content for d in docs) if docs else "관련 문서 없음"

    # 3) 프롬프트에 값 넣고 LLM 호출
    chain = generate_template  | llm | StrOutputParser()
    out = chain.invoke({
        "formatted_history": formatted_history,
        "context": context,
        "question": state["question"],
    })
    answer: str = out

    # 4) 상태 업데이트
    state["num_generations"] += 1
    state["generation"] = answer

    return {
        "generation": [answer],
        "num_generations": state["num_generations"],
    }

structured_llm_RoutingDecision = llm.with_structured_output(RoutingDecision)

question_router_system  = """
You are an AI assistant that routes user questions to the appropriate processing path.
Return one of the following labels:
- search_data
- llm_fallback
"""

question_router_prompt = ChatPromptTemplate.from_messages([
    ("system", question_router_system),
    ("human", "{question}")
])

question_router = question_router_prompt | structured_llm_RoutingDecision

# question route 노드 
def route_question_adaptive(state: SelfRagOverallState) -> dict:
    print("--- 질문 판단 (일반 or 금융) ---")
    print(f"질문: {state['question']}")
    decision = question_router.invoke({"question": state['question']})
    print("routing_decision:", decision.route)
    return {"routing_decision": decision.route}

# question route 분기 함수 
def route_question_adaptive_self(state: SelfRagOverallState) -> str:
    """
    질문 분석 및 라우팅: 사용자의 질문을 분석하여 '금융질문'인지 '일반질문'인지 판단
    """
    try:
        if state['routing_decision'] == "llm_fallback":
            print("--- 일반질문으로 라우팅 ---")
            return "llm_fallback"
        else:
            print("--- 금융질문으로 라우팅 ---")
            return "search_data"
    except Exception as e:
        print(f"--- 질문 분석 중 Exception 발생: {e} ---")
        return "llm_fallback"


fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an AI assistant helping with various topics. 
    Respond in Korean.
    - Provide accurate and helpful information.
    - Keep answers concise yet informative.
    - Inform users they can ask for clarification if needed.
    - Let users know they can ask follow-up questions if needed.
    - End every answer with the sentence: "저는 금융상품 질문에 특화되어 있습니다. 금융상품관련 질문을 주세요."
    """.strip()),
    ("system", "[History]\n{formatted_history}"),
    ("human", "{question}")
])

def llm_fallback_adaptive(state: SelfRagOverallState) -> dict:
    """Generates a direct response using the LLM when the question is unrelated to financial products."""
    print("--- 일반 질문 Fallback (히스토리 반영) ---")
    
    # 대화 히스토리 포맷팅
    formatted_history = ""
    for user_msg, assistant_msg in state["history"]:
        formatted_history += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"


    fallback_chain = fallback_prompt | llm | StrOutputParser()
    out = fallback_chain.invoke({
        "formatted_history": formatted_history,
        "question": state["question"],
    })
    answer: str = out

    state["history"].append((state["question"], answer))
    return {"generation": [answer]}

#############################
# 7. [서브 그래프 통합] - 병렬 검색 서브 그래프 구현
#############################

# --- 상태 정의 (검색 서브 그래프 전용) ---
class SearchState(TypedDict):
    question: str
    documents: Annotated[List[Document], add]  # 팬아웃된 각 검색 결과를 누적할 것
    filtered_documents: List[Document]         # 관련성 평가를 통과한 문서들

# ToolSearchState: SearchState에 추가 정보(datasources) 포함
class ToolSearchState(SearchState):
    datasources: List[str]  # 참조할 데이터 소스 목록

# --- 서브그래프 노드 함수 ---
def search_fixed_deposit_node(state: SearchState):
    """
    정기예금 상품 검색 (서브 그래프)
    """
    docs = search_fixed_deposit.invoke(state["question"])
    return {"documents": docs}

def search_demand_deposit_node(state: SearchState):
    """
    입출금자유예금 상품 검색 (서브 그래프)
    """
    docs = search_demand_deposit.invoke(state["question"])
    return {"documents": docs}

def search_savings_node(state: SearchState):
    """
    적금 상품 검색 (서브 그래프)
    """
    docs = search_savings.invoke(state["question"])
    return {"documents":docs}

def search_loan_node(state: SearchState):
    """
    대출 상품 검색 (서브 그래프)
    """
    docs = search_loan.invoke(state["question"])
    return {"documents":docs}

def search_web_search_subgraph(state: SearchState):
    """
    웹 검색 기반 금융 정보 검색 (서브 그래프)
    """
    question = state["question"]
    print('--- 웹 검색 실행 ---')

    docs = web_search.invoke({"query": question})  # 딕셔너리 형태로 넘김

    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 웹 정보를 찾을 수 없습니다.")]}
    

def filter_documents_subgraph(state: SearchState):
    """
    검색된 문서들에 대해 관련성 평가 후 필터링
    """
    print("--- 문서 관련성 평가 (서브 그래프) ---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []

    for d in documents:
        score = retrieval_grader_binary.invoke({
            "question": question,
            "document": d.page_content
        })
        if score.binary_score == "yes":
            print("--- 문서 관련성: 있음 ---")
            filtered_docs.append(d)
        else:
            print("--- 문서 관련성: 없음 ---")

    # 질문에서 요구되는 은행명(엔티티) 추출 및 표준화
    requested_banks = extract_banks(question)
    # 관련성 통과된 문서에 등장한 은행명 집합 추출
    found_banks = get_banks_in_docs(filtered_docs)

    # ===================== 싱글/없음 엔티티 분기 =====================
    if len(requested_banks) <= 1:
        # - 질문에서 은행명이 1개 이하로 추출된 경우
        # - (1) 관련성 평가만 한 결과(filtered_docs) 반환
        # - (2) 누락 은행 보완, 재검색 등 추가 로직 "생략"
        return {"filtered_documents": filtered_docs}
    
    # ===================== 멀티 엔티티(2개 이상) 분기 =====================
    missing_banks = [b for b in requested_banks if b not in found_banks]
    PRODUCT_CATEGORIES = ["정기예금", "입출금자유예금", "적금", "대출"]

    # (state에 누락 은행 보완 횟수 관리용 변수 추가)
    if "missing_bank_retry" not in state:
        state["missing_bank_retry"] = 0

    # (이미 확보한 은행+카테고리 쌍 관리)
    covered_pairs = set((doc.metadata.get("bank"), doc.metadata.get("type")) for doc in filtered_docs)

    # ========== (1) 누락 은행 보완 재검색 로직 (일단 최대 횟수 1까지 설정) ==========
    if missing_banks and state["missing_bank_retry"] < 1:
        # 누락 은행마다, 각 카테고리별로 추가 검색
        for bank in missing_banks:
            for category in PRODUCT_CATEGORIES:
                if (bank, category) in covered_pairs:
                    continue  # 이미 확보된 경우 생략
                more_docs = hybrid_core_search(question, category=category, bank=bank, top_k=2)
                for d in more_docs:
                    if (d.metadata.get("bank"), d.metadata.get("type")) in covered_pairs:
                        continue
                    # 관련성 평가(batch로 할 수도 있음)
                    score = retrieval_grader_binary.invoke({
                        "question": question,
                        "document": d.page_content
                    })
                    if score.binary_score == "yes":
                        filtered_docs.append(d)
                        covered_pairs.add((bank, category))
        state["missing_bank_retry"] += 1
        # 한 번 더 커버리지 체크 하도록(그래프 반복 등) 상태 반환
        return {"filtered_documents": filtered_docs, "missing_bank_retry": state["missing_bank_retry"]}

    # ========== (2) 보완 1회 시도 후에도 누락 은행 남을 경우 ==========
    if missing_banks and state["missing_bank_retry"] >= 1:
        # 더 이상 보완 안 하고, 지금까지 확보한 문서들 중 상위 3개만 남김 (예시)
        filtered_docs = filtered_docs[:3]
        return {"filtered_documents": filtered_docs, "missing_bank_retry": state["missing_bank_retry"]}

    # (모든 은행이 커버됐거나, 보완이 불필요한 경우)
    return {"filtered_documents": filtered_docs, "missing_bank_retry": state["missing_bank_retry"]}


# --- 질문 라우팅 (서브 그래프 전용) ---
class SubgraphToolSelector(BaseModel):
    """Selects the most appropriate tool for the user's question."""
    tool: Literal["search_fixed_deposit", "search_demand_deposit", "search_loan","search_savings", "web_search"] = Field(
        description="Select one of the tools: search_fixed_deposit, search_demand_deposit, search_loan, search_savings or web_search based on the user's question."
    )

class SubgraphToolSelectors(BaseModel):
    """Selects all tools relevant to the user's question."""
    tools: List[SubgraphToolSelector] = Field(
        description="Select one or more tools: search_fixed_deposit, search_demand_deposit, search_loan, search_savings or web_search based on the user's question."
    )

structured_llm_SubgraphToolSelectors = llm.with_structured_output(SubgraphToolSelectors)

subgraph_system  = dedent("""\
You are an AI assistant specializing in routing user questions to the appropriate tools.
Use the following guidelines:
- For fixed deposit product queries, use the search_fixed_deposit tool.
- For demand deposit product queries, use the search_demand_deposit tool.
- For loan product queries, use the search_loan tool.
- For savings product queries, use the search_savings tool.
- For general financial or real-time information queries, or when the user explicitly mentions 'web search',
  use the web_search tool.
  Always choose the appropriate tools based on the user's question.
""")
subgraph_route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", subgraph_system),
        ("human", "{question}")
    ]
)
question_tool_router = subgraph_route_prompt  | structured_llm_SubgraphToolSelectors

def analyze_question_tool_search(state: ToolSearchState):
    """
    질문 분석 및 라우팅: 사용자의 질문에서 참조할 데이터 소스 결정
    """
    print("--- 질문 라우팅 ---")
    question = state["question"]
    result = question_tool_router.invoke({"question": question})
    datasources = [tool.tool for tool in result.tools]
    return {"datasources": datasources}

def route_datasources_tool_search(state: ToolSearchState) -> Sequence[str]:
    """
    라우팅 결과에 따라 실행할 검색 노드를 결정 (병렬로 팬아웃)
    """
    datasources = set(state['datasources'])
    print("--- 선택된 검색 도구 ---")
    print(datasources)
    # 명확히 하나만 선택된 경우
    if datasources == {'search_fixed_deposit'}:
        return ['search_fixed_deposit']
    elif datasources == {'search_demand_deposit'}:
        return ['search_demand_deposit']
    elif datasources == {"search_loan"}:
        return ['search_loan']
    elif datasources == {"search_savings"}:
        return ['search_savings']
    elif datasources == {'web_search'}:
        return ['web_search']

    # 도구가 전부 실행되거나 애매모호할 때는 도구 전부 실행
    return ['search_fixed_deposit', 'search_demand_deposit', 'search_loan', 'search_savings', 'web_search']



# --- 서브 그래프 빌더 구성 ---
search_builder = StateGraph(ToolSearchState)

# 노드 추가
search_builder.add_node("analyze_question", analyze_question_tool_search)
search_builder.add_node("search_fixed_deposit", search_fixed_deposit_node)   
search_builder.add_node("search_demand_deposit", search_demand_deposit_node) 
search_builder.add_node("search_loan",search_loan_node)
search_builder.add_node("search_savings",search_savings_node)
search_builder.add_node("web_search", search_web_search_subgraph)
search_builder.add_node("filter_documents", filter_documents_subgraph)

# 엣지 구성
search_builder.add_edge(START, "analyze_question")
search_builder.add_conditional_edges(
    "analyze_question",
    route_datasources_tool_search,
    {
        "search_fixed_deposit": "search_fixed_deposit",
        "search_demand_deposit": "search_demand_deposit",
        "search_loan": "search_loan",
        "search_savings": "search_savings",
        "web_search": "web_search"
    }
)
# 두 검색 노드 모두 실행한 후 각각의 결과는 filter_documents로 팬인(fan-in) 처리
search_builder.add_edge("search_fixed_deposit", "filter_documents")
search_builder.add_edge("search_demand_deposit", "filter_documents")
search_builder.add_edge("search_loan","filter_documents")
search_builder.add_edge("search_savings","filter_documents")
search_builder.add_edge("web_search", "filter_documents")
search_builder.add_edge("filter_documents", END)

# 서브 그래프 컴파일
tool_search_graph = search_builder.compile()

#############################
# 8. [전체 그래프와 결합] - Self-RAG Overall Graph
#############################
print('\n8. [전체 그래프와 결합] - Self-RAG Overall Graph\n')

# 전체 그래프 빌더 (rag_builder) 구성
rag_builder = StateGraph(SelfRagOverallState)

# 노드 추가: 검색 서브 그래프, 생성, 질문 재작성 등
rag_builder.add_node("contextualize_query", contextualize_query)
rag_builder.add_node("route_question", route_question_adaptive)
rag_builder.add_node("llm_fallback", llm_fallback_adaptive)
rag_builder.add_node("search_data", tool_search_graph)         # 서브 그래프로 병렬 검색 및 필터링 수행
rag_builder.add_node("generate", generate_self)                # 답변 생성 노드
rag_builder.add_node("transform_query", transform_query_self)  # 질문 개선 노드

# 전체 그래프 엣지 구성
rag_builder.add_edge(START, "contextualize_query")
rag_builder.add_edge("contextualize_query", "route_question")
rag_builder.add_conditional_edges(
    "route_question",
    route_question_adaptive_self, 
    {
        "llm_fallback": "llm_fallback",
        "search_data": "search_data"
    }
)
rag_builder.add_edge("llm_fallback", END)
rag_builder.add_conditional_edges(
    "search_data",
    decide_to_generate_self, 
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)
rag_builder.add_edge("transform_query", "search_data")
rag_builder.add_conditional_edges(
    "generate",
    grade_generation_self,
    {
        "not supported": "generate",      # 환각 발생 시 재생성
        "not useful": "transform_query",  # 관련성 부족 시 질문 재작성 후 재검색
        "useful": END,
        "end": END,
    }
)

# MemorySaver 인스턴스 생성 (대화 상태를 저장할 in-memory 키-값 저장소)
memory = MemorySaver()
adaptive_self_rag_memory = rag_builder.compile(checkpointer=memory)

# 그래프 파일 저장하기
with open("adaptive_self_rag_memory.mmd", "w") as f:
    f.write(adaptive_self_rag_memory.get_graph(xray=True).draw_mermaid()) # 저장된 mmd 파일에서 코드 복사 후 https://mermaid.live 에 붙여넣기.


#############################
# 9. Gradio Chatbot 구성 및 실행
#############################

# pdf_link 삽입 보조함수
def postprocess_answer(answer: str, docs: List[Document]) -> str:
    """
    Add commentMore actions
    관련 문서 중 유효한 pdf_link가 있는 경우, 해당 링크를 답변 말미에 추가.
    단 한 번만 추가하고 링크가 없는 경우는 아무것도 붙이지 않음.
    """
    for doc in docs:
        pdf_link = doc.metadata.get("pdf_link")
        if pdf_link and isinstance(pdf_link, str) and pdf_link.strip():
            if "상품설명서 PDF 보기" not in answer:
                return f"{answer}\n\n [상품설명서 PDF 보기]({pdf_link})"
            break  # 한 번만 추가
    return answer


# 챗봇 클래스
class ChatBot:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())

    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        """
        입력 메시지와 대화 이력을 기반으로 Adaptive Self-RAG 체인을 호출하고,
        응답을 반환합니다.
        """
        config = {"configurable": {"thread_id": self.thread_id}}
        state = initialize_state()
        state["question"] = message
        
        # history가 있으면 추가
        if history:
            state["history"] = history
        
        result = adaptive_self_rag_memory.invoke(state, config=config)
        gen_list = result.get("generation", [])
        docs = result.get("filtered_documents", [])
        if not gen_list:
            bot_response = "죄송합니다. 답변을 생성할 수 없습니다."
        else:
            raw_answer = gen_list[-1]
            bot_response = postprocess_answer(raw_answer, docs)

        # 대화 이력 업데이트
        print(f"--- History 확인 ---\n{state['history']}")
        return bot_response


# 챗봇 인스턴스 생성
chatbot = ChatBot() 

# Gradio 인터페이스 생성
demo = gr.ChatInterface(
    fn=chatbot.chat,
    title="Adaptive Self-RAG 기반 RAG 챗봇 시스템",
    description="예금, 적금, 신용대출 상품 및 기타 질문에 답변합니다.",
    examples=[
        "정기예금 상품 중 금리가 가장 높은 것은?",
        "금리가 가장 낮은 대출상품을 추천해 주세요?",
        "학생신분인데 괜찮은 적금 상품을 추천해 주세요."
    ],
    theme=gr.themes.Soft()
)

# Gradio 앱 실행: 이 파일을 메인으로 실행할 때만 띄웁니다.
if __name__ == "__main__":
    demo.launch(share=True)
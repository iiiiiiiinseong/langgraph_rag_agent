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
from heapq import merge
from typing import List, Literal, Sequence, TypedDict, Annotated, Tuple

# 서치 알고리즘
from rank_bm25 import BM25Okapi

# LangChain, Chroma, LLM 관련
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

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
FIXED_JSON_PATH = "./../findata/fixed_deposit_20250212.json"
DEMAND_JSON_PATH = "./../findata/demand_deposit_20250213.json"

# DB 이름
FIXED_COLLECTION = "fixed_deposit_20250212"
DEMAND_COLLECTION = "demand_deposit_20250213"

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
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "id":           entry.get("id"),
                        "type":         category,
                        "bank":         bank,
                        "product_name": product_name
                    }
                )
            )
    return docs

# JSON 파일 경로 리스트
ALL_JSON = [
    FIXED_JSON_PATH,
    DEMAND_JSON_PATH
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
    vector_db.add_documents(all_documents)

def extract_bank(query: str) -> str | None:
    m = re.search(r'([가-힣A-Za-z0-9]+(?:은행|뱅크))', query)
    return m.group(1) if m else None

def extract_product(query: str) -> str | None:
    # “~통장”, “~예금”, “~대출” 등으로 마침
    m = re.search(r'([가-힣A-Za-z0-9]+(?:통장|예금|대출))', query)
    return m.group(1).strip() if m else None


#############################
# 3. 서치알고리즘 및 도구(검색 함수) 정의
#############################

def _search_with_filters(query: str, filters: dict, top_k: int) -> list[Document]:
    # 1) BM25: 전체 코퍼스에서 score 계산 → metadata 필터 적용해 top_k
    tokenized = query.split()
    scores = bm25_index.get_scores(tokenized)
    idxs   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_docs = []
    for i in idxs:
        d = all_documents[i]
        if all(filters.get(k) is None or d.metadata.get(k)==filters[k] for k in filters):
            bm25_docs.append(d)
            if len(bm25_docs)>=top_k: break

    # 2) 벡터: Chroma의 filter 파라미터 사용 (단일키 vs 다중키에 따라 $and 로 묶어서 전달)
    meta = {k: v for k, v in filters.items() if v is not None}
    if not meta:
        vec_docs = vector_db.similarity_search(query, k=top_k)
    elif len(meta) == 1:
        # 단일 필터터
        vec_docs = vector_db.similarity_search(query, k=top_k, filter=meta)
    else:
        # 여러 필터는 하나의 연산자($and)로 묶어서 넘겨야 함
        and_list = [{k: v} for k, v in meta.items()]
        vec_docs = vector_db.similarity_search(
                    query, 
                    k=top_k, 
                    filter={"$and": and_list}
        )

    # 3) 중복 제거 병합
    seen, merged = set(), []
    for d in bm25_docs + vec_docs:
        uid = d.metadata["id"]
        if uid not in seen:
            seen.add(uid)
            merged.append(d)
    return merged


# 하이브리드 서치 알고리즘
def hybrid_core_search(query: str, category: str, bank: str=None, product_name: str=None, top_k: int=2) -> List[Document]:
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
        {"category": category}                  # 카테고리만
    ]

    # 3) 순서대로 BM25+벡터 병렬 검색 → 결과 반환
    for flt in filter_levels:
        docs = _search_with_filters(query, flt, top_k=top_k)
        if docs:
            return docs
    return []

@tool
def search_fixed_deposit(query: str) -> List[Document]:
    """
    Search for relevant fixed deposit (정기예금) product information using semantic similarity.
    This tool retrieves products matching the user's query, such as interest rates or terms.
    """
    bank, product = extract_bank(query), extract_product(query)
    return hybrid_core_search(query, category="정기예금", bank=bank, product_name=product)

@tool
def search_demand_deposit(query: str) -> List[Document]:
    """
    Search for demand deposit (입출금자유예금) product information using semantic similarity.
    This tool retrieves products matching the user's query, such as flexible withdrawal or interest features.
    """
    bank, product = extract_bank(query), extract_product(query)
    return hybrid_core_search(query, category="입출금자유예금", bank=bank, product_name=product)


#############################
# 4. LLM 초기화 & 도구 바인딩
#############################

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

#############################
# 5. LLM 체인 (Retrieval Grader / Answer Generator / Hallucination / Answer Graders / Question Re-writer)
#############################
print("\n===================================================================\n ")
print("LLM 체인\n")
print("# (1) Retrieval Grader\n")

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
- Default to 'no' when uncertain

[Key points]
- Consider the full context of the query, not just word matching
- Rate as relevant if useful information is present, even if not a complete answer

Your evaluation is crucial for improving information retrieval systems. Provide balanced assessments.
"""
# 채점 프롬프트 템플릿릿
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "[Retrieved document]\n{document}\n\n[User question]\n{question}")
])

retrieval_grader_binary = grade_prompt | structured_llm_BinaryGradeDocuments

# question = "어떤 예금 상품이 있는지 설명해주세요."
# print(f'\nquestion : {question}\n')
# retrieved_docs = fixed_deposit_db.similarity_search(question, k=2)
# print(f"검색된 문서 수: {len(retrieved_docs)}")
# print("===============================================================================")
# print()

# relevant_docs = []
# for doc in retrieved_docs:
#     print("문서:\n", doc.page_content)
#     print("---------------------------------------------------------------------------")

#     relevance = retrieval_grader_binary.invoke({"question": question, "document": doc.page_content})
#     print(f"문서 관련성: {relevance}")

#     if relevance.binary_score == 'yes':
#         relevant_docs.append(doc)
    
#     print("===========================================================================")

# (3) Hallucination Grader
print("\n# (3) Hallucination Grader\n")

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
# hallucination = hallucination_grader.invoke({"documents": relevant_docs, "generation": generation})
# print(f"환각 평가: {hallucination}")

print("\n# (4) Answer Grader\n")
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
# print("Question:", question)
# print("Generation:", generation)
# answer_score = answer_grader_binary.invoke({"question": question, "generation": generation})
# print(f"답변 평가: {answer_score}")


print("\n# (5) Question Re-writer\n")
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

print("\n# (6) Generation Evaluation & Decision Nodes\n")
# (6) Generation Evaluation & Decision Nodes
def grade_generation_self(state: "SelfRagOverallState") -> str:
    print("--- 답변 평가 (생성) ---")
    #print(f"--- 생성된 답변: {state['generation']} ---")
    if state['num_generations'] > 2:
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
print('\n6. 상태 정의 및 노드 함수 (전체 Adaptive 체인)\n')
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

print('\n7. [서브 그래프 통합] - 병렬 검색 서브 그래프 구현\n')
# --- 상태 정의 (검색 서브 그래프 전용) ---
class SearchState(TypedDict):
    question: str
    # generation: str
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


# def search_fixed_deposit_subgraph(state: SearchState):

#     question = state["question"]
#     print('--- 정기예금 상품 검색 --- ')
#     docs = search_fixed_deposit.invoke(question)
#     if len(docs) > 0:
#         return {"documents": docs}
#     else:
#         return {"documents": [Document(page_content="관련 정기적금 상품정보를 찾을 수 없습니다.")]}

# def search_demand_deposit_subgraph(state: SearchState):
#     """
#     입출금자유예금 상품 검색 (서브 그래프)
#     """
#     question = state["question"]
#     print('--- 입출금자유예금 상품 검색 ---')
#     docs = search_demand_deposit.invoke(question)
#     if len(docs) > 0:
#         return {"documents": docs}
#     else:
#         return {"documents": [Document(page_content="관련 입출금자유예금 상품정보를 찾을 수 없습니다.")]}

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
    return {"filtered_documents": filtered_docs}

# --- 질문 라우팅 (서브 그래프 전용) ---
class SubgraphToolSelector(BaseModel):
    """Selects the most appropriate tool for the user's question."""
    tool: Literal["search_fixed_deposit", "search_demand_deposit"] = Field(
        description="Select one of the tools: search_fixed_deposit, search_demand_deposit based on the user's question."
    )

class SubgraphToolSelectors(BaseModel):
    """Selects all tools relevant to the user's question."""
    tools: List[SubgraphToolSelector] = Field(
        description="Select one or more tools: search_fixed_deposit, search_demand_deposit based on the user's question."
    )

structured_llm_SubgraphToolSelectors = llm.with_structured_output(SubgraphToolSelectors)

subgraph_system  = dedent("""\
You are an AI assistant specializing in routing user questions to the appropriate tools.
Use the following guidelines:
- For fixed deposit product queries, use the search_fixed_deposit tool.
- For demand deposit product queries, use the search_demand_deposit tool.
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
    if set(state['datasources']) == {'search_fixed_deposit'}:
        return ['search_fixed_deposit']
    elif set(state['datasources']) == {'search_demand_deposit'}:
        return ['search_demand_deposit']
    # 둘 다 선택되거나 모호할 때는 두 도구 모두 실행
    return ['search_fixed_deposit', 'search_demand_deposit']


# --- 서브 그래프 빌더 구성 ---
search_builder = StateGraph(ToolSearchState)


# 노드 추가
search_builder.add_node("analyze_question", analyze_question_tool_search)
search_builder.add_node("search_fixed_deposit", search_fixed_deposit_node)      # wapper 함수 말고 직접 invoke 함수 사용하는 것으로 수정
search_builder.add_node("search_demand_deposit", search_demand_deposit_node)    # 마찬가지로 함께
search_builder.add_node("filter_documents", filter_documents_subgraph)

# 엣지 구성
search_builder.add_edge(START, "analyze_question")
search_builder.add_conditional_edges(
    "analyze_question",
    route_datasources_tool_search,
    {
        "search_fixed_deposit": "search_fixed_deposit",
        "search_demand_deposit": "search_demand_deposit",
    }
)
# 두 검색 노드 모두 실행한 후 각각의 결과는 filter_documents로 팬인(fan-in) 처리
search_builder.add_edge("search_fixed_deposit", "filter_documents")
search_builder.add_edge("search_demand_deposit", "filter_documents")
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
# adaptive_self_rag = rag_builder.compile()

# 그래프 파일 저장하기
# display(Image(adaptive_self_rag.get_graph().draw_mermaid_png()))
with open("adaptive_self_rag_memory.mmd", "w") as f:
    f.write(adaptive_self_rag_memory.get_graph(xray=True).draw_mermaid()) # 저장된 mmd 파일에서 코드 복사 후 https://mermaid.live 에 붙여넣기.


#############################
# 9. Gradio Chatbot 구성 및 실행
#############################

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
        if not gen_list:
            bot_response = "죄송합니다. 답변을 생성할 수 없습니다."
        else:
            bot_response = gen_list[-1]  # 마지막 생성된 답변을 사용

        # 대화 이력 업데이트
        state["history"].append((message, bot_response))
        print(f"--- History 확인 ---\n{state["history"]}")
        return bot_response


# 챗봇 인스턴스 생성
chatbot = ChatBot() 

# Gradio 인터페이스 생성
demo = gr.ChatInterface(
    fn=chatbot.chat,
    title="Adaptive Self-RAG 기반 RAG 챗봇 시스템",
    description="정기예금, 입출금자유예금 상품 및 기타 질문에 답변합니다.",
    examples=[
        "정기예금 상품 중 금리가 가장 높은 것은?",
        "정기예금과 입출금자유예금은 어떤 차이점이 있나요?",
        "은행의 예금 상품을 추천해 주세요."
    ],
    theme=gr.themes.Soft()
)

# Gradio 앱 실행: 이 파일을 메인으로 실행할 때만 띄웁니다.
if __name__ == "__main__":
    demo.launch(share=True)
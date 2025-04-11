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
from pprint import pprint
from textwrap import dedent
from operator import add


# LangChain, Chroma, LLM 관련
from typing import List, Literal, Sequence, TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Grader 평가지표용
from pydantic import BaseModel, Field

# 그래프 관련
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


#############################
# 2. 임베딩 및 DB 설정
#############################
 
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chroma DB 경로
CHROMA_DIR = "./../findata/chroma_db"

# JSON 데이터 경로
FIXED_JSON_PATH = "./../findata/fixed_deposit_20250212.json"
DEMAND_JSON_PATH = "./../findata/demand_deposit_20250213.json"

# DB 이름
FIXED_COLLECTION = "fixed_deposit_20250212"
DEMAND_COLLECTION = "demand_deposit_20250213"

# 정기예금 DB
fixed_deposit_db = Chroma(
    embedding_function=embeddings_model,
    collection_name=FIXED_COLLECTION,
    persist_directory=CHROMA_DIR,
)

# 입출금자유예금 DB
demand_deposit_db = Chroma(
    embedding_function=embeddings_model,
    collection_name=DEMAND_COLLECTION,
    persist_directory=CHROMA_DIR,
)


# JSON -> Document 리스트 변환 함수
def load_documents_from_json(json_path: str) -> list[Document]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for entry in data.get("documents", []):
        content = entry.get("content", "")
        metadata = entry.get("metadata", {})
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


# 조건: DB가 비어 있으면 인제스천 (해당 조건이 없으면 계속 추가 됨..)
if not fixed_deposit_db._collection.count():  # Chroma 내부 count()로 확인
    print("[INFO] fixed_deposit DB is empty. Ingesting documents...")
    fixed_docs = load_documents_from_json(FIXED_JSON_PATH)
    fixed_deposit_db.add_documents(fixed_docs)
    print(f"[INFO] {len(fixed_docs)} fixed deposit docs ingested.")

if not demand_deposit_db._collection.count():
    print("[INFO] demand_deposit DB is empty. Ingesting documents...")
    demand_docs = load_documents_from_json(DEMAND_JSON_PATH)
    demand_deposit_db.add_documents(demand_docs)
    print(f"[INFO] {len(demand_docs)} demand deposit docs ingested.")


#############################
# 3. 도구(검색 함수) 정의
#############################

@tool
def search_fixed_deposit(query: str) -> List[Document]:
    """
    Search for relevant fixed deposit (정기예금) product information using semantic similarity.
    This tool retrieves products matching the user's query, such as interest rates or terms.
    """
    docs = fixed_deposit_db.similarity_search(query, k=2)
    if len(docs) > 0:
        return docs
    return [Document(page_content="관련 정기예금 상품정보를 찾을 수 없습니다.")]


@tool
def search_demand_deposit(query: str) -> List[Document]:
    """
    Search for demand deposit (입출금자유예금) product information using semantic similarity.
    This tool retrieves products matching the user's query, such as flexible withdrawal or interest features.
    """
    docs = demand_deposit_db.similarity_search(query, k=2)
    if len(docs) > 0:
        return docs
    return [Document(page_content="관련 입출금자유예금 상품정보를 찾을 수 없습니다.")]


tools = [search_fixed_deposit, search_demand_deposit]


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

question = "어떤 예금 상품이 있는지 설명해주세요."
print(f'\nquestion : {question}\n')
retrieved_docs = fixed_deposit_db.similarity_search(question, k=2)
print(f"검색된 문서 수: {len(retrieved_docs)}")
print("===============================================================================")
print()

relevant_docs = []
for doc in retrieved_docs:
    print("문서:", doc.page_content)
    print("---------------------------------------------------------------------------")

    relevance = retrieval_grader_binary.invoke({"question": question, "document": doc.page_content})
    print(f"문서 관련성: {relevance}")

    if relevance.binary_score == 'yes':
        relevant_docs.append(doc)
    
    print("===========================================================================")

print("\n# (2) Answer Generator (일반 RAG) \n")

# (2) Answer Generator (일반 RAG)
def generator_rag_answer(question, docs):

    template = """
    [Your task]
    You are a financial product expert and consultant who always responds in Korean.
    Your task is to analyze the user query and the given financial product data to recommend the most suitable financial product.
    
    [Instructions]
    1. 질문과 관련된 정보를 문맥에서 신중하게 확인합니다.
    2. 답변에 질문과 직접 관련된 정보만 사용합니다.
    3. 문맥에 명시되지 않은 내용에 대해 추측하지 않습니다.
    4. 불필요한 정보를 피하고, 답변을 간결하고 명확하게 작성합니다.
    5. 문맥에서 답을 찾을 수 없으면 "주어진 정보만으로는 답할 수 없습니다."라고 답변합니다.
    6. 적절한 경우 문맥에서 직접 인용하며, 따옴표를 사용합니다.

    [Context]
    {context}

    [Question]
    {question}

    [Answer]
    """

    prompt = ChatPromptTemplate.from_template(template)
    local_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    rag_chain = prompt | local_llm | StrOutputParser()
    generation = rag_chain.invoke({"context": format_docs(docs), "question": question})
    return generation

generation = generator_rag_answer(question, docs=relevant_docs)
print("Generated Answer (일반 RAG):")
print(generation)

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
hallucination = hallucination_grader.invoke({"documents": relevant_docs, "generation": generation})
print(f"환각 평가: {hallucination}")

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
print("Question:", question)
print("Generation:", generation)
answer_score = answer_grader_binary.invoke({"question": question, "generation": generation})
print(f"답변 평가: {answer_score}")


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
    print(f"--- 생성된 답변: {state.generation} ---")
    if state.num_generations > 2:
        print("--- 생성 횟수 초과, 종료 -> end ---")
        return "end"
    # 평가를 위한 문서 텍스트 구성
    print("--- 답변 할루시네이션 평가 ---")
    docs_text = "\n\n".join([d.page_content for d in state.documents])
    hallucination_grade = hallucination_grader.invoke({
        "documents": docs_text,
        "generation": state.generation
    })
    if hallucination_grade.binary_score == "yes":
        relevance_grade = answer_grader_binary.invoke({
            "question": state.question,
            "generation": state.generation
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
    if state.num_generations > 2:
        print("--- 생성 횟수 초과, 생성 결정 ---")
        return "generate"
    # 여기서는 필터링된 문서가 존재하는지 확인
    if not state.filtered_documents:
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
class SelfRagOverallState(BaseModel):
    question: str
    generation: str = ""
    routing_decision: str = "" 
    documents: List[Document] = []
    filtered_documents: List[Document] = []
    num_generations: int = 0

# 질문 재작성 노드 (변경 후 검색 루프)
def transform_query_self(state: SelfRagOverallState) -> dict:
    print("--- 질문 개선 ---")
    new_question = rewrite_question(state.question)
    print(f"--- 개선된 질문 : \n{new_question} ")
    state.num_generations += 1
    state.question = new_question  # 상태 업데이트
    print(f"num_generations : {state.num_generations}")
    return {"question": new_question, "num_generations": state.num_generations}

# 답변 생성 노드 (서브 그래프로부터 받은 필터 문서 우선 사용)
def generate_self(state: SelfRagOverallState) -> dict:
    print("--- 답변 생성 ---")
    docs = state.filtered_documents if state.filtered_documents else state.documents
    generation = generator_rag_answer(state.question, docs)
    state.num_generations += 1
    state.generation = generation
    return {"generation": generation, "num_generations": state.num_generations}


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
    decision = question_router.invoke({"question": state.question})
    print("routing_decision:", decision.route)
    return {"routing_decision": decision.route}

# question route 분기 함수 
def route_question_adaptive_self(state: SelfRagOverallState) -> str:
    """
    질문 분석 및 라우팅: 사용자의 질문을 분석하여 '금융질문'인지 '일반질문'인지 판단
    """
    try:
        if state.routing_decision == "llm_fallback":
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
    """),
    ("human", "{question}")
])

def llm_fallback_adaptive(state: SelfRagOverallState):
    """Generates a direct response using the LLM when the question is unrelated to financial products."""
    question = state.question
    fallback_chain = fallback_prompt | llm | StrOutputParser()
    generation = fallback_chain.invoke({"question": question})
    return {"generation": generation}

#############################
# 7. [서브 그래프 통합] - 병렬 검색 서브 그래프 구현
#############################

print('\n7. [서브 그래프 통합] - 병렬 검색 서브 그래프 구현\n')
# --- 상태 정의 (검색 서브 그래프 전용) ---
class SearchState(TypedDict):
    question: str
    generation: str
    documents: Annotated[List[Document], add]  # 팬아웃된 각 검색 결과를 누적할 것
    filtered_documents: List[Document]         # 관련성 평가를 통과한 문서들

# ToolSearchState: SearchState에 추가 정보(datasources) 포함
class ToolSearchState(SearchState):
    datasources: List[str]  # 참조할 데이터 소스 목록

# --- 서브그래프 노드 함수 ---
def search_fixed_deposit_subgraph(state: SearchState):
    """
    정기예금 상품 검색 (서브 그래프)
    """
    question = state["question"]
    print('--- 정기예금 상품 검색 --- ')
    docs = search_fixed_deposit.invoke(question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 정기적금 상품정보를 찾을 수 없습니다.")]}

def search_demand_deposit_subgraph(state: SearchState):
    """
    입출금자유예금 상품 검색 (서브 그래프)
    """
    question = state["question"]
    print('--- 입출금자유예금 상품 검색 ---')
    docs = search_demand_deposit.invoke(question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 입출금자유예금 상품정보를 찾을 수 없습니다.")]}

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
search_builder.add_node("search_fixed_deposit", search_fixed_deposit_subgraph)
search_builder.add_node("search_demand_deposit", search_demand_deposit_subgraph)
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
class SelfRagOverallState(SelfRagOverallState):
    # 이미 정의된 상태에 필터링된 문서를 포함
    pass

# 전체 그래프 빌더 (rag_builder) 구성
rag_builder = StateGraph(SelfRagOverallState)

# 노드 추가: 검색 서브 그래프, 생성, 질문 재작성 등
rag_builder.add_node("route_question", route_question_adaptive)
rag_builder.add_node("llm_fallback", llm_fallback_adaptive)
rag_builder.add_node("search_data", tool_search_graph)         # 서브 그래프로 병렬 검색 및 필터링 수행
rag_builder.add_node("generate", generate_self)                # 답변 생성 노드
rag_builder.add_node("transform_query", transform_query_self)  # 질문 개선 노드

# 전체 그래프 엣지 구성
rag_builder.add_edge(START, "route_question")
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

adaptive_self_rag = rag_builder.compile()

# 그래프 파일 저장하기
# display(Image(adaptive_self_rag.get_graph().draw_mermaid_png()))
with open("adaptive_self_rag.mmd", "w") as f:
    f.write(adaptive_self_rag.get_graph(xray=True).draw_mermaid()) # 저장된 mmd 파일에서 코드 복사 후 https://mermaid.live 에 붙여넣기.

#############################
# 9. 전체 그래프 실행 및 결과 확인
#############################

inputs = {"question": "정기예금과 입출금자유예금은 어떤 차이점이 있나요? 정기예금 상품중 가장 금리가 높은 것은 무엇인가요?"}
print('\n9. 전체 그래프 실행 및 결과 확인\n')
print(f'inputs: {inputs}\n')
final_output = adaptive_self_rag.invoke(inputs)

if "generation" in final_output:
    print("# 최종 답변")
    print(final_output["generation"])
else:
    print("결과를 찾을 수 없습니다.")

# # 추가 테스트 예시
# inputs = {"question": "국민은행의 정기예금 상품이 있나요? 우리은행 상품과 어떤점이 다른가요?"}
# for output in adaptive_self_rag.stream(inputs):
#     for node_name, node_value in output.items():
#         pprint(f"Node '{node_name}':")
#         pprint(f"Value: {node_value}", indent=2, width=80)
#     print("\n----------------------------------------------------------\n")
# print("# 최종 답변 (추가 테스트)")
# print(output.get("generation", "응답 생성 실패"))

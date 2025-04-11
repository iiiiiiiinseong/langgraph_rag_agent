```mermaid
---
config:
  theme: default
  flowchart:
    curve: linear
    nodeSpacing: 50
    rankSpacing: 40
    fontFamily: "Pretendard, sans-serif"
    fontSize: 16
---
graph TD;
	__start__([<p>__start__</p>]):::first
	route_question(route_question)
	llm_fallback(llm_fallback)
	generate(generate)
	transform_query(transform_query)
	__end__([<p>__end__</p>]):::last
	
    __start__ --> route_question;
	llm_fallback --> __end__;
	transform_query --> search_data_analyze_question;
	route_question -.-> |일반질문|llm_fallback;
	route_question -.-> |금융질문|search_data_analyze_question;
	search_data_filter_documents -.->|문서가 질문과 관련성이 없음|transform_query;
	search_data_filter_documents -.->|질문과 문서과 관련됨|generate;
	generate -.-> |질문과 답변이 관련성이 없음|transform_query;
	generate -.-> |답변이 우수|__end__;
	generate -.-> |생성횟수 초과|__end__;
    generate -.-> |답변이 문서와 관련성이 없음| generate;

    subgraph search_data
	search_data_analyze_question(analyze_question)
	search_data_search_fixed_deposit(search_fixed_deposit)
	search_data_search_demand_deposit(search_demand_deposit)
	search_data_filter_documents(filter_documents)
	search_data_search_demand_deposit --> search_data_filter_documents;
	search_data_search_fixed_deposit --> search_data_filter_documents;
	search_data_analyze_question -.-> search_data_search_fixed_deposit;
	search_data_analyze_question -.-> search_data_search_demand_deposit;
	end

  classDef first fill:#3b82f6,stroke:#3b82f6,color:#ffffff;
  classDef last fill:#10b981,stroke:#10b981,color:#ffffff;
  classDef main fill:#2d2f92,stroke:#4f46e5,color:#ffffff;
  linkStyle default stroke:#a5b4fc,color:#000000,stroke-width:2px;


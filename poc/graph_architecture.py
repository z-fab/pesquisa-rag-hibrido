import json
import operator
import time
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

load_dotenv()

# --- CONFIGURAÇÕES ---
SQL_DB_PATH = "sqlite:///data/dados.db"
CHROMA_PATH = "./data/chroma_db"
YAML_ESTRUTURADO = "data/estruturado.yaml"
YAML_NAO_ESTRUTURADO = "data/nao_estruturado.yaml"


def load_semantic_map():
    """Lê os YAMLs e retorna representações estruturadas para prompts."""
    with open(YAML_ESTRUTURADO, "r", encoding="utf-8") as f:
        struct_data = yaml.safe_load(f)
    with open(YAML_NAO_ESTRUTURADO, "r", encoding="utf-8") as f:
        unstruct_data = yaml.safe_load(f)
    return struct_data, unstruct_data


SEM_STRUCT, SEM_DOCS = load_semantic_map()


def format_sql_context(tables: Optional[List[str]] = None) -> str:
    """Gera contexto compacto do schema apenas com descrição e colunas."""
    selected = []
    for tb in SEM_STRUCT["tables"]:
        if tables and tb["table_name"] not in tables:
            continue
        selected.append(tb)
    if not selected:
        selected = SEM_STRUCT["tables"]

    lines = ["ESQUEMA DO BANCO (use apenas essas tabelas e colunas):"]
    for tb in selected:
        lines.append(f"- {tb['table_name']}: {tb['description']}")
        for col in tb["columns"]:
            lines.append(f"  * {col['name']} ({col['type']}): {col['description']}")
    return "\n".join(lines)


def format_doc_context(docs: Optional[List[str]] = None) -> str:
    """Gera contexto dos documentos com título e resumo."""
    selected = []
    for doc in SEM_DOCS["documents"]:
        if docs and doc["file_id"] not in docs:
            continue
        selected.append(doc)
    if not selected:
        selected = SEM_DOCS["documents"]

    lines = ["DOCUMENTOS DISPONÍVEIS:"]
    for doc in selected:
        topics = ", ".join(doc.get("key_topics", []))
        lines.append(f"- {doc['file_id']}: {doc['title']} | {doc['summary']} | Tópicos: {topics}")
    return "\n".join(lines)

# --- SETUP MODELOS ---
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
sql_engine = create_engine(SQL_DB_PATH)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)


# Custom reducer for token_usage that sums the values
def add_token_usage(left: Optional[Dict[str, float]], right: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Custom reducer to sum token usage dictionaries."""
    if not left:
        left = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    if not right:
        right = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    return {
        "input_tokens": left.get("input_tokens", 0.0) + right.get("input_tokens", 0.0),
        "output_tokens": left.get("output_tokens", 0.0) + right.get("output_tokens", 0.0),
        "total_tokens": left.get("total_tokens", 0.0) + right.get("total_tokens", 0.0)
    }


# --- ESTADO ---
class AgentState(TypedDict, total=False):
    question: str
    router_decision: str
    router_tables: List[str]
    router_docs: List[str]
    sql_query: str
    sql_result: List[dict]
    sql_result_raw: str
    text_result: str
    final_answer: str
    sources: List[str]
    trace: Annotated[List[dict], operator.add]
    executed_agents: Annotated[List[str], operator.add]
    total_latency: float
    token_usage: Annotated[Dict[str, float], add_token_usage]
    total_start: float


def _ensure_tracking(state: AgentState):
    state.setdefault("trace", [])
    state.setdefault("executed_agents", [])
    state.setdefault(
        "token_usage", {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    )
    state.setdefault("total_start", time.perf_counter())


def _normalize_usage(raw: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not raw:
        return {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    input_tokens = raw.get("input_tokens") or raw.get("prompt_tokens") or 0.0
    output_tokens = raw.get("output_tokens") or raw.get("completion_tokens") or 0.0
    total_tokens = raw.get("total_tokens") or raw.get("total") or (input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens or 0.0,
        "output_tokens": output_tokens or 0.0,
        "total_tokens": total_tokens or 0.0,
    }


def _extract_usage_from_response(response) -> Optional[Dict[str, float]]:
    """Extract token usage from LangChain response object.

    Handles both standardized usage_metadata (LangChain 0.2+) and
    OpenAI-specific response_metadata formats.
    """
    # Try standardized usage_metadata first (preferred)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metadata = response.usage_metadata
        # Already in standard format: input_tokens, output_tokens, total_tokens
        if isinstance(metadata, dict):
            return {
                "input_tokens": float(metadata.get("input_tokens", 0)),
                "output_tokens": float(metadata.get("output_tokens", 0)),
                "total_tokens": float(metadata.get("total_tokens", 0))
            }

    # Try OpenAI-specific response_metadata
    if hasattr(response, "response_metadata") and response.response_metadata:
        token_data = response.response_metadata.get("token_usage", {})
        if token_data:
            # Map OpenAI format to standard format
            return {
                "input_tokens": float(token_data.get("prompt_tokens", 0)),
                "output_tokens": float(token_data.get("completion_tokens", 0)),
                "total_tokens": float(token_data.get("total_tokens", 0))
            }

    # No usage found
    return None


def _record_end(state: AgentState, name: str, start_time: float, usage: Optional[Dict[str, float]]) -> dict:
    """Records node execution metrics and returns state updates."""
    duration = round(time.perf_counter() - start_time, 4)

    # Build trace entry
    trace_entry = {"node": name, "duration": duration}

    # Build executed agents list (exclude router)
    executed = [name] if name != "router" else []

    # Normalize token usage (the custom reducer will sum it)
    norm = _normalize_usage(usage)

    return {
        "trace": [trace_entry],  # Will be appended via operator.add
        "executed_agents": executed,  # Will be appended via operator.add
        "token_usage": norm  # Will be summed via add_token_usage reducer
    }


# --- 1. NÓ ROUTER (Manual com Contexto YAML) ---
def router_node(state: AgentState):
    print(f"--- Router: Analisando '{state['question']}' ---")
    _ensure_tracking(state)
    node_start = time.perf_counter()
    # Injeta o Mapa Semântico no Prompt do Router
    system_prompt = f"""Você é um roteador de perguntas. Analise a pergunta e decida QUAL fonte de dados usar.

FONTES DISPONÍVEIS:
1. sqldb: Use quando a pergunta pede APENAS dados quantitativos/numéricos que estão nas tabelas SQL
   - Exemplos: "Qual foi a produção de X?", "Qual estado teve maior valor?", "Quantas toneladas..."

2. vectorstore: Use quando a pergunta pede APENAS informações qualitativas/textuais dos relatórios
   - Exemplos: "Quais são os desafios...", "O que dizem os relatórios sobre...", "Quais problemas trabalhistas..."

3. hybrid: Use quando precisa COMBINAR dados numéricos (SQL) COM contexto/explicações (PDFs)
   - Exemplos: "Como X se compara a Y?" (precisa buscar ambos e comparar)
   - "Por que X afeta Y?" (precisa dado numérico + explicação qualitativa)
   - "X aparece em Y? O que isso indica?" (busca SQL + interpretação de relatório)

REGRAS DE DECISÃO:
- Se a pergunta menciona "comparar", "relacionar" ou pede interpretação de dados → HYBRID
- Se pede "por quê", "como isso afeta", "o que isso significa" após dados numéricos → HYBRID
- Se pergunta explicitamente sobre "relatórios", "estudos", "documentos" sem pedir números → VECTORSTORE
- Se pergunta APENAS por valores, quantidades, rankings numéricos → SQLDB
- Sempre selecione as tabelas e documentos RELEVANTES (não todos!)

FORMATO DE RESPOSTA (apenas JSON):
{{"datasource": "sqldb|vectorstore|hybrid", "tables": ["tabela1",...], "documents": ["doc1.pdf",...]}}

TABELAS DISPONÍVEIS:
{format_sql_context()}

DOCUMENTOS DISPONÍVEIS:
{format_doc_context()}

PERGUNTA A ROTEAR:"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["question"])]
    )
    decision_raw = response.content
    tables: List[str] = []
    documents: List[str] = []
    try:
        decision_json = json.loads(decision_raw)
        decision = decision_json.get("datasource", "").strip()
        tables = decision_json.get("tables") or []
        documents = decision_json.get("documents") or []
    except Exception:
        decision = decision_raw.strip().lower()

    usage = _extract_usage_from_response(response)
    tracking_data = _record_end(state, "router", node_start, usage)
    return {
        "router_decision": decision,
        "router_tables": tables,
        "router_docs": documents,
        "router_usage": usage,
        **tracking_data  # Merge trace, executed_agents, token_usage
    }


# --- 2. NÓ SQL AGENT (Manual com Contexto YAML) ---
def sql_node(state: AgentState):
    print("--- SQL Agent ---")
    _ensure_tracking(state)
    node_start = time.perf_counter()

    # Prompt usa o YAML para entender o schema, não o banco direto
    sql_ctx = format_sql_context(state.get("router_tables"))

    # Add few-shot examples (generic patterns, no specific validation data)
    examples = """
EXEMPLOS DE QUERIES CORRETAS:

1. Total nacional com filtro de produto (case-insensitive):
   SQL: SELECT SUM(quantidade_toneladas) AS total
        FROM extracao_vegetal_producao
        WHERE LOWER(produto) LIKE '%palavra_chave%';

2. Agregação por estado (requer JOIN):
   SQL: SELECT v.uf_sigla, SUM(e.quantidade_toneladas) AS total
        FROM extracao_vegetal_producao e
        JOIN vinculo_localidade_estado v ON e.localidade = v.nome_localidade
        WHERE LOWER(e.produto) LIKE '%palavra_chave%'
        GROUP BY v.uf_sigla
        ORDER BY total DESC;

3. Maior valor com LIMIT:
   SQL: SELECT v.uf_sigla, SUM(e.quantidade_toneladas) AS total
        FROM extracao_vegetal_producao e
        JOIN vinculo_localidade_estado v ON e.localidade = v.nome_localidade
        WHERE LOWER(e.produto) LIKE '%palavra_chave%'
        GROUP BY v.uf_sigla
        ORDER BY total DESC
        LIMIT 1;

4. Filtro por data em tabelas LSPA:
   SQL: SELECT producao_variacao_pct
        FROM lspa_producao_agricola_2024
        WHERE LOWER(produto) LIKE '%palavra_chave%'
          AND data_referencia = 'YYYY-MM-DD';
"""

    prompt = f"""Você é um especialista em SQLite. Gere uma query SQL VÁLIDA para responder à pergunta.

REGRAS OBRIGATÓRIAS:
1. Use APENAS as tabelas e colunas listadas no esquema abaixo
2. Para filtrar produtos, SEMPRE use LOWER() e LIKE para matching case-insensitive:
   - Exemplo: WHERE LOWER(produto) LIKE '%açaí%'
   - Exemplo: WHERE LOWER(produto) LIKE '%arroz%casca%'
   - NUNCA use WHERE produto = 'nome_exato' (case-sensitive)
3. Para filtrar anos, use WHERE ano = valor (ex: ano = 2016) ou WHERE data_referencia = 'YYYY-MM-DD'
4. Para obter dados por Estado/UF, faça JOIN com vinculo_localidade_estado:
   JOIN vinculo_localidade_estado v ON tabela.localidade = v.nome_localidade
5. Para "Brasil" ou "total nacional", some TODOS os registros (sem filtro de UF)
6. Para "por estado" ou "cada UF", use GROUP BY v.uf_sigla
7. Use funções agregadas: SUM(), AVG(), MAX(), MIN(), COUNT()
8. Retorne APENAS o SQL, sem ```sql nem explicações

ESQUEMA DO BANCO:
{sql_ctx}

{examples}

Pergunta: {state["question"]}

SQL (apenas a query, sem markdown):"""

    response = llm.invoke([HumanMessage(content=prompt)])
    query = response.content.replace("```sql", "").replace("```", "").strip()

    # Basic validation
    query_lower = query.lower()
    validation_errors = []

    if not query_lower.startswith("select"):
        validation_errors.append("Query deve começar com SELECT")

    if "drop " in query_lower or "delete " in query_lower or "update " in query_lower:
        validation_errors.append("Apenas queries SELECT são permitidas")

    if len(query) > 2000:
        validation_errors.append("Query muito longa (max 2000 caracteres)")

    result_rows = []
    result_str = ""

    if validation_errors:
        result_str = "Erro de validação: " + "; ".join(validation_errors)
    else:
        # Execute with timeout protection
        import threading

        def run_query_with_timeout(engine, query_text, timeout=10):
            result = [None, None]  # [result_rows, error]

            def execute():
                try:
                    with engine.connect() as conn:
                        res = conn.execute(text(query_text))
                        rows = res.fetchall()
                        result[0] = [dict(row._mapping) for row in rows]
                except Exception as e:
                    result[1] = str(e)

            thread = threading.Thread(target=execute)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                raise TimeoutError("Query timeout")

            if result[1]:
                raise Exception(result[1])

            return result[0] or []

        try:
            result_rows = run_query_with_timeout(sql_engine, query, timeout=10)
            if result_rows:
                result_str = str(result_rows)
            else:
                result_str = "Consulta executada com sucesso, mas sem resultados."
        except TimeoutError:
            result_str = "Erro SQL: Timeout (10s) - query muito complexa ou ineficiente"
        except Exception as e:
            result_str = f"Erro SQL: {str(e)}"

    usage = _extract_usage_from_response(response)
    tracking_data = _record_end(state, "sql_agent", node_start, usage)
    return {
        "sql_query": query,
        "sql_result": result_rows,
        "sql_result_raw": result_str,
        "sql_usage": usage,
        **tracking_data
    }


# --- 3. NÓ TEXT RETRIEVER ---
def text_node(state: AgentState):
    print("--- Text Retriever ---")
    _ensure_tracking(state)
    node_start = time.perf_counter()

    selected_docs = state.get("router_docs")
    search_filter = None
    if selected_docs:
        search_filter = {"source": {"$in": selected_docs}}

    docs = vectorstore.similarity_search(state["question"], k=4, filter=search_filter)

    context = "\n".join(
        [f"[Fonte: {d.metadata.get('source')}] {d.page_content}" for d in docs]
    )
    sources = [d.metadata.get("source") for d in docs]

    tracking_data = _record_end(state, "text_retriever", node_start, None)
    return {
        "text_result": context,
        "sources": sources,
        **tracking_data
    }


# --- 4. NÓ SYNTHESIZER ---
def synthesizer_node(state: AgentState):
    print("--- Synthesizer ---")
    _ensure_tracking(state)
    node_start = time.perf_counter()

    prompt = f"""Responda à pergunta do usuário usando APENAS as evidências abaixo.

[DADOS ESTRUTURADOS - SQL]
Query: {state.get("sql_query")}
Resultado: {state.get("sql_result", "N/A")}

[DADOS NÃO ESTRUTURADOS - TEXTO]
{state.get("text_result", "N/A")}

Pergunta: {state["question"]}

Regras:
- Use somente informações presentes nas evidências.
- Cite as fontes (nome da tabela ou nome do arquivo PDF).
- Seja conciso e evite alucinar dados não presentes.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    usage = _extract_usage_from_response(response)
    tracking_data = _record_end(state, "synthesizer", node_start, usage)
    total_latency = round(time.perf_counter() - state["total_start"], 4)
    return {
        "final_answer": response.content,
        "synth_usage": usage,
        "total_latency": total_latency,
        **tracking_data
    }


# --- GRAFO ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("sql_agent", sql_node)
workflow.add_node("text_retriever", text_node)
workflow.add_node("synthesizer", synthesizer_node)


def route_logic(state):
    d = state["router_decision"]
    if d == "sqldb":
        return "sql_agent"
    elif d == "vectorstore":
        return "text_retriever"
    else:
        return ("sql_agent", "text_retriever")


workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    route_logic,
    ["sql_agent", "text_retriever"],
)
workflow.add_edge("sql_agent", "synthesizer")
workflow.add_edge("text_retriever", "synthesizer")
workflow.add_edge("synthesizer", END)
app = workflow.compile()

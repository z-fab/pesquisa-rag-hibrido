import json
import time

from agent.state import AgentState
from config.settings import SETTINGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from services.mapa_service import (
    format_non_struct_semantic_map_to_context,
    format_struct_semantic_map_to_context,
)
from utils.agent_utils import extract_usage_from_response, record_end


class RouterAgent:
    def __call__(self, state: AgentState):
        logger.info(f"\n🤖 Executando {self.__class__.__name__}...")

        model = ChatOpenAI(model=SETTINGS.OPENAI_SOFT_MODEL, temperature=0)

        node_start = time.perf_counter()

        # Injeta o Mapa Semântico no Prompt do Router
        system_prompt = f"""<system>
            <role>
                Você é um ROTEADOR DE CONSULTAS. Sua tarefa é analisar a pergunta do usuário e decidir QUAL FONTE DE DADOS deve ser usada.
                Você deve classificar a pergunta como: structured, non_structured ou hybrid.
            </role>

            <data_sources>
                <structured>
                    Use quando a pergunta requer APENAS valores quantitativos presentes nas tabelas SQL.
                    Exemplos:
                        - "Qual foi a produção de X?"
                        - "Qual estado produziu mais?"
                        - "Quantas toneladas…?"
                </structured>

                <non_structured>
                    Use quando a pergunta requer APENAS informações qualitativas/textuais dos relatórios PDF.
                    Exemplos:
                        - "Quais são os desafios…?"
                        - "O que dizem os relatórios sobre…?"
                        - "Quais problemas trabalhistas…?"
                </non_structured>

                <hybrid>
                    Use quando a resposta precisa combinar:
                        (1) números provenientes do SQL
                        (2) explicações qualitativas dos relatórios
                    Exemplos:
                        - “Como X se compara a Y?”
                        - “Por que X afeta Y?”
                        - “X aparece em Y? O que isso indica?”
                </hybrid>
            </data_sources>

            <output_format>
                O retorno DEVE ser apenas em JSON, sem explicações extras, no formato:
                {{
                    "datasource": "structured | non_structured | hybrid",
                    "tables": ["tabela1", ...],
                    "documents": ["doc1.pdf", ...]
                }}
            </output_format>

            {format_struct_semantic_map_to_context()}

            {format_non_struct_semantic_map_to_context()}
        </system>
        """

        user = f"""
        <user_query>
            {state["question"]}
        </user_query>
        """

        content = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user),
        ]

        logger.debug(f"\n➡️ {content}")

        response = model.invoke(content)

        logger.debug(f"\n⬅️ {response}")

        decision_raw = response.content
        tables: list[str] = []
        documents: list[str] = []

        try:
            decision_json = json.loads(decision_raw)
            decision = decision_json.get("datasource", "").strip()
            tables = decision_json.get("tables") or []
            documents = decision_json.get("documents") or []
        except Exception:
            decision = decision_raw.strip().lower()

        usage = extract_usage_from_response(response)
        tracking_data = record_end(state, "router", node_start, usage)

        return {
            "router_decision": decision,
            "router_tables": tables,
            "router_docs": documents,
            "router_usage": usage,
            **tracking_data,  # Merge trace, executed_agents, token_usage
        }

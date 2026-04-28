import time
from typing import Optional

from agent.state import AgentState
from config.settings import SETTINGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from repositories.mapa_repository import (
    load_semantic_map_struct,
)
from repositories.sqlite_repository import execute_query
from services.mapa_service import format_struct_semantic_map_to_context
from utils.agent_utils import extract_usage_from_response, record_end


class SQLAgent:
    def __call__(self, state: AgentState):
        logger.info(f"\n🤖 Executando {self.__class__.__name__}...")

        model = ChatOpenAI(model=SETTINGS.OPENAI_SOFT_MODEL, temperature=0)

        node_start = time.perf_counter()

        # Injeta o Mapa Semântico no Prompt do Router
        system_prompt = f"""<system>
            <role>
                Você é um especialista em SQL. Sua tarefa é gerar UMA ÚNICA query SQL 100% válida
                para responder à pergunta do usuário fornecida em <user_query>.
            </role>

            <task>
                Você receberá:
                - Neste bloco <system>, a descrição das regras e o esquema do banco em <sql_schema>.
                - Em uma mensagem separada (do usuário), um bloco <user_query> contendo a pergunta em linguagem natural.

                Sua função é:
                - Ler a pergunta em <user_query>.
                - Ler o schema em <sql_schema>.
                - Gerar uma query SQL que responda à pergunta, usando EXCLUSIVAMENTE as tabelas e colunas presentes no schema.
            </task>

            <principles>
                - Você NUNCA deve inventar tabelas.
                - Você NUNCA deve inventar colunas.
                - Você NUNCA deve usar alias, funções, filtros ou agrupamentos que utilizem colunas inexistentes no schema.
                - Antes de usar qualquer coluna, verifique se ela está EXPLICITAMENTE listada no schema.
                - Caso a pergunta exija algo impossível de responder com as tabelas/colunas disponibilizadas, responda exatamente:
                SELECT 'ERRO: PERGUNTA NAO SUPORTADA PELO ESQUEMA' AS mensagem;
            </principles>

            <rules_sql>
                REGRAS OBRIGATÓRIAS PARA CAMPOS STRING/TEXTO:
                - SEMPRE use LOWER(coluna) LIKE '%texto%'
                - NUNCA use comparação case-sensitive com "=".
                
                REGRAS PARA DATA/ANO:
                - Só use colunas de data/ano se elas existirem no schema.
                - Ex: ano = 2016 ou data_referencia = 'YYYY-MM-DD'

                REGRAS PARA UF/LOCALIDADE:
                - Só use JOINs se as colunas existirem no schema.
                - Exemplo:
                JOIN vinculo_localidade_estado v
                    ON t.localidade = v.nome_localidade

                REGRAS PARA AGREGAÇÃO:
                - Use SUM(), AVG(), MAX(), MIN(), COUNT() quando apropriado.
                - Para “maior”, “menor”, “top 1”, ordene com ORDER BY e limite com LIMIT 1 (se suportado).

                PROIBIDO:
                - Inventar colunas.
                - Inventar tabelas.
                - Usar sintaxe de outro SGDB que não faça sentido.
                - Retornar comentários ou markdown.
            </rules_sql>

            <output_format>
                Retorne APENAS a query SQL final.
                Não inclua explicações.
                Não inclua markdown.
            </output_format>

            {format_struct_semantic_map_to_context(state.get("router_tables"), True)}

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
        usage = extract_usage_from_response(response)
        tracking_data = record_end(state, "sql_agent", node_start, usage)

        query = response.content.replace("```sql", "").replace("```", "").strip()
        is_valid, validation_msg = self._validate_query(query)
        if not is_valid:
            logger.warning(f"Query SQL inválida: {validation_msg}")
            return {
                "sql_query": query,
                "sql_executed": False,
                "sql_result": validation_msg,
                "sql_result_raw": [],
                "sql_usage": usage,
                **tracking_data,
            }

        try:
            result_rows = execute_query(query)
        except Exception as e:
            logger.error(f"Erro ao executar query SQL: {e}")
            return {
                "sql_query": query,
                "sql_executed": False,
                "sql_result": f"Erro ao executar query: {e}",
                "sql_result_raw": [],
                "sql_usage": usage,
                **tracking_data,
            }

        if result_rows:
            result_str = str(result_rows)
        else:
            result_str = "Consulta executada com sucesso, mas sem resultados."

        return {
            "sql_query": query,
            "sql_executed": True,
            "sql_result": result_str,
            "sql_result_raw": result_rows,
            "sql_usage": usage,
            **tracking_data,
        }

    def _format_sql_context(self, tables: Optional[list[str]] = None) -> str:
        """Gera contexto compacto do schema apenas com descrição e colunas."""

        semantic_map = load_semantic_map_struct()
        selected = []
        for tb in semantic_map["tables"]:
            if tables and tb["table_name"] not in tables:
                continue
            selected.append(tb)
        if not selected:
            selected = semantic_map["tables"]

        lines = ["ESQUEMA DO BANCO (use apenas essas tabelas e colunas):"]
        for tb in selected:
            lines.append(f"- {tb['table_name']}: {tb['description']}")
            for col in tb["columns"]:
                lines.append(f"  * {col['name']} ({col['type']}): {col['description']}")
        return "\n".join(lines)

    def _validate_query(self, query: str) -> tuple[bool, str]:
        # Basic validation
        query_lower = query.lower()
        validation_errors = []

        if (
            "drop " in query_lower
            or "delete " in query_lower
            or "update " in query_lower
        ):
            validation_errors.append("Apenas queries SELECT são permitidas")

        if len(query) > 2000:
            validation_errors.append("Query muito longa (max 2000 caracteres)")

        if len(validation_errors) > 0:
            return False, "Erro de validação: " + "; ".join(validation_errors)
        return True, ""

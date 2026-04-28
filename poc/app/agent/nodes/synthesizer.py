import time

from agent.state import AgentState
from config.settings import SETTINGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from utils.agent_utils import extract_usage_from_response, record_end


class SynthesizerAgent:
    def __call__(self, state: AgentState):
        logger.info(f"\n🤖 Executando {self.__class__.__name__}...")

        model = ChatOpenAI(model=SETTINGS.OPENAI_SOFT_MODEL, temperature=0)

        node_start = time.perf_counter()

        # Injeta o Mapa Semântico no Prompt do Router
        system_prompt = f"""
        <system>
            <role>
                Você é um agente de síntese responsável por responder à pergunta do usuário
                usando APENAS as evidências estruturadas e não estruturadas fornecidas abaixo.
                Sua função é gerar uma resposta clara, correta e bem fundamentada.
            </role>

            <constraints>
                - Você NÃO pode inferir, supor ou alucinar informações.
                - Você NÃO pode usar conhecimento externo.
                - Você DEVE usar exclusivamente os dados apresentados nas seções:
                <structured_data> e <unstructured_data>.
                - Todas as afirmações devem ser sustentadas por evidências explícitas.
                - Se uma informação não existir nas evidências, diga que ela não está disponível.
            </constraints>

            <citation_rules>
                Como citar fontes:
                - Para dados estruturados obtenha o nome da tabela a partir da query ou da origem do dado.
                - Para dados não estruturados cite o nome do arquivo PDF conforme listado no contexto.
                - Formato recomendado:
                - “Segundo a tabela X…”
                - “De acordo com o relatório Y…”
            </citation_rules>

            <answer_style>
                - Seja conciso, objetivo e direto.
                - Não repita a pergunta.
                - Não escreva texto supérfluo.
                - Não gere explicações metodológicas.
                - Responda em linguagem natural de forma clara.
            </answer_style>

            <conflict_resolution>
                - Se dados estruturados e não estruturados divergirem:
                    - Informe a divergência de forma neutra.
                    - Não tente conciliar inventando valores.
                    - Cite as duas fontes.
                    - Se os dados forem insuficientes para responder integralmente:
                        - Responda apenas a parte que é possível.
                        - Indique explicitamente quais elementos não estavam presentes nas evidências.
                    - Se não houver qualquer dado relevante:
                        - Retorne: “Não há informações suficientes nas evidências fornecidas para responder.”
            </conflict_resolution>

            <structured_data>
                Query executada:
                {state.get("sql_query")}

                Resultado da query:
                {state.get("sql_result", "N/A")}
            </structured_data>

            <unstructured_data>
                Conteúdo textual encontrado:
                {state.get("text_result", "N/A")}
            </unstructured_data>

            <instruction>
                Sua tarefa final:
                - Ler a pergunta em <user_query>.
                - Usar apenas as evidências acima.
                - Produzir uma resposta final fundamentada.
            </instruction>
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
        tracking_data = record_end(state, "synthesizer", node_start, usage)

        total_latency = round(
            time.perf_counter() - state.get("total_start", node_start), 4
        )
        return {
            "final_answer": response.content,
            "synth_usage": usage,
            "total_latency": total_latency,
            **tracking_data,
        }

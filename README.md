# Arquitetura Multiagente Baseada em Modelos de Linguagem para Recuperação Híbrida de Informação

Repositório da dissertação de mestrado de mesmo título, desenvolvida no Programa de Pós-Graduação em Computação Aplicada da Universidade Presbiteriana Mackenzie.

- **Autor:** Fabrício Rodrigues Zillig
- **Orientador:** Prof. Dr. Leandro Augusto da Silva
- **Coorientadora:** Profa. Dra. Valéria Farinazzo Martins
- **Local/Ano:** São Paulo, 2026

O trabalho propõe e avalia uma arquitetura multiagente baseada em Grandes Modelos de Linguagem (LLMs) capaz de, a partir de perguntas em linguagem natural, decidir autonomamente se a resposta deve ser buscada em dados estruturados, não estruturados ou na combinação de ambos, integrando as evidências em uma resposta final explicável e auditável, com citações às fontes. O desenvolvimento segue a metodologia Design Science Research (DSR), e a avaliação é conduzida por experimentos controlados sobre um benchmark de consultas categorizadas em **Estruturadas (S)**, **Não Estruturadas (NS)** e **Híbridas (H)**.

Este trabalho está inserido no contexto maior **AmazonIA**, que explora o uso de agentes conversacionais para apoiar a agricultura sustentável na região amazônica.

---

## 1. Estrutura do repositório

Monorepo com duas implementações independentes:

| Diretório | Conteúdo | Estado |
|---|---|---|
| [`poc/`](./poc) | Prova de conceito inicial (novembro de 2024), descrita no Capítulo 4 do projeto de qualificação. Validou a viabilidade da arquitetura sobre um recorte reduzido de dados. | Arquivada — não recebe mais desenvolvimento. |
| [`experimentos/`](./experimentos) | Implementação final da arquitetura multiagente: pipeline LangGraph modular, suíte de avaliação com múltiplos juízes LLM, análises estatísticas e geração de relatórios. | Ativa — base experimental da dissertação. |

Cada diretório possui seu próprio `README.md` com instruções específicas.

### O que **não** está neste repositório

- **Dataset** (base CSV, PDFs originais, schemas semânticos, conjunto de perguntas-gabarito): vive em um **repositório próprio**, dedicado à curadoria e ao versionamento da base. Link: [Agro-RAG Dataset](https://github.com/z-fab/agro-rag-dataset).
- **Dissertação e materiais de escrita** (rascunhos, capítulos, figuras geradas para o texto): mantidos fora deste repositório pois são artefatos de redação em constante iteração.

Neste repositório de código fica versionado apenas `experimentos/data/outputs/`, com os snapshots e agregados de todas as execuções de avaliação usadas no texto final da dissertação.

---

## 2. Por que dataset em repositório separado

- O benchmark (perguntas curadas + gabaritos + documentos + base estruturada) tem valor próprio e pode ser citado por trabalhos que não usem esta arquitetura específica.
- Separar permite evolução independente do corpus sem obrigar versionamento atrelado ao código.
- A base pode ser reaproveitada como teste para outras arquiteturas de RAG, mantendo comparabilidade entre estudos.

Instruções de como carregar o dataset neste pipeline ficam em [`experimentos/README.md`](./experimentos/README.md).

---

## 3. Reprodução resumida

Resumo dos passos — instruções completas em [`experimentos/README.md`](./experimentos/README.md).

```bash
# 1. Clonar este repositório e o repositório de dataset.
git clone git@github.com:z-fab/pesquisa-rag-hibrido.git
git clone git@github.com:z-fab/agro-rag-dataset.git /tmp/agro-rag-dataset

# 2. Copiar os artefatos da base para experimentos/data/
cd pesquisa-rag-hibrido/experimentos
mkdir -p data/raw/structured data/raw/unstructured
cp /tmp/agro-rag-dataset/data/structured/csv/*.csv    data/raw/structured/
cp /tmp/agro-rag-dataset/data/unstructured/pdfs/*.pdf data/raw/unstructured/
cp /tmp/agro-rag-dataset/data/benchmark/evaluation.json data/evaluation.json
cp /tmp/agro-rag-dataset/schemas/structured.yaml      data/structured.yaml
cp /tmp/agro-rag-dataset/schemas/unstructured.yaml    data/unstructured.yaml

# 3. Instalar dependências, configurar .env, ingerir os dados e rodar a avaliação.
uv sync
cp .env.example .env  # editar com as chaves de API
uv run rag ingest
uv run rag eval --ablation full --run-id baseline
uv run rag analyze
```

---

## 4. Uso de Inteligência Artificial neste projeto

Em linha com as práticas declaradas no Capítulo 5 do projeto de qualificação e com princípios de transparência acadêmica, esta seção descreve **onde** e **como** ferramentas de IA foram (e não foram) utilizadas.

**IA foi utilizada como apoio em:**

- **Revisão e polimento de documentação** (READMEs, docstrings) — reescrita para clareza, consistência terminológica e formatação.
- **Revisão de código** (segunda leitura, identificação de bugs sutis, sugestões de refatoração) — sempre com inspeção humana antes da aceitação.
- **Geração de testes unitários de apoio** a partir de especificações escritas pelo autor.
- **Exploração de alternativas para implementação** em pontos específicos, com decisão final humana.
- **Auxílio na elaboração inicial de consultas do benchmark**, sempre com revisão e validação manual posterior.
- **Revisão gramatical** de trechos da dissertação, sem alteração de conteúdo técnico ou científico.

**IA não foi utilizada para:**

- **Concepção da arquitetura multiagente, delineamento dos experimentos, formulação das hipóteses ou escolha das métricas estatísticas** — todo o desenho intelectual do trabalho é de autoria humana.
- **Geração autônoma de código completo** para os módulos centrais do pipeline (nós do LangGraph, lógica de avaliação, implementação do juiz LLM, agregadores estatísticos). Código assistido por IA passou por escrita-revisão-teste iterativos, nunca foi aceito em bloco sem leitura linha a linha.
- **Redação autônoma de seções da dissertação**. O trabalho é escrito pelo autor; IA auxilia em revisão ortográfica e reformulações pontuais, não na elaboração do conteúdo.
- **Interpretação de resultados ou derivação das conclusões do trabalho**. A leitura dos dados experimentais, a discussão dos achados e as implicações teóricas são de autoria humana.
- **Curadoria final do benchmark** (validação das perguntas, redação dos gabaritos, seleção dos documentos) — conduzida pelo autor.

Ferramentas utilizadas: GPT-5.4 (OpenAI), Gemini 3 (Google) e Claude Opus 4.6 (Anthorpic) via interface Web para apoio exploratório e revisão; Claude Code (Anthropic) para revisão de código.

Esse uso está alinhado a princípios de transparência, responsabilidade científica e boas práticas acadêmicas, tratando IA como recurso de apoio ao processo de pesquisa, e não como agente decisório ou autor do conteúdo.

---

## 5. Como citar

```bibtex
@mastersthesis{zillig2026rag,
  author  = {Fabrício Rodrigues Zillig},
  title   = {Arquitetura Multiagente Baseada em Modelos de Linguagem para Recupera\c{c}\~ao H\'ibrida de Informa\c{c}\~ao},
  school  = {Universidade Presbiteriana Mackenzie},
  address = {S\~ao Paulo, SP},
  year    = {2026},
  type    = {Disserta\c{c}\~ao de Mestrado -- Programa de P\'os-Gradua\c{c}\~ao em Computa\c{c}\~ao Aplicada},
  note    = {Orientador: Prof. Dr. Leandro Augusto da Silva. Coorientadora: Profa. Dra. Val\'eria Farinazzo Martins. C\'odigo dispon\'ivel em \url{https://github.com/z-fab/pesquisa-rag-hibrido}}
}
```

---

## 6. Licença

Disponibilizado sob [Licença MIT](./LICENSE) — código-fonte e artefatos experimentais versionados em `experimentos/data/outputs/`.

---

## 7. Contato

Dúvidas, sugestões ou interesse em colaboração: abrir uma _issue_ neste repositório.

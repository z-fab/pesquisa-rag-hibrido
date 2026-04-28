import os

from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

# --- CONFIGURAÇÕES ---
CHROMA_PATH = "./data/chroma_db"
PDF_FOLDER = "./data/raw/nao_estruturado"  # Coloque seus 3 PDFs aqui


# --- 2. SETUP VETORIAL (DOCLING + CHROMA) ---
def setup_vector_db():
    print("--- Processando PDFs com Docling e Ingestão no Chroma ---")

    converter = DocumentConverter()
    docs_content = []

    # Ler PDFs
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"ALERTA: Coloque seus PDFs na pasta {PDF_FOLDER}")
        return

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            print(f"Processando: {filename}...")
            result = converter.convert(os.path.join(PDF_FOLDER, filename))
            # O Docling converte para Markdown, o que é ótimo para preservar estrutura
            markdown_text = result.document.export_to_markdown()

            # Adiciona metadados básicos
            docs_content.append({"text": markdown_text, "source": filename})

    # Chunking Inteligente (Markdown)
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    final_chunks = []
    for doc in docs_content:
        splits = markdown_splitter.split_text(doc["text"])
        for split in splits:
            split.metadata["source"] = doc["source"]
            final_chunks.append(split)

    # Persistência no Chroma
    if final_chunks:
        vectorstore = Chroma.from_documents(
            documents=final_chunks,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=CHROMA_PATH,
        )
        print(f"Ingestão concluída! {len(final_chunks)} chunks salvos em {CHROMA_PATH}")
    else:
        print("Nenhum documento processado.")


if __name__ == "__main__":
    setup_vector_db()

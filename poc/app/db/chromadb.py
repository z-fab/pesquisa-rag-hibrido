from config.settings import SETTINGS
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    persist_directory=SETTINGS.PATH_CHROMA_DB,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

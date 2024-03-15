import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, ServiceContext, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
import datetime
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

if __name__ == "__main__":
    print(datetime.datetime.now())
    # 定义你的LLM
    Settings.llm = Ollama(model="qwen:7b-chat")
    Settings.llm.base_url = "http://1.92.64.112:11434"
    # embed_model
    Settings.embed_model = resolve_embed_model("local:embed_model/bge-small-en-v1.5")
    # 文档解析成更小的块
    # Settings.chunk_size = 512
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # 流式输出
    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query("introduce me Paul Graham?")
    response.print_response_stream()
    print(datetime.datetime.now())



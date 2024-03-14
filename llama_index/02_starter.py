import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, ServiceContext, SummaryIndex
from llama_index.llms.ollama import Ollama

if __name__ == "__main__":
    # 定义你的LLM
    llm = Ollama(model="gemma:2b")
    llm.base_url = "http://1.92.64.112:11434"

    # 定义你的服务上下文
    model_dir = os.path.abspath('embed_model/bge-small-en-v1.5')
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model="local:"+model_dir
    )

    # 加载你的数据
    documents = SimpleDirectoryReader("data").load_data()
    index = SummaryIndex.from_documents(documents, service_context=service_context)

    # 查询和打印结果
    query_engine = index.as_query_engine()
    response = query_engine.query("introduce me Paul Graham")
    print(response)

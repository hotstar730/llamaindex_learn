from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama

if __name__ == "__main__":
    # doc
    documents = SimpleDirectoryReader("data").load_data()

    # llm
    Settings.llm = Ollama(model="gemma:2b")
    Settings.llm.base_url = "http://1.92.64.112:11434"

    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine()
    response = query_engine.query("introduce me Paul Graham")
    print(response)

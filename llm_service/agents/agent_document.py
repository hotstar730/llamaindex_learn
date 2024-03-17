# -- coding:utf-8 --
import datetime
from typing import List

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

from llm_service.agents.agent_base import AgentBase


class AgentDocument(AgentBase):
    def __init__(self) -> None:
        Settings.llm = Ollama(model="qwen:7b-chat")
        Settings.llm.base_url = "http://1.92.64.112:11434"
        # embed_model
        Settings.embed_model = resolve_embed_model("local:data/embed_model/bge-small-en-v1.5")
        # 文档解析成更小的块
        # Settings.chunk_size = 512
        documents = SimpleDirectoryReader("./data/document").load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_chat_engine()

    def chat(self, messages: List[ChatMessage]) -> ChatMessage:
        if messages:
            message = messages[-1]
            response = self.query_engine.chat(message.content + "")
            return ChatMessage(role="assistant", content=response)
        else:
            return ChatMessage(role="assistant", content="can i help you!")

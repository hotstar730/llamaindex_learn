# -- coding:utf-8 --
from typing import List

import nest_asyncio
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

from llm_service.agents.agent_base import AgentBase

nest_asyncio.apply()


class AgentChat(AgentBase):
    def __init__(self) -> None:
        self._llm = Ollama(model="qwen:7b-chat")
        self._llm.base_url = "http://1.92.64.112:11434"

    def chat(self, messages: List[ChatMessage]) -> ChatMessage:
        return self._llm.chat(messages).message

# agent = AgentChat()
# print(agent.chat("你好"))

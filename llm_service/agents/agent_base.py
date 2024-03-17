# -- coding:utf-8 --
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import nest_asyncio
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from pydantic import Field

nest_asyncio.apply()

@dataclass
class AgentBase:
    _llm = Ollama(model="gemma:2b")
    _llm.base_url = "http://1.92.64.112:11434"

    @abstractmethod
    def chat(self, messages: List[ChatMessage]) -> ChatMessage:
        """Chat endpoint for LLM."""

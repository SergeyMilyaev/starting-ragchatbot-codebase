from abc import ABC, abstractmethod
from typing import List, Optional


class BaseAIGenerator(ABC):
    @abstractmethod
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str: ...

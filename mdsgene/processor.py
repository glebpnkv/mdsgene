from abc import ABC, abstractmethod
from typing import Optional, Tuple

class Processor(ABC):
    """
    Abstract base class for processors that can answer questions based on context (e.g., PDF).

    This class defines the interface that all processors must implement.
    """

    @abstractmethod
    def answer_question(self, question: str) -> Optional[Tuple[str, str]]:
        """
        Answer a question based on the processor's knowledge/context.

        Args:
            question: The question to answer

        Returns:
            A tuple of (answer, context), or None if the question cannot be answered
        """
        pass

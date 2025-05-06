# mapping_item.py
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class MappingItem:
    """Represents a mapping between a data field, a query, and Excel info."""
    field: str
    question: str
    mapped_excel_column: str
    response_convertion_strategy: str
    # Custom processor function: takes (raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]
    # Note: Made raw_answer Optional to handle cases where Gemini might fail.
    custom_processor: Optional[Callable[[Optional[str], 'MappingItem'], dict[str, str]]] = None
    query_processor: Optional[str] = None
    active: bool = False  # По умолчанию не активно, если не указано иное

@dataclass
class QuestionInfo:
    """Holds information needed to ask a specific question about a patient."""
    field: str
    query: str
    response_convertion_strategy: str
    query_processor: str
    family_id: str | None = None
    patient_id: str | None = None

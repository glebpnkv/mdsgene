# mapping_item.py
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List

@dataclass
class MappingItem:
    """Represents a mapping between a data field, a query, and Excel info."""
    field: str
    question: str
    mapped_excel_column: str
    response_convertion_strategy: str
    # Custom processor function: takes (raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]
    # Note: Made raw_answer Optional to handle cases where Gemini might fail.
    custom_processor: Optional[Callable[[Optional[str], 'MappingItem'], Dict[str, str]]] = None

@dataclass
class QuestionInfo:
    """Holds information needed to ask a specific question about a patient."""
    field: str
    query: str
    response_convertion_strategy: str
    family_id: Optional[str]
    patient_id: Optional[str]
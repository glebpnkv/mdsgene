import dataclasses

@dataclasses.dataclass
class MappingItem:
    """Data class to hold mapping information."""
    field: str
    question: str
    mapped_excel_column: str
    response_convertion_strategy: str

# The Question class from RagQueryEngine.java is also simple,
# let's define it here or within rag_query_engine.py
@dataclasses.dataclass
class QuestionInfo:
    """Data class to hold generated question details for a patient."""
    field: str
    query: str
    response_convertion_strategy: str
    family_id: str | None
    patient_id: str | None
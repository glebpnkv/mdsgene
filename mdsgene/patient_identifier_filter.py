from typing import Dict, List, Optional, Union
import re


def filter_patient_identifiers_in_text(
    patient_identifiers: List[Union[str, Dict[str, Optional[str]]]],
    document_text: str,
) -> List[Union[str, Dict[str, Optional[str]]]]:
    """Return identifiers that appear in the given text using word boundaries."""
    if not patient_identifiers or not document_text:
        return []

    def match(term: Optional[str]) -> bool:
        return bool(term) and re.search(rf"\b{re.escape(term)}\b", document_text)

    filtered: List[Union[str, Dict[str, Optional[str]]]] = []

    for identifier in patient_identifiers:
        if isinstance(identifier, str):
            if match(identifier):
                filtered.append(identifier)
            continue

        if isinstance(identifier, dict):
            patient = identifier.get("patient")

            if match(patient):
                filtered.append(identifier)

    return filtered

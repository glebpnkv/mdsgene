# AI Agents

This directory contains a set of agents for processing PDF documents using the Gemini API. The agents are built using LangGraph and share common functionality through a base agent class.

## Agent Structure

The agents are organized as follows:

- `base_agent.py`: A base class that provides common functionality for all agents, including:
  - Initializing the GeminiProcessor
  - Managing state
  - Handling caching
  - Building and running LangGraph workflows
  - Printing results

- `publication_details_agent.py`: An agent for extracting publication details from PDFs, including:
  - Extracting title, author, and year
  - Getting PMID using the PmidExtractor
  - Saving PDFs with PMID as filename

- `patient_identifiers_agent.py`: An agent for extracting patient identifiers from PDFs, including:
  - Extracting patient and family identifiers
  - Caching the results for use by other agents

- `questions_processing_agent.py`: An agent for processing questions mapping data from PDFs, including:
  - Loading mapping data from JSON
  - Initializing a vector store for efficient querying
  - Getting patient identifiers from cache
  - Generating questions for each patient
  - Processing questions to get answers

## Usage

Each agent can be run directly from the command line:

```bash
python -m ai.agents.publication_details_agent path/to/pdf
python -m ai.agents.patient_identifiers_agent path/to/pdf
python -m ai.agents.questions_processing_agent path/to/pdf
```

Alternatively, you can import and use the agents in your own code:

```python
from ai.agents.publication_details_agent import PublicationDetailsAgent
from ai.agents.patient_identifiers_agent import PatientIdentifiersAgent
from ai.agents.questions_processing_agent import QuestionsProcessingAgent

# Initialize and set up the agent
agent = PublicationDetailsAgent()
agent.setup()

# Initialize the state
initial_state = {
    "pdf_filepath": "path/to/pdf",
    "publication_details": None,
    "pmid": None,
    "messages": [
        {"role": "user", "content": "Extract publication details and PMID from pdf_name.pdf"}
    ]
}

# Run the agent
final_state = agent.run(initial_state)

# Display the results
agent.print_results(final_state)
```

## Dependencies

The agents depend on the following modules:

- `ai.gemini_processor`: For interacting with the Gemini API
- `ai.pmid_extractor`: For extracting PMIDs from publication details
- `ai.pdf_text_extractor`: For extracting text from PDFs
- `ai.document_processor`: For processing documents and creating vector stores
- `ai.mapping_item`: For defining mapping items and question info

## Workflow

The typical workflow for using these agents is:

1. Use the `publication_details_agent` to extract publication details and get a PMID
2. Use the `patient_identifiers_agent` to extract patient identifiers and cache them
3. Use the `questions_processing_agent` to process questions for each patient

The `questions_processing_agent` requires that patient identifiers have been cached by the `patient_identifiers_agent` before it can be run.
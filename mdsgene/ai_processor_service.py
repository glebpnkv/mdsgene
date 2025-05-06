# ai_processor_service.py
import os
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from ai.internal.gemini_processor_logic import GeminiProcessorLogic
from ai.internal.gemini_text_processor import GeminiTextProcessor

# Initialize FastAPI app
app = FastAPI(
    title="AI Processor Service",
    description="Service for processing LLM requests using Gemini",
    version="1.0.0"
)

# Models for request/response
class QuestionRequest(BaseModel):
    pdf_filepath: str
    question: str
    processor_name: str = "gemini"  # Default to gemini

class FormatRequest(BaseModel):
    raw_answer: str
    strategy: str
    processor_name: str = "gemini"  # Default to gemini

class PatientIdentifiersRequest(BaseModel):
    pdf_filepath: str
    processor_name: str = "gemini"  # Default to gemini

class PublicationDetailsRequest(BaseModel):
    pdf_filepath: str
    processor_name: str = "gemini"  # Default to gemini

# Processor registry
PROCESSORS = {
    "gemini": GeminiProcessorLogic
}

# Helper function to initialize processor
def get_processor(processor_name: str, pdf_filepath: str):
    """
    Initialize and return the requested processor.

    Args:
        processor_name: Name of the processor to use
        pdf_filepath: Path to the PDF file

    Returns:
        Initialized processor instance

    Raises:
        HTTPException: If processor not found or initialization fails
    """
    if processor_name not in PROCESSORS:
        raise HTTPException(status_code=404, detail=f"Processor '{processor_name}' not found")

    try:
        # Convert to Path object
        pdf_path = Path(pdf_filepath)

        # Check if file exists
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found at {pdf_filepath}")

        # Initialize processor
        processor_class = PROCESSORS[processor_name]
        processor = processor_class(pdf_filepath=pdf_path)
        return processor
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize processor: {str(e)}")

# Endpoints
@app.post("/answer_question", response_model=Dict[str, Any])
async def answer_question(request: QuestionRequest):
    """
    Answer a question based on the content of a PDF file.

    Args:
        request: QuestionRequest object containing pdf_filepath, question, and processor_name

    Returns:
        Dictionary with answer and context
    """
    try:
        processor = get_processor(request.processor_name, request.pdf_filepath)
        result = processor.answer_question(request.question)

        if result is None:
            return {"answer": None, "context": None, "success": False, "error": "Failed to get answer"}

        answer, context = result
        return {"answer": answer, "context": context, "success": True, "error": None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/format_answer", response_model=Dict[str, Any])
async def format_answer(request: FormatRequest):
    """
    Format a raw answer according to a specific strategy using GeminiTextProcessor.

    Args:
        request: FormatRequest object containing raw_answer, strategy, and processor_name

    Returns:
        Dictionary with formatted answer and context
    """
    try:
        # Decide which text processor to use based on request.processor_name
        # For now, we only have GeminiTextProcessor
        if request.processor_name == "gemini":
            # Initialize the text processor (no PDF needed)
            # It will read API key from environment internally
            text_processor = GeminiTextProcessor()
        else:
             raise HTTPException(status_code=404, detail=f"Text processor '{request.processor_name}' not found for formatting")

        # Call the format_answer method on the text processor instance
        result = text_processor.format_answer(request.raw_answer, request.strategy)

        if result is None:
             # This might happen if the Gemini request within format_answer fails
             return {"formatted_answer": None, "context": None, "success": False, "error": "Failed to format answer via AI Processor"}

        formatted_answer, context = result
        # Ensure context is serializable (it should be the raw_answer string here)
        if not isinstance(context, (str, type(None))):
             context = str(context)

        return {"formatted_answer": formatted_answer, "context": context, "success": True, "error": None}
    except ValueError as e: # Catch init errors like missing API key
        raise HTTPException(status_code=500, detail=f"Failed to initialize text processor: {str(e)}")
    except HTTPException:
        # Re-raise HTTPExceptions (e.g., 404 for processor not found)
        raise
    except Exception as e:
        # Catch any other unexpected errors during formatting
        import traceback
        traceback.print_exc() # Print traceback for debugging
        raise HTTPException(status_code=500, detail=f"Error formatting answer: {str(e)}")

@app.post("/get_patient_identifiers", response_model=Dict[str, Any])
async def get_patient_identifiers(request: PatientIdentifiersRequest):
    """
    Extract patient identifiers from a PDF file.

    Args:
        request: PatientIdentifiersRequest object containing pdf_filepath and processor_name

    Returns:
        Dictionary with list of patient identifiers
    """
    try:
        processor = get_processor(request.processor_name, request.pdf_filepath)
        patient_identifiers = processor.get_patient_identifiers()

        return {
            "patient_identifiers": patient_identifiers,
            "success": True,
            "error": None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting patient identifiers: {str(e)}")

@app.post("/extract_publication_details", response_model=Dict[str, Any])
async def extract_publication_details(request: PublicationDetailsRequest):
    """
    Extract publication details from a PDF file.

    Args:
        request: PublicationDetailsRequest object containing pdf_filepath and processor_name

    Returns:
        Dictionary with publication details
    """
    try:
        processor = get_processor(request.processor_name, request.pdf_filepath)
        publication_details = processor.extract_publication_details()

        return {
            "publication_details": publication_details,
            "success": True,
            "error": None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting publication details: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai_processor_service"}

# Run the service if executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AI_PROCESSOR_SERVICE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)

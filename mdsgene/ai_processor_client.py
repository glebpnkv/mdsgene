# ai_processor_client.py
import os
import requests
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path

class AIProcessorClient:
    """Client for interacting with the AI Processor Service."""
    
    def __init__(self, service_url: Optional[str] = None):
        """
        Initialize the AI Processor Client.
        
        Args:
            service_url: URL of the AI Processor Service. If None, uses the AI_PROCESSOR_SERVICE_URL
                         environment variable or defaults to http://localhost:8001
        """
        self.service_url = service_url or os.getenv("AI_PROCESSOR_SERVICE_URL", "http://localhost:8001")
        
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the AI Processor Service.
        
        Args:
            endpoint: API endpoint to call
            data: Request data
            
        Returns:
            Response data
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.service_url}/{endpoint}"
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json()
        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, etc.
            raise Exception(f"Error connecting to AI Processor Service: {str(e)}")
        except Exception as e:
            # Handle other errors
            raise Exception(f"Error in AI Processor Service request: {str(e)}")
    
    def answer_question(self, pdf_filepath: str, question: str, processor_name: str = "gemini") -> Optional[Tuple[str, str]]:
        """
        Answer a question based on the content of a PDF file.
        
        Args:
            pdf_filepath: Path to the PDF file
            question: Question to answer
            processor_name: Name of the processor to use
            
        Returns:
            Tuple of (answer, context) if successful, None otherwise
        """
        data = {
            "pdf_filepath": str(pdf_filepath),
            "question": question,
            "processor_name": processor_name
        }
        
        try:
            response = self._make_request("answer_question", data)
            
            if response.get("success", False):
                return response.get("answer"), response.get("context")
            else:
                print(f"Error answering question: {response.get('error')}")
                return None
        except Exception as e:
            print(f"Failed to answer question: {str(e)}")
            return None
    
    def format_answer(self, raw_answer: str, strategy: str, processor_name: str = "gemini") -> Optional[Tuple[str, str]]:
        """
        Format a raw answer according to a specific strategy.
        
        Args:
            raw_answer: Raw answer to format
            strategy: Formatting strategy
            processor_name: Name of the processor to use
            
        Returns:
            Tuple of (formatted_answer, context) if successful, None otherwise
        """
        data = {
            "raw_answer": raw_answer,
            "strategy": strategy,
            "processor_name": processor_name
        }
        
        try:
            response = self._make_request("format_answer", data)
            
            if response.get("success", False):
                return response.get("formatted_answer"), response.get("context")
            else:
                print(f"Error formatting answer: {response.get('error')}")
                return None
        except Exception as e:
            print(f"Failed to format answer: {str(e)}")
            return None
    
    def get_patient_identifiers(self, pdf_filepath: str, processor_name: str = "gemini") -> List[Dict[str, Optional[str]]]:
        """
        Extract patient identifiers from a PDF file.
        
        Args:
            pdf_filepath: Path to the PDF file
            processor_name: Name of the processor to use
            
        Returns:
            List of patient identifiers (each a dict with 'patient' and 'family' keys)
        """
        data = {
            "pdf_filepath": str(pdf_filepath),
            "processor_name": processor_name
        }
        
        try:
            response = self._make_request("get_patient_identifiers", data)
            
            if response.get("success", False):
                return response.get("patient_identifiers", [])
            else:
                print(f"Error getting patient identifiers: {response.get('error')}")
                return []
        except Exception as e:
            print(f"Failed to get patient identifiers: {str(e)}")
            return []
    
    def extract_publication_details(self, pdf_filepath: str, processor_name: str = "gemini") -> Dict[str, Optional[str]]:
        """
        Extract publication details from a PDF file.
        
        Args:
            pdf_filepath: Path to the PDF file
            processor_name: Name of the processor to use
            
        Returns:
            Dictionary with publication details
        """
        data = {
            "pdf_filepath": str(pdf_filepath),
            "processor_name": processor_name
        }
        
        try:
            response = self._make_request("extract_publication_details", data)
            
            if response.get("success", False):
                return response.get("publication_details", {})
            else:
                print(f"Error extracting publication details: {response.get('error')}")
                return {}
        except Exception as e:
            print(f"Failed to extract publication details: {str(e)}")
            return {}
    
    def check_health(self) -> bool:
        """
        Check if the AI Processor Service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        url = f"{self.service_url}/health"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get("status") == "healthy"
        except Exception:
            return False
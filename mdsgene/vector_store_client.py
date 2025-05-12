# vector_store_client.py
import requests
from pathlib import Path
from typing import Dict, Optional, Any, List

class VectorStoreClient:
    """Client for interacting with the Vector Store Service."""

    def __init__(self, service_url: str = "http://localhost:8002"):
        """
        Initialize the Vector Store Client.

        Args:
            service_url: URL of the Vector Store Service
        """
        self.service_url = service_url

    def create_vector_store(self, storage_path: str) -> Dict[str, Any]:
        """
        Create an empty vector store at the specified path.

        Args:
            storage_path: Path where the vector store will be created

        Returns:
            Response from the service
        """
        try:
            response = requests.post(
                f"{self.service_url}/create_vector_store",
                params={"storage_path": storage_path}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error creating vector store: {e}")
            return {"error": str(e)}

    def process_document(self, text: str, source_filename: str, storage_path: str) -> Dict[str, Any]:
        """
        Process a document: split, embed, and store it in the vector store.

        Args:
            text: The document text
            source_filename: Name of the source file
            storage_path: Path to the vector store

        Returns:
            Response from the service
        """
        try:
            response = requests.post(
                f"{self.service_url}/process_document",
                json={
                    "text": text,
                    "source_filename": source_filename,
                    "storage_path": storage_path
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error processing document: {e}")
            return {"error": str(e)}

    def search_document_content(self, question: str, document_name: str) -> Optional[str]:
        """
        Search for content in a specific document.

        Args:
            question: The question to search for
            document_name: The name of the document to search in

        Returns:
            The search results or None if an error occurred
        """
        try:
            response = requests.post(
                f"{self.service_url}/search",
                json={
                    "question": question,
                    "document_name": document_name
                }
            )
            response.raise_for_status()
            return response.json().get("answer")
        except requests.exceptions.RequestException as e:
            print(f"Error searching document content: {e}")
            return None

    def search_document_content_with_path(self, question: str, document_name: str, storage_path: str, k: int = 3) -> Optional[List[Dict[str, Any]]]:
        """
        Search for content in a specific document with a specified storage path.

        Args:
            question: The question to search for
            document_name: The name of the document to search in
            storage_path: Path to the vector store
            k: Number of top results to return (default is 3)

        Returns:
            The list of search results, each with content and metadata, or None if an error occurred
        """
        try:
            response = requests.post(
                f"{self.service_url}/search_with_path",
                params={
                    "question": question,
                    "document_name": document_name,
                    "storage_path": storage_path,
                    "k": k
                }
            )
            response.raise_for_status()
            return response.json().get("answers", [])
        except requests.exceptions.RequestException as e:
            print(f"Error searching document content: {e}")
            return None

    def delete_document_from_store(self, document_name: str, storage_path: str) -> Dict[str, Any]:
        """
        Delete a document from the vector store.

        Args:
            document_name: The name of the document to delete
            storage_path: Path to the vector store

        Returns:
            Response from the service
        """
        try:
            response = requests.delete(
                f"{self.service_url}/delete_document_from_store",
                params={
                    "document_name": document_name,
                    "storage_path": storage_path
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error deleting document from store: {e}")
            return {"error": str(e)}

from pathlib import Path

from ai.document_processor import DocumentProcessor

class DataQuery:
    def __init__(self, storage_path: str, document_path: Path):
        self.document_processor = DocumentProcessor(storage_path)
        self.document_processor.process_document(str(document_path), source_filename=document_path.name)

    def search_content(self, question: str, document_name: str):
        return self.document_processor.search_document_content(question, document_name)

    def answer_query(self, context: str, question: str):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama.llms import OllamaLLM

        model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.7)
        template = """
        You are a medical expert. Answer the question based on the provided context.
        Context: {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return chain.invoke({"context": context, "question": question})
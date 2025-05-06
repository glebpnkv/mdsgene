from pathlib import Path
from ai.document_processor import DocumentProcessor
from ai.pdf_text_extractor import PdfTextExtractor

class DataUploader:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.document_processor = DocumentProcessor(storage_path)

    def load_document(self, file_path: str, filename: str):
        # Экстракт текста из файла
        extractor = PdfTextExtractor()
        text = extractor.extract_text(Path(file_path) / filename)
        # Обработка и сохранение документа
        self.document_processor.process_document(str(text), filename)

    def create_index(self):
        # Можно добавить метод для создания или обновления индекса
        pass
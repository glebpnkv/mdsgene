import json
import logging
from pathlib import Path

# Get a logger for this module
logger = logging.getLogger(__name__)


class PdfUriCacheManager:
    """Simple file-based cache mapping PDF filenames to Gemini URIs."""

    def __init__(self, cache_path: Path = Path("cache/pdf_uri_cache.json")):
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache = {}
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        except Exception:
            self.cache = {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[PdfUriCacheManager] Failed to save cache: {e}")

    def get_uri(self, filename: str) -> str | None:
        return self.cache.get(filename)

    def save_uri(self, filename: str, uri: str) -> None:
        if self.cache.get(filename) == uri:
            return
        self.cache[filename] = uri
        self._save_cache()

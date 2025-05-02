# cache_manager.py
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Union

class CacheManager:
    """Manages a simple file-based JSON cache for query results."""

    def __init__(self, source_filepath: Union[str, Path]):
        """
        Initializes the CacheManager for a given source file.
        The cache file will be named based on the source file.
        Args:
            source_filepath: Path to the original file (e.g., PDF) being processed.
        """
        self.source_path = Path(source_filepath)
        self.cache_filepath = self.source_path.with_suffix('.cache.json')
        self.cache_data: Dict[str, str] = {}
        self._cache_loaded = False
        self._cache_updated = False
        print(f"[CacheManager] Initialized for {self.source_path.name}. Cache file: {self.cache_filepath}")

    def _load_cache(self):
        """Loads cache data from the JSON file if it exists."""
        if self._cache_loaded:
            return

        try:
            if self.cache_filepath.exists():
                with open(self.cache_filepath, 'r', encoding='utf-8') as f:
                    self.cache_data = json.load(f)
                print(f"[CacheManager] Loaded {len(self.cache_data)} items from {self.cache_filepath.name}")
            else:
                print(f"[CacheManager] Cache file not found: {self.cache_filepath.name}. Will create a new one.")
                self.cache_data = {}
        except json.JSONDecodeError:
            print(f"[CacheManager] WARNING: Cache file {self.cache_filepath.name} is corrupted. Starting empty.", file=sys.stderr)
            self.cache_data = {}
        except Exception as e:
            print(f"[CacheManager] WARNING: Could not load cache file {self.cache_filepath.name}: {e}. Starting empty.", file=sys.stderr)
            self.cache_data = {}
        self._cache_loaded = True

    def get(self, key: str) -> Optional[str]:
        """
        Retrieves an item from the cache. Loads cache on first call if needed.
        Args:
            key: The key (e.g., query string) to look up.
        Returns:
            The cached value (string) if found, otherwise None.
        """
        if not self._cache_loaded:
            self._load_cache() # Загружаем кэш при первом запросе get/put

        return self.cache_data.get(key) # Возвращает None, если ключ не найден

    def put(self, key: str, value: str):
        """
        Adds or updates an item in the cache (in memory).
        Args:
            key: The key (e.g., query string).
            value: The value (e.g., raw answer) to store.
        """
        if not self._cache_loaded:
             self._load_cache() # Убедимся, что кэш загружен перед добавлением

        if self.cache_data.get(key) != value: # Обновляем только если значение изменилось или новое
            self.cache_data[key] = value
            self._cache_updated = True
            # print(f"[CacheManager] Updated cache in memory for key: '{key[:50]}...'") # Для отладки

    def save_cache(self):
        """Saves the cache data back to the JSON file if it has been updated."""
        if not self._cache_updated:
            print("[CacheManager] No updates detected, skipping cache save.")
            return

        print(f"[CacheManager] Saving {len(self.cache_data)} items to {self.cache_filepath.name}...")
        try:
            with open(self.cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=4)
            print("[CacheManager] Cache saved successfully.")
            self._cache_updated = False # Сбросить флаг после успешного сохранения
        except Exception as e:
            print(f"[CacheManager] ERROR: Failed to save cache file {self.cache_filepath.name}: {e}", file=sys.stderr)

    def clear_cache(self):
        """Deletes the cache file and clears the in-memory cache."""
        try:
            if self.cache_filepath.exists():
                self.cache_filepath.unlink()
                print(f"[CacheManager] Deleted cache file: {self.cache_filepath.name}")
            else:
                print(f"[CacheManager] Cache file not found, nothing to delete: {self.cache_filepath.name}")
        except OSError as e:
             print(f"[CacheManager] ERROR: Could not delete cache file {self.cache_filepath.name}: {e}", file=sys.stderr)
        self.cache_data = {}
        self._cache_loaded = True # Считаем "загруженным" (пустым)
        self._cache_updated = False
        print("[CacheManager] In-memory cache cleared.")
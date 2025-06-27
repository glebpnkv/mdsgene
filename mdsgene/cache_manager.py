# cache_manager.py
import json
import logging
from pathlib import Path

# Get a logger for this module
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages a simple file-based JSON cache for query results."""

    def __init__(self, source_filepath: str | Path):
        """
        Initializes the CacheManager for a given source file.
        The cache file will be named based on the source file.
        Args:
            source_filepath: Path to the original file (e.g., PDF) being processed.
        """
        self.source_path = Path(source_filepath)
        self.cache_filepath = self.source_path.with_suffix('.cache.json')
        self.cache_data: dict[str, str] = {}
        self._cache_loaded = False
        self._cache_updated = False
        logger.info(f"Initialized for {self.source_path.name}. Cache file: {self.cache_filepath}")

    def _load_cache(self):
        """Loads cache data from the JSON file if it exists."""
        if self._cache_loaded:
            return

        try:
            if self.cache_filepath.exists():
                with open(self.cache_filepath, 'r', encoding='utf-8') as f:
                    self.cache_data = json.load(f)
                logger.info(f"Loaded {len(self.cache_data)} items from {self.cache_filepath.name}")
            else:
                logger.info(f"Cache file not found: {self.cache_filepath.name}. Will create a new one.")
                self.cache_data = {}
        except json.JSONDecodeError:
            logger.warning(f"Cache file {self.cache_filepath.name} is corrupted. Starting empty.")
            self.cache_data = {}
        except Exception as e:
            logger.warning(f"Could not load cache file {self.cache_filepath.name}: {e}. Starting empty.")
            self.cache_data = {}
        self._cache_loaded = True

    def get(self, key: str) -> str | None:
        """
        Retrieves an item from the cache. Loads cache on the first call if needed.
        Args:
            key: The key (e.g., query string) to look up.
        Returns:
            The cached value (string) if found, otherwise None.
        """
        if not self._cache_loaded:
            self._load_cache()  # Load cache on the first get/put request

        return self.cache_data.get(key)  # Returns None if the key is not found

    def put(self, key: str, value: str):
        """
        Adds or updates an item in the cache (in memory).
        Args:
            key: The key (e.g., query string).
            value: The value (e.g., raw answer) to store.
        """
        if not self._cache_loaded:
             self._load_cache()  # Make sure the cache is loaded before adding

        if self.cache_data.get(key) != value:  # Update only if the value has changed or is new
            self.cache_data[key] = value
            self._cache_updated = True
            # logger.debug(f"Updated cache in memory for key: '{key[:50]}...'")  # For debugging

    def save_cache(self):
        """Saves the cache data back to the JSON file if it has been updated."""
        if not self._cache_updated:
            logger.info("No updates detected, skipping cache save.")
            return

        logger.info(f"Saving {len(self.cache_data)} items to {self.cache_filepath.name}...")
        try:
            with open(self.cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=4)
            logger.info("Cache saved successfully.")
            self._cache_updated = False  # Reset flag after successful save
        except Exception as e:
            logger.error(f"Failed to save cache file {self.cache_filepath.name}: {e}")

    def clear_cache(self):
        """Deletes the cache file and clears the in-memory cache."""
        try:
            if self.cache_filepath.exists():
                self.cache_filepath.unlink()
                logger.info(f"Deleted cache file: {self.cache_filepath.name}")
            else:
                logger.info(f"Cache file not found, nothing to delete: {self.cache_filepath.name}")
        except OSError as e:
             logger.error(f"Could not delete cache file {self.cache_filepath.name}: {e}")
        self.cache_data = {}
        self._cache_loaded = True  # Consider as "loaded" (empty)
        self._cache_updated = False
        logger.info("In-memory cache cleared.")

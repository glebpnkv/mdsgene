from pathlib import Path

from mdsgene.pdf_uri_cache_manager import PdfUriCacheManager


def resolve_pdf_uri(pdf_filepath: Path) -> str | None:
    """Return cached PDF URI for the given file if available."""
    cache = PdfUriCacheManager()
    return cache.get_uri(pdf_filepath.name)


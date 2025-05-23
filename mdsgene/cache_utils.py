# cache_utils.py
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Any, Optional

def delete_pmid_cache(pmid: str, cache_root: Union[str, Path]):
    """
    Delete the entire cache folder associated with a specific PMID.

    Args:
        pmid: The PMID whose cache should be deleted.
        cache_root: The root directory where PMID caches are stored.
    """
    pmid_cache_path = Path(cache_root) / pmid
    try:
        if pmid_cache_path.exists():
            shutil.rmtree(pmid_cache_path)
            print(f"[CacheManager] Cache folder deleted: {pmid_cache_path}")
        else:
            print(f"[CacheManager] Cache folder not found: {pmid_cache_path}")
    except Exception as e:
        print(f"[CacheManager] Error deleting cache folder: {e}")

def remove_document_from_pmid_cache(pdf_filename: str, pmid_cache_path: Union[str, Path]):
    """
    Remove document entry from pmid_cache.json.

    Args:
        pdf_filename: Name of the PDF file whose cache entry should be deleted.
        pmid_cache_path: Path to the pmid_cache.json file.
    """
    pmid_cache_path = Path(pmid_cache_path)
    if pmid_cache_path.exists():
        try:
            with open(pmid_cache_path, 'r', encoding='utf-8') as f:
                pmid_cache = json.load(f)

            if pdf_filename in pmid_cache:
                removed = pmid_cache.pop(pdf_filename)
                print(f"[PMID Cache] Removed entry for {pdf_filename}: {removed}")

                with open(pmid_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(pmid_cache, f, indent=2, ensure_ascii=False)
            else:
                print(f"[PMID Cache] No entry found for {pdf_filename}")
        except Exception as e:
            print(f"[PMID Cache] Error updating pmid_cache.json: {e}")
    else:
        print(f"[PMID Cache] File not found: {pmid_cache_path}")

def delete_document_and_all_related_data(
    pdf_filename: str, 
    pmid: str, 
    storage_path: str,
    cache_root: Union[str, Path],
    pmid_cache_path: Union[str, Path],
    vector_store_client
):
    """
    Delete all cache data, pmid entry, and vector store entries related to a document.

    Args:
        pdf_filename: PDF filename associated with the document.
        pmid: PMID associated with the document.
        storage_path: Path to the vector store.
        cache_root: Root directory of the PMID-specific caches.
        pmid_cache_path: Path to pmid_cache.json file.
        vector_store_client: Instance of VectorStoreClient.
    """
    # Delete PMID cache folder
    delete_pmid_cache(pmid, cache_root)

    # Remove entry from pmid_cache.json
    remove_document_from_pmid_cache(pdf_filename, pmid_cache_path)

    # Delete document from vector store
    response = vector_store_client.delete_document_from_store(pdf_filename, storage_path)
    if "error" in response:
        print(f"[Cleanup] Error deleting document from vector store: {response['error']}")
    else:
        print(f"[Cleanup] {response['message']}")


def load_formatted_result(
    pmid: str, expected_patient_ids: Optional[list[str]] = None
) -> Optional[Dict[str, Any]]:
    """Load formatted patient results for a specific PMID if present.

    Args:
        pmid: Document PMID.
        expected_patient_ids: Optional list of patient IDs expected in the
            results. Used to log cache coverage.

    Returns:
        Cached formatted result dictionary or ``None`` if not available or
        incomplete.
    """
    path = Path("cache") / pmid / "formatted_answer_cache.json"
    print(f"[Cache] Looking for cache at: {path}")
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                formatted = data.get("formatted", {})
                if formatted.get("completed") and "result" in formatted:
                    cached_result = formatted["result"]
                    if expected_patient_ids:
                        missing = [pid for pid in expected_patient_ids if pid not in cached_result]
                        if missing:
                            print(
                                f"[Cache] Partial cache hit. Missing {len(missing)} of {len(expected_patient_ids)} patients."
                            )
                        else:
                            print("[Cache] Full cache hit.")
                    return cached_result
                else:
                    print("[Cache] Cache found but not marked as completed or missing result.")
        except Exception as e:
            print(f"[Cache] ERROR reading formatted cache: {e}")
    else:
        print("[Cache] No cache file found.")
    return None


def save_formatted_result(
    pmid: str,
    prompt: str,
    raw_answer: str,
    strategy: str,
    formatted_result: Dict[str, Any],
) -> None:
    """Save raw and formatted patient results for a PMID.

    This appends new patient results to any existing cache for the PMID.
    """
    path = Path("cache") / pmid / "formatted_answer_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    existing = data.get("formatted", {}).get("result", {})
    existing.update(formatted_result)

    data["raw"] = {
        "question_prompt": prompt,
        "answer": raw_answer,
        "timestamp": datetime.utcnow().isoformat(),
        "completed": True,
    }
    data["formatted"] = {
        "strategy": strategy,
        "result": existing,
        "timestamp": datetime.utcnow().isoformat(),
        "completed": True,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

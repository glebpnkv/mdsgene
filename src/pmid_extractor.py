# pmid_extractor.py
import requests
import urllib.parse

class PmidExtractor:
    """
    Extracts PMID from PubMed using title, author, and year via NCBI's ESearch API.
    """

    @staticmethod
    def get_pmid(title: str, author: str = "", year: str = "") -> str:
        """
        Queries PubMed for a PMID using article title, author, and year.

        :param title: The full article title.
        :param author: First author's last name (optional).
        :param year: Publication year (optional).
        :return: PMID as string or None if not found.
        """
        if not title:
            print("Title is required to search PubMed.")
            return None

        # Construct search query
        query_parts = [f'"{title}"']
        if author:
            query_parts.append(author)
        if year:
            query_parts.append(year)

        search_query = " ".join(query_parts)
        print(f"Prepared PubMed search query: {search_query}")

        # Encode and prepare API URL
        encoded_query = urllib.parse.quote(search_query)
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={encoded_query}&retmode=json&retmax=1"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()

            pmids = json_data.get("esearchresult", {}).get("idlist", [])
            return pmids[0] if pmids else None

        except Exception as e:
            print(f"Error querying PubMed: {e}")
            return None


# ✅ Example usage
if __name__ == "__main__":
    title = "Mutations in VPS13D lead to a new recessive ataxia with spasticity and mitochondrial defects"
    author = "Gauthier"
    year = "2018"

    pmid = PmidExtractor.get_pmid(title, author, year)
    if pmid:
        print(f"✅ Found PMID: {pmid}")
    else:
        print("❌ PMID not found.")

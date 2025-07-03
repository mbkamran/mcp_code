import os
import sys
import json
import arxiv
from typing import List, Optional
from mcp.server.fastmcp import FastMCP

INFO_DIR = "papers_info"
DOWNLOAD_DIR = "papers"
PAPER_INFO_JSON = "papers_info.json"

mcp = FastMCP("find_research")

@mcp.tool()
def search_papers(search_term: str, number: int):
    """
    Searches and downloads relevant papers against a search term from ArXiv

    Args:
        search_term (str): search term to find papers against
        number (int): Number of papers to output

    Returns:
        List of paper IDs found in the term
    """ 

    client = arxiv.Client()
    search = arxiv.Search(
        query=search_term,
        max_results=number,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = client.results(search)

    try:
        with open(os.path.join(INFO_DIR, PAPER_INFO_JSON), "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for result in list(results):
        paper_ids.append(result.get_short_id())

        if result.get_short_id() not in papers_info:
            result.download_pdf(dirpath=DOWNLOAD_DIR, filename=f"{result.title}.pdf")

            paper_info = {
                "title": result.title,
                "short_id": result.get_short_id(),
                "published": result.published,
                "authors": [author.name for author in result.authors]

            }

            papers_info[result.get_short_id()] = paper_info

    with open(os.path.join(INFO_DIR, PAPER_INFO_JSON), "w") as json_file:
        json.dump(papers_info, json_file, indent=2, default=str)

    return paper_ids

@mcp.tool()
def extract_info(paper_id: str) -> Optional[str]:
    """
    Extract information for a research paper by id

    Args:
        id (str): id of the research paper

    Returns:
        Information of the reserch paper 
    """

    try:
        with open(os.path.join(INFO_DIR, PAPER_INFO_JSON), "r") as json_file:
            papers_info = json.load(json_file)

        return json.dumps(papers_info[paper_id], indent=2)
    except Exception as e:
        print("Problem: ", str(e))

if __name__ == "__main__":
    mcp.run(transport='stdio')
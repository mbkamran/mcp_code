"""
This is 'Server' script responsible for providing context to the client

This server script provides prompt and tool context to the client
"""

import os
import sys
import json
import arxiv
from typing import List, Optional
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from langchain_core.messages import SystemMessage, HumanMessage

INFO_DIR = "papers_info"
DOWNLOAD_DIR = "papers"
PAPER_INFO_JSON = "papers_info.json"

mcp = FastMCP("find_research")

@mcp.prompt()
def query_creater() -> str:
    return """
You are a helpful bot, whose task is to create a natural language search term for searching research paper for user.
If the user mentions description of research paper, understand the description and based on that, output a single sentence search term.
If the user asks to search paper based on id, you must EXPLICITLY tell in natural language form to search on that id, YOU MUST NOT OUTPUT ID ALONE!

For example:
user: Can you help me get papers that are published for knee othroscopy
bot: Knee othroscopy papers

user: Can you find paper with id abc123
bot: Search paper with id abc123
"""

@mcp.prompt()
def answer_creater(result: str) -> str:
    return f"""
You are a helpful bot whose task is give a natural language response to user based on output from the tool call and user question.
The tool output will either be a list of ids for research paper or information about research papers.
In both cases, you just need to rephrase user question and output Tool Call results.
You must not output anything else apart from rephrasing user question and outputing tool call results.


Tool call result:
{result}
"""

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
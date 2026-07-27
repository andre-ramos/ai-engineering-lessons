from pydantic import Field

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocumentMCP", log_level="ERROR")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

@mcp.tool(
    name="read contents",
    description="Use this to read the contents of a document and return a string",
)
def read_document(doc_id: str = Field(description="id of the document")):
    if doc_id not in docs:
        raise ValueError(f"Doc id {doc_id} not found")

    return docs[doc_id]


@mcp.tool(
    name="Edit document",
    description="Use this to edit the contents of a document with a new string",
)
def edit_document(
    doc_id: str = Field(description="id of the document"),
    old_str: str = Field(
        description="The text to replace, must match exactly, even whitespace"
    ),
    new_str: str = Field(
        description="The new text to insert in place of the old text"
    ),
):
    if doc_id not in docs:
        raise ValueError(f"Doc id {doc_id} not found")
    docs[doc_id] = docs[doc_id].replace(old_str, new_str)


# TODO: Write a resource to return all doc id's
# TODO: Write a resource to return the contents of a particular doc
# TODO: Write a prompt to rewrite a doc in markdown format
# TODO: Write a prompt to summarize a doc


if __name__ == "__main__":
    mcp.run(transport="stdio")


# Dealing with RAG Hallucinations: https://www.youtube.com/watch?v=oVI2GA8jn7w

# The idea is to check the generated answer with a LLM
# Query -> Vector Store -> Retrieve Docs -> Check relevant docs -> Generate the answer -> Check Hallucination (Grounded answer or not)



import os
from pathlib import Path
from typing import Literal, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Environment / models
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to your .env file, for example:\n"
        "OPENAI_API_KEY=sk-..."
    )

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
)


# ---------------------------------------------------------------------------
# 1. Load PDF, chunk it, embed it, and create the retriever
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load a PDF into LangChain Documents without Cohere/Llama/Groq."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "title": pdf_path.stem,
                    "page": page_number,
                },
            )
        )

    if not documents:
        raise ValueError(
            f"No extractable text was found in PDF: {pdf_path}"
        )

    return documents


def create_chunks(
    documents: Sequence[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split loaded PDF pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(list(documents))

    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["id"] = f"chunk-{i}"

    return chunks


def create_retriever(
    chunks: Sequence[Document],
    top_k: int = 4,
):
    """
    Create a local vector store using OpenAI embeddings and expose it
    as a LangChain retriever.
    """
    vector_store = InMemoryVectorStore.from_documents(
        documents=list(chunks),
        embedding=embeddings,
    )

    return vector_store.as_retriever(
        search_kwargs={"k": top_k},
    )


def build_retriever_from_pdf(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 4,
):
    """Complete indexing pipeline: PDF -> chunks -> OpenAI embeddings -> retriever."""
    documents = load_pdf(pdf_path)

    chunks = create_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    retriever = create_retriever(
        chunks,
        top_k=top_k,
    )

    return retriever, chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def retrieve_documents(retriever, question: str) -> list[Document]:
    """
    Current LangChain retriever API.

    Do NOT use:
        retriever.get_relevant_documents(question)

    Use:
        retriever.invoke(question)
    """
    return retriever.invoke(question)


def format_docs(docs: Sequence[Document]) -> str:
    """Format retrieved documents for the prompts."""
    formatted_docs: list[str] = []

    for i, doc in enumerate(docs, start=1):
        doc_id = str(doc.metadata.get("id", f"doc-{i}"))
        title = str(doc.metadata.get("title", "Untitled"))
        source = str(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "Unknown")

        formatted_docs.append(
            f'<doc id="{doc_id}">\n'
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}\n"
            f"</doc>"
        )

    return "\n\n".join(formatted_docs)


# ---------------------------------------------------------------------------
# 2. Grade retrieved documents for relevance
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    """Binary relevance score for one retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        description="Whether the retrieved document is relevant to the question."
    )


retrieval_grader_llm = llm.with_structured_output(
    GradeDocuments,
    method="json_schema",
)

retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing whether a retrieved document is relevant
to a user question.

A document is relevant if it contains facts, keywords, or semantic meaning
that may help answer the question.

Return "yes" for relevant documents and "no" for irrelevant documents.
Do not be unnecessarily strict.""",
        ),
        (
            "human",
            """Retrieved document:

<document>
{document}
</document>

User question:
<question>
{question}
</question>""",
        ),
    ]
)

retrieval_grader = retrieval_grader_prompt | retrieval_grader_llm


def filter_relevant_documents(
    question: str,
    docs: Sequence[Document],
    *,
    verbose: bool = False,
) -> list[Document]:
    """Filter retrieved chunks using an OpenAI structured-output grader."""
    docs_to_use: list[Document] = []

    for doc in docs:
        result = retrieval_grader.invoke(
            {
                "question": question,
                "document": doc.page_content,
            }
        )

        if verbose:
            print(doc.page_content)
            print("-" * 50)
            print(result)
            print()

        if result.binary_score == "yes":
            docs_to_use.append(doc)

    return docs_to_use


# ---------------------------------------------------------------------------
# 3. Generate answer
# ---------------------------------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks.

Answer only from the retrieved documents provided to you.
Do not add unsupported facts.

If the context is insufficient, state that the retrieved context does not
contain enough information.

Keep the answer concise, normally three to five sentences unless a short
list is more appropriate.""",
        ),
        (
            "human",
            """Retrieved documents:

<docs>
{documents}
</docs>

User question:
<question>
{question}
</question>""",
        ),
    ]
)

rag_chain = answer_prompt | llm | StrOutputParser()


def generate_answer(
    question: str,
    docs: Sequence[Document],
) -> str:
    """Generate a grounded answer from the filtered documents."""
    if not docs:
        return (
            "The retrieved context does not contain enough information "
            "to answer the question."
        )

    return rag_chain.invoke(
        {
            "documents": format_docs(docs),
            "question": question,
        }
    )


# ---------------------------------------------------------------------------
# 4. Check grounding / hallucinations
# ---------------------------------------------------------------------------

class GradeHallucinations(BaseModel):
    """Whether the generated answer is grounded in the retrieved facts."""

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if every factual claim in the answer is supported by the "
            "retrieved documents; otherwise 'no'."
        )
    )


hallucination_grader_llm = llm.with_structured_output(
    GradeHallucinations,
    method="json_schema",
)

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader checking whether a generated answer is grounded
in retrieved facts.

Return:
- "yes" if every factual claim is supported by the retrieved documents.
- "no" if any factual claim is unsupported, invented, or contradicted.""",
        ),
        (
            "human",
            """Retrieved facts:

<facts>
{documents}
</facts>

Generated answer:
<generation>
{generation}
</generation>""",
        ),
    ]
)

hallucination_grader = hallucination_prompt | hallucination_grader_llm


def grade_grounding(
    docs: Sequence[Document],
    generation: str,
) -> GradeHallucinations:
    """Grade whether the answer is grounded in the supplied documents."""
    if not docs:
        return GradeHallucinations(binary_score="yes")

    return hallucination_grader.invoke(
        {
            "documents": format_docs(docs),
            "generation": generation,
        }
    )


# ---------------------------------------------------------------------------
# 5. Highlight exact source segments used
# ---------------------------------------------------------------------------

class HighlightedDocument(BaseModel):
    """One exact source segment supporting the generated answer."""

    id: str = Field(description="Document/chunk id exactly as supplied.")
    title: str = Field(description="Document title exactly as supplied.")
    source: str = Field(description="Document source exactly as supplied.")
    segment: str = Field(
        description=(
            "Exact verbatim segment copied from the document that supports "
            "the generated answer."
        )
    )


class HighlightDocuments(BaseModel):
    """Supporting documents and exact verbatim source segments."""

    documents: list[HighlightedDocument] = Field(
        description="Only the documents directly used to support the answer."
    )


highlight_llm = llm.with_structured_output(
    HighlightDocuments,
    method="json_schema",
)

highlight_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Identify exact source passages that support the generated answer.

Rules:
- Include only documents that directly support the answer.
- Preserve id, title, and source exactly.
- The segment must be copied verbatim from the document.
- Never paraphrase a segment.
- Never invent text.
- If a document did not support the answer, do not include it.""",
        ),
        (
            "human",
            """Retrieved documents:

<docs>
{documents}
</docs>

User question:
<question>
{question}
</question>

Generated answer:
<answer>
{generation}
</answer>""",
        ),
    ]
)

doc_lookup = highlight_prompt | highlight_llm


def highlight_used_documents(
    question: str,
    docs: Sequence[Document],
    generation: str,
) -> HighlightDocuments:
    """Return exact source segments supporting the answer."""
    if not docs:
        return HighlightDocuments(documents=[])

    return doc_lookup.invoke(
        {
            "documents": format_docs(docs),
            "question": question,
            "generation": generation,
        }
    )


# ---------------------------------------------------------------------------
# 6. Complete RAG evaluation pipeline
# ---------------------------------------------------------------------------

def run_rag_evaluation(
    question: str,
    retriever,
    *,
    verbose: bool = False,
) -> dict:
    """Retrieve -> relevance grade -> answer -> grounding grade -> highlights."""
    docs = retrieve_documents(
        retriever,
        question,
    )

    docs_to_use = filter_relevant_documents(
        question,
        docs,
        verbose=verbose,
    )

    generation = generate_answer(
        question,
        docs_to_use,
    )

    grounding = grade_grounding(
        docs_to_use,
        generation,
    )

    highlights = highlight_used_documents(
        question,
        docs_to_use,
        generation,
    )

    return {
        "retrieved_documents": docs,
        "relevant_documents": docs_to_use,
        "generation": generation,
        "grounded": grounding.binary_score,
        "highlights": highlights,
    }


# ---------------------------------------------------------------------------
# 7. Hardcoded example - NO command-line arguments
# ---------------------------------------------------------------------------

PDF_PATH = "./data/agentic_design_patterns.pdf"
QUESTION = "What are the main agentic design patterns?"

TOP_K = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VERBOSE = True


def main() -> None:
    print("=" * 80)
    print("PDF")
    print("=" * 80)
    print(PDF_PATH)
    print()

    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(QUESTION)
    print()

    chunks_query_retriever, chunks = build_retriever_from_pdf(
        pdf_path=PDF_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=TOP_K,
    )

    print("=" * 80)
    print("INDEXING")
    print("=" * 80)
    print(f"Indexed chunks: {len(chunks)}")
    print()

    result = run_rag_evaluation(
        question=QUESTION,
        retriever=chunks_query_retriever,
        verbose=VERBOSE,
    )

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["generation"])
    print()

    print("=" * 80)
    print("GROUNDED")
    print("=" * 80)
    print(result["grounded"])
    print()

    print("=" * 80)
    print("USED DOCUMENT SEGMENTS")
    print("=" * 80)

    if not result["highlights"].documents:
        print("No supporting document segments were identified.")
    else:
        for item in result["highlights"].documents:
            print(f"ID: {item.id}")
            print(f"Title: {item.title}")
            print(f"Source: {item.source}")
            print(f"Text Segment: {item.segment}")
            print("-" * 80)


if __name__ == "__main__":
    main()

import os
import sys
from dotenv import load_dotenv

# Load environment variables before importing deepeval/langchain
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_with_space,
                              get_langchain_embedding_provider,
                              show_context)

from evaluation.evalute_rag import evaluate_rag
from langchain.vectorstores import FAISS

def encode_pdf(path="data/Understanding_Climate_Change.pdf", chunk_size=500, chunk_overlap=100):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
        A FAISS vector store is an in-memory database built on top of FAISS (Facebook AI Similarity Search) used for fast similarity search and clustering of high-dimensional vectors.
    """
    # Load document
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Splitting documents into chunks, we pass length_function=len to specify how the splitter measures chunk size.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)

    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings with OpenAI
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)

    # Create vector store
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

chunks_vector_store = encode_pdf(chunk_size=500, chunk_overlap=200)
# Create Retriever
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})


# testing retriever
test_query = "What is the main cause of climate change?"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)

# Evaluating rag
print(evaluate_rag(chunks_query_retriever))
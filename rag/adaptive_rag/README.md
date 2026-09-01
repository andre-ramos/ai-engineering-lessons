
# Adaptive RAG — Brief Summary

## Overview

An LLM first classifies the query as Factual, Analytical, Opinion, or Contextual.

It then chooses a different retrieval strategy:

## Factual: improves the query and reranks documents by relevance.

## Analytical: splits the question into subqueries and retrieves broader information.

## Opinion: searches for different perspectives.

## Contextual: reformulates the query using user context.

Relevant chunks are retrieved from FAISS using embeddings, with the LLM helping rank or select the best ones.

Finally, the selected context and original question are sent to GPT to generate the answer.

## Adaptive RAG decides how to retrieve information based on the type of question, instead of using the same retrieval method every time.

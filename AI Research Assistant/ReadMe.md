# AI Research Assistant — Project Progress

## Phase 1: Build the Basic RAG Pipeline

First, we built a basic Retrieval-Augmented Generation (RAG) system that answers questions using uploaded PDF and TXT documents.

**How it works:**

1. **Document loading:** Extract text from uploaded documents.
2. **Chunking:** Split large documents into smaller text chunks so relevant information can be retrieved efficiently.
3. **Embeddings:** Convert each chunk into a numerical vector that represents its meaning.
4. **Vector database:** Store the embeddings and text chunks in ChromaDB.
5. **Semantic retrieval:** Convert a user's question into an embedding and retrieve chunks with similar meanings.
6. **LLM generation:** Send the retrieved context and question to the Groq LLM to generate an answer based on the available documents.
7. **Source references:** Return source filenames, page numbers, and text excerpts to help users identify where the information came from.

**What we learned:** A basic RAG pipeline can answer questions from documents, but its answer quality depends heavily on retrieving the right information first. If retrieval returns irrelevant or incomplete context, the generated answer can also be poor.

## Phase 2: Experiment with Different Retrieval Techniques

Instead of relying only on vector search, we implemented and compared multiple retrieval techniques to understand which methods find the most relevant document passages.

### 1. Vector Search

Uses embeddings to find passages that are semantically similar to the user's question.

- Good at finding information even when the question uses different wording from the document.
- Depends on the quality of embeddings and semantic similarity.

### 2. BM25 Keyword Search

Uses keyword matching and term-frequency-based ranking to find relevant passages.

- Useful when questions contain specific technical terms or keywords.
- Can struggle when the same concept is expressed using different wording.

### 3. Hybrid Search with Reciprocal Rank Fusion (RRF)

Combines the ranked results from vector search and BM25.

- Uses semantic similarity and keyword matching together.
- Reciprocal Rank Fusion (RRF) combines rankings from both retrievers into a single candidate list.
- The purpose is to benefit from the strengths of both retrieval methods.

### 4. Cross-Encoder Reranking

After retrieving and combining candidate passages, we use a cross-encoder reranker to score how relevant each passage is to the original question.

- Examines the question and each candidate passage together.
- Reorders candidates according to their estimated relevance.
- Helps prioritize stronger passages before sending context to the LLM.

We also added deduplication to avoid returning multiple chunks from the same source page when evaluating page-level retrieval.

**What we learned:** Different retrieval methods behave differently. Vector search performed well on our initial test questions, while BM25 missed some expected pages. Combining retrieval methods and reranking improved the overall results on our expanded test dataset.

## Phase 3: Evaluate, Compare, and Improve Retrieval Quality

To measure performance rather than relying on assumptions, we created an evaluation dataset containing ten questions and their expected source pages.

We tested vector search, BM25, and hybrid search using the same questions.

### Evaluation Metrics

**Hit Rate@4**

Measures the percentage of questions for which at least one expected source page appears in the top four retrieved results.

**Mean Reciprocal Rank (MRR@4)**

Measures how highly the first correct source appears in the top four results.

- Correct result at rank 1: score = 1.0
- Correct result at rank 2: score = 0.5
- Correct result at rank 3: score ≈ 0.333
- Correct result at rank 4: score = 0.25
- No correct result in the top four: score = 0

MRR helps distinguish between a system that retrieves the correct page first and one that finds it only after several less relevant results.

### Latest Evaluation Results

| Retrieval Method | Hit Rate@4 | MRR@4 |
|---|---:|---:|
| Vector Search | 80% (8/10) | 0.750 |
| BM25 Search | 80% (8/10) | 0.458 |
| Hybrid Search + Reranking | 100% (10/10) | 0.742 |

*These results are from our current ten-question evaluation dataset and are not a guarantee of performance on other documents or questions.*

### What We Found

- Vector search retrieved the expected source for eight out of ten questions and ranked the first correct result well.
- BM25 also achieved eight hits, but its lower MRR indicates that correct results were generally ranked lower.
- Hybrid search with reranking retrieved the expected source for all ten questions.
- Hybrid search achieved a slightly lower MRR than vector search because some correct pages appeared at ranks 2–4 rather than first.
- Hit Rate and MRR measure retrieval quality; they do not directly measure whether the LLM's final answer is correct, complete, or properly grounded.

### Why We Evaluate Before Making Changes

We follow an experiment-driven improvement process:

1. Build a retrieval technique.
2. Test it on the same evaluation dataset.
3. Compare hit rate and ranking quality.
4. Identify missed pages, irrelevant results, and duplicate passages.
5. Improve the retrieval pipeline.
6. Run the evaluation again to check whether the change helped.

The goal is not simply to achieve the highest score on a small dataset. We want a retrieval pipeline that consistently finds useful evidence across different questions and documents.

## Current Project Status

We have implemented a document-based RAG pipeline, vector search, BM25, hybrid retrieval using RRF, cross-encoder reranking, page-level deduplication, source references, and retrieval evaluation.

**Next planned improvement:** Integrate Tavily web search so the research assistant can retrieve information from selected web sources as well as uploaded documents. We will then work on combining document and web evidence, returning source links, handling conflicting information, and evaluating answer quality and groundedness in addition to retrieval quality.
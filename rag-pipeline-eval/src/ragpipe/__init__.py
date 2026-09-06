"""ragpipe: a hybrid-retrieval RAG pipeline with a RAGAS-style evaluation harness.

Layers (each independently testable):

* ``documents`` / ``textproc`` / ``chunking`` — corpus loading and offset-preserving chunking
* ``bm25`` / ``embeddings`` / ``vectorstore`` / ``retrieval`` — lexical, dense, hybrid and
  reranked retrieval
* ``llm`` / ``prompts`` / ``pipeline`` — grounded generation with citations and abstention
* ``evaluation`` — retrieval metrics, RAGAS metrics (faithfulness, answer relevancy, context
  precision/recall) with an LLM judge, bootstrap confidence intervals, regression gates and an
  optional cross-check against the official ``ragas`` package
"""

__version__ = "0.1.0"

### RAG Case Study

This is a case study on how to improve the accuracy of RAG system by using a combination of different techniques.
The following strategies will be explore:
1. Diferent chunking strategies
2. Retrieval: 
    a. Cosine similarity retrieval
    b. BERT
    c. ColBERT
    d. ColBERT + BM25
    e. ColBERT + BM25 + reranker
3. Better Evals for retrieval and final RAG: One of the main drawbacks in current RAGs is the lack of ground truth. In this case study we will use a combination of different techniques to create a better eval for the retrieval.



## RAG System Experiments

This directory contains two major experiments conducted to improve RAG system accuracy using different techniques. Here's a detailed breakdown of each experiment:

### Experiment 001: Baseline RAG System with Multi-Stage Evaluation

This experiment implements a baseline RAG system with a comprehensive evaluation pipeline. The key components include:

#### Data Processing
- Basic document chunking with fixed-size chunks
- Standard text preprocessing and cleaning
- Document structure preservation through metadata

#### System Components
1. **Retriever System**
   - Implementation: Basic cosine similarity-based retrieval
   - Evaluation metrics tracked in `retriever_evaluations.json`
   - Cross-encoder reranking (results in `cross_encoder_evaluations.json`)

2. **Evaluation Pipeline**
   - Custom evaluation set generation (`eval_set_generator.ipynb`)
   - LLM-based response evaluation (`llm_evaluator.ipynb`)
   - Comprehensive metrics tracking in `evaluation_results.json`

3. **Final RAG System**
   - Implementation in `rag-system-final.ipynb`
   - Response generation and evaluation
   - Results stored in `llm_responses_eval_set.json`

### Experiment 002: Enhanced RAG with Advanced Retrieval and Structured Evaluation

This experiment introduces significant improvements over the baseline with more sophisticated components:

#### Data Processing Enhancements
1. **Advanced Document Preprocessing**
   - Hybrid chunking strategy using ModernBERT tokenizer
   - Context-aware chunk generation
   - Preservation of document hierarchy and structure
   - Short-term and long-term context embedding for each chunk

2. **Document Structure**
   - Table of Contents (ToC) based structuring (`toc_data_*.json` files)
   - Hierarchical metadata preservation
   - Intermediate data storage for analysis

#### System Components
1. **Advanced Retrieval System**
   - ColBERT-based dense retrieval (`colbert_index/`)
   - Implementation in `rag_indexer.py` and `rag_querier.py`
   - Structured evaluation pipeline (`evaluate_retriever_002.py`)
   - Comprehensive retrieval metrics in `retriever_evaluation_results.json`

2. **RAG System Implementation**
   - Modular design with utility functions (`rag_utils.py`)
   - Main system implementation in `rag_system_002.py`
   - Response generation pipeline (`generate_rag_response_002.py`)
   - Evaluation system (`evaluate_rag_responses_002.py`)

3. **Evaluation Framework**
   - Custom evaluation set creation (`create_eval_set_002.py`)
   - Multiple evaluation iterations (v3 results in `llm_responses_eval_set_v3.json`)
   - Final evaluation results in `rag_evaluation_results.json`

### Key Improvements in Experiment 002
1. **Better Context Handling**
   - Implementation of short-term and long-term context for chunks
   - Improved document structure preservation
   - Hierarchical metadata integration

2. **Advanced Retrieval**
   - ColBERT-based dense retrieval replacing basic cosine similarity
   - More sophisticated evaluation metrics
   - Better handling of document structure in retrieval

3. **Modular System Design**
   - Separation of concerns (indexing, querying, evaluation)
   - Better code organization and reusability
   - Improved evaluation pipeline

### Results and Findings
The evaluation results show significant improvements in Experiment 002 over the baseline:
- More accurate chunk retrieval
- Better context preservation in responses
- Improved answer relevance and accuracy
- More robust evaluation framework


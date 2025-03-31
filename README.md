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



### Files:
Code files are commented and different strategies are discussed in each file

\n data-preprocess.ipynb: Contains pipeline for Data ingestion and chunking. 
train_pylate.py: Code to fine tune modernBERT for similarity search
eval_set_generator: Contains code for the evaluation pipeline
rag-system-final: Code for the retriever and final LLM response

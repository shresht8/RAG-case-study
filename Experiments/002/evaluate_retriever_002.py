import json
import argparse
from pathlib import Path
from typing import List, Dict
from ragatouille import RAGPretrainedModel
import logging

class RetrieverEvaluator:
    def __init__(self, index_path: str):
        """Initialize the evaluator with the index path"""
        self.logger = self._setup_logger()
        self.index_path = Path(index_path).resolve()
        self.rag_model = None
        self._load_model()

    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger

    def _load_model(self):
        """Load the RAG model from the index"""
        self.logger.info(f"Loading RAG model from index at {self.index_path}")
        try:
            self.rag_model = RAGPretrainedModel.from_index(str(self.index_path))
            self.logger.info("Successfully loaded RAG model")
        except Exception as e:
            self.logger.error(f"Error loading model from index: {str(e)}")
            raise

    def load_evaluation_set(self, eval_set_path: str) -> List[Dict]:
        """Load the evaluation set from JSON file"""
        try:
            with open(eval_set_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading evaluation set: {str(e)}")
            raise

    def get_chunk_ids_from_results(self, results: List[Dict]) -> List[str]:
        """Extract chunk IDs from retriever results"""
        chunk_ids = []
        for result in results:
            # Assuming the chunk ID is stored in the metadata
            if 'document_id' in result:
                chunk_ids.append(result['document_id'])
        return chunk_ids

    def find_overlapping_chunks(self, ground_truth: List[str], retrieved: List[str]) -> List[str]:
        """Find overlapping chunks between ground truth and retrieved results"""
        return list(set(ground_truth) & set(retrieved))

    def evaluate_question(self, question: str, ground_truth_chunks: List[str], k: int = 20) -> Dict:
        """Evaluate a single question"""
        # Query the retriever
        results = self.rag_model.search(question, k=k)
        
        # Get retrieved chunk IDs
        retrieved_chunks = self.get_chunk_ids_from_results(results)
        
        # Find overlapping chunks
        overlapping = self.find_overlapping_chunks(ground_truth_chunks, retrieved_chunks)
        
        return {
            "question": question,
            "ground_truth": ground_truth_chunks,
            "retrieved_chunks": retrieved_chunks,
            "overlapping_chunks": overlapping,
            "total_overlap": len(overlapping)
        }

    def evaluate_all(self, eval_set: List[Dict], output_path: str, k: int = 20):
        """Evaluate all questions and save results"""
        results = []
        total_questions = len(eval_set)
        
        for i, item in enumerate(eval_set, 1):
            self.logger.info(f"Evaluating question {i}/{total_questions}")
            result = self.evaluate_question(item['question'], item['chunk_ids'], k)
            results.append(result)

        # Save results
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG retriever performance')
    parser.add_argument('index_path', help='Path to the RAG index directory')
    parser.add_argument('--eval_set', default='Experiments/002/retriever_evaluation_set.json',
                      help='Path to evaluation set JSON file')
    parser.add_argument('--output', default='Experiments/002/retriever_evaluation_results.json',
                      help='Path to save evaluation results')
    parser.add_argument('--k', type=int, default=20,
                      help='Number of results to retrieve per query (default: 20)')
    
    args = parser.parse_args()
    
    try:
        evaluator = RetrieverEvaluator(args.index_path)
        eval_set = evaluator.load_evaluation_set(args.eval_set)
        evaluator.evaluate_all(eval_set, args.output, args.k)
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
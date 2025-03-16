# run  on ubuntu using:  python rag_querier.py .ragatouille/colbert/indexes/Experiment_002 --k 10 

from pathlib import Path
import argparse
import sys
import logging
from typing import Dict, List
from ragatouille import RAGPretrainedModel

class RAGQuerier:
    def __init__(self, index_path: Path):
        """Initialize the RAG querier with a specific index"""
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

    def _get_user_input(self) -> tuple[str, Dict]:
        """Get query and metadata filters from user"""
        print("\n=== RAG Query System ===\n")
        
        # Get search query
        query = input("Enter your search query: ").strip()
        while not query:
            print("Query cannot be empty. Please try again.")
            query = input("Enter your search query: ").strip()
            
        print("\nEnter metadata filters (press Enter to skip):")
        metadata_filters = {}
        
        # Get metadata filters
        headers = [
            ("Header 1", "Header 1"),
            ("Header 2", "Header 2"),
            ("Header 3", "Header 3"),
            ("Header 4", "Header 4"),
            ("Header 5", "Header 5")
        ]
        
        for display_name, field_name in headers:
            value = input(f"Enter {display_name}: ").strip()
            if value:  # Only add non-empty values to filters
                metadata_filters[field_name] = value
                
        return query, metadata_filters
        
    def _filter_by_metadata(self, results: List[Dict], metadata_filters: Dict) -> List[Dict]:
        """Filter results based on metadata criteria"""
        if not metadata_filters:
            return results
            
        filtered_results = []
        for result in results:
            doc_metadata = result.get('document_metadata', {})
            matches_all_filters = all(
                doc_metadata.get(key) == filter_value 
                for key, filter_value in metadata_filters.items()
            )
            if matches_all_filters:
                filtered_results.append(result)
                
        return filtered_results
        
    def search(self, k: int = 10) -> None:
        """Perform search based on user input"""
        try:
            # Get user input
            query, metadata_filters = self._get_user_input()
            
            # Log search parameters
            self.logger.info(f"Searching with query: '{query}'")
            if metadata_filters:
                self.logger.info(f"Applying metadata filters: {metadata_filters}")
            
            # Perform search and filter results
            results = self.rag_model.search(query, k=k)
            filtered_results = self._filter_by_metadata(results, metadata_filters)
            
            # Display results
            self._display_results(filtered_results)
            
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            raise
            
    def _display_results(self, results: List[Dict]):
        """Display search results in a formatted way"""
        if not results:
            print("\nNo results found matching your criteria.")
            return
            
        print(f"\n=== Found {len(results)} matching results ===\n")
        
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:200]}...")  # Show first 200 chars
            print(f"Document ID: {result['document_id']}")
            if 'document_metadata' in result:
                print("Metadata:")
                for key, value in result['document_metadata'].items():
                    print(f"  {key}: {value}")
            print("-" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Query RAG index with metadata filtering')
    parser.add_argument('index_path', help='Path to the index directory')
    parser.add_argument('--k', type=int, default=10, help='Number of results to retrieve (default: 10)')
    
    args = parser.parse_args()
    
    try:
        querier = RAGQuerier(args.index_path)
        querier.search(k=args.k)
        
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nSearch failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
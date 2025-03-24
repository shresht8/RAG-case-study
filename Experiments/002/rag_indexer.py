# Build index using: python rag_indexer.py 002
# Index path will be .ragatouille/colbert/indexes/Experiment_002

from rag_utils import DocumentIndexer
from pathlib import Path
import argparse
import sys
import logging

class RAGIndexBuilder:
    def __init__(self, experiment_number: str):
        """
        Initialize the RAG index builder for a specific experiment
        
        Args:
            experiment_number: The experiment number to work with
        """
        self.experiment_number = experiment_number
        self.indexer = DocumentIndexer(experiment_number)
        
    def build_index(self) -> Path:
        """
        Build the index for the experiment
        
        Returns:
            Path: Location of the created index
        """
        try:
            print(f"\nBuilding index for experiment {self.experiment_number}...")
            print("This may take a while depending on the size of your document collection.")
            
            # Build the index
            index_path = self.indexer.create_index()
            
            print(f"\nSuccess! Index built at: {index_path}")
            return index_path
            
        except FileNotFoundError as e:
            print(f"\nError: Could not find document chunks file for experiment {self.experiment_number}")
            print(f"Please ensure the file exists at: {self.indexer.chunks_path}")
            raise
            
        except Exception as e:
            print(f"\nError building index: {str(e)}")
            raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Build RAG index for a specific experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python rag_indexer.py 001
  python rag_indexer.py experiment_001
        """
    )
    
    parser.add_argument(
        'experiment_number',
        help='Experiment number/name (e.g., 001 or experiment_001)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Initialize and run index builder
        builder = RAGIndexBuilder(args.experiment_number)
        index_path = builder.build_index()
        
        # Final success message
        print("\nIndex building completed successfully!")
        print(f"You can find your index at: {index_path}")
        print("\nTo use this index for searching, make note of this path.")
        
    except KeyboardInterrupt:
        print("\n\nIndex building interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed to build index: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
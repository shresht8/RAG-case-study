### Implement guardrails to ensure the LLM is not hallucinating
### Implement guardrails to ensure no PII is exposed
### Implement guardrails to make sure response is safe and does not contain offensive content

# To run: python rag_system_002.py .ragatouille/colbert/indexes/Experiment_002

from ragatouille import RAGPretrainedModel
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional, Dict, List, Annotated
import argparse
import logging
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class AnswerWithCitation(BaseModel):
    """Validates and structures the final response with citations."""
    is_relevant: bool = Field(
        description='Whether the query is relevant to the provided context. If not relevant, return False.'
    )
    answer: Annotated[
        str, 
        Field(
            description='Final answer to user query if relevant information is available. Otherwise, indicate cannot help.'
        )
    ]
    
    citation: Optional[Dict[str, str]] = Field(
        None,
        description="Citation mapping chunk ids to their content. Must include all relevant chunks."
    )

    @field_validator('is_relevant', 'answer', 'citation')
    def validate_response(cls, v, info: ValidationInfo):
        if info.field_name == 'answer':
            is_relevant = info.data.get('is_relevant')
            if not is_relevant and v != "I cannot help with that":
                raise ValueError("Answer must be 'I cannot help with that' if query is not relevant.")
        if info.field_name == 'citation':
            is_relevant = info.data.get('is_relevant')
            if not is_relevant and v is not None:
                raise ValueError("Citation must be None if query is not relevant.")
            if is_relevant and v is None:
                raise ValueError("Citation must be provided if query is relevant.")
        return v

class RAGSystem:
    def __init__(self, index_path: str):
        """Initialize the RAG system."""
        self.logger = self._setup_logger()
        self.index_path = Path(index_path).resolve()
        self.rag_model = None
        self.client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))
        self._load_model()

    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger

    def _load_model(self):
        """Load the RAG model from the index."""
        self.logger.info(f"Loading RAG model from index at {self.index_path}")
        try:
            self.rag_model = RAGPretrainedModel.from_index(str(self.index_path))
            self.logger.info("Successfully loaded RAG model")
        except Exception as e:
            self.logger.error(f"Error loading model from index: {str(e)}")
            raise

    def get_user_input(self) -> tuple[str, Dict]:
        """Get query and metadata filters from user."""
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
            if value:
                metadata_filters[field_name] = value
                
        return query, metadata_filters

    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved results into context string."""
        formatted_chunks = []
        for result in results:
            chunk_text = (
                f"Source: {result['document_id']}\n"
                f"Content: {result['content']}\n"
            )
            formatted_chunks.append(chunk_text)
            
        return "\n".join(formatted_chunks)

    def create_messages(self, query: str, context: str) -> List[Dict]:
        """Create formatted messages for the LLM."""
        system_prompt = """You are a helpful assistant. Answer the question based on the provided context only. 
        If you cannot find the answer in the context, say so. Always provide citations for your answers."""
        
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]

    def process_query(self, query: str, metadata_filters: Dict, k: int = 10):
        """Process a query through the RAG system."""
        try:
            # Search for relevant documents
            self.logger.info(f"Searching with query: '{query}'")
            results = self.rag_model.search(query, k=k)
            
            # Format context
            context = self.format_context(results)
            
            # Create messages for LLM
            messages = self.create_messages(query, context)
            
            # Get response from LLM
            response = self.client.chat.completions.create(
                response_model=AnswerWithCitation,
                model="gpt-4o",
                messages=messages,
                max_retries=3
            )
            
            # Display results
            print("\n=== Response ===")
            print(f"\nAnswer: {response.answer}")
            
            if response.citation:
                print("\nCitations:")
                for chunk_id, content in response.citation.items():
                    print(f"\n{chunk_id}: {content}")
                    
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='RAG System with LLM integration')
    parser.add_argument('index_path', help='Path to the RAG index directory')
    
    args = parser.parse_args()
    
    try:
        rag_system = RAGSystem(args.index_path)
        while True:
            try:
                query, metadata_filters = rag_system.get_user_input()
                rag_system.process_query(query, metadata_filters)
                
                # Ask if user wants to continue
                continue_search = input("\nWould you like to ask another question? (y/n): ").lower()
                if continue_search != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n\nSearch interrupted by user.")
                break
                
    except Exception as e:
        print(f"\nSystem error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
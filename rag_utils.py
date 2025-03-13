import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from ragatouille import RAGPretrainedModel

# Configuration
BASE_EXPERIMENTS_PATH = "Experiments"
MODEL_NAME = "colbert-ir/colbertv2.0" #modernBERT_text_similarity_finetune or colbert-ir/colbertv2.0
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = "INFO"

def setup_logger(name):
    """Configure and return a logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

class DocumentIndexer:
    def __init__(self, experiment_number: str):
        self.logger = setup_logger(__name__)
        self.experiment_number = experiment_number
        self.experiment_path = Path(BASE_EXPERIMENTS_PATH) / experiment_number
        self.chunks_path = self.experiment_path / "document_chunks.json"
        self.index_path = self.experiment_path / "index"
        self.rag_model = None
        
    def validate_chunk(self, chunk: Dict, chunk_idx: int) -> Dict:
        """
        Validate and extract required fields from a document chunk
        
        Args:
            chunk: Document chunk dictionary
            chunk_idx: Index of the chunk for error reporting
            
        Returns:
            Dict containing validated content and metadata
        """
        # Check if chunk is a dictionary
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {chunk_idx} is not a dictionary")
            
        # Extract text content from the appropriate field
        text_content = None
        
        # Try different possible field names for content
        content_fields = ['text', 'content', 'body', 'chunk_content']  # Added 'chunk_content'
        
        for field in content_fields:
            if field in chunk:
                text_content = chunk[field]
                break
                
        if not text_content:
            self.logger.error(f"Chunk structure: {json.dumps(chunk, indent=2)}")
            raise ValueError(
                f"Chunk {chunk_idx} is missing text content (tried fields: {', '.join(content_fields)})"
            )
        
        # Extract metadata - check for both chunk_metadata and direct metadata fields
        metadata = {}
        if 'chunk_metadata' in chunk:
            metadata = chunk['chunk_metadata']
        else:
            # Create metadata dictionary excluding content and known system fields
            excluded_fields = content_fields + ['chunk_id']
            metadata = {
                k: v for k, v in chunk.items() 
                if k not in excluded_fields and v is not None
            }
        
        # Get or generate chunk ID
        chunk_id = str(chunk.get('chunk_id', f'chunk_{chunk_idx}'))
        
        return {
            'content': text_content,
            'metadata': metadata,
            'id': chunk_id
        }
        
    def load_document_chunks(self) -> tuple[List[str], List[Dict], List[str]]:
        """Load and parse document chunks from JSON file"""
        self.logger.info(f"Loading document chunks from {self.chunks_path}")
        
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Document chunks file not found at {self.chunks_path}")
        
        try:
            with open(self.chunks_path, 'r') as f:
                chunks_data = json.load(f)
                
            # Ensure we have a list of chunks
            if not isinstance(chunks_data, list):
                raise ValueError("Document chunks must be a list")
                
            if not chunks_data:
                raise ValueError("Document chunks list is empty")
                
            # Process each chunk
            documents = []
            metadata = []
            doc_ids = []
            
            self.logger.info(f"Found {len(chunks_data)} chunks to process")
            
            for idx, chunk in enumerate(chunks_data):
                try:
                    validated_chunk = self.validate_chunk(chunk, idx)
                    documents.append(validated_chunk['content'])
                    metadata.append(validated_chunk['metadata'])
                    doc_ids.append(validated_chunk['id'])
                except Exception as e:
                    self.logger.error(f"Error processing chunk {idx}: {str(e)}")
                    raise
                    
            self.logger.info(f"Successfully processed {len(documents)} document chunks")
            
            # Log a sample chunk for verification
            if documents:
                self.logger.info("Sample document chunk structure:")
                self.logger.info(f"Content preview: {documents[0][:100]}...")
                self.logger.info(f"Metadata: {metadata[0]}")
                self.logger.info(f"ID: {doc_ids[0]}")
                
            return documents, metadata, doc_ids
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in chunks file: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading document chunks: {str(e)}")
            raise
            
    def create_index(self) -> Path:
        """Create a new index with metadata"""
        self.logger.info(f"Creating new index for experiment {self.experiment_number}")
        
        try:
            # Load the model
            self.rag_model = RAGPretrainedModel.from_pretrained(MODEL_NAME)
            self.logger.info("Successfully loaded RAG model")
            
            # Load documents and metadata
            documents, metadata, doc_ids = self.load_document_chunks()
            
            # Create the index
            index_path = self.rag_model.index(
                index_name=str(self.index_path),
                collection=documents,
                document_ids=doc_ids,
                document_metadatas=metadata
            )
            
            self.logger.info(f"Successfully created index at {index_path}")
            return index_path
            
        except Exception as e:
            self.logger.error(f"Error creating index: {str(e)}")
            raise
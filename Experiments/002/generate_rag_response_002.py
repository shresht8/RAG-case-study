from ragatouille import RAGPretrainedModel
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional, Dict, List, Annotated
import argparse
import logging
import json
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from prompts import system_prompt_rag_system_002

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class AnswerWithCitation(BaseModel):
    """Validates and structures the final response."""
    is_relevant: bool = Field(
        description='Whether the query is relevant to the provided context. If not relevant, return False.'
    )
    answer: str = Field(
        description='Final answer to user query if relevant information is available. Use all the relevant information to answer the question.'
          'Dont use any other information than the ones provided in the context. Otherwise, indicate cannot help.'
    )
    citation: Optional[Dict[str, str]] = Field(
        None,
        description='Citation mapping chunk ids to their relevant content. The content must be a substring of the chunk content.'
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
            
            if v is not None:
                try:
                    # Get retrieved chunk IDs from validation context
                    retrieved_chunk_ids = info.context.get('retrieved_chunk_ids', [])
                    assert retrieved_chunk_ids, "Retrieved chunk IDs must be provided for citation validation"
                    
                    # Validate only chunk IDs
                    for chunk_id in v.keys():
                        assert chunk_id in retrieved_chunk_ids, f"Cited chunk ID {chunk_id} not found in retrieved chunks"
                except AssertionError as e:
                    raise ValueError(str(e))
        
        return v


    def process_citations(self, retrieved_chunks: Dict[str, str]) -> Dict[str, str]:
        """Process citations by matching first and last two words of the citation."""
        if not self.citation or not retrieved_chunks:
            return None

        processed_citations = {}
        for chunk_id, cited_content in self.citation.items():
            try:
                full_chunk = retrieved_chunks[chunk_id]
                cited_content = cited_content.strip()
                
                # Convert both strings to lowercase for comparison
                chunk_lower = full_chunk.lower()
                cited_lower = cited_content.lower()
                
                if cited_lower in chunk_lower:
                    # If exact match found (case-insensitive), use the original citation
                    processed_citations[chunk_id] = cited_content
                else:
                    # Split into words and get first and last two words
                    words = [w for w in cited_lower.split() if w]
                    if len(words) < 4:  # Need at least 4 words for the matching
                        processed_citations[chunk_id] = cited_content
                        continue
                    
                    first_two = ' '.join(words[:2])
                    last_two = ' '.join(words[-2:])
                    
                    # Find positions of first and last two words
                    start_pos = chunk_lower.find(first_two)
                    end_pos = chunk_lower.find(last_two)
                    
                    if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                        # Extract the substring from original text (maintaining case)
                        # Include the length of the last two words to get complete text
                        end_pos = end_pos + len(last_two)
                        processed_citations[chunk_id] = full_chunk[start_pos:end_pos]
                    else:
                        # If no valid match found, keep original citation
                        processed_citations[chunk_id] = cited_content
                        
            except Exception as e:
                # On any error, keep the original citation
                processed_citations[chunk_id] = cited_content
                
        return processed_citations

class RAGResponseGenerator:
    def __init__(self, index_path: str):
        self.logger = self._setup_logger()
        self.index_path = Path(index_path).resolve()
        self.rag_model = None
        self.client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))
        self._load_model()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def _load_model(self):
        self.logger.info(f"Loading RAG model from index at {self.index_path}")
        try:
            self.rag_model = RAGPretrainedModel.from_index(str(self.index_path))
            self.logger.info("Successfully loaded RAG model")
        except Exception as e:
            self.logger.error(f"Error loading model from index: {str(e)}")
            raise

    def format_context(self, results: List[Dict]) -> str:
        formatted_chunks = []
        for result in results:
            chunk_text = (
                f"Source: {result['document_id']}\n"
                f"Content: {result['content']}\n"
            )
            formatted_chunks.append(chunk_text)
        return "\n".join(formatted_chunks)

    def create_messages(self, query: str, context: str) -> List[Dict]:
        system_prompt = system_prompt_rag_system_002
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

    def process_query(self, query: str, k: int = 10) -> Dict:
        try:
            # Search for relevant documents
            results = self.rag_model.search(query, k=k)
            
            # Create a dictionary of retrieved chunks and list of chunk IDs
            retrieved_chunks = {
                result['document_id']: result['content'] 
                for result in results
            }
            retrieved_chunk_ids = list(retrieved_chunks.keys())
            
            # Format context
            context = self.format_context(results)
            
            # Create messages for LLM
            messages = self.create_messages(query, context)
            
            # Get response from LLM
            response = self.client.chat.completions.create(
                model="gpt-4o",
                response_model=AnswerWithCitation,
                messages=messages,
                max_retries=3,
                validation_context={"retrieved_chunk_ids": retrieved_chunk_ids}  # Only pass chunk IDs
            )
            
            # Process citations to get exact substrings
            processed_citations = response.process_citations(retrieved_chunks)
            
            return {
                "llm_response": response.answer,
                "is_relevant": response.is_relevant,
                "cited_chunk_ids": processed_citations,
                "retrieved_chunk_ids": retrieved_chunk_ids
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return None

    def process_evaluation_set(self, eval_set_path: str, output_path: str):
        # Load evaluation set
        with open(eval_set_path, 'r') as f:
            eval_set = json.load(f)

        # Process each question
        results = []
        for item in eval_set:
            self.logger.info(f"Processing question: {item['question']}")
            
            # Map fields from evaluation set to output structure
            record = {
                "question": item["question"],
                "ground_truth_answer": item["answer"],  # Map from "answer" to "ground_truth_answer"
                "difficulty": item["difficulty"],
                "ground_truth_chunk_ids": item["chunk_ids"],  # Map from "chunk_ids" to "ground_truth_chunk_ids"
                "document": item["document"]
            }
            
            # Process the query
            response = self.process_query(item["question"])
            if response:
                record.update({
                    "retrieved_chunk_ids": response["retrieved_chunk_ids"],
                    "llm_response": response["llm_response"],
                    "is_relevant": response["is_relevant"],
                    "cited_chunk_ids": response["cited_chunk_ids"]
                })
            else:
                record.update({
                    "retrieved_chunk_ids": [],
                    "llm_response": None,
                    "is_relevant": False,
                    "cited_chunk_ids": None
                })
            
            results.append(record)

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate RAG responses for evaluation set')
    parser.add_argument('index_path', help='Path to the RAG index directory')
    parser.add_argument('--eval_set', default='Experiments/002/retriever_evaluation_set.json',
                      help='Path to evaluation set JSON file')
    parser.add_argument('--output', default='Experiments/002/llm_responses_eval_set_v2.json',
                      help='Path to save responses')
    
    args = parser.parse_args()
    
    try:
        generator = RAGResponseGenerator(args.index_path)
        generator.process_evaluation_set(args.eval_set, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
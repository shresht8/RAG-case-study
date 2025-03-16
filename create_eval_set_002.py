# create_eval_set.py
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import logging
from typing import Dict, List
import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    GenerationConfig,
)
from sentence_transformers.cross_encoder import CrossEncoder
from prompts import system_prompt_QA_eval_bot

class EvalSetGenerator:
    def __init__(self, experiment_dir: str, project_id: str = None, location: str = "australia-southeast1"):
        """
        Initialize the evaluation set generator.
        
        Args:
            experiment_dir: Directory for experiment files
            project_id: Google Cloud project ID
            location: Google Cloud location
        """
        self.experiment_dir = Path(experiment_dir)
        self.logger = self._setup_logger()
        self._init_vertex_ai(project_id, location)
        self.cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")
        
        # Response schema for question generation
        self.response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                    },
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of chunk IDs that the question and answer are based on"
                    }
                },
                "required": ["question", "answer", "difficulty", "chunk_ids"],
            },
        }

    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def _init_vertex_ai(self, project_id: str, location: str):
        """Initialize Vertex AI with project settings"""
        if project_id:
            vertexai.init(project=project_id, location=location)

    def format_document_chunks(self) -> Dict[str, str]:
        """
        Read and format document chunks from JSON file.
        
        Returns:
            Dictionary mapping document names to their formatted content
        """
        chunks_path = self.experiment_dir / "document_chunks.json"
        self.logger.info(f"Reading chunks from {chunks_path}")
        
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading chunks file: {str(e)}")
            raise

        formatted_docs = {}
        
        # Group chunks by document name (extracted from chunk_id)
        for chunk in chunks_data:
            # Extract document name from chunk_id (assuming format: DOC_NAME_chunk_N)
            doc_name = "_".join(chunk['chunk_id'].split("_")[:-2])
            
            if doc_name not in formatted_docs:
                formatted_docs[doc_name] = f"{doc_name}:\n\n"
            
            # Format chunk with content and metadata
            formatted_docs[doc_name] += "----x----\n"
            formatted_docs[doc_name] += f"chunk_id: {chunk['chunk_id']}\n"
            formatted_docs[doc_name] += f"chunk_content: {chunk['chunk_content']}\n"
            
            # Add metadata if present
            if 'chunk_metadata' in chunk:
                formatted_docs[doc_name] += "metadata:\n"
                for header, value in chunk['chunk_metadata'].items():
                    formatted_docs[doc_name] += f"  {header}: {value}\n"
            
            formatted_docs[doc_name] += "\n"

        return formatted_docs

    def generate_questions(self, context: str, num_questions: int = 10) -> str:
        """
        Generate questions using Vertex AI.
        
        Args:
            context: Document context to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            JSON string containing generated questions
        """
        model = GenerativeModel("gemini-1.5-pro-002")
        
        # System prompt template
        system_prompt = system_prompt_QA_eval_bot
        
        try:
            response = model.generate_content(
                system_prompt.format(chunk_set=context, num_questions=num_questions),
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=self.response_schema
                ),
            )
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating questions: {str(e)}")
            raise

    def create_final_eval_set(self, num_questions_per_doc: int = 10, max_chunks: int = 20):
        """
        Create and save the final evaluation set using both LLM and cross-encoder.
        
        Args:
            num_questions_per_doc: Number of questions to generate per document
            max_chunks: Maximum number of chunks to include in ground truth
        """
        try:
            # Format documents
            formatted_docs = self.format_document_chunks()
            
            # Generate questions for each document
            all_questions = []
            for doc_name, doc_content in formatted_docs.items():
                self.logger.info(f"Generating questions for {doc_name}")
                questions = json.loads(
                    self.generate_questions(doc_content, num_questions_per_doc)
                )
                for q in questions:
                    q['document'] = doc_name
                all_questions.extend(questions)

            # Process with cross-encoder and create final ground truth
            final_eval_set = []
            for question in all_questions:
                # Get cross-encoder scores for all chunks
                scores = []
                for chunk in formatted_docs[question['document']].split('----x----'):
                    if 'chunk_id:' in chunk:
                        chunk_id = chunk.split('chunk_id:')[1].split('\n')[0].strip()
                        chunk_content = chunk.split('chunk_content:')[1].split('\n')[0].strip()
                        score = self.cross_encoder.predict([(question['question'], chunk_content)])
                        scores.append((chunk_id, float(score)))

                # Sort by score
                scores.sort(key=lambda x: x[1], reverse=True)
                top_chunks = [s[0] for s in scores[:max_chunks]]

                # Find overlapping chunks
                llm_chunks = set(question['chunk_ids'])
                cross_encoder_chunks = set(top_chunks[:10])
                overlapping_chunks = list(llm_chunks.intersection(cross_encoder_chunks))

                # Create final ground truth
                final_ground_truth = overlapping_chunks.copy()
                remaining_slots = max_chunks - len(final_ground_truth)
                
                # Add remaining chunks from cross-encoder results
                if remaining_slots > 0:
                    for chunk_id in top_chunks:
                        if chunk_id not in final_ground_truth:
                            final_ground_truth.append(chunk_id)
                            remaining_slots -= 1
                            if remaining_slots == 0:
                                break

                # Create simplified evaluation entry with overlapping_chunks
                eval_entry = {
                    'question': question['question'],
                    'answer': question['answer'],
                    'difficulty': question['difficulty'],
                    'chunk_ids': final_ground_truth,  # This is the final ground truth
                    'document': question['document'],
                    'overlapping_chunks': overlapping_chunks  # List of chunks that overlap between LLM and cross-encoder
                }
                final_eval_set.append(eval_entry)

            # Save final evaluation set
            output_path = self.experiment_dir / "retriever_evaluation_set.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_eval_set, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved final evaluation set to {output_path}")
            return final_eval_set

        except Exception as e:
            self.logger.error(f"Error creating final evaluation set: {str(e)}")
            raise

def main():
    PROJECT_ID = os.getenv("PROJECT_ID") 
    generator = EvalSetGenerator(
        experiment_dir="Experiments/002",
        project_id=PROJECT_ID
    )
    generator.create_final_eval_set()

if __name__ == "__main__":
    main()
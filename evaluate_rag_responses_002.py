# Run using: python evaluate_rag_responses_002.py Experiments/002

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
import json
import os
import instructor
from openai import OpenAI
from pathlib import Path
import argparse
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class EvaluationMetrics(BaseModel):
    """Validates and structures the evaluation metrics."""
    relevancy_score: float = Field(
        allowed_values=[0, 1],
        description="Score indicating how relevant the response is to the query. 0 if not relevant, 1 if relevant."
    )
    correctness_score: float = Field(
        allowed_values=[0, 0.5, 1],
        description="Score indicating factual correctness compared to ground truth. 0 if incorrect, 0.5 if partially correct, 1 if correct."
    )
    context_alignment_score: float = Field(
        allowed_values=[0, 0.5, 1],
        description="Score indicating how well response aligns with retrieved chunks. 0 if wrong citations, 0.5 if some correct citations, 1 if all citations correct and properly used."
    )
    citation_score: float = Field(
        allowed_values=[0, 0.5, 1],
        description="Score indicating citation accuracy. 0 if citations not accurate, 0.5 if partially accurate, 1 if fully accurate and properly used."
    )
    safety_score: float = Field(
        allowed_values=[0, 1],
        description="Score indicating safety of the response. 0 if not safe, 1 if safe."
    )
    feedback: str = Field(
        description="Detailed feedback explaining the scores"
    )

    # @field_validator('relevancy_score', 'correctness_score', 'context_alignment_score', 'citation_score', 'safety_score')
    # def validate_score_values(cls, v, field):
    #     allowed_values = field.field_info.json_schema_extra['allowed_values']
    #     if v not in allowed_values:
    #         raise ValueError(f"{field.name} must be one of {allowed_values}")
    #     return v

class RAGEvaluator:
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o",
        max_retries: int = 3
    ):
        self.client = instructor.patch(OpenAI(api_key=openai_api_key))
        self.model = model
        self.max_retries = max_retries

    def load_document_chunks(self, experiment_dir: str) -> Dict[str, Dict]:
        """Load document chunks from JSON file"""
        chunks_file = os.path.join(experiment_dir, "document_chunks.json")
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
        return {chunk["chunk_id"]: chunk for chunk in chunks_data}

    def format_retrieved_chunks(self, chunk_ids: List[str], chunks_lookup: Dict[str, Dict]) -> str:
        """Format retrieved chunks into a single string"""
        formatted_chunks = []
        for chunk_id in chunk_ids:
            if chunk_id in chunks_lookup:
                chunk = chunks_lookup[chunk_id]
                metadata = chunk.get('chunk_metadata', {})
                metadata_str = "\n".join(f"{key}: {value}" for key, value in metadata.items()) if metadata else "No metadata"
                
                formatted_chunk = (
                    f"Chunk ID: {chunk_id}\n"
                    f"Metadata:\n{metadata_str}\n"
                    f"Content: {chunk['chunk_content']}\n"
                )
                formatted_chunks.append(formatted_chunk)
        return "\n".join(formatted_chunks)

    def format_cited_chunks(self, cited_chunks: Optional[Dict[str, str]]) -> str:
        """Format cited chunks into a single string"""
        if cited_chunks is None:
            return "No citations provided"
            
        formatted_citations = []
        for chunk_id, content in cited_chunks.items():
            # Try to get metadata from chunks_lookup if available
            chunk_metadata = {}
            if hasattr(self, 'chunks_lookup') and chunk_id in self.chunks_lookup:
                chunk_metadata = self.chunks_lookup[chunk_id].get('chunk_metadata', {})
            
            metadata_str = "\n".join(f"{key}: {value}" for key, value in chunk_metadata.items()) if chunk_metadata else "No metadata"
            
            formatted_citation = (
                f"Chunk ID: {chunk_id}\n"
                f"Metadata:\n{metadata_str}\n"
                f"Content: {content}\n"
            )
            formatted_citations.append(formatted_citation)
        return "\n".join(formatted_citations)

    def evaluate_response(self, record: Dict[str, Any], chunks_lookup: Dict[str, Dict]) -> EvaluationMetrics:
        """Evaluate a single response record"""
        
        # Format retrieved chunks
        retrieved_chunks_str = self.format_retrieved_chunks(
            record['retrieved_chunk_ids'], 
            chunks_lookup
        )
        
        # Format cited chunks
        cited_chunks_str = self.format_cited_chunks(record['cited_chunk_ids'])
        
        evaluation_prompt = f"""
        Please evaluate the following RAG system response:

        Question: {record['question']}
        Ground Truth Answer: {record['ground_truth_answer']}
        LLM Response: {record['llm_response']}
        Is Relevant: {record['is_relevant']}

        Retrieved Chunks:
        {retrieved_chunks_str}

        Cited Chunks:
        {cited_chunks_str}

        Evaluate based on:
        1. Relevancy: Is the response relevant to the query and is the is_relevant field correct? 0 if not relevant and 1 if relevant.
        2. Correctness: How factually correct is the response compared to the ground truth?
          0 if incorrect - The answer doesnt match the ground truth answer at all.
          0.5 if partially correct - The answer matches some part of the ground truth answer.
          1 if correct - The answer contains all the information present in the ground truth answer.
        3. Context Alignment: Does the response align with the retrieved chunks? 
          0 if LLM doesn't pick the right citations from the retrieved chunks.
          0.5 if LLM picks some of the right citations from the retrieved chunks but not all.
          1 if LLM picks the right citations from the retrieved chunks and uses them properly.
        4. Citation Accuracy: Are the citations accurate and properly used? 
          0 if not accurate - The citations are not accurate. None of the citations present are relevant to the question or response.
          0.5 if partially accurate - The citations are partially accurate. The citations present are relevant to the question but only contains some of the 
          information present in the LLM response.
          1 if accurate - The citations are accurate. The citations present are relevant to the question and the response given is based on the citations.
        5. Safety: Is the response safe and appropriate? 0 if not safe and 1 if safe.

        """

        try:
            evaluation = self.client.chat.completions.create(
                model=self.model,
                response_model=EvaluationMetrics,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of RAG systems."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_retries=self.max_retries
            )
            return evaluation
        except Exception as e:
            print(f"Error evaluating question '{record['question']}': {str(e)}")
            return None

    def evaluate_experiment(self, experiment_dir: str) -> Dict[str, Any]:
        """Evaluate all responses in an experiment"""
        
        # Load the responses
        responses_file = os.path.join(experiment_dir, "llm_responses_eval_set.json")
        with open(responses_file, 'r') as f:
            responses = json.load(f)

        # Load document chunks
        chunks_lookup = self.load_document_chunks(experiment_dir)

        # Evaluate each record
        evaluations = []
        for record in responses:
            print(f"Evaluating question: {record['question']}")
            evaluation = self.evaluate_response(record, chunks_lookup)
            if evaluation:
                evaluations.append(evaluation.model_dump())

        # Calculate average scores
        avg_scores = {
            "avg_relevancy_score": sum(e["relevancy_score"] for e in evaluations) / len(evaluations),
            "avg_correctness_score": sum(e["correctness_score"] for e in evaluations) / len(evaluations),
            "avg_context_alignment_score": sum(e["context_alignment_score"] for e in evaluations) / len(evaluations),
            "avg_citation_score": sum(e["citation_score"] for e in evaluations) / len(evaluations),
            "avg_safety_score": sum(e["safety_score"] for e in evaluations) / len(evaluations)
        }

        # Prepare final output
        final_output = {
            "experiment_dir": experiment_dir,
            "individual_evaluations": evaluations,
            "average_scores": avg_scores
        }

        # Save results
        output_file = os.path.join(experiment_dir, "rag_evaluation_results.json")
        with open(output_file, 'w') as f:
            json.dump(final_output, f, indent=2)

        print(f"\nEvaluation results saved to: {output_file}")
        print("\nAverage Scores:")
        for metric, score in avg_scores.items():
            print(f"{metric}: {score:.3f}")

        return final_output

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG responses')
    parser.add_argument('experiment_dir', help='Path to experiment directory')
    parser.add_argument('--model', default='gpt-4o',
                      help='OpenAI model to use for evaluation')
    
    args = parser.parse_args()
    
    try:
        evaluator = RAGEvaluator(OPENAI_API_KEY, args.model)
        results = evaluator.evaluate_experiment(args.experiment_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
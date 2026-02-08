"""
Ollama LLM client for entity and relation extraction
"""
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional
import ollama
from .config import get_config


class OllamaClient:
    """Client for interacting with Ollama LLMs with proper error handling"""
    
    def __init__(self, max_retries: int = 3, timeout: int = 60, 
                 max_text_length: int = 4000, batch_size: int = 10):
        self.config = get_config().ollama
        self.prompts = get_config().prompts
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Make a call to Ollama LLM with retry logic and timeout"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                # Note: valid ollama-python args might vary by version, assume standard chat
                response = ollama.chat(
                    model=self.config.model,
                    messages=messages,
                    options={'num_predict': 2048} # timeout not directly supported in all python client versions, rely on server/OS
                )
                
                if 'message' not in response or 'content' not in response['message']:
                    raise ValueError(f"Unexpected response structure: {response}")
                
                return response['message']['content']
                
            except Exception as e:
                self.logger.warning(f"Ollama API call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All retry attempts failed for Ollama API")
                    raise RuntimeError(f"Failed to call Ollama after {self.max_retries} attempts: {e}") from e
                
                time.sleep(2 ** attempt)  # Exponential backoff
        return "" # Should be unreachable due to raise above

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response with robust error handling"""
        if not text:
             # Log warning but return None to be handled by caller
            self.logger.warning("Empty response text from LLM")
            return None
        
        # Multiple extraction strategies
        json_str = None
        
        # Strategy 1: JSON in markdown code blocks
        code_block_pattern = r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                json_str = candidate
                break
            except json.JSONDecodeError:
                continue
        
        # Strategy 2: Raw JSON patterns
        if not json_str:
            json_patterns = [
                r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Basic object
                r'(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'  # Basic array
            ]
            
            for pattern in json_patterns:
                for match in re.finditer(pattern, text, re.DOTALL):
                    candidate = match.group(1).strip()
                    try:
                        json.loads(candidate) # Validate syntax
                        json_str = candidate
                        break
                    except json.JSONDecodeError:
                        continue
                if json_str:
                    break
        
        if not json_str:
            self.logger.warning(f"Could not extract valid JSON from response: {text[:200]}...")
            return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse extracted JSON: {e}")
            return None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using LLM
        Returns: {"PERSON": [...], "ORG": [...], ...}
        """
        if not text:
             return {}

        prompt = self.prompts.entity_extraction.format(text=text[:self.max_text_length])
        
        system_prompt = "You are a precise entity extraction system. Return only valid JSON."
        
        try:
            response = self._call_llm(prompt, system_prompt)
            result = self._extract_json(response)
            
            if result and isinstance(result, dict) and 'entities' in result:
                 # Validate entity structure
                entities = result['entities']
                validated_entities = {}
                if isinstance(entities, dict):
                    for entity_type, entity_list in entities.items():
                        if isinstance(entity_list, list):
                            # Clean and dedup
                            valid_list = list(set([str(e).strip() for e in entity_list if isinstance(e, (str, int, float))]))
                            if valid_list:
                                validated_entities[entity_type] = valid_list
                return validated_entities
            
            return {}
        except Exception as e:
             self.logger.error(f"Entity extraction failed: {e}")
             return {}

    def extract_entities_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """Extract entities from multiple texts with batch processing"""
        if not texts:
            return []
        
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for text in batch:
                try:
                    entities = self.extract_entities(text)
                    results.append(entities)
                except Exception as e:
                    self.logger.error(f"Failed to extract entities from text batch: {e}")
                    results.append({})  # Return empty dict for failed extractions
        return results
    
    def extract_relations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract relationships from text using LLM
        Returns: [{"subject": "X", "relation": "Y", "object": "Z"}, ...]
        """
        if not text:
            return []

        prompt = self.prompts.relation_extraction.format(text=text[:self.max_text_length])
        
        system_prompt = "You are a precise relation extraction system. Return only valid JSON array."
        
        try:
            response = self._call_llm(prompt, system_prompt)
            result = self._extract_json(response)
            
            if result and isinstance(result, list):
                # Validate structure
                valid_relations = []
                for rel in result:
                    if isinstance(rel, dict) and all(k in rel for k in ['subject', 'relation', 'object']):
                        valid_relations.append(rel)
                return valid_relations
            
            return []
        except Exception as e:
            self.logger.error(f"Relation extraction failed: {e}")
            return []
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the text"""
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text[:self.max_text_length]}"
        try:
            return self._call_llm(prompt)
        except Exception:
            return ""
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using Ollama"""
        try:
            response = ollama.embeddings(
                model=get_config().chroma.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return []

    def evaluate_extraction_quality(self, texts: List[str], 
                                  ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate extraction quality against ground truth"""
        if len(texts) != len(ground_truth):
            raise ValueError("Texts and ground truth must have same length")
        
        results = []
        for text, gt in zip(texts, ground_truth):
            start_time = time.time()
            try:
                extracted = {
                    'entities': self.extract_entities(text),
                    'relations': self.extract_relations(text)
                }
                extraction_time = time.time() - start_time
                
                results.append({
                    'extracted': extracted,
                    'ground_truth': gt,
                    'extraction_time': extraction_time
                })
            except Exception as e:
                self.logger.error(f"Evaluation failed for text: {e}")
                results.append({
                    'extracted': {'entities': {}, 'relations': []},
                    'ground_truth': gt,
                    'extraction_time': 0,
                    'error': str(e)
                })
        
        # Calculate metrics
        return self._calculate_metrics(results)
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate evaluation metrics for entity extraction (Precision/Recall/F1)"""
        if not results:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "avg_time": 0.0}
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_time = 0
        valid_extractions = 0
        
        for result in results:
            if 'error' in result:
                continue
                
            extracted = result['extracted']
            gt = result['ground_truth']
            extraction_time = result['extraction_time']
            
            # Entity metrics
            extracted_entities = self._flatten_entities(extracted.get('entities', {}))
            gt_entities = self._flatten_entities(gt.get('entities', {}))
            
            if gt_entities:
                # Precision = TP / (TP + FP) -> Correctly extracted / Total extracted
                # Recall = TP / (TP + FN) -> Correctly extracted / Total ground truth
                
                intersection = extracted_entities & gt_entities
                precision = len(intersection) / len(extracted_entities) if extracted_entities else 0
                recall = len(intersection) / len(gt_entities)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                valid_extractions += 1
            
            total_time += extraction_time
        
        return {
            "precision": total_precision / valid_extractions if valid_extractions > 0 else 0,
            "recall": total_recall / valid_extractions if valid_extractions > 0 else 0,
            "f1": total_f1 / valid_extractions if valid_extractions > 0 else 0,
            "avg_time": total_time / len(results) if results else 0,
            "success_rate": valid_extractions / len(results) if results else 0
        }
    
    def _flatten_entities(self, entities: Dict[str, List[str]]) -> set:
        """Flatten entity dict to set of entities for metric calculation"""
        flat = set()
        for entity_list in entities.values():
            flat.update(map(str, entity_list))
        return flat

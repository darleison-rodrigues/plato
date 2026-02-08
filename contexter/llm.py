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
        self.timeout = timeout # Note: timeout is not directly used by ollama.chat in typical Python client versions, but kept for potential future use or server-side timeouts.
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
                # Note: valid ollama-python args might vary by version, assume standard chat.
                # Timeout is typically handled at the OS/network level or via HTTP client configurations if using direct HTTP calls.
                response = ollama.chat(
                    model=self.config.model,
                    messages=messages,
                    options={'num_predict': 2048} # num_predict is an example option, can be tuned.
                )
                
                if 'message' not in response or 'content' not in response['message']:
                    raise ValueError(f"Unexpected response structure: {response}")
                
                return response['message']['content']
                
            except Exception as e:
                self.logger.warning(f"Ollama API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All retry attempts failed for Ollama API")
                    # Raising a specific RuntimeError to indicate failure after retries.
                    raise RuntimeError(f"Failed to call Ollama after {self.max_retries} attempts: {e}") from e
                
                # Exponential backoff: waits 2, 4, 8... seconds
                time.sleep(2 ** attempt)  
        # This return statement is technically unreachable due to the raise in the last attempt.
        # It can be removed or kept as a safeguard, but it indicates a logic path that should not be hit.
        return "" 

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response with robust error handling"""
        if not text:
            self.logger.warning("Empty response text from LLM, cannot extract JSON.")
            return None
        
        json_str = None
        
        # Strategy 1: JSON in markdown code blocks (common LLM output format)
        # Handles both ```json ... ``` and ``` ... ```
        code_block_pattern = r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                json_str = candidate
                break # Found a valid JSON in a code block
            except json.JSONDecodeError:
                continue # Try next match if this one is not valid JSON
        
        # Strategy 2: Raw JSON patterns if not found in code blocks
        if not json_str:
            # These patterns attempt to find the outermost JSON structure (object or array)
            # They are simplified and might not catch all edge cases of deeply nested or malformed JSON.
            json_patterns = [
                r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Basic object pattern
                r'(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'  # Basic array pattern
            ]
            
            for pattern in json_patterns:
                for match in re.finditer(pattern, text, re.DOTALL):
                    candidate = match.group(1).strip()
                    try:
                        json.loads(candidate) # Validate syntax
                        json_str = candidate
                        break # Found a valid JSON structure
                    except json.JSONDecodeError:
                        continue # Try next match
                if json_str:
                    break # Exit outer loop if JSON found
        
        if not json_str:
            self.logger.warning(f"Could not extract any valid JSON structure from response. Truncated response: {text[:200]}...")
            return None
        
        try:
            # Final attempt to parse the identified JSON string
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse extracted JSON string '{json_str[:100]}...': {e}")
            return None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using LLM.
        Returns: {"PERSON": [...], "ORG": [...], ...}
        """
        if not text:
             return {}

        # Truncate text if it exceeds max_text_length to prevent LLM errors or excessive costs.
        prompt = self.prompts.entity_extraction.format(text=text[:self.max_text_length])
        
        # System prompt guides the LLM to behave as a precise extraction system and return JSON.
        system_prompt = "You are a precise entity extraction system. Return only valid JSON."
        
        try:
            response = self._call_llm(prompt, system_prompt)
            result = self._extract_json(response)
            
            # Validate the structure of the extracted result
            if result and isinstance(result, dict) and 'entities' in result:
                entities = result['entities']
                validated_entities = {}
                if isinstance(entities, dict):
                    for entity_type, entity_list in entities.items():
                        if isinstance(entity_list, list):
                            # Clean, deduplicate, and ensure all entities are strings.
                            valid_list = list(set([
                                str(e).strip() for e in entity_list 
                                if isinstance(e, (str, int, float)) and str(e).strip() # Ensure not empty after stripping
                            ]))
                            if valid_list:
                                validated_entities[entity_type] = valid_list
                return validated_entities
            else:
                self.logger.warning(f"Entity extraction returned unexpected format or missing 'entities' key: {result}")
                return {}
        except Exception as e:
             self.logger.error(f"Entity extraction failed: {e}")
             return {}

    def extract_entities_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple texts.
        Note: This implementation processes texts sequentially within a batch loop.
        For true parallel LLM calls, a different batching strategy (e.g., concurrent requests) would be needed.
        """
        if not texts:
            return []
        
        all_extracted_entities = []
        # Process texts in chunks defined by batch_size
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            for text in batch_texts:
                try:
                    entities = self.extract_entities(text)
                    all_extracted_entities.append(entities)
                except Exception as e:
                    self.logger.error(f"Failed to extract entities for a text in batch (error: {e}). Appending empty result.")
                    all_extracted_entities.append({})  # Append empty dict for failed extractions
        return all_extracted_entities
    
    def extract_relations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract relationships from text using LLM.
        Returns: [{"subject": "X", "relation": "Y", "object": "Z"}, ...]
        """
        if not text:
            return []

        prompt = self.prompts.relation_extraction.format(text=text[:self.max_text_length])
        
        # System prompt guides the LLM to behave as a precise extraction system and return a JSON array.
        system_prompt = "You are a precise relation extraction system. Return only valid JSON array."
        
        try:
            response = self._call_llm(prompt, system_prompt)
            result = self._extract_json(response)
            
            # Validate the structure of the extracted result
            if result and isinstance(result, list):
                valid_relations = []
                for rel in result:
                    # Ensure each relation is a dictionary with required keys
                    if isinstance(rel, dict) and all(k in rel for k in ['subject', 'relation', 'object']):
                        # Further clean/validate individual fields if necessary
                        valid_relations.append({
                            'subject': str(rel['subject']).strip(),
                            'relation': str(rel['relation']).strip(),
                            'object': str(rel['object']).strip()
                        })
                return valid_relations
            else:
                self.logger.warning(f"Relation extraction returned unexpected format or not a list: {result}")
                return []
        except Exception as e:
            self.logger.error(f"Relation extraction failed: {e}")
            return []
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the text using LLM"""
        # Simple prompt for summarization, respecting max_text_length for input.
        # The 'max_length' parameter refers to words in the summary.
        prompt = f"Summarize the following text in approximately {max_length} words or less:\n\n{text[:self.max_text_length]}"
        try:
            summary = self._call_llm(prompt)
            # Basic post-processing: trim if summary is too long (LLM might not always respect word counts strictly).
            # A more robust approach would involve token counting, but word count is a good heuristic.
            return ' '.join(summary.split()[:max_length]) if summary else ""
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return ""
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using Ollama's embedding model"""
        if not text:
            return []
        try:
            # Uses the embedding model configured in ChromaDB settings.
            response = ollama.embeddings(
                model=get_config().chroma.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            self.logger.error(f"Error generating embedding for text snippet: {e}")
            return []

    # Removed: evaluate_extraction_quality method - Considered "fantasy code" as it's for evaluation and benchmarking,
    # not core operational functionality of the LLM client.
    # Removed: _calculate_metrics method - Helper for evaluate_extraction_quality, also removed.
    # Removed: _flatten_entities method - Helper for _calculate_metrics, also removed.

    # Removed the unused methods and their helpers, simplifying the client to its core operational functions.
    # The remaining methods are: __init__, _call_llm, _extract_json, extract_entities, extract_entities_batch, extract_relations, summarize, generate_embedding.
    # These cover direct LLM interaction, JSON parsing, specific extraction tasks, summarization, and embedding generation.
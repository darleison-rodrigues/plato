"""
Ollama LLM client for entity and relation extraction
"""
import json
import re
from typing import Dict, List, Any
import ollama
from .config import get_config


class OllamaClient:
    """Client for interacting with Ollama LLMs"""
    
    def __init__(self):
        self.config = get_config().ollama
        self.prompts = get_config().prompts
        
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Make a call to Ollama LLM"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = ollama.chat(
                model=self.config.model,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""
    
    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response, handling markdown code blocks"""
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Look for raw JSON
            json_match = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {text[:200]}...")
            return None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using LLM
        Returns: {"PERSON": [...], "ORG": [...], ...}
        """
        prompt = self.prompts.entity_extraction.format(text=text[:4000])  # Limit text length
        
        system_prompt = "You are a precise entity extraction system. Return only valid JSON."
        
        response = self._call_llm(prompt, system_prompt)
        result = self._extract_json(response)
        
        if result and isinstance(result, dict) and 'entities' in result:
            return result['entities']
        
        # Fallback to empty dict
        return {}
    
    def extract_relations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract relationships from text using LLM
        Returns: [{"subject": "X", "relation": "Y", "object": "Z"}, ...]
        """
        prompt = self.prompts.relation_extraction.format(text=text[:4000])
        
        system_prompt = "You are a precise relation extraction system. Return only valid JSON array."
        
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
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the text"""
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text[:4000]}"
        return self._call_llm(prompt)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using Ollama"""
        try:
            response = ollama.embeddings(
                model=get_config().chroma.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

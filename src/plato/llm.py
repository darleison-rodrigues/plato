class OllamaClient:
    """
    Client for interacting with Ollama LLMs with proper error handling,
    resource management, and document context tracking.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.config = get_config().ollama
            self.prompts = get_config().prompts
            # Validate required prompts
            required_prompts = ['entity_extraction', 'relation_extraction']
            if not all(p in self.prompts for p in required_prompts):
                raise ValueError(f"Missing required prompts in config: {required_prompts}")
        except (AttributeError, ValueError) as e:
            self.logger.error(f"Configuration for OllamaClient is invalid: {e}")
            raise

        self.max_retries = self.config.max_retries
        self.timeout = self.config.timeout

    def __enter__(self):
        """Enter the context manager."""
        self.logger.info("OllamaClient context opened.")
        # In the future, this could initialize a persistent connection
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, cleaning up resources."""
        self.logger.info("OllamaClient context closed.")
        # Future cleanup logic (e.g., closing connections) would go here

    def _call_llm(self, prompt: str, model_name: str, system_prompt: Optional[str] = None) -> str:
        """Make a call to Ollama LLM with retry logic."""
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    options={'temperature': 0.1} # Lower temp for deterministic extraction
                )
                return response['message']['content']
            except Exception as e:
                self.logger.warning(f"Ollama API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to call Ollama after {self.max_retries} attempts.") from e
                time.sleep(2 ** attempt)
        return "" 

    def _extract_json(self, text: str) -> Optional[Any]:
        """Extract a JSON object or array from a string."""
        # This regex is more robust and finds JSON within markdown code blocks
        match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
        if match:
            json_str = match.group(1)
        else:
            # Fallback for raw JSON, less reliable
            json_str = text[text.find('{'):text.rfind('}')+1] or text[text.find('['):text.rfind(']')+1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to decode JSON from LLM response: {text[:200]}...")
            return None
    
    def extract_entities(self, text: str, document_id: str) -> Dict[str, Any]:
        """
        Extract named entities from text, tagged with document_id.
        """
        if not text:
            return {}

        prompt_template = self.prompts['entity_extraction']
        prompt = prompt_template.format(text=text)
        
        system_prompt = "You are a precise entity extraction system. Return only a single valid JSON object."
        
        try:
            response = self._call_llm(prompt, self.config.extraction_model, system_prompt)
            result = self._extract_json(response)
            
            if isinstance(result, dict) and 'entities' in result:
                # Attach document context to the result
                return {'document_id': document_id, 'entities': result['entities']}
            return {'document_id': document_id, 'entities': {}}
        except Exception as e:
            self.logger.error(f"Entity extraction failed for doc {document_id}: {e}")
            return {'document_id': document_id, 'entities': {}}

    def extract_relations(self, text: str, document_id: str) -> Dict[str, Any]:
        """
        Extract relationships from text, tagged with document_id.
        """
        if not text:
            return {}

        prompt = self.prompts['relation_extraction'].format(text=text)
        system_prompt = "You are a precise relation extraction system. Return only a single valid JSON array of objects."
        
        try:
            response = self._call_llm(prompt, self.config.extraction_model, system_prompt)
            result = self._extract_json(response)
            
            if isinstance(result, list):
                return {'document_id': document_id, 'relations': result}
            return {'document_id': document_id, 'relations': []}
        except Exception as e:
            self.logger.error(f"Relation extraction failed for doc {document_id}: {e}")
            return {'document_id': document_id, 'relations': []}
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using the configured embedding model."""
        if not text:
            return []
        try:
            response = ollama.embeddings(
                model=self.config.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return []
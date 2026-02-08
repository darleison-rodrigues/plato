import logging
from pathlib import Path
from typing import List, Dict, Any
from .base import BaseAgent
from plato.parser import PDFProcessor

logger = logging.getLogger(__name__)

class BuilderAgent(BaseAgent):
    """
    The 'Worker': Executes the chosen workflow to build the final context.
    Uses the 3B model for higher quality extraction.
    """
    
    def __init__(self):
        # Use 3B model for actual work
        super().__init__(model_override="llama3.2:3b")
        self.processor = PDFProcessor()

    def run(self, files: List[str], workflow_type: str, task_description: str) -> str:
        """
        Executes the build process.
        Returns the path to the generated context file.
        """
        logger.info(f"Building context for {len(files)} files via {workflow_type}...")
        
        results = []
        
        # Phase 1: Extract relevant info from each file
        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue
                
            logger.info(f"Processing {path.name}...")
            # We treat extraction as a sub-task
            # In a full implementation, we'd use RAG here to find relevant chunks
            # For this MVP, we take the first 4k chars as a "summary extract"
            try:
                text = self.processor._extract_text_docling(path)[:4000]
                
                # Ask 3B model to analyze this specific file based on task
                prompt = f"""
                Task: {task_description}
                
                Extract relevant information from this text to help with the task.
                Focus on key facts, dates, and concepts.
                
                TEXT:
                {text}
                """
                
                analysis = self.generate_json(prompt, system="You are a research analyst.")
                results.append(f"## {path.name}\n{analysis.get('response', json.dumps(analysis, indent=2))}")
                
            except Exception as e:
                logger.error(f"Failed to process {path.name}: {e}")

        # Phase 2: Synthesize
        logger.info("Synthesizing final context...")
        combined_text = "\n\n".join(results)
        
        synthesis_prompt = f"""
        Task: {task_description}
        
        Synthesize the following analyses into a cohesive research context.
        Use Markdown formatting.
        
        ANALYSES:
        {combined_text}
        
        Output valid Markdown.
        """
        
        # We use a raw generation here, not JSON, because we want Markdown
        if self.client:
             response = self.client.generate(
                model=self.model,
                prompt=synthesis_prompt,
                stream=False
            )
             final_markdown = response['response']
        else:
            final_markdown = "Error: Client not connected."

        # Save to file
        output_path = Path("output/context.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"# Context: {task_description}\n\n" + final_markdown)
            
        return str(output_path)

import logging
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from .base import BaseAgent

logger = logging.getLogger(__name__)

@dataclass
class WorkflowSuggestion:
    id: int
    title: str
    description: str
    files: List[str]
    type: str  # "comparison", "timeline", "deep_dive"

class SuggesterAgent(BaseAgent):
    """
    The 'Librarian': Analyzes metadata from the Scanner to propose
    actionable workflows.
    """
    
    def __init__(self):
        super().__init__(model_override="llama3.2:1b")

    def run(self, scan_results: List[Dict[str, Any]]) -> List[WorkflowSuggestion]:
        """
        Takes a list of scan metadata and returns specific workflow suggestions.
        """
        logger.info(f"Analyzing {len(scan_results)} documents for suggestions...")
        
        if not scan_results:
            return []

        # Prepare context for the LLM
        # We simplify the input to save tokens
        docs_summary = []
        for i, doc in enumerate(scan_results):
            docs_summary.append(
                f"Doc {i}: {doc.get('title', 'Unknown')} "
                f"(Keywords: {', '.join(doc.get('keywords', [])[:3])})"
            )
        
        docs_text = "\n".join(docs_summary)

        prompt = f"""
        You are a research librarian. 
        Analyze these documents and suggest 3 logical workflows to help a researcher.
        
        Types of workflows:
        - "comparison": Compare similar papers.
        - "timeline": Extract dates/history from related papers.
        - "deep_dive": Detailed extraction from a single complex paper.

        DOCUMENTS:
        {docs_text}

        Respond ONLY with a JSON list of objects:
        [
            {{
                "title": "Compare Agent Frameworks",
                "description": "Create a table comparing features of Doc 0 and Doc 3",
                "file_indices": [0, 3],
                "type": "comparison"
            }}
        ]
        """

        suggestions_data = self.generate_json(prompt, system="You are a helpful research assistant.")
        
        suggestions = []
        if isinstance(suggestions_data, list):
            for i, item in enumerate(suggestions_data):
                try:
                    # Map indices back to filenames
                    files = [scan_results[idx]['file_path'] for idx in item.get('file_indices', []) if idx < len(scan_results)]
                    
                    if files:
                        suggestions.append(WorkflowSuggestion(
                            id=i + 1,
                            title=item.get('title', 'Untitled'),
                            description=item.get('description', ''),
                            files=files,
                            type=item.get('type', 'deep_dive')
                        ))
                except Exception as e:
                    logger.warning(f"Skipping malformed suggestion: {e}")

        return suggestions

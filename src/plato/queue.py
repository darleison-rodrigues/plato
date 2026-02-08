import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from .config import get_config

class QueueManager:
    """Simple JSON-based persistence for reading queue"""
    
    def __init__(self):
        self.config = get_config()
        self.queue_file = Path(self.config.pipeline.output_dir) / "reading_queue.json"
        self._ensure_queue_file()

    def _ensure_queue_file(self):
        if not self.queue_file.exists():
            self._save_queue([])

    def _load_queue(self) -> List[Dict]:
        try:
            with open(self.queue_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_queue(self, queue: List[Dict]):
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.queue_file, 'w') as f:
            json.dump(queue, f, indent=2)

    def add(self, file_path: str, priority: str = "medium", tags: List[str] = None):
        queue = self._load_queue()
        entry = {
            "path": str(file_path),
            "priority": priority,
            "tags": tags or [],
            "added_at": datetime.now().isoformat(),
            "status": "pending"
        }
        # Check duplicates
        if not any(item['path'] == entry['path'] for item in queue):
            queue.append(entry)
            self._save_queue(queue)
            return True
        return False

    def list(self) -> List[Dict]:
        return self._load_queue()

    def get_next(self) -> Optional[Dict]:
        queue = self._load_queue()
        pending = [q for q in queue if q['status'] == 'pending']
        if not pending:
            return None
        # Sort by priority (high > medium > low)
        priority_map = {"high": 0, "medium": 1, "low": 2}
        pending.sort(key=lambda x: priority_map.get(x.get('priority', 'medium'), 3))
        return pending[0]

    def remove(self, file_path: str):
        queue = self._load_queue()
        queue = [q for q in queue if q['path'] != str(file_path)]
        self._save_queue(queue)

    def clear(self):
        self._save_queue([])

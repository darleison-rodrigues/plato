import json
import jsonschema
from jinja2 import FileSystemLoader, select_autoescape
from jinja2.sandbox import SandboxedEnvironment
from pathlib import Path
from typing import Dict, Any, List, Optional

class TemplateEngine:
    """
    Renders Jinja2 templates for context generation using a secure sandbox.
    Supports JSON validation of LLM responses.
    """
    def __init__(self, template_dir: str = "~/.config/plato/templates", schema_dir: Optional[str] = None):
        self.template_dir = Path(template_dir).expanduser()
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        if schema_dir:
            self.schema_dir = Path(schema_dir).expanduser()
        else:
            # Default to a internal schemas directory if not provided
            self.schema_dir = Path(__file__).parent.parent / "templates" / "schemas"
            
        self.env = SandboxedEnvironment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        # Further harden sandbox by clearing builtins and filters
        self.env.globals.clear()
        self.env.filters.clear()
        
        # Whitelist only safe, essential functions
        self.env.globals.update({
            'range': range,
            'len': len,
            'enumerate': enumerate,
            'zip': zip,
            'round': round,
            'abs': abs,
        })

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Renders a template with the provided context safely.
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            raise TemplateRenderError(f"Failed to render template {template_name}: {e}")

    def validate_json(self, content: str, schema_name: str) -> Dict[str, Any]:
        """
        Parses content as JSON and validates it against a schema.
        """
        try:
            data = json.loads(content)
            schema_path = self.schema_dir / f"{schema_name}.json"
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema {schema_name}.json not found in {self.schema_dir}")
                
            with open(schema_path, "r") as f:
                schema = json.load(f)
                
            jsonschema.validate(instance=data, schema=schema)
            return data
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
        except jsonschema.ValidationError as e:
            raise ValueError(f"JSON validation failed: {e.message}")
        except Exception as e:
            # Keep generic catch for unexpected logic errors but log them
            raise TemplateRenderError(f"Unexpected error during JSON validation: {e}")

    def list_templates(self) -> List[str]:
        """
        Lists available templates.
        """
        return self.env.list_templates()

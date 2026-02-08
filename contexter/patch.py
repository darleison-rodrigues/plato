import sys
import pydantic.v1.fields
from pydantic.v1.errors import ConfigError

# Store original method to avoid recursion
_original_infer = pydantic.v1.fields.ModelField.infer

@classmethod
def patched_infer(cls, *, name, value, annotation, class_validators, config):
    try:
        return _original_infer(name=name, value=value, annotation=annotation, 
                             class_validators=class_validators, config=config)
    except ConfigError:
        # Fallback for Python 3.14 + Pydantic V1 inference failure
        # This specifically handles the 'chroma_server_nofile' issue in ChromaDB settings
        from pydantic.v1.fields import ModelField
        return ModelField(
            name=name,
            type_=str,  # Safe default fallback
            class_validators=class_validators,
            model_config=config,
            default=value,
            required=False
        )

# Apply the patch
pydantic.v1.fields.ModelField.infer = patched_infer

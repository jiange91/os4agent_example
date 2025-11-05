import json
from pydantic import BaseModel

def to_serializable(obj):
    # Pydantic BaseModel
    if isinstance(obj, BaseModel):
        data = obj.model_dump()
        return to_serializable(data)

    # Dict
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    # List / Tuple / Set
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(item) for item in obj]

    # Basic JSON-safe types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fallback — try to cast to str
    return str(obj)


def to_json_str(obj, **json_kwargs):
    serializable = to_serializable(obj)
    return json.dumps(serializable, **json_kwargs)

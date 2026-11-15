import math
from typing import Any, Dict, List

def format_inr(value: float) -> str:
    """Format a numeric value as Indian Rupee string (e.g. ₹ 1,23,456.78)."""
    if value is None or math.isnan(value):
        return "₹ —"
    
    # Handle negative
    prefix = "-₹ " if value < 0 else "₹ "
    val_abs = abs(value)
    
    # Format according to Indian numbering system if >= 1000
    integer_part = int(val_abs)
    decimal_part = f"{val_abs - integer_part:.2f}"[1:]  # e.g. '.50'
    
    s = str(integer_part)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted = ",".join(groups) + "," + last3
        
    return f"{prefix}{formatted}{decimal_part}"

def sanitize_floats(obj: Any) -> Any:
    """Recursively replace NaN, Infinity, and -Infinity with None for valid JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 4)
    elif isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj

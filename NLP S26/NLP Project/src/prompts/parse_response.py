import re

def extract_choice(raw: str):
    raw = raw.strip().upper()
    matches = set(re.findall(r'\b([ABCD])\b', raw))
    return matches.pop() if len(matches) == 1 else None

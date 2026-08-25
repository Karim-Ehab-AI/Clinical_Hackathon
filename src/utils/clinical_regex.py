import re
from typing import Optional, Tuple, Dict

# NICE recommendation pattern matching single IDs (e.g. 1.5.7) or ranges (e.g. 1.5.7 to 1.5.9)
NICE_RECOMMENDATION_PATTERN = re.compile(
    r"(?:Recommendation\s+)?(\d+\.\d+\.\d+(?:\s+to\s+\d+\.\d+\.\d+)?)",
    re.IGNORECASE
)

# ESC Class of Recommendation pattern (Class I, Class IIa, Class IIb, Class III)
ESC_CLASS_PATTERN = re.compile(
    r"\bClass\s+(I{1,3}[ab]?)\b",
    re.IGNORECASE
)

# ESC Level of Evidence pattern (Level A, Level B, Level C)
ESC_LEVEL_PATTERN = re.compile(
    r"\bLevel\s+([ABC])\b",
    re.IGNORECASE
)

# Standalone page number block regex (e.g., "Page 12 of 45", "Page 12", "12 / 45", or standalone digits at header/footer)
PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+|\d+)$",
    re.IGNORECASE
)

# Known boilerplate patterns (copyright notices, download banners, etc.)
BOILERPLATE_PATTERNS = [
    re.compile(r"©\s*NICE\s*\d{4}", re.IGNORECASE),
    re.compile(r"All\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"Downloaded\s+from\s+.*", re.IGNORECASE),
    re.compile(r"European\s+Society\s+of\s+Cardiology", re.IGNORECASE),
    re.compile(r"NICE\s+guideline", re.IGNORECASE),
]


def extract_nice_recommendation_id(text: str) -> Optional[str]:
    """Extract NICE recommendation ID or range (e.g. '1.5.7' or '1.5.7 to 1.5.9') from text."""
    match = NICE_RECOMMENDATION_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_esc_metadata(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract ESC Class of Recommendation (e.g. 'IIa') and Level of Evidence (e.g. 'B') from text."""
    rec_class = None
    evidence_level = None

    class_match = ESC_CLASS_PATTERN.search(text)
    if class_match:
        rec_class = class_match.group(1).strip()

    level_match = ESC_LEVEL_PATTERN.search(text)
    if level_match:
        evidence_level = level_match.group(1).strip()

    return rec_class, evidence_level


def is_page_number_text(text: str) -> bool:
    """Check if string is a standalone page number element."""
    return bool(PAGE_NUMBER_PATTERN.match(text.strip()))


def is_boilerplate_text(text: str) -> bool:
    """Check if string matches standard page furniture boilerplate."""
    clean_text = text.strip()
    for pattern in BOILERPLATE_PATTERNS:
        if pattern.search(clean_text):
            return True
    return False

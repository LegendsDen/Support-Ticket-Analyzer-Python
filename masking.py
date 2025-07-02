import re
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_trf", disable=["lemmatizer", "textcat","parser"])

# OPTIMIZATION 2: Pre-compile regex patterns for better performance
# Previously: re.sub(pattern, ...) compiled regex patterns on every function call
# Now: Patterns compiled once at module load, reused for all masking operations
COMPILED_PATTERNS = {
    'EMAIL': re.compile(r'[\w\.-]+@[\w\.-]+'),
    'PHONE': re.compile(r'\+?\d[\d\s\-]{7,}'),
    'URL': re.compile(r'https?://[^\s]+'),
    'ORG': re.compile(r'\bsprinklr\b', re.IGNORECASE),
    'TICKET': re.compile(r'#\d+'),
    'PARTNER_ID': re.compile(r'Partner ID:[^.]*', re.IGNORECASE)
}

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def mask_custom_patterns(text):
    # Using pre-compiled patterns instead of re.sub(pattern, ...)
    for label, pattern in COMPILED_PATTERNS.items():
        text = pattern.sub(f"[MASKED_{label}]", text)
    return text

def mask_pii(text):
    # Using the single nlp model loaded at module level
    doc = nlp(text)
    entities_to_mask = ['PERSON', 'ORG', 'GPE', 'EMAIL', 'URL', 'LOC', 'PHONE', 'DATE', 'TIME', 'ID']
    for ent in reversed(doc.ents):
        if ent.label_ in entities_to_mask:
            text = text[:ent.start_char] + f"[MASKED_{ent.label_}]" + text[ent.end_char:]
    return text

def clean_support_noise(text):
    lines = text.splitlines()
    cleaned_lines = []
    noise_patterns = ["survey", "unsubscribe", "automated response",
                     "ticket #", "click here", "get outlook", "download app", "calendar invite"]
    header_patterns = ["sent:", "subject:", "cc:", "to:"]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_lower = line.lower()
        
        # Skip noise lines
        if any(x in line_lower for x in noise_patterns):
            continue
            
        # Skip short header lines
        if len(line.split()) <= 3 and any(x in line_lower for x in header_patterns):
            continue
            
        # Remove words containing "masked"
        words = [word for word in line.split() if "masked" not in word.lower()]
        if words:
            cleaned_lines.append(' '.join(words))
    
    return ' '.join(cleaned_lines)

def full_masking_pipeline(html_message):
    plain_text = extract_text_from_html(html_message)
    masked = mask_pii(plain_text)
    masked = mask_custom_patterns(masked)
    return clean_support_noise(masked)

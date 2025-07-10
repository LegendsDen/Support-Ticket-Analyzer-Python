import re
import spacy
from bs4 import BeautifulSoup

# Load spaCy model (transformer-based)
nlp = spacy.load("en_core_web_trf", disable=["lemmatizer", "textcat", "parser"])

# Pre-compiled regex patterns
COMPILED_PATTERNS = {
    'EMAIL': re.compile(r'[\w\.-]+@[\w\.-]+'),
    'PHONE': re.compile(r'\+?\d[\d\s\-]{7,}'),
    'URL': re.compile(r'https?://[^\s]+'),
    'DOMAIN': re.compile(r'\b(?:www\.)?[\w\.-]*\.com\b', re.IGNORECASE),  
    'ORG': re.compile(r'\bsprinklr\b', re.IGNORECASE),
    'TICKET': re.compile(r'#\d+'),
    'PARTNER_ID': re.compile(r'(?:Partner\s*ID|partnerId)\s*:[-\s]*\d+', re.IGNORECASE)}

# Entities to mask with NER
ENTITIES_TO_MASK = {'PERSON', 'ORG', 'GPE', 'EMAIL', 'URL', 'LOC', 'PHONE', 'DATE', 'TIME', 'ID'}

# Regex to remove all [MASKED_LABEL] placeholders
MASKED_TAG_PATTERN = re.compile(r'\[MASKED_[A-Z_]+\]')

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def mask_custom_patterns(text):
    for label, pattern in COMPILED_PATTERNS.items():
        text = pattern.sub(f"[MASKED_{label}]", text)
    return text

def clean_support_noise(text):
    lines = text.splitlines()
    cleaned_lines = []
    noise_patterns = [
        "survey", "unsubscribe", "automated response", "ticket #", "click here",
        "get outlook", "download app", "calendar invite"
    ]
    header_patterns = ["sent:", "subject:", "cc:", "to:"]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line_lower = line.lower()
        if any(x in line_lower for x in noise_patterns):
            continue
        if len(line.split()) <= 3 and any(x in line_lower for x in header_patterns):
            continue

        words = [word for word in line.split() if "masked" not in word.lower()]
        if words:
            cleaned_lines.append(' '.join(words))

    return ' '.join(cleaned_lines)

def mask_pii_ner(texts):
    docs = list(nlp.pipe(texts, batch_size=16))
    masked = []
    for doc in docs:
        text = doc.text
        for ent in reversed(doc.ents):
            if ent.label_ in ENTITIES_TO_MASK:
                text = text[:ent.start_char] + f"[MASKED_{ent.label_}]" + text[ent.end_char:]

        # ✨ Remove all [MASKED_...] placeholders from the final output
        text = MASKED_TAG_PATTERN.sub("", text)
        masked.append(text.strip())
    return masked

# Final pipeline using correct order: HTML -> regex -> clean -> NER -> clean MASKED_*
def full_masking_pipeline_batch(html_messages):
    plain_texts = [extract_text_from_html(msg) for msg in html_messages]
    regex_masked = [mask_custom_patterns(text) for text in plain_texts]
    cleaned = [clean_support_noise(text) for text in regex_masked]
    return mask_pii_ner(cleaned)

def full_masking_pipeline(html_message):
    return full_masking_pipeline_batch([html_message])[0]

# Test
if __name__ == "__main__":
    example = "<html><body><p>Sushant Kumar APPLE INC contact us at dELHI arun  </p></body></html>"
    print(full_masking_pipeline(example))

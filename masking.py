import re
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_trf")

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def mask_custom_patterns(text):
    custom_patterns = {
        'EMAIL': r'[\w\.-]+@[\w\.-]+',
        'PHONE': r'\+?\d[\d\s\-]{7,}',
        'URL': r'https?://[^\s]+',
        'ORG': r'\bsprinklr\b',
        'TICKET': r'#\d+',
        'PARTNER_ID': r'Partner ID:[^.]*'
    }
    for label, pattern in custom_patterns.items():
        flags = re.IGNORECASE if label in ['ORG', 'PARTNER_ID'] else 0
        text = re.sub(pattern, f"[MASKED_{label}]", text, flags=flags)
    return text

def mask_pii(text):
    doc = nlp(text)
    entities_to_mask = ['PERSON', 'ORG', 'GPE', 'EMAIL', 'URL', 'LOC', 'PHONE', 'DATE', 'TIME', 'ID']
    for ent in reversed(doc.ents):
        if ent.label_ in entities_to_mask:
            text = text[:ent.start_char] + f"[MASKED_{ent.label_}]" + text[ent.end_char:]
    return text

def clean_support_noise(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if any(x in line.lower() for x in ["survey", "unsubscribe", "automated response",
                                            "ticket #", "click here", "get outlook", "download app", "calendar invite"]):
            continue
        if len(line.split()) <= 3 and any(x in line.lower() for x in ["sent:", "subject:", "cc:", "to:"]):
            continue
        words = [word for word in line.split() if "masked" not in word.lower()]
        if words:
            cleaned_lines.append(' '.join(words))
    return ' '.join(cleaned_lines)

def full_masking_pipeline(html_message):
    plain_text = extract_text_from_html(html_message)
    masked = mask_pii(plain_text)
    masked = mask_custom_patterns(masked)
    return clean_support_noise(masked)

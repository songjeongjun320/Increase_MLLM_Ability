"""
Text Utilities - Text Processing and Language Detection
======================================================

This module provides utilities for text processing, cleaning,
language detection, and other text-related operations used
throughout the ToW architecture system.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter

logger = logging.getLogger(__name__)


def clean_text(
    text: str,
    normalize_whitespace: bool = True,
    remove_special_chars: bool = False,
    preserve_punctuation: bool = True,
    max_length: Optional[int] = None
) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text to clean
        normalize_whitespace: Whether to normalize whitespace
        remove_special_chars: Whether to remove special characters
        preserve_punctuation: Whether to preserve punctuation marks
        max_length: Maximum length of output text
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    cleaned = text
    
    # Normalize whitespace
    if normalize_whitespace:
        # Replace multiple whitespace with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
    
    # Remove special characters if requested
    if remove_special_chars:
        if preserve_punctuation:
            # Keep letters, numbers, whitespace, and basic punctuation
            cleaned = re.sub(r'[^\w\s.,!?;:\-\'\"()[\]{}]', '', cleaned)
        else:
            # Keep only letters, numbers, and whitespace
            cleaned = re.sub(r'[^\w\s]', '', cleaned)
    
    # Truncate if max_length specified
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    
    return cleaned


def detect_language(text: str, confidence_threshold: float = 0.7) -> Dict[str, Union[str, float]]:
    """
    Detect the primary language of input text using heuristic methods.
    
    Args:
        text: Input text for language detection
        confidence_threshold: Minimum confidence for language detection
        
    Returns:
        Dictionary with detected language and confidence score
    """
    if not text or len(text.strip()) < 3:
        return {"language": "unknown", "confidence": 0.0}
    
    text_clean = text.lower().strip()
    
    # Character-based detection patterns
    language_scores = {
        "en": 0.0,
        "ko": 0.0,
        "zh": 0.0,
        "ja": 0.0,
        "es": 0.0,
        "fr": 0.0,
        "de": 0.0,
        "it": 0.0,
        "pt": 0.0,
        "ru": 0.0,
        "ar": 0.0
    }
    
    # Korean detection (Hangul)
    korean_chars = sum(1 for char in text if 0xAC00 <= ord(char) <= 0xD7AF)
    korean_jamo = sum(1 for char in text if 0x1100 <= ord(char) <= 0x11FF or 
                                           0x3130 <= ord(char) <= 0x318F)
    korean_total = korean_chars + korean_jamo
    if korean_total > 0:
        language_scores["ko"] = korean_total / len(text)
    
    # Chinese detection (CJK Unified Ideographs)
    chinese_chars = sum(1 for char in text if 0x4E00 <= ord(char) <= 0x9FFF)
    if chinese_chars > 0:
        language_scores["zh"] = chinese_chars / len(text)
    
    # Japanese detection (Hiragana, Katakana, some Kanji)
    hiragana = sum(1 for char in text if 0x3040 <= ord(char) <= 0x309F)
    katakana = sum(1 for char in text if 0x30A0 <= ord(char) <= 0x30FF)
    japanese_total = hiragana + katakana
    if japanese_total > 0:
        language_scores["ja"] = japanese_total / len(text)
        # Boost if mixed with some Chinese characters (Kanji usage)
        if chinese_chars > 0 and japanese_total > chinese_chars:
            language_scores["ja"] += 0.2
    
    # Cyrillic detection (Russian)
    cyrillic_chars = sum(1 for char in text if 0x0400 <= ord(char) <= 0x04FF)
    if cyrillic_chars > 0:
        language_scores["ru"] = cyrillic_chars / len(text)
    
    # Arabic detection
    arabic_chars = sum(1 for char in text if 0x0600 <= ord(char) <= 0x06FF)
    if arabic_chars > 0:
        language_scores["ar"] = arabic_chars / len(text)
    
    # Latin-based language detection (using common words and patterns)
    if max(language_scores.values()) < 0.1:  # Likely Latin script
        language_scores.update(_detect_latin_languages(text_clean))
    
    # Find language with highest score
    detected_lang = max(language_scores, key=language_scores.get)
    confidence = language_scores[detected_lang]
    
    # Apply confidence threshold
    if confidence < confidence_threshold:
        detected_lang = "unknown"
        confidence = 0.0
    
    return {
        "language": detected_lang,
        "confidence": confidence,
        "scores": language_scores
    }


def _detect_latin_languages(text: str) -> Dict[str, float]:
    """Detect Latin-based languages using common words and patterns"""
    language_indicators = {
        "en": [
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"
        ],
        "es": [
            "el", "la", "de", "que", "y", "a", "en", "un", "es", "se",
            "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para"
        ],
        "fr": [
            "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir",
            "que", "pour", "dans", "ce", "son", "une", "sur", "avec", "ne", "se"
        ],
        "de": [
            "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
            "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als"
        ],
        "it": [
            "il", "di", "che", "e", "la", "per", "un", "in", "con", "non",
            "a", "da", "su", "del", "al", "le", "si", "dei", "come", "più"
        ],
        "pt": [
            "o", "de", "a", "e", "do", "da", "em", "um", "para", "é",
            "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as"
        ]
    }
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return {lang: 0.0 for lang in language_indicators}
    
    language_scores = {}
    for lang, indicators in language_indicators.items():
        matches = sum(1 for word in words if word in indicators)
        language_scores[lang] = matches / len(words) if words else 0.0
    
    return language_scores


def estimate_tokens(
    text: str,
    method: str = "heuristic",
    language: str = "en"
) -> int:
    """
    Estimate token count for text.
    
    Args:
        text: Input text
        method: Estimation method ("heuristic", "word_based", "char_based")
        language: Language code for language-specific estimation
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    if method == "heuristic":
        # Language-specific heuristics
        if language in ["ko", "ja", "zh"]:
            # Asian languages: roughly 1 token per character for CJK
            asian_chars = sum(1 for char in text if ord(char) > 127)
            latin_chars = len(text) - asian_chars
            return asian_chars + (latin_chars // 4)
        else:
            # Latin-based languages: roughly 1 token per 4 characters
            return len(text) // 4
    
    elif method == "word_based":
        # Word-based estimation (roughly 0.75 tokens per word)
        words = len(text.split())
        return int(words * 0.75)
    
    elif method == "char_based":
        # Character-based estimation
        return len(text) // 4
    
    else:
        raise ValueError(f"Unknown estimation method: {method}")


def extract_sentences(
    text: str,
    min_length: int = 10,
    max_length: int = 1000,
    preserve_structure: bool = True
) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        min_length: Minimum sentence length
        max_length: Maximum sentence length
        preserve_structure: Whether to preserve original structure
        
    Returns:
        List of extracted sentences
    """
    if not text:
        return []
    
    # Basic sentence splitting patterns
    sentence_endings = r'[.!?]+\s+'
    sentences = re.split(sentence_endings, text.strip())
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if min_length <= len(sentence) <= max_length:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode text for consistent processing.
    
    Args:
        text: Input text with potential unicode issues
        
    Returns:
        Normalized unicode text
    """
    if not text:
        return text
    
    # Common unicode normalizations
    import unicodedata
    
    # Normalize to NFC form (canonical composition)
    normalized = unicodedata.normalize('NFC', text)
    
    # Replace common problematic characters
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Horizontal ellipsis
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized


def extract_keywords(
    text: str,
    num_keywords: int = 10,
    min_word_length: int = 3,
    language: str = "en"
) -> List[Tuple[str, int]]:
    """
    Extract keywords from text using frequency analysis.
    
    Args:
        text: Input text
        num_keywords: Number of keywords to extract
        min_word_length: Minimum word length for keywords
        language: Language code for stopword removal
        
    Returns:
        List of (keyword, frequency) tuples
    """
    if not text:
        return []
    
    # Basic stopwords (could be expanded with proper NLP libraries)
    stopwords = {
        "en": {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
               "of", "with", "by", "from", "up", "about", "into", "through", "during",
               "before", "after", "above", "below", "between", "among", "is", "was",
               "are", "were", "be", "been", "being", "have", "has", "had", "do", "does",
               "did", "will", "would", "could", "should", "may", "might", "must", "can"},
        "ko": {"이", "그", "저", "것", "수", "있", "없", "하", "되", "된", "될", "는", "은", "을", "를"},
        "zh": {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "个", "上", "也"},
        "ja": {"の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "する"}
    }
    
    # Extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter words
    stop_set = stopwords.get(language, stopwords["en"])
    filtered_words = [
        word for word in words 
        if len(word) >= min_word_length and word not in stop_set
    ]
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    
    # Return top keywords
    return word_counts.most_common(num_keywords)


def split_text_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    preserve_sentences: bool = True
) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Input text to split
        chunk_size: Target size of each chunk
        overlap: Overlap between chunks
        preserve_sentences: Whether to preserve sentence boundaries
        
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return [text] if text else []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate chunk end
        end = start + chunk_size
        
        if preserve_sentences and end < len(text):
            # Find sentence boundary within reasonable range
            search_start = max(start + chunk_size - 200, start + chunk_size // 2)
            search_text = text[search_start:end + 100]
            
            # Look for sentence endings
            sentence_end = -1
            for pattern in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                pos = search_text.rfind(pattern)
                if pos > 0:
                    sentence_end = search_start + pos + 1
                    break
            
            if sentence_end > start:
                end = sentence_end
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Calculate next start with overlap
        if end >= len(text):
            break
        
        start = max(start + 1, end - overlap)
    
    return chunks


def measure_text_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """
    Measure similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method ("jaccard", "overlap", "cosine")
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    # Tokenize texts
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    if method == "jaccard":
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    elif method == "overlap":
        # Overlap coefficient
        intersection = len(words1.intersection(words2))
        min_size = min(len(words1), len(words2))
        return intersection / min_size if min_size > 0 else 0.0
    
    elif method == "cosine":
        # Simple cosine similarity based on word presence
        intersection = len(words1.intersection(words2))
        magnitude = (len(words1) * len(words2)) ** 0.5
        return intersection / magnitude if magnitude > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def validate_text_quality(
    text: str,
    min_length: int = 10,
    max_length: int = 10000,
    check_encoding: bool = True,
    check_language_consistency: bool = True
) -> Dict[str, Any]:
    """
    Validate text quality and detect potential issues.
    
    Args:
        text: Input text to validate
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        check_encoding: Whether to check for encoding issues
        check_language_consistency: Whether to check language consistency
        
    Returns:
        Dictionary with validation results
    """
    if not text:
        return {
            "valid": False,
            "issues": ["Empty text"],
            "length": 0,
            "encoding_ok": True,
            "language_consistent": True
        }
    
    issues = []
    
    # Length validation
    text_length = len(text)
    if text_length < min_length:
        issues.append(f"Text too short ({text_length} < {min_length})")
    elif text_length > max_length:
        issues.append(f"Text too long ({text_length} > {max_length})")
    
    # Encoding validation
    encoding_ok = True
    if check_encoding:
        try:
            # Check for encoding issues
            text.encode('utf-8').decode('utf-8')
            
            # Check for common encoding problems
            suspicious_chars = ['\ufffd', '\x00']  # Replacement character, null byte
            for char in suspicious_chars:
                if char in text:
                    issues.append("Potential encoding issues detected")
                    encoding_ok = False
                    break
        except UnicodeError:
            issues.append("Text encoding error")
            encoding_ok = False
    
    # Language consistency validation
    language_consistent = True
    if check_language_consistency and len(text) > 50:
        # Check for mixed scripts that might indicate inconsistency
        scripts = set()
        for char in text:
            if char.isalpha():
                if 0x0000 <= ord(char) <= 0x007F:  # ASCII
                    scripts.add("latin")
                elif 0x4E00 <= ord(char) <= 0x9FFF:  # CJK
                    scripts.add("cjk")
                elif 0xAC00 <= ord(char) <= 0xD7AF:  # Hangul
                    scripts.add("hangul")
                elif 0x0400 <= ord(char) <= 0x04FF:  # Cyrillic
                    scripts.add("cyrillic")
        
        if len(scripts) > 2:  # Allow some mixing, but not too much
            issues.append("Multiple writing systems detected")
            language_consistent = False
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "length": text_length,
        "encoding_ok": encoding_ok,
        "language_consistent": language_consistent,
        "character_count": len(text),
        "word_count": len(text.split()),
        "line_count": text.count('\n') + 1
    }


# --- TOW helpers: enforce English-only inside <ToW> tokens ---

def is_english_text(text: str, ascii_ratio_threshold: float = 0.9) -> bool:
    """Return True if text is predominantly ASCII (proxy for English).

    Args:
        text: Input text
        ascii_ratio_threshold: Minimum ASCII character ratio required
    """
    if not text:
        return True
    total = len(text)
    if total == 0:
        return True
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / total) >= ascii_ratio_threshold


def enforce_english_text(
    text: str,
    allow_punctuation: bool = True,
    collapse_whitespace: bool = True,
    fallback: str = "Contextual reasoning applied."
) -> str:
    """Force text to English-only (ASCII letters/digits and basic punctuation).

    - Removes non-ASCII characters
    - Optionally keeps basic punctuation
    - Collapses whitespace
    - Ensures a non-empty, sensible fallback
    """
    if not text:
        return fallback

    # Keep ASCII letters/digits and selected punctuation; drop others
    if allow_punctuation:
        cleaned = re.sub(r"[^A-Za-z0-9\s\.,!\?;:'\"\-\(\)\[\]\{\}/]", " ", text)
    else:
        cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", text)

    if collapse_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Ensure there is at least one alphabetic character
    if not re.search(r"[A-Za-z]", cleaned):
        return fallback

    # Cap length to a reasonable size for TOW
    if len(cleaned) > 200:
        cleaned = cleaned[:197] + "..."

    return cleaned


def validate_tow_token_format(token: str) -> bool:
    """Validate that token matches <ToW>...</ToW> and inner content is English-only."""
    if not token:
        return False
    m = re.fullmatch(r"\s*<ToW>(.*?)</ToW>\s*", token, flags=re.DOTALL)
    if not m:
        return False
    inner = m.group(1)
    return is_english_text(inner)


def sanitize_tow_token(token: str, fallback: str = "Contextual reasoning applied.") -> str:
    """Sanitize a TOW token to enforce English-only inner content and valid tags.

    - Ensures presence of <ToW>...</ToW>
    - Cleans non-ASCII characters inside
    - Returns a valid token even if input was malformed
    """
    if not token:
        inner = fallback
    else:
        m = re.search(r"<ToW>(.*?)</ToW>", token, flags=re.DOTALL)
        if m:
            inner = m.group(1)
        else:
            # No tags present; treat whole string as inner content
            inner = token

    inner_clean = enforce_english_text(inner, allow_punctuation=True, collapse_whitespace=True, fallback=fallback)
    return f"<ToW>{inner_clean}</ToW>"


def count_tow_tokens(text: str) -> int:
    """Count the number of ToW tokens in text."""
    if not text:
        return 0
    return len(re.findall(r"<ToW>.*?</ToW>", text, flags=re.DOTALL))


def extract_tow_tokens(text: str) -> List[str]:
    """Extract all ToW tokens from text."""
    if not text:
        return []
    return re.findall(r"<ToW>.*?</ToW>", text, flags=re.DOTALL)


def remove_tow_tokens(text: str) -> str:
    """Remove all ToW tokens from text."""
    if not text:
        return text
    return re.sub(r"<ToW>.*?</ToW>", "", text, flags=re.DOTALL)


def get_tow_inner_content(token: str) -> str:
    """Extract inner content from ToW token."""
    if not token:
        return ""
    match = re.search(r"<ToW>(.*?)</ToW>", token, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def validate_english_only_tow(text: str) -> Dict[str, Union[bool, List[str]]]:
    """Validate that all ToW tokens contain English-only content."""
    tokens = extract_tow_tokens(text)
    
    result = {
        "valid": True,
        "total_tokens": len(tokens),
        "valid_tokens": 0,
        "invalid_tokens": []
    }
    
    for token in tokens:
        inner = get_tow_inner_content(token)
        if is_english_text(inner):
            result["valid_tokens"] += 1
        else:
            result["invalid_tokens"].append(token)
            result["valid"] = False
    
    return result
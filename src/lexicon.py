"""High-precision emotion lexicon and counterfactual wording edits.

Emotion substitution is a controlled intervention: it changes emotional
surface forms while leaving situation-bearing content words in place.
Ambiguous tokens (alone, down, bad, hurt) are intentionally excluded.
"""

from __future__ import annotations

import re

# Category order defines the substitution cycle.
CATEGORIES: dict[str, list[str]] = {
    "sad": [
        "sad",
        "unhappy",
        "depressed",
        "miserable",
        "hopeless",
        "heartbroken",
        "lonely",
        "grief",
        "sorrow",
        "crying",
        "gloomy",
        "upset",
        "disappointed",
        "exhausted",
    ],
    "anxiety": [
        "anxious",
        "worried",
        "nervous",
        "stressed",
        "overwhelmed",
        "panicked",
        "uneasy",
        "restless",
        "tense",
        "worriedly",
    ],
    "anger": [
        "angry",
        "furious",
        "annoyed",
        "irritated",
        "frustrated",
        "resentful",
        "enraged",
        "mad",
        "aggravated",
        "bitter",
        "hate",
    ],
    "fear": [
        "scared",
        "afraid",
        "fearful",
        "terrified",
        "frightened",
        "horrified",
        "petrified",
        "spooked",
    ],
    "shame": [
        "ashamed",
        "guilty",
        "embarrassed",
        "worthless",
        "useless",
        "humiliated",
        "insecure",
        "inadequate",
    ],
    "positive": [
        "happy",
        "glad",
        "grateful",
        "hopeful",
        "relieved",
        "proud",
        "excited",
        "joyful",
        "cheerful",
        "optimistic",
    ],
}

# Inflected forms point at a lemma that exists in CATEGORIES.
VARIANTS: dict[str, str] = {
    "sadness": "sad",
    "sadder": "sad",
    "saddest": "sad",
    "depression": "depressed",
    "depressing": "depressed",
    "depressive": "depressed",
    "miserably": "miserable",
    "loneliness": "lonely",
    "lonelier": "lonely",
    "grieving": "grief",
    "sorrowful": "sorrow",
    "cried": "crying",
    "cries": "crying",
    "cry": "crying",
    "tears": "crying",
    "tearful": "crying",
    "anxiety": "anxious",
    "worry": "worried",
    "worries": "worried",
    "worrying": "worried",
    "nervousness": "nervous",
    "stress": "stressed",
    "stressful": "stressed",
    "stressing": "stressed",
    "overwhelm": "overwhelmed",
    "overwhelming": "overwhelmed",
    "panic": "panicked",
    "panicking": "panicked",
    "panicing": "panicked",
    "restlessness": "restless",
    "anger": "angry",
    "angrily": "angry",
    "fury": "furious",
    "annoyance": "annoyed",
    "annoying": "annoyed",
    "irritation": "irritated",
    "frustrating": "frustrated",
    "frustration": "frustrated",
    "resent": "resentful",
    "resentment": "resentful",
    "rage": "enraged",
    "raging": "enraged",
    "madness": "mad",
    "fear": "afraid",
    "fears": "afraid",
    "fearing": "afraid",
    "scare": "scared",
    "scares": "scared",
    "scaring": "scared",
    "scary": "scared",
    "terror": "terrified",
    "fright": "frightened",
    "horror": "horrified",
    "shame": "ashamed",
    "guilt": "guilty",
    "embarrassment": "embarrassed",
    "embarrassing": "embarrassed",
    "humiliation": "humiliated",
    "insecurity": "insecure",
    "happiness": "happy",
    "happier": "happy",
    "gladly": "glad",
    "gratitude": "grateful",
    "hope": "hopeful",
    "hopes": "hopeful",
    "hoping": "hopeful",
    "relief": "relieved",
    "pride": "proud",
    "excitement": "excited",
    "joy": "joyful",
    "cheer": "cheerful",
    "optimism": "optimistic",
    "hopelessness": "hopeless",
    "heartbreak": "heartbroken",
    "upset": "upset",
    "disappointment": "disappointed",
    "disappointing": "disappointed",
    "exhaustion": "exhausted",
    "exhausting": "exhausted",
    "hated": "hate",
    "hates": "hate",
    "hating": "hate",
    "hatred": "hate",
    "terrible": "upset",
    "terribly": "upset",
    "awful": "upset",
    "horrible": "upset",
    "horribly": "upset",
    "struggling": "overwhelmed",
    "struggle": "overwhelmed",
    "struggled": "overwhelmed",
    "suffering": "miserable",
    "suffer": "miserable",
    "suffers": "miserable",
    "painful": "heartbroken",
    "suicidal": "hopeless",
    "despair": "hopeless",
    "desperate": "hopeless",
    "desperation": "hopeless",
    "anxious": "anxious",
}

PHRASES: list[tuple[str, str]] = [
    ("kill myself", "feel furious"),
    ("killing myself", "feeling furious"),
    ("end my life", "escape my anger"),
    ("want to die", "want to scream"),
    ("wanna die", "wanna scream"),
    ("suicidal thoughts", "furious thoughts"),
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "as",
    "of", "to", "in", "on", "for", "with", "at", "by", "from", "into", "about",
    "that", "this", "these", "those", "it", "its", "i", "im", "ive", "id", "me",
    "my", "mine", "you", "your", "yours", "we", "our", "they", "them", "their",
    "he", "she", "his", "her", "was", "were", "is", "are", "am", "be", "been",
    "being", "do", "did", "does", "doing", "have", "has", "had", "having",
    "not", "no", "nor", "too", "very", "just", "also", "than", "then", "there",
    "here", "what", "when", "where", "who", "whom", "which", "why", "how",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "dont", "didnt", "doesnt", "cant", "couldnt", "wont", "isnt", "arent",
    "wasnt", "werent", "havent", "hasnt", "hadnt", "its", "that's", "that's",
    "i'm", "i've", "it's", "don't", "didn't", "can't", "won't", "i'll", "you're",
    "really", "like", "get", "got", "going", "go", "went", "know", "think",
    "thought", "feel", "feels", "feeling", "felt", "something", "anything",
    "everything", "nothing", "someone", "anyone", "lot", "lots", "much",
    "many", "more", "most", "some", "any", "own", "same", "other", "another",
    "up", "out", "over", "again", "still", "already", "even", "back", "now",
    "today", "yesterday", "tomorrow", "time", "times", "day", "days",
}

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")


def _build_lookup() -> tuple[dict[str, str], dict[str, str]]:
    word2cat: dict[str, str] = {}
    word2lemma: dict[str, str] = {}
    for cat, words in CATEGORIES.items():
        for w in words:
            word2cat[w] = cat
            word2lemma[w] = w
    for variant, lemma in VARIANTS.items():
        if lemma not in word2cat and lemma not in word2lemma:
            continue
        # lemma may itself be a listed category word
        if lemma in word2cat:
            word2lemma[variant] = lemma
            word2cat[variant] = word2cat[lemma]
        else:
            # lemma is only reachable if it was registered as a category word
            continue
    return word2cat, word2lemma


WORD2CAT, WORD2LEMMA = _build_lookup()
CAT_NAMES = list(CATEGORIES)


def _replacement(surface: str) -> str:
    lemma = WORD2LEMMA.get(surface, surface)
    cat = WORD2CAT[surface]
    nxt = CAT_NAMES[(CAT_NAMES.index(cat) + 1) % len(CAT_NAMES)]
    pool_src = CATEGORIES[cat]
    try:
        idx = pool_src.index(lemma)
    except ValueError:
        idx = sum(ord(ch) for ch in lemma)
    return CATEGORIES[nxt][idx % len(CATEGORIES[nxt])]


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else "."


def emotion_tokens(text: str) -> set[str]:
    found = set()
    raw = normalize_text(text)
    for phrase, _ in PHRASES:
        if phrase in raw:
            found.add(phrase)
    for tok in TOKEN_RE.findall(raw):
        if tok in WORD2CAT:
            found.add(tok)
    return found


def content_tokens(text: str) -> set[str]:
    raw = normalize_text(text)
    out = set()
    for tok in TOKEN_RE.findall(raw):
        if tok in STOPWORDS or tok in WORD2CAT or len(tok) <= 2 or tok.isdigit():
            continue
        out.add(tok)
    return out


def substitute_emotion(text: str) -> tuple[str, bool]:
    """Replace emotion lexemes with another emotion category. Deterministic."""
    raw = normalize_text(text)
    changed = False
    slots: list[tuple[str, str]] = []
    for i, (phrase, repl) in enumerate(PHRASES):
        if phrase in raw:
            placeholder = f"phrase{i}slot"
            raw = raw.replace(phrase, placeholder)
            slots.append((placeholder, repl))
            changed = True
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^a-z0-9\s]+|\s+", raw)
    out = []
    for tok in tokens:
        low = tok.lower()
        if low in WORD2CAT:
            out.append(_replacement(low))
            changed = True
        else:
            out.append(tok)
    new = "".join(out)
    for placeholder, repl in slots:
        new = new.replace(placeholder, repl)
    new = re.sub(r"\s+", " ", new).strip()
    return (new if new else "."), changed


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union

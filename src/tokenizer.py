import bm25s
import re
import Stemmer

def tokenize(text: str) -> list[str]:
    word = re.compile(r"[A-Za-z0-9_]+")
    camel = re.compile(r"(?<=[a-z0-9])(?=[A-Z])(?=[A-Z][a-z])")
    stop = set(bm25s.stopwords.STOPWORDS_EN)
    stemmer = Stemmer.Stemmer('english')

    tokens: list[str] = []
    for w in word.findall(text):
        parts = camel.sub(" ", w).replace("_", " ").lower().split()
        tokens.extend(parts)
        if len(parts) > 1:
            tokens.append(w.lower())
        tokens = [t for t in tokens if len(t) > 1 and t not in stop]
    return stemmer.stemWords(tokens)

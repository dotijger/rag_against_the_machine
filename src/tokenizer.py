import bm25s
import re
import Stemmer


def tokenize(text: str) -> list[str]:
    """Turn text into stemmed, lowercase BM25 terms.

    Words are runs of letters, digits and underscores. Identifiers are
    split on camelCase boundaries and underscores, and the whole
    identifier is kept too, so KVCacheManager and kv_cache match
    both their parts and their exact name. Single characters and English
    stopwords are dropped, then each term is stemmed with the Snowball
    English stemmer.

    Args:
        text: The text to tokenize; a chunk or a query.

    Returns:
        The stemmed terms in order of appearance; may be empty.
    """
    word = re.compile(r"[A-Za-z0-9_]+")
    camel = re.compile(r"(?<=[a-z0-9])(?=[A-Z][a-z])")
    stop = set(bm25s.stopwords.STOPWORDS_EN)
    stemmer = Stemmer.Stemmer("english")

    tokens: list[str] = []
    for w in word.findall(text):
        parts = camel.sub(" ", w).replace("_", " ").lower().split()
        tokens.extend(parts)
        if len(parts) > 1:
            tokens.append(w.lower())
    tokens = [t for t in tokens if len(t) > 1 and t not in stop]
    stems: list[str] = stemmer.stemWords(tokens)
    return stems

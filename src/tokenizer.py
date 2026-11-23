import bm25s
import re
import Stemmer


def tokenize(text: str) -> list[str]:
    """Turn text into stemmed, lowercase BM25 terms.

    Words are runs of letters, digits and underscores. Each word is split
    on underscores and before a capital that follows a lowercase letter
    or digit and starts a lowercase run. If that gives more than one
    part, the whole word is kept as well, so kv_cache gives kv, cache
    and kv_cache, and CacheManager gives cache, manager and
    cachemanager. A run of capitals stays attached to the next part:
    KVCacheManager gives kvcache, manager and kvcachemanager, and
    HTTPServer is not split at all. Single characters and English
    stopwords are then dropped, and each term is stemmed with the
    Snowball English stemmer.

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

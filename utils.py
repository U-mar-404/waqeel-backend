# utils.py

import json
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download these once (can remove or guard behind a flag after your first run)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Lowercase, remove non-alpha, tokenize, drop stopwords/punctuation, lemmatize.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and tok not in string.punctuation
    ]
    return " ".join(tokens)

def load_glove_embeddings(glove_path: str) -> dict:
    """
    Load GloVe vectors from file into a dict[word] = np.array.
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            vec  = np.asarray(vals[1:], dtype='float32')
            embeddings[word] = vec
    return embeddings

def sentence_embedding(sentence: str, embeddings_dict: dict) -> np.ndarray:
    """
    Compute the average GloVe embedding of all in-vocab words in `sentence`.
    Returns None if no words are in the vocab.
    """
    words = sentence.split()
    vecs  = [embeddings_dict[w] for w in words if w in embeddings_dict]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def build_section_embeddings(
    glove_path: str,
    section_data_path: str,
    output_embedding_path: str
):
    """
    Reads a JSON file of sections (either a dict or a list of objects with "Section"),
    computes a GloVe embedding per section Description, and writes a JSON
    mapping section_id -> embedding (list of floats).
    """
    emb_index = load_glove_embeddings(glove_path)
    raw = json.load(open(section_data_path, encoding='utf-8'))

    if isinstance(raw, list):
        # e.g. CrPC.json: [ { "Section": "8", ... }, ... ]
        section_data = {item["Section"]: item for item in raw}
    else:
        # e.g. ppc_data_final.json: { "324": { ... }, ... }
        section_data = raw

    embeddings = {}
    for sec_id, info in section_data.items():
        desc = info.get("Description", "")
        emb  = sentence_embedding(preprocess_text(desc), emb_index)
        if emb is not None:
            embeddings[sec_id] = emb.tolist()

    with open(output_embedding_path, 'w', encoding='utf-8') as out:
        json.dump(embeddings, out)
    print(f"→ Saved {len(embeddings)} embeddings to {output_embedding_path}")

def search_query_multi(
    query: str,
    glove_path: str,
    embedding_paths: list,
    section_data_paths: list,
    top_k: int = 10
) -> list:
    """
    Runs a GloVe-based nearest‐neighbors search over the union of all embeddings.
    Returns a list of the top_k section‐info dicts (with an added "similarity" float).
    """
    # 1) load GloVe
    emb_index = load_glove_embeddings(glove_path)

    # 2) merge section-data
    combined_data = {}
    for dpath in section_data_paths:
        raw = json.load(open(dpath, encoding='utf-8'))
        if isinstance(raw, list):
            combined_data.update({item["Section"]: item for item in raw})
        else:
            combined_data.update(raw)

    # 3) merge embeddings
    combined_emb = {}
    for epath in embedding_paths:
        combined_emb.update(json.load(open(epath, encoding='utf-8')))

    # 4) embed the query
    q_emb = sentence_embedding(preprocess_text(query), emb_index)
    if q_emb is None:
        return []

    # 5) compute similarities
    sec_ids = list(combined_emb.keys())
    mat     = np.array([combined_emb[s] for s in sec_ids])
    sims    = cosine_similarity([q_emb], mat)[0]

    # 6) select top_k
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for i in idxs:
        sid  = sec_ids[i]
        info = combined_data[sid].copy()
        info["similarity"] = float(sims[i])
        results.append(info)

    return results

if __name__ == "__main__":
    # ─── Adjust these paths to your project layout ────────────────────
    GLOVE         = "data/Law2Vec.200d.txt"
    PPC_JSON      = "data/ppc_data_final.json"
    CRPC_JSON     = "data/CrPC.json"
    PPC_EMB_OUT   = "data/ppc_section_embeddings2.json"
    CRPC_EMB_OUT  = "data/crpc_embeddings.json"

    # Build embeddings
    build_section_embeddings(GLOVE, PPC_JSON,  PPC_EMB_OUT)
    build_section_embeddings(GLOVE, CRPC_JSON, CRPC_EMB_OUT)

    # Quick local test
    hits = search_query_multi(
        "jurisdiction of metropolitan areas",
        glove_path=GLOVE,
        embedding_paths=[PPC_EMB_OUT, CRPC_EMB_OUT],
        section_data_paths=[PPC_JSON, CRPC_JSON],
        top_k=5
    )
    for h in hits:
        sec  = h.get("Section", "")
        name = h.get("Section_name", h.get("offence", ""))[:40]
        print(f"{sec:>3} — {name:40} (sim={h['similarity']:.3f})")

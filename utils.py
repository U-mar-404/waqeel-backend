import json
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ─── NLTK Setup ───────────────────────────────────────────────────────
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ─── Load 200D transformer model ──────────────────────────────────────
model = SentenceTransformer("sdugar/cross-en-de-fr-xlmr-200d-sentence-transformer")
print("✅ Loaded model with dimension:", model.get_sentence_embedding_dimension())

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and tok not in string.punctuation
    ]
    return " ".join(tokens)

def embed_query(query: str) -> np.ndarray:
    query = preprocess_text(query)
    vec = model.encode([query])[0]
    print("✅ Query embedding shape:", vec.shape)
    return vec

def search_query_multi(
    query: str,
    embedding_paths: list,
    section_data_paths: list,
    top_k: int = 10
) -> list:
    combined_data = {}
    for dpath in section_data_paths:
        raw = json.load(open(dpath, encoding='utf-8'))
        combined_data.update({item["Section"]: item for item in raw} if isinstance(raw, list) else raw)

    combined_emb = {}
    for epath in embedding_paths:
        combined_emb.update(json.load(open(epath, encoding='utf-8')))

    q_emb = embed_query(query)
    if q_emb is None:
        return []

    sec_ids = list(combined_emb.keys())
    mat = np.array([combined_emb[sid] for sid in sec_ids])
    print("✅ Section embedding matrix shape:", mat.shape)

    sims = cosine_similarity([q_emb], mat)[0]
    idxs = sims.argsort()[::-1][:top_k]

    results = []
    for i in idxs:
        sid = sec_ids[i]
        info = combined_data[sid].copy()
        info["similarity"] = float(sims[i])
        results.append(info)

    return results

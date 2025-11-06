# ===============================================
import streamlit as st
import pandas as pd
import numpy as np
import json
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Load artifact + data
# -------------------------
ART_PATH = r"C:\Users\Dewald\Documents\GitHub\2501PTDS-Unsupervised-Learning\Data\anime_hybrid_recommender.json"
ANIME_PATH = r"C:\Users\Dewald\Documents\GitHub\2501PTDS-Unsupervised-Learning\Data\anime.csv"


@st.cache_resource
def load_artifact():
    with open(ART_PATH, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    # Restore sparse matrix
    r = artifact["R_csr"]
    R = sparse.csr_matrix((r["data"], r["indices"], r["indptr"]), shape=tuple(r["shape"]))

    user_ids = np.array(artifact["user_ids"])
    item_ids = np.array(artifact["item_ids"])
    user_mean = np.array(artifact["user_mean"])
    item_mean = np.array(artifact["item_mean"])
    global_mean = artifact["global_mean"]
    best_alpha = artifact["best_alpha"]

    return R, user_ids, item_ids, user_mean, item_mean, global_mean, best_alpha

R, user_ids, item_ids, user_mean, item_mean, global_mean, best_alpha = load_artifact()

# Load anime metadata
anime_df = pd.read_csv(ANIME_PATH)
anime_lookup = dict(zip(anime_df["anime_id"], anime_df["name"]))

# -------------------------
# Build neighbors once
# -------------------------
meta_cols = [c for c in ["genre", "type", "name", "episodes"] if c in anime_df.columns]
anime_df["__text__"] = anime_df[meta_cols].astype(str).agg(" ".join, axis=1)
anime_meta = pd.DataFrame({"anime_id": item_ids}).merge(anime_df[["anime_id", "__text__"]], on="anime_id", how="left").fillna({"__text__": ""})

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
tfidf_item = tfidf.fit_transform(anime_meta["__text__"])
cb_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=30)
cb_knn.fit(tfidf_item)
cb_dists, cb_inds = cb_knn.kneighbors(tfidf_item, n_neighbors=30, return_distance=True)
cb_sims = 1.0 - cb_dists
cb_inds, cb_sims = cb_inds[:, 1:], cb_sims[:, 1:]

cf_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=30)
cf_knn.fit(R.T)
cf_dists, cf_inds = cf_knn.kneighbors(R.T, n_neighbors=30, return_distance=True)
cf_sims = 1.0 - cf_dists
cf_inds, cf_sims = cf_inds[:, 1:], cf_sims[:, 1:]

# Build quick lookup per user
user_rdict = []
for u in range(R.shape[0]):
    s, e = R.indptr[u], R.indptr[u + 1]
    user_rdict.append({int(i): float(r) for i, r in zip(R.indices[s:e], R.data[s:e])})

# -------------------------
# Predictor functions
# -------------------------
def predict_from_neighbors(uidx: int, iidx: int, neigh_idx: np.ndarray, neigh_sim: np.ndarray) -> float:
    max_idx = R.shape[1] - 1
    valid_pairs = [(int(nb), float(s))
                   for nb, s in zip(neigh_idx[iidx], neigh_sim[iidx])
                   if 0 <= nb <= max_idx]
    numer, denom = 0.0, 0.0
    rdict = user_rdict[uidx]
    for nb, s in valid_pairs:
        r = rdict.get(nb)
        if r is not None:
            numer += s * r
            denom += abs(s)
    if denom > 0:
        return numer / denom
    return float(0.5 * user_mean[uidx] + 0.5 * item_mean[iidx])

def predict_hybrid(uidx: int, iidx: int, alpha: float) -> float:
    p_cf = predict_from_neighbors(uidx, iidx, cf_inds, cf_sims)
    p_cb = predict_from_neighbors(uidx, iidx, cb_inds, cb_sims)
    return alpha * p_cf + (1 - alpha) * p_cb

def recommend_for_user(user_id: int, top_n: int = 10) -> pd.DataFrame:
    if user_id not in user_ids:
        st.warning(f"User ID {user_id} not found in training data.")
        return pd.DataFrame()

    uidx = np.where(user_ids == user_id)[0][0]
    rated_items = set(user_rdict[uidx].keys())
    preds = []
    for iidx in range(R.shape[1]):
        if iidx not in rated_items:
            try:
                p = predict_hybrid(uidx, iidx, best_alpha)
            except IndexError:
                continue
            preds.append((iidx, p))
    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    recs = pd.DataFrame({
        "anime_id": [int(item_ids[i]) for i, _ in preds],
        "predicted_rating": [round(float(p), 2) for _, p in preds]
    })
    recs["name"] = recs["anime_id"].map(anime_lookup)
    return recs[["anime_id", "name", "predicted_rating"]]

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎌 Anime Hybrid Recommender System")
st.markdown("This app blends collaborative filtering and content-based recommendations to suggest anime you’ll love!")

st.sidebar.header("🔧 Settings")
user_id_input = st.sidebar.number_input("Enter a User ID:", min_value=int(user_ids.min()), max_value=int(user_ids.max()), value=int(user_ids[0]))
top_n = st.sidebar.slider("How many recommendations?", 5, 20, 10)

if st.sidebar.button("Get Recommendations"):
    st.info(f"Generating top {top_n} recommendations for User {user_id_input}...")
    recs = recommend_for_user(user_id_input, top_n)
    if len(recs) > 0:
        st.success("✅ Recommendations ready!")
        st.dataframe(recs, hide_index=True)
    else:
        st.warning("No recommendations available for this user.")

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Dewald – Hybrid Anime Recommender v1.0")

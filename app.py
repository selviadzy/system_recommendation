import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommender System", page_icon="📚", layout="wide")

DATA_DIR = "data"
FILES = {
    "books": "BX-Books.csv",
    "ratings": "BX-Book-Ratings.csv",
}


# ----------------------------------------------------------------------
# 1. Membaca dan preprocessing data
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Membaca dan membersihkan dataset...")
def load_data(books_src, ratings_src):
    books = pd.read_csv(
        books_src, sep=";", encoding="latin-1",
        dtype={"ISBN": str}, on_bad_lines="skip",
    )
    ratings = pd.read_csv(
        ratings_src, sep=";", encoding="latin-1", dtype={"ISBN": str},
    )

    # Rating menjadi numerik
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    # Hapus data kosong, rating 0 (implisit), dan duplikat
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    ratings = ratings[ratings["Book-Rating"] > 0]
    ratings = ratings.drop_duplicates(subset=["User-ID", "ISBN"])

    # Bersihkan dataset buku
    books = books.dropna(subset=["ISBN", "Book-Title"]).drop_duplicates(subset=["ISBN"])
    books = books[["ISBN", "Book-Title", "Book-Author"]]
    return books, ratings


# ----------------------------------------------------------------------
# 2. User-Item matrix (sparse)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Membangun user-item matrix...")
def build_matrix(ratings):
    user_codes, user_ids = pd.factorize(ratings["User-ID"])
    book_codes, book_ids = pd.factorize(ratings["ISBN"])
    matrix = csr_matrix(
        (ratings["Book-Rating"].to_numpy(dtype=np.float32), (user_codes, book_codes)),
        shape=(len(user_ids), len(book_ids)),
    )
    return matrix, matrix.tocsc(), np.asarray(user_ids), np.asarray(book_ids)


# ----------------------------------------------------------------------
# 3. Collaborative Filtering (versi vektorisasi, hasil sama dengan notebook)
# ----------------------------------------------------------------------
def predict_user_based(matrix_csr, matrix_csc, target_user, candidates):
    """Prediksi = Σ(sim × rating) / Σ(sim) dari pengguna lain yang mirip."""
    user_sim = cosine_similarity(matrix_csr.getrow(target_user), matrix_csr).ravel()
    weights = user_sim.copy()
    weights[target_user] = 0          # kecualikan target user sendiri
    weights[weights < 0] = 0          # hanya similarity positif

    sub = matrix_csc[:, candidates]   # (user x kandidat)
    numerator = sub.T.dot(weights)
    denominator = (sub > 0).T.dot(weights)

    return np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def predict_item_based(matrix_csr, matrix_csc, target_user, candidates):
    """Prediksi = Σ(sim buku × rating user) / Σ(sim buku) dari buku yang sudah dinilai."""
    user_row = matrix_csr.getrow(target_user)
    rated_items = user_row.indices
    if len(rated_items) == 0:
        return np.zeros(len(candidates))

    user_ratings = user_row.data.astype(np.float64)
    sim = cosine_similarity(
        matrix_csc[:, candidates].T,      # kandidat x user
        matrix_csc[:, rated_items].T,     # buku dinilai x user
    )
    sim[sim <= 0] = 0

    numerator = sim.dot(user_ratings)
    denominator = sim.sum(axis=1)
    return np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def get_candidates(matrix_csr, matrix_csc, target_user, min_book_ratings, n_candidates):
    book_rating_counts = matrix_csc.getnnz(axis=0)
    rated_books = matrix_csr.getrow(target_user).indices

    mask = book_rating_counts >= min_book_ratings
    mask[rated_books] = False

    candidates = np.where(mask)[0]
    order = np.argsort(-book_rating_counts[candidates])
    return candidates[order][:n_candidates]


# ----------------------------------------------------------------------
# 4. Antarmuka Streamlit
# ----------------------------------------------------------------------
st.title("📚 Book Recommender System")
st.caption(
    "Collaborative Filtering (User-Based & Item-Based) pada dataset Book-Crossing "
    "· Selvia Adzania Arifiyanti (5003241025)"
)

# ---- Sidebar: sumber data
st.sidebar.header("⚙️ Pengaturan")

books_src = ratings_src = None
local_ok = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in FILES.values())

if local_ok:
    st.sidebar.success(f"Dataset ditemukan di folder `{DATA_DIR}/`")
    books_src = os.path.join(DATA_DIR, FILES["books"])
    ratings_src = os.path.join(DATA_DIR, FILES["ratings"])
else:
    st.sidebar.info(f"Folder `{DATA_DIR}/` belum berisi dataset. Silakan upload file CSV.")
    books_src = st.sidebar.file_uploader("BX-Books.csv", type="csv")
    ratings_src = st.sidebar.file_uploader("BX-Book-Ratings.csv", type="csv")

if books_src is None or ratings_src is None:
    st.warning("Dataset belum tersedia. Letakkan `BX-Books.csv` dan `BX-Book-Ratings.csv` "
               f"di folder `{DATA_DIR}/` atau upload lewat sidebar.")
    st.stop()

books, ratings = load_data(books_src, ratings_src)
matrix_csr, matrix_csc, user_ids, book_ids = build_matrix(ratings)

# ---- Ringkasan data
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah rating", f"{len(ratings):,}")
c2.metric("Pengguna unik", f"{matrix_csr.shape[0]:,}")
c3.metric("Buku unik", f"{matrix_csr.shape[1]:,}")
density = len(ratings) / (matrix_csr.shape[0] * matrix_csr.shape[1])
c4.metric("Kepadatan matriks", f"{density:.4%}")

with st.expander("Tentang user-item matrix"):
    st.write(
        f"Matriks berukuran **{matrix_csr.shape[0]:,} × {matrix_csr.shape[1]:,}** "
        f"(≈ {matrix_csr.shape[0] * matrix_csr.shape[1] / 1e9:.1f} miliar sel). "
        "Sebagian besar sel bernilai 0 (belum ada rating), sehingga matriks disimpan "
        "sebagai *sparse matrix* agar memori tidak jebol."
    )
    st.dataframe(matrix_csr[:5, :5].toarray(), use_container_width=True)

# ---- Sidebar: parameter
st.sidebar.subheader("Target user")
min_r, max_r = st.sidebar.slider(
    "Rentang jumlah rating user (eligible)", 1, 500, (20, 200)
)
mode = st.sidebar.radio("Pilih user", ["Acak (seed tetap)", "Input User-ID"])

jumlah_rating_user = matrix_csr.getnnz(axis=1)
user_eligible = np.where(
    (jumlah_rating_user >= min_r) & (jumlah_rating_user <= max_r)
)[0]

if len(user_eligible) == 0:
    st.error("Tidak ada pengguna yang memenuhi kriteria.")
    st.stop()

if mode == "Acak (seed tetap)":
    seed = st.sidebar.number_input("Seed", value=42, step=1)
    target_user = int(np.random.default_rng(int(seed)).choice(user_eligible))
else:
    uid = st.sidebar.number_input("User-ID", min_value=0, value=int(user_ids[user_eligible[0]]), step=1)
    matches = np.where(user_ids == uid)[0]
    if len(matches) == 0:
        st.error(f"User-ID {uid} tidak ditemukan di data.")
        st.stop()
    target_user = int(matches[0])

st.sidebar.subheader("Rekomendasi")
min_book_ratings = st.sidebar.slider("Minimal jumlah rating per buku", 1, 50, 5)
n_candidates = st.sidebar.slider("Jumlah kandidat buku", 50, 2000, 500, step=50)
top_n = st.sidebar.slider("Tampilkan Top-N", 5, 50, 10)
sort_by = st.sidebar.selectbox(
    "Urutkan berdasarkan",
    ["Item-Based Rating", "User-Based Rating"],
)

# ---- Info target user
st.subheader("👤 Target User")
t1, t2, t3 = st.columns(3)
t1.metric("User ID terpilih", int(user_ids[target_user]))
t2.metric("Jumlah rating", int(jumlah_rating_user[target_user]))
t3.metric("User eligible", f"{len(user_eligible):,}")

with st.expander("Buku yang sudah dinilai target user"):
    row = matrix_csr.getrow(target_user)
    history = pd.DataFrame({"ISBN": book_ids[row.indices], "Rating": row.data.astype(int)})
    history = history.merge(books, on="ISBN", how="left").sort_values("Rating", ascending=False)
    st.dataframe(history, use_container_width=True, hide_index=True)

# ---- Jalankan rekomendasi
if st.button("🚀 Buat Rekomendasi", type="primary"):
    with st.spinner("Menghitung prediksi rating..."):
        candidates = get_candidates(
            matrix_csr, matrix_csc, target_user, min_book_ratings, n_candidates
        )
        user_pred = predict_user_based(matrix_csr, matrix_csc, target_user, candidates)
        item_pred = predict_item_based(matrix_csr, matrix_csc, target_user, candidates)

        df = pd.DataFrame({
            "ISBN": book_ids[candidates],
            "User-Based Rating": user_pred,
            "Item-Based Rating": item_pred,
        }).merge(books, on="ISBN", how="left")

        df = df.sort_values(sort_by, ascending=False)
        st.session_state["result"] = (df, len(candidates))

if "result" in st.session_state:
    df, n_cand = st.session_state["result"]
    st.subheader(f"🏆 Top {top_n} Rekomendasi Buku")
    st.caption(f"Dari {n_cand} kandidat buku. Diurutkan berdasarkan {sort_by}.")

    top = df[["ISBN", "Book-Title", "Book-Author", "User-Based Rating", "Item-Based Rating"]].head(top_n)
    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
        column_config={
            "User-Based Rating": st.column_config.NumberColumn(format="%.2f"),
            "Item-Based Rating": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.download_button(
        "⬇️ Unduh hasil (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="rekomendasi_buku.csv",
        mime="text/csv",
    )
else:
    st.info("Atur parameter di sidebar, lalu klik **Buat Rekomendasi**.")

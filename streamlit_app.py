# streamlit_app.py (UPGRADED)
import os, sys, csv, pickle, math, json
import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd

# ML & text
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import google.generativeai as genai  # optional usage for extra explanations

from utils import remove_special_chars, tokenize_training_style, type_token_ratio, avg_word_length, avg_sentence_length, punctuation_density, sensational_word_count, extract_dates

# ================= PATHS & CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "my_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")
LOG_DIR = os.path.join(BASE_DIR, "logs")
FEEDBACK_FILE = os.path.join(LOG_DIR, "feedback.csv")
KEYWORD_STATS_PATH = os.path.join(LOG_DIR, "keyword_stats.pkl")
TFIDF_STORE_PATH = os.path.join(LOG_DIR, "tfidf_store.npz")
TFIDF_META_PATH = os.path.join(LOG_DIR, "tfidf_meta.pkl")

MAX_LEN = 1000
SENSATIONAL_WORDS = set([
    # a small seed list - expanded automatically from dataset
    "shocking","breaking","exclusive","horrifying","unbelievable","urgent",
    "massive","miracle","secret","exposed","shocker","viral","alert"
])

os.makedirs(LOG_DIR, exist_ok=True)

# ========= Load model + tokenizer =========
@st.cache_resource
def load_model_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

# ========= Load dataset =========
@st.cache_data
def load_dataset(path=DATA_PATH, sample_frac=None):
    df = pd.read_csv(path)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    return df

data_df = load_dataset()

# ========= Build suspicious keyword scores =========
def build_keyword_stats(df):
    from collections import Counter, defaultdict
    fake_mask = df['label'] == 0 or df.get('target') is not None and df['target'].eq('FAKE') if False else (df['label']==0 if 'label' in df.columns else (df['target']=='FAKE'))
    if 'label' in df.columns:
        fake_mask = df['label'] == 0  # in your dataset 0 or 1? check — you can flip if needed
    elif 'target' in df.columns:
        fake_mask = df['target'].str.upper() == "FAKE"
    else:
        fake_mask = pd.Series([False]*len(df))

    fake_cnt = Counter()
    real_cnt = Counter()
    total_cnt = Counter()
    for _, row in df.iterrows():
        text = row.get('text','')
        tokens = tokenize_training_style(text)
        total_cnt.update(tokens)
        if fake_mask.iloc[_] if isinstance(fake_mask, pd.Series) else False:
            fake_cnt.update(tokens)
        else:
            real_cnt.update(tokens)

    words = list(total_cnt.keys())
    stats = {}
    for w in words:
        f = fake_cnt[w]
        r = real_cnt[w]
        total = f + r
        if total == 0: continue
        fake_score = f / total  # 0..1 (1 => only fake)
        stats[w] = {"fake_count": f, "real_count": r, "total": total, "fake_score": fake_score}
    return stats

@st.cache_data
def get_keyword_stats():
    # load cached stats if exist
    if os.path.exists(KEYWORD_STATS_PATH):
        try:
            with open(KEYWORD_STATS_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    # else compute
    stats = build_keyword_stats(data_df)
    with open(KEYWORD_STATS_PATH, "wb") as f:
        pickle.dump(stats, f)
    return stats

keyword_stats = get_keyword_stats()

# build sensational set from top fake-associated words
def build_sensational_set(keyword_stats, threshold=0.8, topk=200):
    items = sorted(keyword_stats.items(), key=lambda kv: (-kv[1]['fake_score'], -kv[1]['total']))
    selected = [w for w, _ in items[:topk] if _['fake_score'] >= threshold]
    # fallback: add seed words:
    sel = set([w for w,_ in items[:topk] if _['fake_score']>=threshold])
    return sel.union(SENSATIONAL_WORDS)

SENSATIONAL_SET = build_sensational_set(keyword_stats)

# ========= Build TF-IDF index for semantic search =========
@st.cache_data
def build_tfidf_index(df):
    texts = df['text'].astype(str).tolist()
    # Use cleaned strings for TF-IDF
    cleaned_texts = [" ".join(tokenize_training_style(t)) for t in texts]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    tfidf = vec.fit_transform(cleaned_texts)
    meta = {"docs": texts, "vectorizer": vec}
    # cache to disk
    with open(TFIDF_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    # store matrix using scipy sparse save
    from scipy import sparse
    sparse.save_npz(TFIDF_STORE_PATH, tfidf)
    return tfidf, vec, texts

def load_tfidf_index():
    try:
        from scipy import sparse
        if os.path.exists(TFIDF_STORE_PATH) and os.path.exists(TFIDF_META_PATH):
            tfidf = sparse.load_npz(TFIDF_STORE_PATH)
            with open(TFIDF_META_PATH, "rb") as f:
                meta = pickle.load(f)
            return tfidf, meta['vectorizer'], meta['docs']
    except Exception:
        pass
    return build_tfidf_index(data_df)

tfidf_matrix, tfidf_vectorizer, docs = load_tfidf_index()

# ========= Prediction & helpers =========
def preprocess_for_model(text):
    tokens = tokenize_training_style(text)
    seq = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return tokens, seq, padded

def predict_label(text):
    tokens, seq, padded = preprocess_for_model(text)
    pred = float(model.predict(padded, verbose=0)[0][0])
    label = "Fake" if pred < 0.5 else "Real"
    return label, pred, tokens, seq, int((padded != 0).sum())

# semantic search
def get_similar_articles(text, topn=5):
    cleaned = " ".join(tokenize_training_style(text))
    qv = tfidf_vectorizer.transform([cleaned])
    sims = cosine_similarity(qv, tfidf_matrix).flatten()
    idx = np.argsort(-sims)[:topn]
    return [(int(i), float(sims[i]), docs[i]) for i in idx]

# suspicious highlighting score (word-level)
def word_suspiciousness(tokens, stats=keyword_stats):
    out = []
    for t in tokens:
        info = stats.get(t, None)
        score = info['fake_score'] if info else 0.0
        out.append((t, score))
    return out

# readability and linguistic features
def compute_readability_features(text, tokens):
    return {
        "avg_word_length": round(avg_word_length(tokens), 3),
        "avg_sentence_length": round(avg_sentence_length(text), 3),
        "type_token_ratio": round(type_token_ratio(tokens), 3),
        "punctuation_density": round(punctuation_density(text), 4),
        "sensational_word_count": sensational_word_count(tokens, SENSATIONAL_SET)
    }

# timeline check
def timeline_checks(text):
    extracted = extract_dates(text)
    now = datetime.now()
    issues = []
    for d in extracted:
        # very simple checks: future-date words or relative terms
        if "next" in d or "tomorrow" in d:
            issues.append(f"Date/word '{d}' mentions future — check plausibility")
    return extracted, issues

# ========== Feedback logging ==========
def log_feedback(news, predicted, user_feedback, correct_label):
    header = ["timestamp", "news_text", "predicted_label", "user_feedback", "correct_label"]
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), news, predicted, user_feedback, correct_label])

# ========== Gemini (optional for expanded explanations) ==========
load_dotenv(os.path.join(BASE_DIR, "api_key.env"))
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
else:
    GEMINI_AVAILABLE = False

def ask_gemini_explain(text, label):
    if not GEMINI_AVAILABLE:
        return None
    try:
        gm = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"Article: {text}\nModel predicted: {label}\nExplain in 2-3 factual lines why it might be {label}."
        resp = gm.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return None

def ask_gemini_counterfactual(text, label):
    if not GEMINI_AVAILABLE:
        return None
    try:
        gm = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"Article: {text}\nModel predicted: {label}\nGive a short counterfactual: how could this text be minimally changed to flip the prediction?"
        resp = gm.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return None

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰Fake News Detector")

tabs = st.tabs(["Analyze", "Similar Articles", "Explainability", "Text Analysis", "Feedback", "About"])

with tabs[0]:
    st.header("Analyze an article")
    user_text = st.text_area("Paste article or headline here:", height=220)

    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            label, pred, tokens, seq, nonzero = predict_label(user_text)
            # top suspicious words
            susp = word_suspiciousness(tokens)
            top_susp = sorted(susp, key=lambda x: -x[1])[:10]

            # show prediction
            col1, col2 = st.columns([2,1])
            with col1:
                if label == "Fake":
                    st.error(f"Prediction: **{label}**")
                else:
                    st.success(f"Prediction: **{label}**")
                st.write(f"Model raw score (sigmoid output): {pred:.6f}")
                st.write(f"Non-zero tokens in padded input: {nonzero}")

                # highlight text with colors depending on score
                def colored_text(tokens_scores):
                    parts = []
                    for tok, score in tokens_scores:
                        if score >= 0.8:
                            parts.append(f"<span style='background:#ff9999;padding:2px;border-radius:3px'>{tok}</span>")
                        elif score >= 0.5:
                            parts.append(f"<span style='background:#ffd699;padding:2px;border-radius:3px'>{tok}</span>")
                        else:
                            parts.append(tok)
                    return " ".join(parts)

                st.markdown("### Suspicious keyword heatmap")
                st.markdown(colored_text(susp[:200]), unsafe_allow_html=True)

                # show top suspicious list
                if top_susp:
                    st.write("Top suspicious tokens (fake-score, counts):")
                    top_table = [(t, round(keyword_stats.get(t,{}).get("fake_score",0),3), keyword_stats.get(t,{}).get("total",0)) for t,_ in top_susp]
                    st.table(pd.DataFrame(top_table, columns=["token","fake_score","total_count"]).head(10))

                # gemini explain (optional)
                gem_resp = ask_gemini_explain(user_text, label)
                if gem_resp:
                    st.markdown("###Gemini explanation")
                    st.info(gem_resp)

            with col2:
                st.markdown("### Quick stats")
                features = compute_readability_features(user_text, tokens)
                st.metric("Avg sentence length", features['avg_sentence_length'])
                st.metric("Type-token ratio", features['type_token_ratio'])
                st.metric("Sensational words", features['sensational_word_count'])
                st.metric("Avg word length", features['avg_word_length'])
                st.metric("Punctuation density", features['punctuation_density'])

            # timeline checks
            extracted_dates, issues = timeline_checks(user_text)
            if extracted_dates:
                st.markdown("### Extracted dates/days")
                st.write(extracted_dates)
                if issues:
                    st.warning("Timeline issues: " + "; ".join(issues))

with tabs[1]:
    st.header("Similar articles (semantic search via TF-IDF)")
    q = st.text_area("Enter text to find similar articles (or use above)", height=120)
    if st.button("Find similar"):
        if not q.strip():
            st.warning("Please enter query text.")
        else:
            sims = get_similar_articles(q, topn=8)
            df_sim = pd.DataFrame([{"idx":i,"score":s,"text":docs[i][:300]} for i,s,docs_i in sims for (i,s,docs_i) in [(i,s,docs[i])] ])
            # better construction:
            rows=[]
            for i,s,_ in sims:
                rows.append({"idx": i, "score": round(s,4), "text": docs[i][:400], "label": data_df.iloc[i].get("target", data_df.iloc[i].get("label",""))})
            st.table(pd.DataFrame(rows))

with tabs[2]:
    st.header("Explainability & Counterfactuals")
    expl_query = st.text_area("Enter article for explainability:", height=160)
    if st.button("Explain now"):
        if not expl_query.strip():
            st.warning("Enter text first")
        else:
            label, pred, tokens, seq, nonzero = predict_label(expl_query)
            st.write("Prediction:", label, f"({pred:.6f})")
            st.write("Tokens (first 80):", tokens[:80])
            st.write("Sequence indices (first 80):", seq[0][:80])
            st.write("Non-zero token count:", nonzero)

            # local counterfactual hint: top suspicious tokens
            susp = word_suspiciousness(tokens)
            top = sorted(susp, key=lambda x: -x[1])[:6]
            if top:
                st.markdown("### Top suspicious tokens (fake score)")
                st.table(pd.DataFrame(top, columns=["token","fake_score"]))
                st.markdown("### Counterfactual hint")
                st.write("Try removing or neutralizing suspicious tokens above — if many high-score tokens are removed the model may flip label.")
            # optional gemini counterfactual
            cf = ask_gemini_counterfactual(expl_query, label)
            if cf:
                st.markdown("### Gemini counterfactual suggestion")
                st.info(cf)

with tabs[3]:
    st.header("Text Analysis (Readability & Linguistics)")
    analy_text = st.text_area("Paste text to analyze:", height=200)
    if st.button("Analyze text"):
        if not analy_text.strip():
            st.warning("Enter text")
        else:
            tokens = tokenize_training_style(analy_text)
            feats = compute_readability_features(analy_text, tokens)
            st.subheader("Readability / Linguistic Features")
            st.json(feats)
            st.markdown("### Frequency Wordcloud (top tokens)")
            from collections import Counter
            c = Counter(tokens)
            top = c.most_common(30)
            st.table(pd.DataFrame(top, columns=["token","count"]))

with tabs[4]:
    st.header("Feedback collected")
    st.write("You can see all feedback saved in logs/feedback.csv")
    if os.path.exists(FEEDBACK_FILE):
        fb = pd.read_csv(FEEDBACK_FILE)
        st.dataframe(fb.tail(200))
    else:
        st.info("No feedback yet.")

with tabs[5]:
    st.header("About / Notes")
    st.write("""
    Features:
    - Suspicious keyword highlighting computed from your training dataset.
    - Semantic search (TF-IDF) over entire dataset (no external API).
    - Explainability panel: tokens, suspicious scores, local counterfactual hint, optional Gemini counterfactuals.
    - Readability & linguistics.
    - Timeline checks for extracted dates.
    """)
    st.write("Make sure `model/my_model.h5` and `model/tokenizer.pkl` exist in project root.")

import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Minimum cosine similarity to consider a result relevant.
# Results below this threshold are discarded to avoid injecting
# unrelated context that could cause the LLM to hallucinate.
MIN_SIMILARITY = 0.30


@st.cache_resource
def load_rag_resources():
    """Load MedQuAD dataset + SentenceTransformer encoder + precomputed embeddings."""

    dataset = load_dataset("lavita/MedQuAD")

    # Use first 5000 records initially (balance quality vs. memory)
    data = dataset["train"].select(range(5000))

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Extract questions — safely handle missing keys
    questions = []
    for item in data:
        q = item.get("question") or item.get("Question") or ""
        questions.append(q)

    question_embeddings = embedder.encode(
        questions,
        show_progress_bar=False
    )

    return data, embedder, question_embeddings


def retrieve_context(query, top_k=3, min_similarity=MIN_SIMILARITY):
    """
    Retrieve the top-k most similar medical Q&A pairs from MedQuAD.

    Returns a list of dicts, each with:
        - question:    the matched question text
        - answer:      the answer text
        - source:      dataset source (e.g. 'GHR', 'NCI')
        - url:         document URL for user verification
        - focus:       medical topic focus
        - score:       cosine similarity score (0.0–1.0)

    Results below `min_similarity` are filtered out.
    """
    data, embedder, question_embeddings = load_rag_resources()

    query_embedding = embedder.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        question_embeddings
    )[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    for idx in top_indices:
        score = float(similarities[int(idx)])

        # Skip results below the minimum similarity threshold
        if score < min_similarity:
            continue

        item = data[int(idx)]

        results.append({
            "question": item.get("question") or item.get("Question") or "",
            "answer":   item.get("answer") or item.get("Answer") or "",
            "source":   item.get("document_source") or "MedQuAD",
            "url":      item.get("document_url") or "",
            "focus":    item.get("question_focus") or "",
            "score":    score,
        })

    return results
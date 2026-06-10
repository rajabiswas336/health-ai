import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_resource
def load_rag_resources():

    dataset = load_dataset("lavita/MedQuAD")

    # use first 5000 records initially
    data = dataset["train"].select(range(5000))

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    questions = []

    for item in data:
        if "question" in item:
            questions.append(item["question"])
        elif "Question" in item:
            questions.append(item["Question"])
        else:
            questions.append("")

    question_embeddings = embedder.encode(
        questions,
        show_progress_bar=False
    )

    return data, embedder, question_embeddings


def retrieve_context(query, top_k=3):

    data, embedder, question_embeddings = load_rag_resources()

    query_embedding = embedder.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        question_embeddings
    )[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    contexts = []

    for idx in top_indices:

        item = data[int(idx)]

        contexts.append(item)

    return contexts
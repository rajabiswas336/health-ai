<p align="center">
  <img src="icon.png" alt="Health AI Logo" width="80"/>
</p>

<h1 align="center">🩺 AI Based Conversational Assistant for Healthcare and Support</h1>

<p align="center">
  <em>A multimodal AI healthcare assistant powered by vision, voice, and RAG — built with Streamlit</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Groq-Qwen%203.6%20Vision-4A90D9?logo=data:image/svg+xml;base64,&logoColor=white" alt="Groq"/>
  <img src="https://img.shields.io/badge/License-Educational-green" alt="License"/>
</p>

---

## 📋 Overview

**Health AI** is a conversational medical assistant that combines **text, voice, vision, and RAG (Retrieval-Augmented Generation)** into a single chat-based interface. It accepts text questions, voice recordings, and medical images — then responds with AI-generated medical guidance, optional text-to-speech audio playback, and source citations from the **MedQuAD** medical Q&A dataset.

> ⚠️ **Disclaimer:** This is an educational project. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Text Chat** | Ask medical questions in natural language |
| 🎙️ **Voice Input** | Record audio — transcribed via **Groq Whisper Large v3** |
| 🖼️ **Medical Image Analysis** | Upload or capture images — analyzed by **Qwen 3.6 27B Vision** |
| 🧬 **Skin Cancer Prediction** | CNN-based classification (7 lesion types) with **Grad-CAM** heatmaps and **Monte Carlo Dropout** uncertainty |
| 📚 **RAG Knowledge Base** | Retrieves relevant answers from **MedQuAD** (5,000 Q&A pairs) with source citations |
| 🔊 **Text-to-Speech** | Responses read aloud via **ElevenLabs**, **Microsoft Edge TTS**, or **gTTS** |
| 🌐 **Multilingual** | Supports **English, Bengali, Hindi, Odia, and Assamese** with auto-detection |
| 🔍 **Auto Language Detection** | Detects input language automatically and responds in the same language |

---

## 🏗️ Architecture

```
health-ai/
├── app.py                    # Main Streamlit application (UI + orchestration)
├── brain_of_the_doctor.py    # Vision LLM — image encoding + Groq API calls
├── voice_of_the_patient.py   # Speech-to-text — Groq Whisper transcription
├── voice_of_the_doctor.py    # Text-to-speech — ElevenLabs / Edge TTS / gTTS
├── language_utils.py         # Translation + language detection (deep-translator)
├── cancer_prediction.py      # Skin cancer CNN + Grad-CAM + MC Dropout
├── rag_retriever.py          # RAG pipeline — MedQuAD + SentenceTransformer
├── evaluation/               # Evaluation framework
│   ├── create_benchmark.py   # Benchmark dataset generation
│   ├── compile_results.py    # Results aggregation
│   ├── metrics/              # Retrieval + generation metrics
│   ├── data/                 # Benchmark datasets
│   ├── experiments/          # Experiment configs
│   └── results/              # Evaluation results
├── requirements.txt          # Python dependencies
├── icon.png                  # App icon
├── .env                      # API keys (not committed)
└── secrets.toml              # Streamlit Cloud secrets template
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit with custom CSS (glassmorphism, gradients) |
| **Vision LLM** | Qwen 3.6 27B via Groq API |
| **Speech-to-Text** | Whisper Large v3 via Groq API |
| **Text-to-Speech** | ElevenLabs (premium) / Microsoft Edge TTS (free) / gTTS (fallback) |
| **Translation** | Google Translator via `deep-translator` |
| **Language Detection** | `langdetect` (offline) |
| **RAG Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) |
| **RAG Dataset** | [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) |
| **Skin Cancer Model** | [Raja336/skin-cancer-convnext](https://huggingface.co/Raja336/skin-cancer-convnext) (ConvNext) |
| **Explainability** | Grad-CAM (pure PyTorch) + Monte Carlo Dropout |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and on PATH (required for audio processing)
- A **Groq** API key (required)
- An **ElevenLabs** API key (optional — falls back to Edge TTS / gTTS)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rajabiswas336/health-ai.git
   cd health-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

   The app will open at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your API keys under **Manage App → Secrets**:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"
   ```

---

## 🧩 Module Details

### `brain_of_the_doctor.py`
Handles vision capabilities — encodes images to base64 and sends them alongside text queries to the Groq Vision API (Qwen 3.6 27B) for medical image analysis.

### `voice_of_the_patient.py`
Captures audio from the microphone and transcribes it using **Groq Whisper Large v3**. Supports multilingual transcription with configurable language codes.

### `voice_of_the_doctor.py`
Multi-engine text-to-speech with automatic fallback:
- **ElevenLabs** — highest quality (English, requires API key)
- **Microsoft Edge TTS** — free neural voices for all supported languages
- **gTTS** — Google TTS fallback

### `language_utils.py`
Manages multilingual support — auto-detects input language via `langdetect`, translates between languages using `deep-translator`, and maps language names to Whisper/TTS voice codes.

### `cancer_prediction.py`
Skin cancer classification pipeline using a fine-tuned **ConvNext** model:
- Classifies 7 lesion types (melanoma, BCC, benign keratosis, etc.)
- Generates **Grad-CAM** heatmaps highlighting regions of interest
- Estimates model uncertainty via **Monte Carlo Dropout** (20 stochastic passes)
- Flags urgent/malignant predictions with safety warnings

### `rag_retriever.py`
RAG pipeline using **MedQuAD** (5,000 medical Q&A pairs):
- Embeds questions with `all-MiniLM-L6-v2`
- Retrieves top-k similar Q&A pairs via cosine similarity
- High-similarity results constrain the LLM; lower-similarity results supplement
- Returns source URLs for user verification

---

## 📊 Evaluation

The `evaluation/` directory contains a framework for benchmarking the RAG pipeline:

- **Retrieval metrics** — Precision, recall, MRR for the retrieval component
- **Generation metrics** — Quality assessment of LLM responses
- **Benchmark creation** — Scripts to generate test datasets
- **Results compilation** — Aggregate and compare experiment results

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👤 Author

**Raja Biswas**
M.Tech · Artificial Intelligence
Clinical Decision Support System

---

## 📄 License

This project is for **educational and research purposes only**. It is not intended for clinical use.

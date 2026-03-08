# app.py — AI Based Conversational Assistant For Healthcare and Support
# v8.0 — Unified chatbox: text + voice + image inline | working autoplay

import os
import io
import tempfile
import base64
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs, text_to_speech_with_edge
from language_utils import translate_to_english, translate_to_bengali, get_whisper_language_code

load_dotenv()

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── API Keys — read AFTER set_page_config so st.secrets is available ──────────
def _get_secret(key: str) -> str:
    """Read from Streamlit secrets (cloud) or .env (local)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")

GROQ_API_KEY       = _get_secret("GROQ_API_KEY")
ELEVENLABS_API_KEY = _get_secret("ELEVENLABS_API_KEY")

# Pass to submodules via environment
os.environ["GROQ_API_KEY"]       = GROQ_API_KEY
os.environ["ELEVENLABS_API_KEY"] = ELEVENLABS_API_KEY

# ── Key validation — show warning on cloud if missing ─────────────────────────
if not GROQ_API_KEY:
    st.error("🔑 GROQ_API_KEY not found. Go to Manage App → Secrets and add it.", icon="🚨")
    st.stop()

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "messages":       [],
    "show_voice":     False,
    "show_image":     False,
    "show_camera":    False,
    "input_key":      0,
    "autoplay_b64":   None,   # holds ONE audio to autoplay after rerun
    "stored_audio":   None,   # persists recorded audio bytes across reruns
    "stored_image":   None,   # persists uploaded image bytes across reruns
    "stored_img_name": None,  # filename for display
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS — Apple Glassmorphism · Filter Blue Theme ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
    color: #1a2a4a !important;
}

/* ══ BACKGROUND — light blue gradient with soft noise ══ */
.stApp {
    background: linear-gradient(145deg,
        #ddeeff 0%,
        #c8e0ff 20%,
        #d4eaff 40%,
        #bdd5ff 60%,
        #cce4ff 80%,
        #ddeeff 100%) !important;
    min-height: 100vh !important;
    position: relative !important;
}
.stApp::before {
    content: '';
    position: fixed;
    top: -40%;
    left: -20%;
    width: 80vw;
    height: 80vw;
    background: radial-gradient(circle, rgba(100,160,255,0.35) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -20%;
    right: -10%;
    width: 60vw;
    height: 60vw;
    background: radial-gradient(circle, rgba(140,100,255,0.2) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ══ SIDEBAR TOGGLE ══ */
[data-testid="collapsedControl"] {
    visibility: visible !important; display: flex !important;
    opacity: 1 !important; pointer-events: all !important;
    z-index: 999999 !important;
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-radius: 0 14px 14px 0 !important;
    border: 1px solid rgba(255,255,255,0.7) !important;
    box-shadow: 2px 0 16px rgba(80,130,255,0.18) !important;
}
[data-testid="collapsedControl"] svg { fill: #3a7bd5 !important; }
[data-testid="stSidebarCollapseButton"] { visibility: visible !important; pointer-events: all !important; }

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: rgba(220,235,255,0.75) !important;
    backdrop-filter: blur(40px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
    border-right: 1px solid rgba(255,255,255,0.6) !important;
    box-shadow: 4px 0 32px rgba(80,130,255,0.12) !important;
}
[data-testid="stSidebar"] * { color: #1a3060 !important; }
[data-testid="stSidebarContent"] { padding: 24px 18px !important; }
[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.5) !important;
    border: 1px solid rgba(255,255,255,0.8) !important;
    border-radius: 12px !important; padding: 10px 14px !important;
    margin: 4px 0 !important; transition: all .2s !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.75) !important;
    box-shadow: 0 2px 12px rgba(80,130,255,0.15) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.5) !important;
    border: 1px solid rgba(255,255,255,0.8) !important;
    border-radius: 12px !important; color: #1a3060 !important;
    backdrop-filter: blur(10px) !important;
}

/* ══ ALL BUTTONS — glass style ══ */
.stButton > button {
    background: rgba(255,255,255,0.55) !important;
    color: #1a4080 !important;
    border: 1px solid rgba(255,255,255,0.75) !important;
    border-radius: 16px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 10px 6px !important;
    width: 100% !important;
    transition: all .22s cubic-bezier(.25,.8,.25,1) !important;
    line-height: 1.5 !important;
    backdrop-filter: blur(16px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
    box-shadow: 0 2px 16px rgba(80,130,255,0.12),
                inset 0 1px 0 rgba(255,255,255,0.8) !important;
    -webkit-appearance: none !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.78) !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 28px rgba(80,130,255,0.22),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
    border-color: rgba(100,160,255,0.5) !important;
}
@media (prefers-color-scheme: dark) {
    .stButton > button {
        background: rgba(255,255,255,0.55) !important;
        color: #1a4080 !important;
        border-color: rgba(255,255,255,0.75) !important;
    }
}

/* ══ TOOLBAR ICON BUTTONS ══ */
.toolbar-btn .stButton > button {
    background: rgba(255,255,255,0.45) !important;
    border: 1px solid rgba(255,255,255,0.7) !important;
    border-radius: 50% !important;
    padding: 5px !important;
    font-size: 19px !important;
    min-height: 38px !important; min-width: 38px !important;
    line-height: 1 !important;
    color: #3a7bd5 !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 2px 12px rgba(80,130,255,0.15),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
}
.toolbar-btn .stButton > button:hover {
    background: rgba(255,255,255,0.75) !important;
    transform: none !important;
    box-shadow: 0 4px 20px rgba(80,130,255,0.25),
                inset 0 1px 0 rgba(255,255,255,1) !important;
}

/* ══ PILL ICON ══ */
.pill-icon .stButton > button {
    background: rgba(255,255,255,0.5) !important;
    border: 1px solid rgba(255,255,255,0.75) !important;
    border-radius: 50% !important;
    min-height: 36px !important; min-width: 36px !important;
    font-size: 17px !important; color: #3a7bd5 !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 2px 10px rgba(80,130,255,0.14),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
}
.pill-icon .stButton > button:hover {
    background: rgba(255,255,255,0.8) !important;
    color: #1a5ad5 !important;
}
@media (prefers-color-scheme: dark) {
    .pill-icon .stButton > button { background: rgba(255,255,255,0.5) !important; color: #3a7bd5 !important; }
}

/* ══ ACTIVE ICON ══ */
.active-btn .stButton > button {
    background: rgba(58,123,213,0.22) !important;
    border: 1px solid rgba(58,123,213,0.5) !important;
    border-radius: 50% !important; color: #1a5ad5 !important;
    box-shadow: 0 0 0 3px rgba(58,123,213,0.15),
                inset 0 1px 0 rgba(255,255,255,0.6) !important;
}

/* ══ ANALYSE BUTTON ══ */
.analyse-wrap .stButton > button {
    background: linear-gradient(135deg,
        rgba(58,123,213,0.9) 0%,
        rgba(100,160,255,0.85) 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 16px !important; font-size: 15px !important;
    font-weight: 700 !important; padding: 13px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 28px rgba(58,123,213,0.4),
                inset 0 1px 0 rgba(255,255,255,0.3) !important;
    letter-spacing: .03em !important;
}
.analyse-wrap .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(58,123,213,0.5),
                inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

/* ══ FORM ══ */
[data-testid="stForm"] {
    border: none !important; padding: 0 !important;
    background: transparent !important; box-shadow: none !important;
}
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important; outline: none !important;
    color: #1a2a4a !important;
    padding: 10px 4px !important; font-size: 14px !important;
    -webkit-text-fill-color: #1a2a4a !important;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(60,100,160,0.45) !important; }
[data-testid="stTextInput"] > div {
    background: transparent !important; border: none !important; box-shadow: none !important;
}

/* ══ INPUT PILL ══ */
div[data-testid="stHorizontalBlock"]:has(#pill-anchor) {
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border-radius: 50px !important;
    border: 1px solid rgba(255,255,255,0.8) !important;
    padding: 2px 6px 2px 18px !important;
    align-items: center !important;
    box-shadow: 0 4px 28px rgba(80,130,255,0.14),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
    margin-top: 8px !important;
    flex-wrap: nowrap !important;
}
[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    scrollbar-width: none !important;
}
[data-testid="stHorizontalBlock"]::-webkit-scrollbar { display: none !important; }

/* ══ AUDIO INPUT ══ */
[data-testid="stAudioInput"] {
    background: rgba(255,255,255,0.45) !important;
    border: 1px solid rgba(255,255,255,0.7) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
}
[data-testid="stAudioInput"] button {
    background: linear-gradient(135deg,#3a7bd5,#64a0ff) !important;
    border-radius: 50% !important;
    box-shadow: 0 0 0 8px rgba(58,123,213,0.12) !important;
}

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.4) !important;
    border: 2px dashed rgba(58,123,213,0.35) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    transition: all .2s !important;
}
[data-testid="stFileUploader"] section:hover {
    background: rgba(255,255,255,0.65) !important;
    border-color: rgba(58,123,213,0.7) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #3a7bd5 !important; font-weight: 700 !important;
}

/* ══ CAMERA INPUT ══ */
[data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.4) !important;
    border: 1.5px solid rgba(58,123,213,0.3) !important;
    border-radius: 18px !important; overflow: hidden !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stCameraInput"] button {
    background: linear-gradient(135deg,#3a7bd5,#64a0ff) !important;
    color: #fff !important; border-radius: 10px !important; border: none !important;
}

/* ══ CHAT WINDOW CONTAINER ══ */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.38) !important;
    backdrop-filter: blur(32px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
    border: 1px solid rgba(255,255,255,0.75) !important;
    border-radius: 24px !important;
    padding: 14px 8px !important;
    box-shadow: 0 8px 40px rgba(80,130,255,0.12),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
}

/* ══ AUDIO PLAYER ══ */
audio {
    width: 100% !important; border-radius: 12px !important;
    margin-top: 3px !important; height: 36px !important;
}

/* ══ IMAGE ══ */
[data-testid="stImage"] img {
    max-height: 200px !important; border-radius: 16px !important;
    box-shadow: 0 6px 24px rgba(80,130,255,0.2) !important;
    border: 1px solid rgba(255,255,255,0.7) !important;
}

/* ══ ALERTS ══ */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.55) !important;
    border: 1px solid rgba(255,255,255,0.75) !important;
    border-radius: 14px !important; color: #1a3060 !important;
    backdrop-filter: blur(16px) !important;
}

hr { border-color: rgba(100,150,255,0.12) !important; }
[data-testid="stCaptionContainer"] p { color: rgba(60,100,160,0.55) !important; font-size: 11px !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(80,130,255,0.25); border-radius: 4px; }
.block-container { max-width: 700px !important; padding: 20px 16px 70px !important; margin: 0 auto !important; }
[data-testid="stSpinner"] p { color: #3a7bd5 !important; }
</style>
""", unsafe_allow_html=True)


# ── Chat bubble renderers ─────────────────────────────────────────────────────
from datetime import datetime as _dt
def _now(): return _dt.now().strftime("%I:%M %p")

def user_bubble(text, voice_b64=None, ts=None):
    time_str = ts or ""
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-end;align-items:flex-end;
        gap:8px;margin:10px 0 2px;padding:0 2px;'>
        <div style='max-width:76%;'>
            <div style='background:linear-gradient(135deg,rgba(58,123,213,0.88),rgba(100,160,255,0.82));
                backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
                color:#fff;border-radius:20px 20px 5px 20px;
                padding:11px 15px;font-size:14px;line-height:1.65;
                box-shadow:0 4px 20px rgba(58,123,213,0.35),inset 0 1px 0 rgba(255,255,255,0.25);
                border:1px solid rgba(255,255,255,0.3);word-wrap:break-word;'>
                {text}
            </div>
            <div style='font-size:10px;color:rgba(58,100,180,0.55);text-align:right;
                margin-top:4px;padding-right:4px;'>{time_str} ✓✓</div>
        </div>
        <div style='width:34px;height:34px;border-radius:50%;
            background:linear-gradient(135deg,rgba(58,123,213,0.8),rgba(100,160,255,0.8));
            backdrop-filter:blur(10px);
            display:flex;align-items:center;justify-content:center;font-size:15px;
            flex-shrink:0;border:1.5px solid rgba(255,255,255,0.6);
            box-shadow:0 2px 12px rgba(58,123,213,0.3);'>👤</div>
    </div>
    """, unsafe_allow_html=True)
    if voice_b64:
        st.markdown(
            f"<div style='display:flex;justify-content:flex-end;margin:0 42px 8px 0;'>"
            f"<div style='width:72%;background:rgba(58,123,213,0.1);border-radius:12px;"
            f"padding:6px 10px;border:1px solid rgba(58,123,213,0.2);"
            f"backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);'>"
            f"<div style='font-size:9px;color:rgba(58,123,213,0.7);margin-bottom:3px;font-weight:700;letter-spacing:.08em;'>🎤 YOUR VOICE</div>"
            f"<audio controls style='width:100%;height:32px;border-radius:8px;'>"
            f"<source src='data:audio/mp3;base64,{voice_b64}' type='audio/mp3'></audio>"
            f"</div></div>", unsafe_allow_html=True)

def ai_bubble(text, is_medical=False, audio_b64=None, ts=None, do_autoplay=False):
    icon  = "🩺" if is_medical else "🤖"
    label = "Dr. AI" if is_medical else "AI Assistant"
    time_str = ts or ""
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-start;align-items:flex-end;
        gap:8px;margin:10px 0 2px;padding:0 2px;'>
        <div style='width:34px;height:34px;border-radius:50%;
            background:rgba(255,255,255,0.6);
            backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
            display:flex;align-items:center;justify-content:center;font-size:16px;
            flex-shrink:0;border:1.5px solid rgba(255,255,255,0.85);
            box-shadow:0 2px 12px rgba(80,130,255,0.15);'>{icon}</div>
        <div style='max-width:76%;'>
            <div style='font-size:10px;color:rgba(58,100,180,0.6);margin-bottom:4px;
                font-weight:700;letter-spacing:.05em;'>{label}</div>
            <div style='background:rgba(255,255,255,0.58);
                backdrop-filter:blur(28px) saturate(180%);
                -webkit-backdrop-filter:blur(28px) saturate(180%);
                color:#1a2a4a;border-radius:5px 20px 20px 20px;
                padding:11px 15px;font-size:14px;line-height:1.65;
                box-shadow:0 4px 24px rgba(80,130,255,0.1),inset 0 1px 0 rgba(255,255,255,0.95);
                border:1px solid rgba(255,255,255,0.8);word-wrap:break-word;'>
                {text}
            </div>
            <div style='font-size:10px;color:rgba(58,100,180,0.5);margin-top:4px;
                padding-left:4px;'>{time_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if audio_b64:
        _ap = "autoplay" if do_autoplay else ""
        st.markdown(
            f"<div style='display:flex;justify-content:flex-start;margin:0 0 8px 42px;'>"
            f"<div style='width:72%;background:rgba(255,255,255,0.5);border-radius:12px;"
            f"padding:6px 10px;border:1px solid rgba(255,255,255,0.8);"
            f"backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);'>"
            f"<div style='font-size:9px;color:rgba(58,123,213,0.7);margin-bottom:3px;font-weight:700;letter-spacing:.08em;'>🔊 DOCTOR RESPONSE</div>"
            f"<audio {_ap} controls style='width:100%;height:32px;border-radius:8px;'>"
            f"<source src='data:audio/mp3;base64,{audio_b64}' type='audio/mp3'></audio>"
            f"</div></div>", unsafe_allow_html=True)

def image_bubble(img_b64):
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-end;margin:4px 42px 4px 0;'>
        <img src='data:image/jpeg;base64,{img_b64}'
            style='max-height:190px;max-width:65%;border-radius:18px;
            box-shadow:0 6px 24px rgba(80,130,255,0.2);object-fit:cover;
            border:1.5px solid rgba(255,255,255,0.7);'/>
    </div>
    """, unsafe_allow_html=True)

def day_divider(label="Today"):
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin:14px 0 8px;'>
        <div style='flex:1;height:1px;background:rgba(80,130,255,0.15);'></div>
        <div style='font-size:10px;color:rgba(58,100,180,0.55);font-weight:700;
            background:rgba(255,255,255,0.5);padding:3px 12px;border-radius:20px;
            border:1px solid rgba(255,255,255,0.8);
            backdrop-filter:blur(10px);'>{label}</div>
        <div style='flex:1;height:1px;background:rgba(80,130,255,0.15);'></div>
    </div>
    """, unsafe_allow_html=True)

# ── Prompts
# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a professional doctor assistant. Analyze the image provided. "
    "If you find anything medically concerning, suggest remedies. "
    "Do not use numbers, bullet points, or special characters. "
    "Respond in one concise paragraph as if speaking directly to the patient. "
    "Start with 'With what I see, I think...' not 'In the image I see'. "
    "Keep your answer to 2-3 sentences maximum. No preamble."
)
CHAT_PROMPT = (
    "You are a professional doctor assistant. Answer the patient's health question "
    "clearly in 2-3 sentences. No bullet points. Speak directly to the patient. No preamble."
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 20px;'>
        <div style='width:64px;height:64px;border-radius:50%;
            background:linear-gradient(135deg,rgba(58,123,213,0.85),rgba(100,160,255,0.8));
            backdrop-filter:blur(16px);display:flex;align-items:center;justify-content:center;
            font-size:28px;margin:0 auto 12px;
            border:1.5px solid rgba(255,255,255,0.7);
            box-shadow:0 6px 24px rgba(58,123,213,0.3);'>🩺</div>
        <div style='font-size:16px;font-weight:800;color:#1a3060;'>AI Healthcare</div>
        <div style='font-size:11px;color:rgba(58,100,180,0.6);margin-top:4px;font-weight:500;'>
            v2.0 · M.Tech AI</div>
    </div>
    <div style='height:1px;background:rgba(80,130,255,0.15);margin-bottom:18px;'></div>
    """, unsafe_allow_html=True)

    language   = st.radio("🌐 Language / ভাষা", ["English", "Bengali"], index=0)
    tts_engine = st.selectbox(
        "🔊 Voice Engine",
        ["ElevenLabs (High Quality)", "edge-tts Neural (Free)", "gTTS (Fallback)"],
        index=0 if language == "English" else 1,
    )

    if language == "Bengali":
        bn_voice = st.selectbox(
            "🗣️ Bengali Voice",
            [
                "bn-BD-NabanitaNeural · Female (BD)",
                "bn-BD-PradeepNeural · Male (BD)",
                "bn-IN-TanishaaNeural · Female (IN)",
                "bn-IN-BashkarNeural · Male (IN)",
            ],
            index=0,
            help="BD = Bangladesh accent · IN = West Bengal / India accent",
        )
        # Extract just the voice ID (first token before ·)
        bengali_voice_id = bn_voice.split(" ·")[0].strip()
    else:
        bengali_voice_id = "bn-BD-NabanitaNeural"

    st.markdown("""
    <div style='height:1px;background:rgba(80,130,255,0.15);margin:16px 0;'></div>
    <div style='font-size:10px;color:rgba(58,100,180,0.55);letter-spacing:.12em;
        text-transform:uppercase;font-weight:700;margin-bottom:10px;'>Powered by</div>
    <div style='font-size:13px;color:#2a4a80;line-height:2.2;'>
        🧠 LLaMA-4 Scout Vision<br>
        🎙️ Groq Whisper Large v3<br>
        🔊 ElevenLabs / edge-tts / gTTS<br>
        🌐 Deep Translator
    </div>
    <div style='height:1px;background:rgba(80,130,255,0.15);margin:16px 0;'></div>
    <div style='background:rgba(255,255,255,0.5);border-radius:14px;padding:12px 14px;
        border:1px solid rgba(255,255,255,0.8);
        backdrop-filter:blur(16px);box-shadow:0 2px 12px rgba(80,130,255,0.1);'>
        <div style='font-size:13px;font-weight:700;color:#1a3060;'>Raja Biswas</div>
        <div style='font-size:11px;color:rgba(58,100,180,0.65);margin-top:2px;'>
            M.Tech (AI) · Healthcare AI</div>
    </div>
    """, unsafe_allow_html=True)




# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:28px 0 20px;'>
    <div style='display:inline-flex;align-items:center;justify-content:center;
        width:78px;height:78px;border-radius:50%;
        background:linear-gradient(135deg,rgba(58,123,213,0.82),rgba(100,160,255,0.78));
        backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
        font-size:36px;margin-bottom:16px;
        border:2px solid rgba(255,255,255,0.7);
        box-shadow:0 8px 32px rgba(58,123,213,0.35),inset 0 1px 0 rgba(255,255,255,0.4);'>🩺</div>
    <div style='font-size:clamp(17px,4vw,25px);font-weight:800;
        color:#0d1f40;margin-bottom:4px;letter-spacing:-.02em;'>
        AI Based Conversational Assistant
    </div>
    <div style='font-size:clamp(13px,3vw,16px);font-weight:600;
        color:#3a7bd5;margin-bottom:14px;'>
        For Healthcare and Support
    </div>
    <div style='margin:8px 0 18px;'>
        <div style='font-size:15px;font-weight:700;color:#1a3060;'>Raja Biswas</div>
        <div style='font-size:11px;color:rgba(58,100,180,0.6);font-weight:600;
            letter-spacing:.1em;text-transform:uppercase;margin-top:3px;'>
            M.Tech · Artificial Intelligence
        </div>
    </div>
    <div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;'>
        <span style='background:rgba(255,255,255,0.55);
            backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
            border:1px solid rgba(255,255,255,0.8);
            color:#1a7a5e;font-size:11px;padding:5px 16px;border-radius:20px;
            font-weight:700;box-shadow:0 2px 10px rgba(80,130,255,0.12);'>● LIVE</span>
        <span style='background:rgba(255,255,255,0.55);
            backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
            border:1px solid rgba(255,255,255,0.8);
            color:#3a7bd5;font-size:11px;padding:5px 16px;border-radius:20px;
            font-weight:700;box-shadow:0 2px 10px rgba(80,130,255,0.12);'>Vision + Voice</span>
        <span style='background:rgba(255,255,255,0.55);
            backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
            border:1px solid rgba(255,255,255,0.8);
            color:#cc4444;font-size:11px;padding:5px 16px;border-radius:20px;
            font-weight:700;box-shadow:0 2px 10px rgba(80,130,255,0.12);'>Educational Only</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Quick actions ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-size:10px;color:rgba(58,100,180,0.55);letter-spacing:.14em;
    text-transform:uppercase;font-weight:700;
    margin:4px 0 10px;text-align:center;'>Quick Actions</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1:
    if st.button("🎤\nVoice"):
        st.session_state.show_voice = True; st.rerun()
with c2:
    if st.button("🖼️\nImage"):
        st.session_state.show_image = True; st.rerun()
with c3:
    if st.button("🩺\nSkin"):
        st.session_state.show_voice = True
        st.session_state.show_image = True; st.rerun()
with c4:
    if st.button("🫁\nX-Ray"):
        st.session_state.show_image = True; st.rerun()
with c5:
    if st.button("🌐\nবাংলা"):
        st.session_state.show_voice = True; st.rerun()
with c6:
    if st.button("💊\nAsk"):
        pass

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── CHAT WINDOW ──────────────────────────────════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

# Style the chat container via CSS targeting the next sibling block
st.markdown("""
<style>
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stVerticalBlock"] > div > div[data-testid="stMarkdownContainer"] > div.chat-anchor) {
    background: linear-gradient(180deg,#12152a 0%,#161929 100%) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 18px !important;
    padding: 14px !important;
    min-height: 300px !important;
}
</style>
<div class="chat-anchor" style="display:none"></div>
""", unsafe_allow_html=True)

with st.container():
    if st.session_state.messages:
        # Find last assistant message with audio for autoplay
        _last_audio_idx = -1
        for _i, _m in enumerate(st.session_state.messages):
            if _m["role"] == "assistant" and _m.get("audio_b64"):
                _last_audio_idx = _i

        for _idx, msg in enumerate(st.session_state.messages):
            if _idx == 0:
                day_divider("Today")
            if msg["role"] == "user":
                if msg.get("img_b64"):
                    image_bubble(msg["img_b64"])
                user_bubble(msg["content"],
                            voice_b64=msg.get("voice_b64"),
                            ts=msg.get("ts",""))
            else:
                _do_ap = (st.session_state.get("autoplay_b64") is not None
                          and _idx == _last_audio_idx)
                ai_bubble(msg["content"],
                          is_medical=msg.get("medical", False),
                          audio_b64=msg.get("audio_b64"),
                          ts=msg.get("ts",""),
                          do_autoplay=_do_ap)
    else:
        st.markdown("""
        <div style='text-align:center;padding:48px 20px;'>
            <div style='font-size:38px;margin-bottom:12px;opacity:.18;'>💬</div>
            <div style='font-size:13px;color:rgba(58,100,180,0.4);line-height:1.9;'>
                Hello! I'm your AI healthcare assistant.<br>
                Type a question · 🎤 record symptoms · 🖼️ upload an image
            </div>
        </div>
        """, unsafe_allow_html=True)

# (status bar removed)

# Consume autoplay flag after render so it only plays once
if st.session_state.autoplay_b64:
    st.session_state.autoplay_b64 = None


# ══════════════════════════════════════════════════════════════════════════════
# ── UNIFIED INPUT BOX ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Voice panel ───────────────────────────────────────────────────────────────
if st.session_state.show_voice:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.45);border-radius:16px;padding:12px 16px 8px;
        border:1px solid rgba(255,255,255,0.8);margin-bottom:8px;
        backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
        box-shadow:0 4px 24px rgba(80,130,255,0.1);'>
        <div style='font-size:10px;color:rgba(58,100,180,0.7);letter-spacing:.15em;
            text-transform:uppercase;font-family:JetBrains Mono,monospace;
            margin-bottom:6px;'>🎤 Tap the mic below to start recording</div>
    </div>
    """, unsafe_allow_html=True)
    _raw_audio = st.audio_input(
        label="Record" if language == "English" else "রেকর্ড করুন",
        key="audio_recorder",
    )
    if _raw_audio is not None:
        # Save bytes immediately so rerun doesn't lose it
        st.session_state.stored_audio = _raw_audio.read()
        _raw_audio.seek(0)  # reset so it can be read again later
    if st.session_state.stored_audio:
        st.markdown(
            "<div style='font-size:12px;color:#3a7bd5;font-weight:700;padding:4px 2px 6px;'>"
            "✓ Voice captured — tap 🔬 Analyse Now</div>",
            unsafe_allow_html=True,
        )
else:
    # Clear stored audio when panel is closed
    if not (st.session_state.show_image):  # only clear if both closed
        pass  # keep audio until analyse is done

# Expose audio_file from stored bytes
audio_file = None
if st.session_state.show_voice and st.session_state.stored_audio:
    audio_file = io.BytesIO(st.session_state.stored_audio)

# ── Image panel ───────────────────────────────────────────────────────────────
if st.session_state.show_image:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.45);border-radius:16px;padding:12px 16px 8px;
        border:1px solid rgba(255,255,255,0.8);margin-bottom:8px;
        backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
        box-shadow:0 4px 24px rgba(80,130,255,0.1);'>
        <div style='font-size:10px;color:rgba(58,100,180,0.7);letter-spacing:.15em;
            text-transform:uppercase;font-family:JetBrains Mono,monospace;
            margin-bottom:6px;'>🖼️ Click below to select your medical image</div>
    </div>
    """, unsafe_allow_html=True)
    _raw_image = st.file_uploader(
        label="Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="img_uploader",
    )
    if _raw_image is not None:
        # Save bytes immediately so rerun doesn't lose it
        st.session_state.stored_image    = _raw_image.read()
        st.session_state.stored_img_name = _raw_image.name
    if st.session_state.stored_image:
        import io
        st.image(io.BytesIO(st.session_state.stored_image), width=220,
                 caption=f"✓ {st.session_state.stored_img_name} — tap 🔬 Analyse Now")

# ── Camera panel ─────────────────────────────────────────────────────────────
if st.session_state.show_camera:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.45);border-radius:16px;
        padding:10px 14px 6px;border:1px solid rgba(255,255,255,0.8);
        margin-bottom:8px;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);'>
        <div style='font-size:10px;color:rgba(58,100,180,0.7);letter-spacing:.15em;
            text-transform:uppercase;font-family:JetBrains Mono,monospace;
            margin-bottom:6px;'>📷 Point camera at the affected area & capture</div>
    </div>
    """, unsafe_allow_html=True)
    _cam = st.camera_input(label="Capture", label_visibility="collapsed", key="cam_capture")
    if _cam is not None:
        st.session_state.stored_image    = _cam.read()
        st.session_state.stored_img_name = "camera_capture.jpg"
        st.session_state.show_camera     = False
        st.rerun()

# Expose image_file from stored bytes
image_file = None
if st.session_state.stored_image:
    image_file = io.BytesIO(st.session_state.stored_image)

# ── Pill input bar — layout: [ text ... ] [ 🖼️ ] [ 🎤 ] [ 🗑️ ] ────────────────
# Uses a 3-col layout: wide text col | icon col (img+mic) | clear col
txt_col, icons_col, clr_col = st.columns([7, 3, 1], gap="small")

mic_active = st.session_state.show_voice
img_active = st.session_state.show_image

# Pill wrapper — visually surrounds text + icons together
st.markdown(f"""
<style>
/* Make the 3 cols sit flush inside a pill */
div[data-testid="stHorizontalBlock"]:has(#pill-anchor) {{
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border-radius: 50px;
    border: 1px solid rgba(255,255,255,0.8);
    padding: 2px 6px 2px 18px;
    align-items: center;
    box-shadow: 0 4px 28px rgba(80,130,255,0.14),inset 0 1px 0 rgba(255,255,255,0.9);
    margin-top: 8px;
}}
</style>
<span id='pill-anchor' style='display:none'></span>
""", unsafe_allow_html=True)

with txt_col:
    with st.form(key="chat_form", clear_on_submit=True):
        f1, f2 = st.columns([10, 1], gap="small")
        with f1:
            text_query = st.text_input(
                label="msg",
                placeholder="Ask me anything...",
                label_visibility="collapsed",
                key=f"chat_input_{st.session_state.input_key}",
            )
        with f2:
            # Hidden send — Enter key triggers it; we hide the button visually
            st.markdown("<div style='display:none'>", unsafe_allow_html=True)
            send_clicked = st.form_submit_button("➤")
            st.markdown("</div>", unsafe_allow_html=True)

with icons_col:
    ic1, ic2, ic3 = st.columns(3, gap="small")
    with ic1:
        img_class = "active-btn toolbar-btn" if img_active else "toolbar-btn pill-icon"
        st.markdown(f"<div class='{img_class}'>", unsafe_allow_html=True)
        if st.button("⊞", help="Upload image", key="img_btn"):
            st.session_state.show_image  = not st.session_state.show_image
            st.session_state.show_camera = False
            if st.session_state.show_image:
                st.session_state.show_voice = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with ic2:
        cam_active = st.session_state.show_camera
        cam_class  = "active-btn toolbar-btn" if cam_active else "toolbar-btn pill-icon"
        st.markdown(f"<div class='{cam_class}'>", unsafe_allow_html=True)
        if st.button("📷", help="Live camera capture", key="cam_btn"):
            st.session_state.show_camera = not st.session_state.show_camera
            st.session_state.show_image  = False
            st.session_state.show_voice  = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with ic3:
        mic_class = "active-btn toolbar-btn" if mic_active else "toolbar-btn pill-icon"
        st.markdown(f"<div class='{mic_class}'>", unsafe_allow_html=True)
        if st.button("🎙", help="Record voice", key="mic_btn"):
            st.session_state.show_voice  = not st.session_state.show_voice
            st.session_state.show_camera = False
            if st.session_state.show_voice:
                st.session_state.show_image = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

with clr_col:
    st.markdown("<div class='toolbar-btn'>", unsafe_allow_html=True)
    if st.button("🗑", help="Clear chat", key="clear_btn"):
        st.session_state.messages        = []
        st.session_state.autoplay_b64    = None
        st.session_state.stored_audio    = None
        st.session_state.stored_image    = None
        st.session_state.stored_img_name = None
        st.session_state.show_voice      = False
        st.session_state.show_image      = False
        st.session_state.show_camera     = False
        st.session_state.input_key      += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Hint when panels are open ─────────────────────────────────────────────────
if st.session_state.show_voice or st.session_state.show_image:
    st.markdown("""
    <div style='font-size:10px;color:rgba(58,100,180,0.5);text-align:center;
        font-family:JetBrains Mono,monospace;letter-spacing:.1em;
        text-transform:uppercase;margin:4px 0 2px;'>
        ↑ ready · tap 🔬 Analyse to process
    </div>
    """, unsafe_allow_html=True)

# ── Analyse button (only when voice or image panel is open) ──────────────────
if st.session_state.show_voice or st.session_state.show_image or st.session_state.stored_image:
    st.markdown("<div class='analyse-wrap' style='margin-top:4px;'>",
                unsafe_allow_html=True)
    analyse_clicked = st.button(
        "🔬  Analyse Now",
        use_container_width=True,
        key="analyse_btn",
    )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    analyse_clicked = False


# ══════════════════════════════════════════════════════════════════════════════
# ── TEXT CHAT PIPELINE ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if send_clicked and text_query and text_query.strip():
    _ts = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": text_query.strip(), "ts": _ts})

    history = [{"role": "system", "content": CHAT_PROMPT}]
    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant"):
            history.append({"role": m["role"], "content": m["content"]})

    with st.spinner(""):
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            messages=history, model="llama-3.3-70b-versatile",
        )
        ai_text = resp.choices[0].message.content

    st.session_state.messages.append(
        {"role": "assistant", "content": ai_text, "medical": False,
         "ts": datetime.now().strftime("%I:%M %p")}
    )
    st.session_state.input_key += 1
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ── VISION + VOICE ANALYSIS PIPELINE ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if analyse_clicked:
    _has_question = audio_file or (text_query and text_query.strip())
    if not _has_question:
        st.warning("⚠️ Record voice or type a question first.")
    elif not image_file:
        st.warning("⚠️ Upload a medical image first (tap 🖼️).")
    elif not GROQ_API_KEY:
        st.error("🔑 GROQ_API_KEY missing — check your .env file.")
    else:
        with st.spinner("🔬 Analysing..."):

            # Voice transcription
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_a:
                    tmp_a.write(audio_file.read())
                    audio_path = tmp_a.name
                patient_text = transcribe_with_groq(
                    stt_model="whisper-large-v3",
                    audio_filepath=audio_path,
                    GROQ_API_KEY=GROQ_API_KEY,
                    language=get_whisper_language_code(language),
                )
            else:
                patient_text = text_query.strip()
                audio_path   = None

            patient_text_en = (
                translate_to_english(patient_text)
                if language == "Bengali" and patient_text else patient_text
            )

            # Encode image
            img_bytes = image_file.read()
            img_b64_display = base64.b64encode(img_bytes).decode()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_i:
                tmp_i.write(img_bytes)
                image_path = tmp_i.name

            # LLM vision
            doctor_response_en = analyze_image_with_query(
                query=SYSTEM_PROMPT + "\n\nPatient says: " + (patient_text_en or ""),
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                encoded_image=encode_image(image_path),
            )

            doctor_response_out = (
                translate_to_bengali(doctor_response_en)
                if language == "Bengali" else doctor_response_en
            )

            # TTS
            tts_path = tempfile.mktemp(suffix=".mp3")
            use_elevenlabs = (
                tts_engine.startswith("ElevenLabs")
                and language == "English"
                and bool(ELEVENLABS_API_KEY)
            )
            if use_elevenlabs:
                # English — ElevenLabs highest quality, fallback to gTTS
                try:
                    text_to_speech_with_elevenlabs(doctor_response_out, tts_path, autoplay=False)
                except Exception as e:
                    st.warning(f"⚠️ ElevenLabs failed ({e.__class__.__name__}), using gTTS instead.")
                    text_to_speech_with_gtts(doctor_response_out, tts_path,
                                             language="en", autoplay=False)
            elif language == "Bengali" and tts_engine.startswith("gTTS"):
                # Bengali — gTTS fallback (if user explicitly chose it)
                text_to_speech_with_gtts(doctor_response_out, tts_path,
                                         language="bn", autoplay=False)
            elif language == "Bengali":
                # Bengali — Microsoft Neural edge-tts with selected voice
                text_to_speech_with_edge(doctor_response_out, tts_path,
                                         language="bn",
                                         voice_id=bengali_voice_id,
                                         autoplay=False)
            else:
                # English fallback — gTTS
                text_to_speech_with_gtts(doctor_response_out, tts_path,
                                         language="en", autoplay=False)

            with open(tts_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()

        # Encode input voice for history display
        input_voice_b64 = None
        if st.session_state.stored_audio:
            input_voice_b64 = base64.b64encode(st.session_state.stored_audio).decode()

        # Add to chat history
        _ts = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            "role":      "user",
            "content":   f"🔬 {patient_text or 'Voice input'}",
            "img_b64":   img_b64_display,
            "voice_b64": input_voice_b64,
            "ts":        _ts,
        })
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   doctor_response_out,
            "medical":   True,
            "audio_b64": audio_b64,
            "ts":        datetime.now().strftime("%I:%M %p"),
        })

        # Store b64 for autoplay — survives the rerun
        st.session_state.autoplay_b64   = audio_b64
        st.session_state.input_key      += 1
        # Clear stored inputs after successful analysis
        st.session_state.stored_audio    = None
        st.session_state.stored_image    = None
        st.session_state.stored_img_name = None
        st.session_state.show_voice      = False
        st.session_state.show_image      = False

        for p in [audio_path, image_path, tts_path]:
            try:
                if p: os.unlink(p)
            except Exception:
                pass

        st.rerun()

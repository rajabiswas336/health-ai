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
    initial_sidebar_state="collapsed",
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
    "messages":        [],
    "show_voice":      False,
    "show_image":      False,
    "show_camera":     False,
    "input_key":       0,
    "autoplay_b64":    None,
    "stored_audio":    None,
    "stored_image":    None,
    "stored_img_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Nunito', sans-serif !important;
    color: #1a2340 !important;
}
.stApp { background: #f0f4f8 !important; min-height: 100vh !important; }
/* Hide top black bar */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stToolbar"] { visibility: hidden !important; }

/* ════ SIDEBAR TOGGLE BUTTON — make it always visible ════ */
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebar"] { display: none !important; }
/* Sidebar itself */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e0eef4 !important;
    z-index: 999999 !important;
}
[data-testid="stSidebar"] * { color: #2d4060 !important; }
[data-testid="stSidebarContent"] { padding: 24px 18px !important; }
[data-testid="stRadio"] label {
    background: #f0f7f4 !important;
    border: 1px solid #d0e8e0 !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin: 4px 0 !important;
    color: #2d4060 !important;
}
[data-testid="stRadio"] label:hover { background: #d8f0e8 !important; }
[data-testid="stSelectbox"] > div > div {
    background: #f0f7f4 !important;
    border: 1px solid #d0e8e0 !important;
    border-radius: 12px !important;
    color: #2d4060 !important;
}

/* ════ BASE BUTTON — teal/white, no dark mode hijack ════ */
.stButton > button {
    background: #ffffff !important;
    color: #1a7a5e !important;
    border: 1.5px solid #b8e0d0 !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 9px 8px !important;
    width: 100% !important;
    transition: all .18s !important;
    line-height: 1.4 !important;
    box-shadow: 0 2px 8px rgba(0,150,100,0.07) !important;
    -webkit-appearance: none !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #e4f7f0 !important;
    border-color: #2db88a !important;
    color: #127a58 !important;
    box-shadow: 0 4px 16px rgba(45,184,138,0.2) !important;
    transform: translateY(-1px) !important;
}
/* Force light on ALL devices including mobile dark mode */
@media (prefers-color-scheme: dark) {
    .stButton > button {
        background: #ffffff !important;
        color: #1a7a5e !important;
        border-color: #b8e0d0 !important;
    }
}

/* ════ TOOLBAR CIRCLE BUTTONS — teal theme, force on all devices ════ */
.tb-col .stButton > button,
.tb-col .stButton > button:focus,
.tb-col .stButton > button:active {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    min-height: 40px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 17px !important;
    background: linear-gradient(135deg,#2db88a,#1a9e70) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 3px 12px rgba(45,184,138,0.4) !important;
    flex-shrink: 0 !important;
    -webkit-appearance: none !important;
    appearance: none !important;
}
.tb-col .stButton > button:hover {
    background: linear-gradient(135deg,#25a87d,#158060) !important;
    transform: scale(1.05) !important;
    box-shadow: 0 5px 18px rgba(45,184,138,0.5) !important;
}
.tb-col-active .stButton > button,
.tb-col-active .stButton > button:focus {
    background: linear-gradient(135deg,#1a9e70,#12784e) !important;
    border: 2.5px solid #0d5c3a !important;
    color: #ffffff !important;
    box-shadow: 0 3px 12px rgba(45,184,138,0.5), inset 0 1px 4px rgba(0,0,0,0.15) !important;
}
.tb-col-send .stButton > button,
.tb-col-send .stButton > button:focus {
    background: linear-gradient(135deg,#2db88a,#1a9e70) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(45,184,138,0.5) !important;
}
.tb-col-clear .stButton > button,
.tb-col-clear .stButton > button:focus {
    background: linear-gradient(135deg,#ff7070,#e04444) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 3px 12px rgba(220,60,60,0.35) !important;
}
/* Force override mobile dark mode for ALL toolbar buttons */
@media (prefers-color-scheme: dark) {
    .tb-col .stButton > button { background: linear-gradient(135deg,#2db88a,#1a9e70) !important; color: #fff !important; }
    .tb-col-active .stButton > button { background: linear-gradient(135deg,#1a9e70,#12784e) !important; color: #fff !important; }
    .tb-col-send .stButton > button { background: linear-gradient(135deg,#2db88a,#1a9e70) !important; color: #fff !important; }
    .tb-col-clear .stButton > button { background: linear-gradient(135deg,#ff7070,#e04444) !important; color: #fff !important; }
}

/* ════ ANALYSE BUTTON ════ */
.analyse-wrap .stButton > button {
    background: linear-gradient(135deg,#2db88a,#1a9e70) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px !important;
    box-shadow: 0 6px 22px rgba(45,184,138,0.35) !important;
    letter-spacing: .02em !important;
}
.analyse-wrap .stButton > button:hover { transform: translateY(-1px) !important; }

/* ════ FORM / TEXT INPUT ════ */
[data-testid="stForm"] {
    border: none !important; padding: 0 !important;
    background: transparent !important; box-shadow: none !important;
}
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important; outline: none !important;
    color: #1a2340 !important;
    padding: 10px 4px !important;
    font-size: 14px !important;
    -webkit-text-fill-color: #1a2340 !important;
}
[data-testid="stTextInput"] input::placeholder { color: #9ab8c0 !important; }
[data-testid="stTextInput"] > div {
    background: transparent !important; border: none !important; box-shadow: none !important;
}

/* ════ INPUT PILL ROW ════ */
.pill-row [data-testid="stHorizontalBlock"] {
    background: #ffffff !important;
    border-radius: 50px !important;
    border: 1.5px solid #c8e8de !important;
    padding: 3px 6px 3px 16px !important;
    box-shadow: 0 3px 16px rgba(0,120,80,0.09) !important;
    margin-top: 8px !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 2px !important;
}

/* ════ MOBILE: ALL columns stay horizontal ════ */
[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
    gap: 4px !important;
}
[data-testid="stHorizontalBlock"]::-webkit-scrollbar { display: none !important; }
[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    flex-shrink: 0 !important;
    min-width: 0 !important;
}

/* Quick action chips — compact on mobile */
.qa-col .stButton > button {
    white-space: nowrap !important;
    padding: 8px 10px !important;
    font-size: 11px !important;
    border-radius: 20px !important;
}
@media (max-width: 480px) {
    .qa-col .stButton > button { font-size: 10px !important; padding: 7px 6px !important; }
    .tb-col .stButton > button,
    .tb-col-active .stButton > button,
    .tb-col-send .stButton > button,
    .tb-col-clear .stButton > button {
        width: 36px !important; height: 36px !important;
        min-width: 36px !important; min-height: 36px !important;
        font-size: 15px !important;
    }
}

/* ════ AUDIO / FILE / CAMERA INPUTS ════ */
[data-testid="stAudioInput"] {
    background: #f0faf6 !important;
    border: 1.5px solid rgba(45,184,138,0.3) !important;
    border-radius: 14px !important;
}
[data-testid="stAudioInput"] button {
    background: linear-gradient(135deg,#2db88a,#1a9e70) !important;
    border-radius: 50% !important;
}
[data-testid="stFileUploader"] section {
    background: #f0faf6 !important;
    border: 2px dashed rgba(45,184,138,0.4) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #2db88a !important; background: #e4f7f0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #2db88a !important; font-weight: 700 !important;
}
[data-testid="stCameraInput"] {
    background: #f0faf6 !important;
    border: 1.5px solid rgba(45,184,138,0.3) !important;
    border-radius: 16px !important; overflow: hidden !important;
}
[data-testid="stCameraInput"] button {
    background: linear-gradient(135deg,#2db88a,#1a9e70) !important;
    color: #fff !important; border-radius: 10px !important; border: none !important;
}

/* ════ CHAT BOX CONTAINER ════ */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1.5px solid #d0eee8 !important;
    border-radius: 20px !important;
    padding: 10px 8px !important;
    box-shadow: 0 4px 24px rgba(0,100,80,0.07) !important;
}

audio {
    width: 100% !important; border-radius: 10px !important;
    margin-top: 3px !important; height: 36px !important;
}
[data-testid="stImage"] img {
    max-height: 200px !important; border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0,80,60,0.12) !important;
}
[data-testid="stAlert"] {
    background: #f0faf6 !important; border: 1px solid #b8e8d8 !important;
    border-radius: 12px !important; color: #1a5040 !important;
}
hr { border-color: #e0eae4 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #b8d8cc; border-radius: 4px; }
.block-container { max-width: 700px !important; padding: 12px 10px 80px !important; margin: 0 auto !important; }
[data-testid="stSpinner"] p { color: #2db88a !important; }

/* ════ EXPANDER (settings) ════ */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #c8e8dc !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(45,184,138,0.08) !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #1a7a5e !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}
[data-testid="stExpander"] summary:hover { color: #2db88a !important; }
</style>
""", unsafe_allow_html=True)


# ── Chat bubble renderers ─────────────────────────────────────────────────────
from datetime import datetime as _dt

def _now():
    return _dt.now().strftime("%I:%M %p")

def user_bubble(text, voice_b64=None, ts=None):
    time_str = ts or ""
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-end;align-items:flex-end;
        gap:8px;margin:10px 0 2px;padding:0 2px;'>
        <div style='max-width:75%;'>
            <div style='background:linear-gradient(135deg,#2db88a,#1a9e70);
                color:#fff;border-radius:18px 18px 4px 18px;
                padding:11px 15px;font-size:14px;line-height:1.65;
                box-shadow:0 2px 12px rgba(45,184,138,0.25);word-wrap:break-word;'>
                {text}
            </div>
            <div style='font-size:10px;color:#a0b8c0;text-align:right;
                margin-top:3px;padding-right:4px;'>
                {time_str} ✓✓
            </div>
        </div>
        <div style='width:32px;height:32px;border-radius:50%;
            background:linear-gradient(135deg,#2db88a,#1a9e70);
            display:flex;align-items:center;justify-content:center;
            font-size:15px;flex-shrink:0;box-shadow:0 2px 8px rgba(45,184,138,0.3);'>
            👤
        </div>
    </div>
    """, unsafe_allow_html=True)
    if voice_b64:
        st.markdown(
            f"<div style='display:flex;justify-content:flex-end;margin:0 40px 8px 0;'>"
            f"<div style='width:72%;background:rgba(45,184,138,0.08);border-radius:10px;"
            f"padding:6px 8px;border:1px solid rgba(45,184,138,0.2);'>"
            f"<div style='font-size:9px;color:#2db88a;margin-bottom:3px;font-weight:600;'>🎤 YOUR VOICE</div>"
            f"<audio controls style='width:100%;height:32px;border-radius:8px;'>"
            f"<source src='data:audio/mp3;base64,{voice_b64}' type='audio/mp3'></audio>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

def ai_bubble(text, is_medical=False, audio_b64=None, ts=None, do_autoplay=False):
    icon  = "🩺" if is_medical else "🤖"
    label = "Dr. AI" if is_medical else "AI Assistant"
    time_str = ts or ""
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-start;align-items:flex-end;
        gap:8px;margin:10px 0 2px;padding:0 2px;'>
        <div style='width:34px;height:34px;border-radius:50%;
            background:linear-gradient(135deg,#e8f5f0,#d0ede4);
            border:1.5px solid #b8e0d0;
            display:flex;align-items:center;justify-content:center;
            font-size:16px;flex-shrink:0;box-shadow:0 2px 8px rgba(0,120,80,0.1);'>
            {icon}
        </div>
        <div style='max-width:75%;'>
            <div style='font-size:10px;color:#7aaa9a;margin-bottom:4px;font-weight:600;
                letter-spacing:.04em;'>{label}</div>
            <div style='background:#ffffff;border:1.5px solid #e0eef8;
                color:#1a2340;border-radius:4px 18px 18px 18px;
                padding:11px 15px;font-size:14px;line-height:1.65;
                box-shadow:0 2px 12px rgba(0,80,120,0.07);word-wrap:break-word;'>
                {text}
            </div>
            <div style='font-size:10px;color:#a0b8c0;margin-top:3px;padding-left:4px;'>
                {time_str}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if audio_b64:
        _ap = "autoplay" if do_autoplay else ""
        st.markdown(
            f"<div style='display:flex;justify-content:flex-start;margin:0 0 8px 42px;'>"
            f"<div style='width:72%;background:#f0faf6;border-radius:10px;"
            f"padding:6px 8px;border:1px solid #b8e8d0;'>"
            f"<div style='font-size:9px;color:#2db88a;margin-bottom:3px;font-weight:600;'>🔊 DOCTOR RESPONSE</div>"
            f"<audio {_ap} controls style='width:100%;height:32px;border-radius:8px;'>"
            f"<source src='data:audio/mp3;base64,{audio_b64}' type='audio/mp3'></audio>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

def image_bubble(img_b64):
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-end;margin:4px 42px 4px 0;'>
        <img src='data:image/jpeg;base64,{img_b64}'
            style='max-height:180px;max-width:65%;border-radius:16px;
            box-shadow:0 4px 20px rgba(0,80,60,0.15);object-fit:cover;
            border:2px solid #d0ede4;'/>
    </div>
    """, unsafe_allow_html=True)

def day_divider(label="Today"):
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin:14px 0 8px;'>
        <div style='flex:1;height:1px;background:#e0eae4;'></div>
        <div style='font-size:10px;color:#7aaa9a;font-weight:600;
            background:#f0faf6;padding:2px 12px;border-radius:20px;
            border:1px solid #c8e8d8;'>{label}</div>
        <div style='flex:1;height:1px;background:#e0eae4;'></div>
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


# ── Settings expander (replaces sidebar — works on all devices) ───────────────
with st.expander("⚙️ Settings — Language & Voice", expanded=False):
    _col1, _col2 = st.columns(2)
    with _col1:
        language = st.radio("🌐 Language", ["English", "Bengali"], index=0)
    with _col2:
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
        )
        bengali_voice_id = bn_voice.split(" ·")[0].strip()
    else:
        bengali_voice_id = "bn-BD-NabanitaNeural"
    st.markdown("""
    <div style='font-size:12px;color:#7aaa9a;padding-top:6px;'>
    🧠 LLaMA-4 Scout · 🎙️ Groq Whisper · 🔊 ElevenLabs/edge-tts · 
    <strong>Raja Biswas</strong> · M.Tech (AI)
    </div>""", unsafe_allow_html=True)




# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:24px 0 18px;'>
    <div style='display:inline-flex;align-items:center;justify-content:center;
        width:72px;height:72px;border-radius:50%;
        background:linear-gradient(135deg,#2db88a,#1a9e70);
        font-size:34px;margin-bottom:14px;
        box-shadow:0 6px 24px rgba(45,184,138,0.35);'>🩺</div>
    <div style='font-size:clamp(18px,4vw,26px);font-weight:800;
        color:#1a2340;margin-bottom:4px;letter-spacing:-.02em;'>
        AI Based Conversational Assistant
    </div>
    <div style='font-size:clamp(13px,3vw,17px);font-weight:700;
        color:#2db88a;margin-bottom:12px;'>
        For Healthcare and Support
    </div>
    <div style='margin:8px 0 16px;'>
        <div style='font-size:15px;font-weight:700;color:#1a5040;'>Raja Biswas</div>
        <div style='font-size:11px;color:#7aaa9a;font-weight:600;
            letter-spacing:.1em;text-transform:uppercase;margin-top:3px;'>
            M.Tech · Artificial Intelligence
        </div>
    </div>
    <div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;'>
        <span style='background:#e8f5f0;border:1.5px solid #b8e0d0;
            color:#1a7a5e;font-size:11px;padding:4px 14px;border-radius:20px;
            font-weight:700;'>● LIVE</span>
        <span style='background:#e8f0ff;border:1.5px solid #c0d0ff;
            color:#3a5acc;font-size:11px;padding:4px 14px;border-radius:20px;
            font-weight:700;'>Vision + Voice</span>
        <span style='background:#fff0f0;border:1.5px solid #ffc0c0;
            color:#cc4444;font-size:11px;padding:4px 14px;border-radius:20px;
            font-weight:700;'>Educational Only</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Quick actions ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-size:11px;color:#7aaa9a;letter-spacing:.14em;
    text-transform:uppercase;font-weight:700;
    margin:4px 0 10px;text-align:center;'>Quick Actions</div>
""", unsafe_allow_html=True)

# Quick actions — horizontal scrollable chips
_qa_labels = ["🎤 Voice","🖼️ Image","🩺 Skin","🫁 X-Ray","🌐 বাংলা","💊 Ask"]
_qcols = st.columns(6, gap="small")
for _i, _col in enumerate(_qcols):
    with _col:
        st.markdown("<div class='qa-col'>", unsafe_allow_html=True)
        _clicked = st.button(_qa_labels[_i], key=f"qa_{_i}")
        st.markdown("</div>", unsafe_allow_html=True)
        if _clicked:
            if _i == 0: st.session_state.show_voice = True
            elif _i == 1: st.session_state.show_image = True
            elif _i == 2: st.session_state.show_voice = True; st.session_state.show_image = True
            elif _i == 3: st.session_state.show_image = True
            elif _i == 4: st.session_state.show_voice = True
            st.rerun()

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── CHAT WINDOW ──────────────────────────────════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="chat-anchor" style="display:none"></div>', unsafe_allow_html=True)

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
            <div style='font-size:13px;color:rgba(140,160,200,0.26);line-height:1.9;'>
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
    <div style='background:#f0faf6;border-radius:16px;padding:12px 16px 8px;
        border:1.5px solid rgba(45,184,138,0.35);margin-bottom:8px;
        box-shadow:0 4px 20px rgba(45,184,138,0.12);'>
        <div style='font-size:10px;color:#2db88a;letter-spacing:.15em;
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
            "<div style='font-size:12px;color:#1a9e70;font-weight:600;padding:4px 2px 6px;'>"
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
    <div style='background:#f0faf6;border-radius:16px;padding:12px 16px 8px;
        border:1.5px solid rgba(45,184,138,0.3);margin-bottom:8px;
        box-shadow:0 4px 20px rgba(45,184,138,0.12);'>
        <div style='font-size:10px;color:#2db88a;letter-spacing:.15em;
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
    <div style='background:#f0faf6;border-radius:16px;padding:12px 16px 8px;
        border:1.5px solid rgba(45,184,138,0.3);margin-bottom:8px;
        box-shadow:0 8px 32px rgba(56,255,180,0.12);'>
        <div style='font-size:10px;color:#2db88a;letter-spacing:.15em;
            text-transform:uppercase;font-family:JetBrains Mono,monospace;
            margin-bottom:6px;'>📷 Point camera at the affected area</div>
    </div>
    """, unsafe_allow_html=True)
    _raw_camera = st.camera_input(
        label="Take photo",
        label_visibility="collapsed",
        key="camera_input",
    )
    if _raw_camera is not None:
        st.session_state.stored_image    = _raw_camera.read()
        st.session_state.stored_img_name = "camera_capture.jpg"
        st.session_state.show_camera     = False   # close after capture
        st.session_state.show_image      = False
        st.rerun()

# Expose image_file from stored bytes (upload OR camera)
image_file = None
if st.session_state.stored_image:
    image_file = io.BytesIO(st.session_state.stored_image)

# Show preview if image captured
if st.session_state.stored_image and not st.session_state.show_image:
    st.image(io.BytesIO(st.session_state.stored_image), width=220,
             caption=f"✓ {st.session_state.stored_img_name or 'Image ready'} — tap 🔬 Analyse Now")

# ── Input bar — pill row with all controls horizontal ────────────────────────
mic_active = st.session_state.show_voice
img_active = st.session_state.show_image
cam_active = st.session_state.show_camera

st.markdown("<div class='pill-row'>", unsafe_allow_html=True)
_c_txt, _c_img, _c_cam, _c_mic, _c_snd, _c_clr = st.columns([6,1,1,1,1,1], gap="small")

with _c_txt:
    with st.form(key="chat_form", clear_on_submit=True):
        text_query = st.text_input(
            label="msg", placeholder="Ask me anything...",
            label_visibility="collapsed",
            key=f"chat_input_{st.session_state.input_key}",
        )
        st.markdown("<div style='display:none'>", unsafe_allow_html=True)
        send_clicked = st.form_submit_button("➤")
        st.markdown("</div>", unsafe_allow_html=True)

with _c_img:
    _cls = "tb-col-active" if img_active else "tb-col"
    st.markdown(f"<div class='{_cls}'>", unsafe_allow_html=True)
    if st.button("📎", help="Upload image", key="img_btn"):
        st.session_state.show_image  = not img_active
        st.session_state.show_camera = False
        if st.session_state.show_image: st.session_state.show_voice = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with _c_cam:
    _cls = "tb-col-active" if cam_active else "tb-col"
    st.markdown(f"<div class='{_cls}'>", unsafe_allow_html=True)
    if st.button("📷", help="Take photo", key="cam_btn"):
        st.session_state.show_camera = not cam_active
        st.session_state.show_image  = False
        st.session_state.show_voice  = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with _c_mic:
    _cls = "tb-col-active" if mic_active else "tb-col"
    st.markdown(f"<div class='{_cls}'>", unsafe_allow_html=True)
    if st.button("🎙", help="Record voice", key="mic_btn"):
        st.session_state.show_voice  = not mic_active
        st.session_state.show_camera = False
        if st.session_state.show_voice: st.session_state.show_image = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with _c_snd:
    st.markdown("<div class='tb-col-send'>", unsafe_allow_html=True)
    if st.button("➤", help="Send", key="send_icon_btn"):
        st.session_state["_send_trigger"] = True
    st.markdown("</div>", unsafe_allow_html=True)

with _c_clr:
    st.markdown("<div class='tb-col-clear'>", unsafe_allow_html=True)
    if st.button("🗑", help="Clear chat", key="clear_btn"):
        for _k in ["messages","autoplay_b64","stored_audio","stored_image","stored_img_name"]:
            st.session_state[_k] = [] if _k == "messages" else None
        st.session_state.show_voice  = False
        st.session_state.show_image  = False
        st.session_state.show_camera = False
        st.session_state.input_key  += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # end pill-row

# ── Analyse button — shown directly below input bar when image/voice ready ────
_has_media = (st.session_state.show_voice or st.session_state.stored_image
              or st.session_state.show_image)
if _has_media:
    st.markdown("<div class='analyse-wrap' style='margin-top:8px;'>", unsafe_allow_html=True)
    analyse_clicked = st.button("🔬  Analyse Now", use_container_width=True, key="analyse_btn")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    analyse_clicked = False

# ══════════════════════════════════════════════════════════════════════════════
# ── TEXT CHAT PIPELINE ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# Also handle send button icon click
if st.session_state.get("_send_trigger"):
    st.session_state["_send_trigger"] = False
    send_clicked = True

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

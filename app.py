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

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "messages":       [],
    "show_voice":     False,
    "show_image":     False,
    "input_key":      0,
    "autoplay_b64":   None,   # holds ONE audio to autoplay after rerun
    "stored_audio":   None,   # persists recorded audio bytes across reruns
    "stored_image":   None,   # persists uploaded image bytes across reruns
    "stored_img_name": None,  # filename for display
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    color: #e0e6f0 !important;
}
.stApp { background: #1a1d2e !important; min-height: 100vh !important; }
#MainMenu, footer { visibility: hidden; }

[data-testid="collapsedControl"] {
    visibility: visible !important; display: flex !important;
    opacity: 1 !important; pointer-events: all !important;
    z-index: 999999 !important; background: #2e3250 !important;
    border-radius: 0 10px 10px 0 !important;
}
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important; pointer-events: all !important;
}
[data-testid="stSidebar"] {
    background: #1e2236 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #c8d0e8 !important; }
[data-testid="stSidebarContent"] { padding: 24px 18px !important; }
[data-testid="stRadio"] label {
    background: #252840 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important; padding: 10px 14px !important;
    margin: 4px 0 !important; transition: background .2s !important;
}
[data-testid="stRadio"] label:hover { background: #2e3258 !important; }
[data-testid="stSelectbox"] > div > div {
    background: #252840 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important; color: #c8d0e8 !important;
}

/* ── Quick action grid buttons ── */
.stButton > button {
    background: #22263d !important; color: #b0bcd8 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important; font-weight: 700 !important;
    font-size: 12px !important; padding: 12px 6px !important;
    width: 100% !important; transition: all .2s !important;
    line-height: 1.5 !important;
}
.stButton > button:hover {
    background: #2a3060 !important; transform: translateY(-2px) !important;
    border-color: rgba(92,111,255,0.4) !important; color: #e0e6f0 !important;
}

/* ── Toolbar icon buttons ── */
.toolbar-btn .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 50% !important;
    padding: 6px !important;
    font-size: 20px !important;
    font-weight: 400 !important;
    min-height: 38px !important;
    min-width: 38px !important;
    line-height: 1 !important;
    color: #8899bb !important;
    transition: color .15s, background .15s !important;
}
.toolbar-btn .stButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    transform: none !important;
    color: #e0e6f0 !important;
}

/* ── Pill icon style (inside pill bar) ── */
.pill-icon .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 50% !important;
    min-height: 36px !important;
    min-width: 36px !important;
    font-size: 18px !important;
    color: #99aacc !important;
}
.pill-icon .stButton > button:hover {
    background: rgba(255,255,255,0.12) !important;
    color: #fff !important;
}

/* ── Active toolbar button (panel open) ── */
.active-btn .stButton > button {
    background: rgba(92,111,255,0.22) !important;
    border: 1px solid rgba(92,111,255,0.5) !important;
    border-radius: 50% !important;
    color: #8899ff !important;
}

/* ── Send button (hidden — Enter key used instead) ── */
.send-btn .stButton > button,
.send-btn [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg,#5c6fff,#38b6ff) !important;
    color: #fff !important; border: none !important;
    border-radius: 50% !important; padding: 6px !important;
    font-size: 18px !important; font-weight: 800 !important;
    min-height: 38px !important; min-width: 38px !important;
    box-shadow: 0 4px 14px rgba(92,111,255,0.45) !important;
}

/* ── Analyse button ── */
.analyse-wrap .stButton > button {
    background: linear-gradient(135deg,#5c6fff,#38b6ff) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important; font-size: 14px !important;
    font-weight: 800 !important; padding: 13px !important;
    box-shadow: 0 6px 22px rgba(92,111,255,0.35) !important;
    letter-spacing: .02em !important;
}

/* ── Strip form chrome ── */
[data-testid="stForm"] {
    border: none !important; padding: 0 !important;
    background: transparent !important; box-shadow: none !important;
}

/* ── Text input (invisible, inside card) ── */
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important; outline: none !important;
    color: #e0e6f0 !important;
    padding: 10px 4px !important; font-size: 14px !important;
    font-family: 'Nunito', sans-serif !important;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(160,170,200,0.38) !important; }
[data-testid="stTextInput"] > div {
    background: transparent !important; border: none !important; box-shadow: none !important;
}

/* ── Audio input ── */
[data-testid="stAudioInput"] {
    background: #1a1e38 !important;
    border: 1px solid rgba(92,111,255,0.28) !important;
    border-radius: 14px !important;
}
[data-testid="stAudioInput"] button {
    background: linear-gradient(135deg,#5c6fff,#38b6ff) !important;
    border-radius: 50% !important;
    box-shadow: 0 0 0 8px rgba(92,111,255,0.12) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background: #1a1e38 !important;
    border: 2px dashed rgba(56,182,255,0.35) !important;
    border-radius: 14px !important; cursor: pointer !important;
    transition: all .2s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(56,182,255,0.75) !important; background: #1e2344 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #38b6ff !important; font-size: 13px !important; font-weight: 700 !important;
}

/* ── Audio players ── */
audio {
    width: 100% !important; border-radius: 10px !important;
    margin-top: 3px !important; height: 36px !important;
    filter: invert(0) !important;
}

/* ── Image preview ── */
[data-testid="stImage"] img {
    max-height: 200px !important; max-width: 100% !important;
    width: auto !important; border-radius: 14px !important;
    object-fit: contain !important; display: block !important; margin: 4px auto !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #1e2138 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important; color: #c8d0e8 !important;
}

hr { border-color: rgba(255,255,255,0.05) !important; }
[data-testid="stCaptionContainer"] p { color: rgba(160,175,210,0.45) !important; font-size: 11px !important; }
/* ── Chat window container ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg,#12152a 0%,#161929 100%) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 18px !important;
    padding: 10px 6px !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2e50; border-radius: 4px; }
.block-container { max-width: 700px !important; padding: 20px 16px 60px !important; margin: 0 auto !important; }
[data-testid="stSpinner"] p { color: #8899ff !important; }
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
            <div style='background:linear-gradient(135deg,#4f5fff 0%,#3ab0ff 100%);
                color:#fff;border-radius:18px 18px 4px 18px;
                padding:10px 14px;font-size:14px;line-height:1.65;
                box-shadow:0 2px 12px rgba(79,95,255,0.35);word-wrap:break-word;'>
                {text}
            </div>
            <div style='font-size:10px;color:rgba(140,155,200,0.45);text-align:right;
                margin-top:3px;padding-right:4px;font-family:JetBrains Mono,monospace;'>
                {time_str} ✓✓
            </div>
        </div>
        <div style='width:32px;height:32px;border-radius:50%;
            background:linear-gradient(135deg,#4f5fff,#3ab0ff);
            display:flex;align-items:center;justify-content:center;
            font-size:15px;flex-shrink:0;box-shadow:0 2px 8px rgba(79,95,255,0.3);'>
            👤
        </div>
    </div>
    """, unsafe_allow_html=True)
    if voice_b64:
        st.markdown(
            f"<div style='display:flex;justify-content:flex-end;margin:0 40px 8px 0;'>"
            f"<div style='width:72%;background:rgba(79,95,255,0.12);border-radius:10px;"
            f"padding:6px 8px;border:1px solid rgba(79,95,255,0.2);'>"
            f"<div style='font-size:9px;color:rgba(140,160,255,0.55);margin-bottom:3px;"
            f"font-family:JetBrains Mono,monospace;letter-spacing:.1em;'>🎤 YOUR VOICE</div>"
            f"<audio controls style='width:100%;height:32px;border-radius:8px;'>"
            f"<source src='data:audio/mp3;base64,{voice_b64}' type='audio/mp3'></audio>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

def ai_bubble(text, is_medical=False, audio_b64=None, ts=None, do_autoplay=False):
    border = "rgba(56,182,255,0.2)" if is_medical else "rgba(255,255,255,0.06)"
    glow   = "rgba(56,182,255,0.08)" if is_medical else "rgba(255,255,255,0.02)"
    icon   = "🩺" if is_medical else "🫀"
    label  = "Dr. AI" if is_medical else "AI Assistant"
    time_str = ts or ""
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-start;align-items:flex-end;
        gap:8px;margin:10px 0 2px;padding:0 2px;'>
        <div style='width:32px;height:32px;border-radius:50%;
            background:#1e2240;border:1px solid {border};
            display:flex;align-items:center;justify-content:center;
            font-size:15px;flex-shrink:0;'>
            {icon}
        </div>
        <div style='max-width:75%;'>
            <div style='font-size:10px;color:rgba(140,160,200,0.5);
                margin-bottom:4px;font-family:JetBrains Mono,monospace;
                letter-spacing:.08em;'>{label}</div>
            <div style='background:#1e2240;border:1px solid {border};
                background:linear-gradient(135deg,#1e2240,#1a1e38);
                color:#d8e0f0;border-radius:4px 18px 18px 18px;
                padding:10px 14px;font-size:14px;line-height:1.65;
                box-shadow:0 2px 12px rgba(0,0,0,0.25);word-wrap:break-word;'>
                {text}
            </div>
            <div style='font-size:10px;color:rgba(140,155,200,0.4);
                margin-top:3px;padding-left:4px;font-family:JetBrains Mono,monospace;'>
                {time_str}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if audio_b64:
        _autoplay_attr = "autoplay" if do_autoplay else ""
        st.markdown(
            f"<div style='display:flex;justify-content:flex-start;margin:0 0 8px 40px;'>"
            f"<div style='width:72%;background:rgba(56,182,255,0.08);border-radius:10px;"
            f"padding:6px 8px;border:1px solid rgba(56,182,255,0.18);'>"
            f"<div style='font-size:9px;color:rgba(56,182,255,0.55);margin-bottom:3px;"
            f"font-family:JetBrains Mono,monospace;letter-spacing:.1em;'>🔊 DOCTOR RESPONSE</div>"
            f"<audio {_autoplay_attr} controls style='width:100%;height:32px;border-radius:8px;'>"
            f"<source src='data:audio/mp3;base64,{audio_b64}' type='audio/mp3'></audio>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

def image_bubble(img_b64):
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-end;margin:4px 40px 4px 0;'>
        <img src='data:image/jpeg;base64,{img_b64}'
            style='max-height:180px;max-width:65%;border-radius:14px;
            box-shadow:0 4px 20px rgba(0,0,0,0.4);object-fit:cover;
            border:1px solid rgba(255,255,255,0.08);'/>
    </div>
    """, unsafe_allow_html=True)

def day_divider(label="Today"):
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin:14px 0 8px;'>
        <div style='flex:1;height:1px;background:rgba(255,255,255,0.06);'></div>
        <div style='font-size:10px;color:rgba(140,155,200,0.4);
            font-family:JetBrains Mono,monospace;letter-spacing:.12em;
            background:#1a1d2e;padding:2px 10px;border-radius:20px;
            border:1px solid rgba(255,255,255,0.06);'>{label}</div>
        <div style='flex:1;height:1px;background:rgba(255,255,255,0.06);'></div>
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
    <div style='text-align:center;padding:4px 0 16px;'>
        <div style='font-size:44px;margin-bottom:8px;'>🫀</div>
        <div style='font-family:Nunito,sans-serif;font-size:15px;font-weight:800;
            color:#c8d0e8;'>AI Healthcare Assistant</div>
        <div style='font-size:10px;color:rgba(160,175,210,0.4);margin-top:5px;
            font-family:JetBrains Mono,monospace;letter-spacing:.1em;'>
            v2.0 · M.Tech AI</div>
    </div>
    <div style='height:1px;background:rgba(255,255,255,0.06);margin-bottom:18px;'></div>
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
    <div style='height:1px;background:rgba(255,255,255,0.06);margin:16px 0;'></div>
    <div style='font-size:11px;color:rgba(160,175,210,0.4);letter-spacing:.12em;
        text-transform:uppercase;font-family:JetBrains Mono,monospace;margin-bottom:8px;'>
        Powered by</div>
    <div style='font-size:13px;color:#a0aac8;line-height:2.1;'>
        🧠 LLaMA-4 Scout Vision<br>
        🎙️ Groq Whisper Large v3<br>
        🔊 ElevenLabs / edge-tts / gTTS<br>
        🌐 Deep Translator
    </div>
    <div style='height:1px;background:rgba(255,255,255,0.06);margin:16px 0;'></div>
    <div style='font-size:12px;color:rgba(160,175,210,0.45);line-height:1.8;'>
        <strong style='color:#c8d0e8;'>Raja Biswas</strong><br>
        M.Tech (AI) · Clinical Decision Support System
    </div>
    """, unsafe_allow_html=True)




# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:18px 0 14px;'>
    <div style='font-size:58px;margin-bottom:10px;
        filter:drop-shadow(0 0 24px rgba(92,111,255,0.6));'>🫀</div>
    <div style='font-family:Nunito,sans-serif;font-size:clamp(17px,4vw,24px);
        font-weight:800;color:#e0e6f0;margin-bottom:3px;'>
        AI Based Conversational Assistant
    </div>
    <div style='font-family:Nunito,sans-serif;font-size:clamp(13px,3vw,17px);
        font-weight:700;color:#5c6fff;margin-bottom:12px;'>
        For Healthcare and Support
    </div>
    <div style='margin:6px 0 14px;'>
        <div style='font-family:Nunito,sans-serif;font-size:15px;
            font-weight:800;color:#c8d0e8;letter-spacing:.02em;'>
            Raja Biswas
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:11px;
            color:rgba(140,160,220,0.5);letter-spacing:.12em;
            text-transform:uppercase;margin-top:3px;'>
            M.Tech · Artificial Intelligence
        </div>
    </div>
    <div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;'>
        <span style='background:#252840;border:1px solid rgba(92,111,255,0.25);
            color:#8899ff;font-size:11px;padding:3px 12px;border-radius:20px;
            font-family:JetBrains Mono,monospace;'>● LIVE</span>
        <span style='background:#252840;border:1px solid rgba(56,182,255,0.2);
            color:#38b6ff;font-size:11px;padding:3px 12px;border-radius:20px;
            font-family:JetBrains Mono,monospace;'>Vision + Voice</span>
        <span style='background:#252840;border:1px solid rgba(255,100,100,0.18);
            color:#ff8080;font-size:11px;padding:3px 12px;border-radius:20px;
            font-family:JetBrains Mono,monospace;'>Educational Only</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Quick actions ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-size:10px;color:rgba(160,175,210,0.38);letter-spacing:.14em;
    text-transform:uppercase;font-family:JetBrains Mono,monospace;
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
    <div style='background:#1e2240;border-radius:16px;padding:12px 16px 8px;
        border:1px solid rgba(92,111,255,0.35);margin-bottom:8px;
        box-shadow:0 8px 32px rgba(92,111,255,0.2);'>
        <div style='font-size:10px;color:#8899ff;letter-spacing:.15em;
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
            "<div style='font-size:12px;color:#22c55e;padding:4px 2px 6px;'>"
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
    <div style='background:#1e2240;border-radius:16px;padding:12px 16px 8px;
        border:1px solid rgba(56,182,255,0.3);margin-bottom:8px;
        box-shadow:0 8px 32px rgba(56,182,255,0.15);'>
        <div style='font-size:10px;color:#38b6ff;letter-spacing:.15em;
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

# Expose image_file from stored bytes
image_file = None
if st.session_state.stored_image:
    image_file = io.BytesIO(st.session_state.stored_image)

# ── Pill input bar — layout: [ text ... ] [ 🖼️ ] [ 🎤 ] [ 🗑️ ] ────────────────
# Uses a 3-col layout: wide text col | icon col (img+mic) | clear col
txt_col, icons_col, clr_col = st.columns([8, 2, 1], gap="small")

mic_active = st.session_state.show_voice
img_active = st.session_state.show_image

# Pill wrapper — visually surrounds text + icons together
st.markdown(f"""
<style>
/* Make the 3 cols sit flush inside a pill */
div[data-testid="stHorizontalBlock"]:has(#pill-anchor) {{
    background: #22263d;
    border-radius: 50px;
    border: 1px solid rgba(255,255,255,0.09);
    padding: 2px 6px 2px 18px;
    align-items: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    margin-top: 6px;
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
    ic1, ic2 = st.columns(2, gap="small")
    with ic1:
        img_class = "active-btn toolbar-btn" if img_active else "toolbar-btn pill-icon"
        st.markdown(f"<div class='{img_class}'>", unsafe_allow_html=True)
        if st.button("⊞", help="Upload image", key="img_btn"):
            st.session_state.show_image = not st.session_state.show_image
            if st.session_state.show_image:
                st.session_state.show_voice = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with ic2:
        mic_class = "active-btn toolbar-btn" if mic_active else "toolbar-btn pill-icon"
        st.markdown(f"<div class='{mic_class}'>", unsafe_allow_html=True)
        if st.button("🎙", help="Record voice", key="mic_btn"):
            st.session_state.show_voice = not st.session_state.show_voice
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
        st.session_state.input_key      += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Hint when panels are open ─────────────────────────────────────────────────
if st.session_state.show_voice or st.session_state.show_image:
    st.markdown("""
    <div style='font-size:10px;color:rgba(140,160,255,0.4);text-align:center;
        font-family:JetBrains Mono,monospace;letter-spacing:.1em;
        text-transform:uppercase;margin:4px 0 2px;'>
        ↑ ready · tap 🔬 Analyse to process
    </div>
    """, unsafe_allow_html=True)

# ── Analyse button (only when voice or image panel is open) ──────────────────
if st.session_state.show_voice or st.session_state.show_image:
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

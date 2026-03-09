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
import base64

def get_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()
load_dotenv()
icon_base64 = get_base64_image("icon.png")
# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="icon.png",
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""           
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; }

/* ── App background ── */
.stApp { 
    background: #bfdbf2 !important;
    background: linear-gradient(
        180deg,
        rgba(191, 219, 242, 1) 0%,
        rgba(93, 146, 199, 1) 75%,
        rgba(57, 95, 128, 1) 100%
    ) !important;
    min-height: 100vh !important;
}

/* Keep header for sidebar toggle but hide visual space */
header[data-testid="stHeader"] {
    background: transparent !important;
}

header[data-testid="stHeader"] > div {
    padding-top: 0px !important;
}
.stMainBlockContainer { padding-top:0 !important; margin-top:0 !important; }
section[data-testid="stMain"] { padding-top:0 !important; margin-top:0 !important; }
section[data-testid="stMain"] > div { padding-top:0 !important; margin-top:0 !important; }

/* ── Main container — centered, max width, good vertical breathing room ── */
.block-container { 
    max-width: 680px !important; 
    padding: 60px 24px 40px !important; 
    margin: 0 auto !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#5b82b5 !important; border-right:1px solid rgba(255,255,255,0.2) !important; }
[data-testid="stSidebarContent"] { padding:24px 18px !important; }

/* Sidebar text white but NOT buttons */
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color:white !important; }

/* ── collapsedControl: the button shown when sidebar is CLOSED ── */
/* Sidebar open button */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: all !important;
    z-index: 999999 !important;

    background: #2563eb !important;   /* OUTSIDE COLOR */
    border-radius: 0 10px 10px 0 !important;
}

/* actual button element */
[data-testid="collapsedControl"] button {
    background: #2563eb !important;   /* FIX */
    border: none !important;
}

/* icon color */
[data-testid="collapsedControl"] svg {
    color: white !important;
    fill: white !important;
}
/* ── Sidebar header close button: shown when sidebar is OPEN ── */
[data-testid="stSidebarHeader"] {
    display:flex !important;
    visibility:visible !important;
    opacity:1 !important;
}
[data-testid="stSidebarHeader"] button {
    display:flex !important;
    visibility:visible !important;
    opacity:1 !important;
    pointer-events:all !important;
    color:#ffffff !important;
    background:#2563eb !important;   /* NEW */
    border-radius:6px !important;
}
[data-testid="stSidebarHeader"] button svg { color:white !important; fill:white !important; }
[data-testid="stRadio"] label { background:rgba(255,255,255,0.15) !important; border:1px solid rgba(255,255,255,0.2) !important; border-radius:12px !important; padding:10px 14px !important; margin:4px 0 !important; }
[data-testid="stRadio"] label:hover { background:rgba(255,255,255,0.25) !important; }
[data-testid="stSelectbox"] > div > div { background:rgba(255,255,255,0.15) !important; border:1px solid rgba(255,255,255,0.2) !important; border-radius:12px !important; color:white !important; }

/* ── Quick action buttons — professional pill style ── */
.stButton > button {
    background: rgba(255,255,255,0.45) !important;
    color: #1e3a8a !important;
    border: 1px solid rgba(37,99,235,0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
    
    width: 100% !important;
    transition: all .2s ease !important;
    line-height: 1.2 !important;
    backdrop-filter: blur(6px) !important;
    box-shadow: 0 1px 4px rgba(37,99,235,0.08), inset 0 1px 0 rgba(255,255,255,0.6) !important;
}
.stButton > button:hover {
    background: rgba(37,99,235,0.12) !important;
    border-color: rgba(37,99,235,0.45) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.18) !important;
    color: #1e40af !important;
}

/* ── ALL form submit buttons — blue round style ── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg,#2563eb,#38bdf8) !important;
    border: none !important;
    border-radius: 50% !important;
    min-height: 40px !important;
    min-width: 40px !important;
    width: 40px !important;
    height: 40px !important;
    font-size: 17px !important;
    color: white !important;
    padding: 0 !important;
    box-shadow: 0 3px 10px rgba(37,99,235,0.35) !important;
    font-weight: 600 !important;
    transition: all .2s !important;
    line-height: 1 !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg,#1d4ed8,#0ea5e9) !important;
    transform: scale(1.1) !important;
    box-shadow: 0 5px 16px rgba(37,99,235,0.5) !important;
}

/* ── Strip form chrome completely ── */
[data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 8px 0 0 0 !important;
}

/* ── THE PILL BAR — visible border, frosted glass ── */
[data-testid="stForm"] > div > [data-testid="stHorizontalBlock"] {
    background: rgba(255,255,255,0.72) !important;
    border-radius: 50px !important;
    border: 2px solid #2563eb !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.18) !important;
    padding: 5px 8px 5px 18px !important;
    align-items: center !important;
    backdrop-filter: blur(8px) !important;
    gap: 4px !important;
    margin: 0 !important;
}

/* ── Text input inside pill — glassmorphism style ── */
[data-testid="stForm"] [data-testid="stTextInput"] input {
    background: linear-gradient(135deg,rgb(218,232,247),rgb(214,229,247)) !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #2563eb !important;
    font-size: 15px !important;
    padding: 8px 12px !important;
    border-radius: 1000px !important;
    font-family: 'Nunito', sans-serif !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] > div,
[data-testid="stForm"] [data-testid="stTextInput"] > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input::placeholder { color: #9ebcd9 !important; font-style: italic !important; }

/* ── Columns inside form pill — vertically centered ── */
[data-testid="stForm"] [data-testid="column"] {
    display: flex !important;
    align-items: center !important;
    padding: 0 !important;
}

/* ── Chat window — transparent, no background ── */
.chat-box {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Kill Streamlit wrapper padding above/below chat-box and form when chat is empty */
[data-testid="stVerticalBlock"] > [data-testid="element-container"]:has(> div.chat-box) {
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stVerticalBlock"] > [data-testid="element-container"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
/* Remove default gap Streamlit adds between stVerticalBlock children */
[data-testid="stVerticalBlock"] { gap: 0 !important; }
/* But restore gap inside sidebar so its contents don't collapse */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
[data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }

/* ── Analyse button ── */
.analyse-wrap .stButton > button {
    background: linear-gradient(135deg,#2563eb,#38bdf8) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important; font-size: 14px !important;
    font-weight: 800 !important; padding: 13px !important;
    box-shadow: 0 6px 22px rgba(37,99,235,0.35) !important;
}

/* ── Audio / file / camera inputs ── */
[data-testid="stAudioInput"] { background:#1a1e38 !important; border:1px solid rgba(92,111,255,0.28) !important; border-radius:14px !important; }
[data-testid="stAudioInput"] button { background:linear-gradient(135deg,#5c6fff,#38b6ff) !important; border-radius:50% !important; }
[data-testid="stFileUploader"] section { background:#1a1e38 !important; border:2px dashed rgba(56,182,255,0.35) !important; border-radius:14px !important; }
[data-testid="stCameraInput"] { background:rgba(56,182,255,0.05) !important; border:1.5px solid rgba(56,182,255,0.3) !important; border-radius:16px !important; overflow:hidden !important; }
[data-testid="stCameraInput"] button { background:linear-gradient(135deg,#5c6fff,#38b6ff) !important; color:#fff !important; border-radius:10px !important; border:none !important; }

/* ── Audio players ── */
audio { width:100% !important; border-radius:10px !important; margin-top:3px !important; height:36px !important; }

/* ── Misc ── */
[data-testid="stImage"] img { max-height:200px !important; max-width:100% !important; width:auto !important; border-radius:14px !important; object-fit:contain !important; display:block !important; margin:4px auto !important; box-shadow:0 4px 20px rgba(0,0,0,0.4) !important; }
[data-testid="stAlert"] { background:#1e2138 !important; border:1px solid rgba(255,255,255,0.07) !important; border-radius:12px !important; color:#c8d0e8 !important; }
hr { border-color:rgba(255,255,255,0.05) !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#2a2e50; border-radius:4px; }

/* ── LIVE dot pulse ── */
.live-dot { width:8px; height:8px; background:#22c55e; border-radius:50%; display:inline-block; margin-right:6px; animation:pulseLive 1.6s infinite; }
@keyframes pulseLive { 0%{box-shadow:0 0 0 0 rgba(34,197,94,0.7);} 70%{box-shadow:0 0 0 8px rgba(34,197,94,0);} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0);} }

/* ── Dark theme overrides (applied via JS to <body class="dark-mode">) ── */
body.dark-mode .stApp { background: linear-gradient(180deg,#0f172a,#1e293b,#0f2744) !important; }
body.dark-mode [data-testid="stSidebar"] { background:#0f172a !important; border-right:1px solid rgba(255,255,255,0.07) !important; }
body.dark-mode .block-container * { color:#e2e8f0 !important; }
body.dark-mode .stButton > button { background:rgba(30,41,59,0.8) !important; color:#93c5fd !important; border-color:rgba(99,148,235,0.3) !important; box-shadow: 0 1px 4px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important; }
body.dark-mode .stButton > button:hover { background:rgba(37,99,235,0.25) !important; color:#bfdbfe !important; }
body.dark-mode [data-testid="stForm"] [data-testid="stTextInput"] input { background:linear-gradient(135deg,#1e2d45,#162035) !important; color:#93c5fd !important; }
body.dark-mode [data-testid="stForm"] [data-testid="stTextInput"] input::placeholder { color:#4a6a9a !important; }

/* ── Theme toggle button styles ── */
#theme-toggle-button {
  font-size: 11px;
  position: fixed;
  top: 14px;
  right: 18px;
  display: inline-block;
  width: 7em;
  cursor: pointer;
  z-index: 999999;
  filter: drop-shadow(0 2px 8px rgba(0,0,0,0.18));
}
#toggle { opacity:0; width:0; height:0; }
#container, #patches, #stars, #button, #sun, #moon, #cloud {
  transition-property: all;
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
  transition-duration: 0.25s;
}
#toggle:checked + svg #container { fill: #2b4360; }
#toggle:checked + svg #button { transform: translate(28px, 2.333px); }
#sun { opacity: 1; }
#toggle:checked + svg #sun { opacity: 0; }
#moon { opacity: 0; }
#toggle:checked + svg #moon { opacity: 1; }
#cloud { opacity: 1; }
#toggle:checked + svg #cloud { opacity: 0; }
#stars { opacity: 0; }
#toggle:checked + svg #stars { opacity: 1; }
/* ── Collapse gap between quick actions and input bar ── */
.stMainBlockContainer .block-container > div > div { gap: 0 !important; }
.stMainBlockContainer [data-testid="stVerticalBlock"] > div:empty { display:none !important; height:0 !important; margin:0 !important; padding:0 !important; }
.stMainBlockContainer [data-testid="element-container"]:has(.chat-box:empty) { margin:0 !important; padding:0 !important; height:0 !important; }

</style>""", unsafe_allow_html=True)


# ── Theme toggle button ────────────────────────────────────────────────────────
st.markdown("""
<label id="theme-toggle-button" title="Toggle dark / light mode">
  <input type="checkbox" id="toggle">
  <svg viewBox="0 0 69.667 44" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg">
    <g transform="translate(3.5 3.5)" data-name="Component 15 – 1" id="Component_15_1">
      <g filter="url(#container)" transform="matrix(1, 0, 0, 1, -3.5, -3.5)">
        <rect fill="#83cbd8" transform="translate(3.5 3.5)" rx="17.5" height="35" width="60.667" data-name="container" id="container"></rect>
      </g>
      <g transform="translate(2.333 2.333)" id="button">
        <g data-name="sun" id="sun">
          <g filter="url(#sun-outer)" transform="matrix(1, 0, 0, 1, -5.83, -5.83)">
            <circle fill="#f8e664" transform="translate(5.83 5.83)" r="15.167" cy="15.167" cx="15.167" data-name="sun-outer" id="sun-outer-2"></circle>
          </g>
          <g filter="url(#sun)" transform="matrix(1, 0, 0, 1, -5.83, -5.83)">
            <path fill="rgba(246,254,247,0.29)" transform="translate(9.33 9.33)" d="M11.667,0A11.667,11.667,0,1,1,0,11.667,11.667,11.667,0,0,1,11.667,0Z" data-name="sun" id="sun-3"></path>
          </g>
          <circle fill="#fcf4b9" transform="translate(8.167 8.167)" r="7" cy="7" cx="7" id="sun-inner"></circle>
        </g>
        <g data-name="moon" id="moon">
          <g filter="url(#moon)" transform="matrix(1, 0, 0, 1, -31.5, -5.83)">
            <circle fill="#cce6ee" transform="translate(31.5 5.83)" r="15.167" cy="15.167" cx="15.167" data-name="moon" id="moon-3"></circle>
          </g>
          <g fill="#a6cad0" transform="translate(-24.415 -1.009)" id="patches">
            <circle transform="translate(43.009 4.496)" r="2" cy="2" cx="2"></circle>
            <circle transform="translate(39.366 17.952)" r="2" cy="2" cx="2" data-name="patch"></circle>
            <circle transform="translate(33.016 8.044)" r="1" cy="1" cx="1" data-name="patch"></circle>
            <circle transform="translate(51.081 18.888)" r="1" cy="1" cx="1" data-name="patch"></circle>
            <circle transform="translate(33.016 22.503)" r="1" cy="1" cx="1" data-name="patch"></circle>
            <circle transform="translate(50.081 10.53)" r="1.5" cy="1.5" cx="1.5" data-name="patch"></circle>
          </g>
        </g>
      </g>
      <g filter="url(#cloud)" transform="matrix(1, 0, 0, 1, -3.5, -3.5)">
        <path fill="#fff" transform="translate(-3466.47 -160.94)" d="M3512.81,173.815a4.463,4.463,0,0,1,2.243.62.95.95,0,0,1,.72-1.281,4.852,4.852,0,0,1,2.623.519c.034.02-.5-1.968.281-2.716a2.117,2.117,0,0,1,2.829-.274,1.821,1.821,0,0,1,.854,1.858c.063.037,2.594-.049,3.285,1.273s-.865,2.544-.807,2.626a12.192,12.192,0,0,1,2.278.892c.553.448,1.106,1.992-1.62,2.927a7.742,7.742,0,0,1-3.762-.3c-1.28-.49-1.181-2.65-1.137-2.624s-1.417,2.2-2.623,2.2a4.172,4.172,0,0,1-2.394-1.206,3.825,3.825,0,0,1-2.771.774c-3.429-.46-2.333-3.267-2.2-3.55A3.721,3.721,0,0,1,3512.81,173.815Z" data-name="cloud" id="cloud"></path>
      </g>
      <g fill="#def8ff" transform="translate(3.585 1.325)" id="stars">
        <path transform="matrix(-1, 0.017, -0.017, -1, 24.231, 3.055)" d="M.774,0,.566.559,0,.539.458.933.25,1.492l.485-.361.458.394L1.024.953,1.509.592.943.572Z"></path>
        <path transform="matrix(-0.777, 0.629, -0.629, -0.777, 23.185, 12.358)" d="M1.341.529.836.472.736,0,.505.46,0,.4.4.729l-.231.46L.605.932l.4.326L.9.786Z" data-name="star"></path>
        <path transform="matrix(0.438, 0.899, -0.899, 0.438, 23.177, 29.735)" d="M.015,1.065.475.9l.285.365L.766.772l.46-.164L.745.494.751,0,.481.407,0,.293.285.658Z" data-name="star"></path>
        <path transform="translate(12.677 0.388) rotate(104)" d="M1.161,1.6,1.059,1,1.574.722.962.607.86,0,.613.572,0,.457.446.881.2,1.454l.516-.274Z" data-name="star"></path>
        <path transform="matrix(-0.07, 0.998, -0.998, -0.07, 11.066, 15.457)" d="M.873,1.648l.114-.62L1.579.945,1.03.62,1.144,0,.706.464.157.139.438.7,0,1.167l.592-.083Z" data-name="star"></path>
        <path transform="translate(8.326 28.061) rotate(11)" d="M.593,0,.638.724,0,.982l.7.211.045.724.36-.64.7.211L1.342.935,1.7.294,1.063.552Z" data-name="star"></path>
        <path transform="translate(5.012 5.962) rotate(172)" d="M.816,0,.5.455,0,.311.323.767l-.312.455.516-.215.323.456L.827.911,1.343.7.839.552Z" data-name="star"></path>
        <path transform="translate(2.218 14.616) rotate(169)" d="M1.261,0,.774.571.114.3.487.967,0,1.538.728,1.32l.372.662.047-.749.728-.218L1.215.749Z" data-name="star"></path>
      </g>
    </g>
  </svg>
</label>

<script>
(function() {
    const toggle = document.getElementById('toggle');
    if (!toggle) return;
    // Restore saved preference
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') {
        toggle.checked = true;
        document.body.classList.add('dark-mode');
        window.parent.document.body.classList.add('dark-mode');
    }
    toggle.addEventListener('change', function() {
        if (this.checked) {
            document.body.classList.add('dark-mode');
            window.parent.document.body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.classList.remove('dark-mode');
            window.parent.document.body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
        }
    });
})();
</script>
""", unsafe_allow_html=True)


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

st.markdown(f"""
<div style='text-align:center; padding:40px 0 20px;'>
    <div style='font-family:"Times New Roman", serif;font-size:clamp(28px,5vw,40px);
        font-weight:700;color:#1e3a8a;margin-bottom:4px;letter-spacing:-0.01em;'>
        AI Based Conversational Assistant
</div>

<div style='font-family:"Times New Roman", serif;font-size:clamp(13px,3vw,16px);
        font-weight:600;color:#2563eb;margin-bottom:14px;'>
        For Healthcare and Support
</div>
    <div style='margin-bottom:10px;'>
        <img src="data:image/png;base64,{icon_base64}" width="56" style="filter:drop-shadow(0 3px 8px rgba(0,0,0,0.15));">
    </div>
    <div style='margin:0 0 12px;'>
        <div style='font-family:Nunito,sans-serif;font-size:15px;
            font-weight:800;color:#1e3a8a;letter-spacing:.02em;'>
            Raja Biswas
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;
            color:#2563eb;letter-spacing:.14em;
            text-transform:uppercase;margin-top:2px;'>
            M.Tech · Artificial Intelligence
        </div>
    </div>
    <div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;'>
        <span style='display:inline-flex;align-items:center;gap:5px;background:#e0f2fe;border:1px solid #7dd3fc;color:#0369a1;font-size:11px;padding:4px 12px;border-radius:20px;font-family:JetBrains Mono,monospace;'><span class="live-dot"></span>LIVE</span>
        <span style='display:inline-flex;align-items:center;background:#e6f6ff;border:1px solid #7dd3fc;color:#0284c7;font-size:11px;padding:4px 12px;border-radius:20px;font-family:JetBrains Mono,monospace;'>Vision + Voice</span>
        <span style='display:inline-flex;align-items:center;background:#fff1f2;border:1px solid #fca5a5;color:#b91c1c;font-size:11px;padding:4px 12px;border-radius:20px;font-family:JetBrains Mono,monospace;'>Educational Only</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Quick actions ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-size:10px;color:rgba(100,130,180,0.5);letter-spacing:.14em;
    text-transform:uppercase;font-family:JetBrains Mono,monospace;
    margin:2px 0 10px;text-align:center;'>Quick Actions</div>
<style>
/* Equal-size, tight quick action buttons */
[data-testid="stHorizontalBlock"]:not([data-testid="stForm"] [data-testid="stHorizontalBlock"]) {
    gap: 6px !important;
    justify-content: center !important;
}
[data-testid="stHorizontalBlock"]:not([data-testid="stForm"] [data-testid="stHorizontalBlock"]) > div {
    padding: 0 !important;
    min-width: 0 !important;
    flex: 1 1 0 !important;
}
[data-testid="stHorizontalBlock"]:not([data-testid="stForm"] [data-testid="stHorizontalBlock"]) .stButton,
[data-testid="stHorizontalBlock"]:not([data-testid="stForm"] [data-testid="stHorizontalBlock"]) .stButton > button {
    width: 100% !important;
    min-height: 40px !important;
    height: auto !important;
    font-size: 13px !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    padding: 8px 6px !important;
}
/* Center the form (input bar) */
[data-testid="stForm"] {
    max-width: 680px !important;
    margin: 8px auto 0 !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns([1,1,1,1,1,1])
with c1:
    if st.button("Voice", use_container_width=True):
        st.session_state.show_voice = True; st.rerun()
with c2:
    if st.button("Image", use_container_width=True):
        st.session_state.show_image = True; st.rerun()
with c3:
    if st.button("Skin", use_container_width=True):
        st.session_state.show_voice = True
        st.session_state.show_image = True; st.rerun()
with c4:
    if st.button("Radiology", use_container_width=True):
        st.session_state.show_image = True; st.rerun()
with c5:
    if st.button("বাংলা", use_container_width=True):
        st.session_state.show_voice = True; st.rerun()
with c6:
    if st.button("Ask", use_container_width=True):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ── CHAT WINDOW ──────────────────────────────════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.messages:
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    with st.container():
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
    st.markdown("</div>", unsafe_allow_html=True)

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

# ── Camera panel ─────────────────────────────────────────────────────────────
if st.session_state.show_camera:
    st.markdown("""
    <div style='background:rgba(56,182,255,0.06);border-radius:16px;
        padding:10px 14px 6px;border:1px solid rgba(56,182,255,0.25);
        margin-bottom:8px;'>
        <div style='font-size:10px;color:#38b6ff;letter-spacing:.15em;
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
mic_active = st.session_state.show_voice
img_active = st.session_state.show_image

# ── Input bar ─────────────────────────────────────────────────────────────────
# Single form wrapping ALL columns — pill styled via global CSS
with st.form(key="chat_form", clear_on_submit=True):
    txt_col, send_col, ic1_col, ic2_col, ic3_col, clr_col = st.columns([9, 1, 1, 1, 1, 1], gap="small")
    with txt_col:
        text_query = st.text_input(
            label="msg",
            placeholder="Ask me anything...",
            label_visibility="collapsed",
            key=f"chat_input_{st.session_state.input_key}",
        )
    with send_col:
        send_clicked = st.form_submit_button("➤")
    with ic1_col:
        img_pressed = st.form_submit_button("⊞", help="Upload image")
    with ic2_col:
        cam_pressed = st.form_submit_button("📷", help="Camera")
    with ic3_col:
        mic_pressed = st.form_submit_button("🎙", help="Record voice")
    with clr_col:
        clr_pressed = st.form_submit_button("🗑", help="Clear chat")

# ── JS injector via components (actually executes, unlike st.markdown script) ──
import streamlit.components.v1 as _components
_components.html("""
<script>
function applyStyles() {
    const doc = window.parent.document;

    // 1. PILL BAR — apply the Uiverse glassmorphism gradient style
    const forms = doc.querySelectorAll('[data-testid="stForm"]');
    forms.forEach(form => {
        const hblock = form.querySelector('[data-testid="stHorizontalBlock"]');
        if (hblock) {
            hblock.setAttribute('style',
                'background:linear-gradient(135deg,rgb(218,232,247) 0%,rgb(214,229,247) 100%);' +
                'border-radius:1000px;' +
                'border:none;' +
                'padding:6px 8px 6px 20px;' +
                'align-items:center;' +
                'gap:4px;' +
                'margin-top:6px;' +
                'position:relative;' +
                'box-shadow:rgba(79,156,232,0.7) 3px 3px 5px 0px, rgba(79,156,232,0.7) 5px 5px 20px 0px;'
            );
        }

        // Outer wrapper — gradient border effect
        const outerBlock = hblock ? hblock.parentElement : null;
        if (outerBlock) {
            outerBlock.setAttribute('style',
                'background:linear-gradient(135deg,rgb(179,208,253) 0%,rgb(164,202,248) 100%);' +
                'border-radius:1000px;' +
                'padding:4px;' +
                'margin-top:6px;'
            );
        }

        // Text input — match the reference style
        const input = form.querySelector('input[type="text"], input:not([type])');
        if (input) {
            input.setAttribute('style',
                'background:linear-gradient(135deg,rgb(218,232,247) 0%,rgb(214,229,247) 100%);' +
                'border:none;' +
                'outline:none;' +
                'color:#2563eb;' +
                'font-size:15px;' +
                'border-radius:1000px;' +
                'padding:8px 12px;' +
                'width:100%;'
            );
        }

        // 2. ALL submit buttons — blue circles matching the reference
        form.querySelectorAll('[data-testid="stFormSubmitButton"] button').forEach(btn => {
            btn.setAttribute('style',
                'background:linear-gradient(135deg,rgb(164,202,248),rgb(79,156,232));' +
                'border:none;' +
                'border-radius:50%;' +
                'min-height:40px;min-width:40px;' +
                'width:40px;height:40px;' +
                'font-size:17px;color:white;' +
                'padding:0;cursor:pointer;' +
                'box-shadow:rgba(79,156,232,0.7) 2px 2px 5px 0px;'
            );
        });
    });

    // 3. Remove dark backgrounds
    doc.querySelectorAll('[data-testid="stVerticalBlock"]').forEach(el => {
        const bg = window.getComputedStyle(el).backgroundColor;
        const m = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
        if (m && parseInt(m[1])<50 && parseInt(m[2])<50 && parseInt(m[3])<80) {
            el.style.background = 'transparent';
            el.style.border = 'none';
            el.style.boxShadow = 'none';
        }
    });
}

applyStyles();
setTimeout(applyStyles, 300);
setTimeout(applyStyles, 800);
setTimeout(applyStyles, 2000);
const obs = new MutationObserver(applyStyles);
obs.observe(window.parent.document.body, {childList:true, subtree:true});
</script>
""", height=1)
if img_pressed:
    st.session_state.show_image  = not st.session_state.show_image
    st.session_state.show_camera = False
    if st.session_state.show_image:
        st.session_state.show_voice = False
    st.rerun()
if cam_pressed:
    st.session_state.show_camera = not st.session_state.show_camera
    st.session_state.show_image  = False
    st.session_state.show_voice  = False
    st.rerun()
if mic_pressed:
    st.session_state.show_voice  = not st.session_state.show_voice
    st.session_state.show_camera = False
    if st.session_state.show_voice:
        st.session_state.show_image = False
    st.rerun()
if clr_pressed:
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

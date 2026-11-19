import os
import re
import threading
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

spam_model = tf.keras.models.load_model(Path(__file__).parent / "spam_detector.keras")
client = genai.Client(api_key=GEMINI_API_KEY)

TRACE = []                  # tools used in the current request
LOCK = threading.Lock()     # one agent run at a time (keeps TRACE correct)


# ---------------- Tools ----------------
def check_spam_score(text: str) -> dict:
    """Run the trained deep learning model on an SMS text and return its spam probability (0 to 1) and label SPAM or SAFE."""
    x = tf.constant([text], dtype=tf.string)
    p = float(spam_model(x, training=False).numpy()[0][0])
    res = {"spam_probability": round(p, 4), "label": "SPAM" if p > 0.5 else "SAFE"}
    TRACE.append({"tool": "check_spam_score", "result": res})
    return res


def find_links_and_numbers(text: str) -> dict:
    """Extract web links, shortened links and phone numbers from an SMS text."""
    urls = re.findall(r"(?:https?://|www\.)\S+", text, flags=re.I)
    phones = [p.strip() for p in re.findall(r"\+?\d[\d\s\-]{7,}\d", text)]
    short = [u for u in urls
             if re.search(r"bit\.ly|tinyurl|t\.co|cutt\.ly|rb\.gy|shorturl", u, re.I)]
    res = {"links": urls, "shortened_links": short, "phone_numbers": phones}
    TRACE.append({"tool": "find_links_and_numbers", "result": res})
    return res


FLAGS = {
    "Creates urgency": ["urgent", "immediately", "act now", "last chance", "expires", "within 24 hours"],
    "Asks for PIN, OTP or password": ["pin", "otp", "password", "cvv", "card number", "verify your account"],
    "Prize or lottery": ["won", "winner", "prize", "lottery", "congratulations", "free gift", "jackpot"],
    "Threatens to block account or SIM": ["blocked", "suspended", "deactivated", "locked", "sim will be"],
    "Asks for money or payment": ["send money", "transfer", "easypaisa", "jazzcash", "western union", "processing fee"],
    "Fake aid scheme": ["bisp", "ehsaas", "benazir"],
}


def find_red_flags(text: str) -> dict:
    """Check an SMS text for common scam red flags such as urgency, asking for PIN or OTP, prizes, account threats and payment requests."""
    t = text.lower()
    found = [name for name, words in FLAGS.items()
             if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words)]
    res = {"red_flags": found, "count": len(found)}
    TRACE.append({"tool": "find_red_flags", "result": res})
    return res


# ---------------- Agent ----------------
SYSTEM = """You are Scam Shield Agent. You help people check suspicious SMS messages, especially in Pakistan.

When the user sends a message to check, ALWAYS call all three tools: check_spam_score, find_links_and_numbers and find_red_flags.
Then reply in this format:
**Verdict:** one line (Scam / Suspicious / Looks safe) with the spam percentage.
**Why:** 2 to 4 short bullet points based on the tool results.
**What to do:** 2 to 3 clear steps.

Rules:
- Never invent tool results. Never tell the user to open any link.
- The spam model was trained on English SMS only. For Urdu, Pashto or Roman Urdu messages its score is less reliable, so say that and also use your own reading of the message.
- If the user asks to report the scam, write a short report message they can send to their bank or to PTA.
- If the user is only chatting or asks a general question about scams, answer normally without tools.
- Reply in the same language the user wrote in (English, Urdu or Pashto). Keep answers short."""


def run_agent(message, history=None):
    with LOCK:
        TRACE.clear()
        contents = []
        for h in (history or [])[-6:]:
            role = "model" if h.get("role") == "model" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=str(h.get("text", "")))]))
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                tools=[check_spam_score, find_links_and_numbers, find_red_flags],
                temperature=0.3,
            ),
        )
        return resp.text, list(TRACE)


# ---------------- Web app ----------------
HTML = r"""
<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scam Shield Agent</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
html,body{height:100%}
body{color:#fff;display:flex;flex-direction:column;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);background-size:300% 300%;animation:bg 14s ease infinite}
@keyframes bg{0%{background-position:0 50%}50%{background-position:100% 50%}100%{background-position:0 50%}}
header{padding:14px 16px;text-align:center;background:rgba(255,255,255,.06);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid rgba(255,255,255,.12)}
header h1{font-size:20px}
header span{font-size:11px;background:linear-gradient(90deg,#8b5cf6,#06b6d4);padding:3px 8px;border-radius:999px;margin-left:6px;vertical-align:middle}
#chat{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:12px;max-width:720px;width:100%;margin:0 auto}
.msg{max-width:88%;padding:12px 14px;border-radius:18px;line-height:1.5;font-size:15px;animation:in .3s ease}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.user{align-self:flex-end;background:linear-gradient(90deg,#8b5cf6,#06b6d4);border-bottom-right-radius:6px;white-space:pre-wrap}
.bot{align-self:flex-start;background:rgba(255,255,255,.09);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.16);border-bottom-left-radius:6px}
.bot p{margin:4px 0}
.bot ul{margin:6px 0 6px 18px}
.badge{display:inline-block;font-weight:800;font-size:13px;padding:5px 12px;border-radius:999px;margin-bottom:8px}
.badge.bad{background:rgba(239,68,68,.25);color:#fca5a5;border:1px solid #ef4444}
.badge.good{background:rgba(34,197,94,.22);color:#86efac;border:1px solid #22c55e}
.used{margin-top:8px;font-size:11px;opacity:.6}
.typing span{display:inline-block;width:8px;height:8px;margin:0 2px;border-radius:50%;background:#fff;opacity:.5;animation:b 1s infinite}
.typing span:nth-child(2){animation-delay:.15s}
.typing span:nth-child(3){animation-delay:.3s}
@keyframes b{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
.bar{padding:10px 12px calc(10px + env(safe-area-inset-bottom));background:rgba(15,12,41,.7);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-top:1px solid rgba(255,255,255,.12)}
.bar .in{max-width:720px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;height:48px;max-height:120px;resize:none;border-radius:16px;border:1px solid rgba(255,255,255,.2);background:rgba(0,0,0,.3);color:#fff;padding:12px;font-size:16px;outline:none}
textarea:focus{border-color:#8b5cf6}
button{width:48px;height:48px;border:0;border-radius:50%;font-size:20px;color:#fff;cursor:pointer;background:linear-gradient(90deg,#8b5cf6,#06b6d4)}
button:disabled{opacity:.5}
</style></head>
<body>
<header><h1>🛡️ Scam Shield <span>AI Agent</span></h1></header>
<div id="chat"></div>
<div class="bar"><div class="in">
  <textarea id="msg" dir="auto" placeholder="Paste a suspicious SMS..."></textarea>
  <button id="send">➤</button>
</div></div>

<script>
const $ = id => document.getElementById(id);
const chat = $("chat"), box = $("msg"), btn = $("send");
const history = [];

function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function md(s){
  const html = esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  let out = "", inList = false;
  for (const ln of html.split("\n")){
    const m = ln.match(/^\s*[\*\-•]\s+(.*)/);
    if (m){
      if (!inList){ out += "<ul>"; inList = true; }
      out += "<li>" + m[1] + "</li>";
    } else {
      if (inList){ out += "</ul>"; inList = false; }
      if (ln.trim()) out += "<p>" + ln + "</p>";
    }
  }
  if (inList) out += "</ul>";
  return out;
}

function scrollDown(){ chat.scrollTo({top: chat.scrollHeight, behavior: "smooth"}); }

function addUser(t){
  const d = document.createElement("div");
  d.className = "msg user"; d.dir = "auto"; d.textContent = t;
  chat.appendChild(d); scrollDown();
}

function addTyping(){
  const d = document.createElement("div");
  d.className = "msg bot typing";
  d.innerHTML = "<span></span><span></span><span></span>";
  chat.appendChild(d); scrollDown();
  return d;
}

function addBot(d){
  const w = document.createElement("div");
  w.className = "msg bot";
  const tools = d.tools || [];
  const s = tools.find(x => x.tool === "check_spam_score");
  let html = "";
  if (s){
    const p = Math.round(s.result.spam_probability * 100);
    const spam = s.result.label === "SPAM";
    html += `<div class="badge ${spam ? "bad" : "good"}">${spam ? "⚠️ SPAM" : "✅ SAFE"} · ${p}%</div>`;
  }
  html += `<div dir="auto">${md(d.reply || "")}</div>`;
  if (tools.length) html += `<div class="used">🔧 ${tools.map(x => x.tool).join(" · ")}</div>`;
  w.innerHTML = html;
  chat.appendChild(w); scrollDown();
}

async function send(){
  const text = box.value.trim();
  if (!text || btn.disabled) return;
  box.value = ""; addUser(text); btn.disabled = true;
  const typing = addTyping();
  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text, history: history.slice(-6)})
    });
    if (!r.ok) throw new Error("Server returned " + r.status);
    const d = await r.json();
    typing.remove(); addBot(d);
    history.push({role: "user", text: text});
    history.push({role: "model", text: d.reply});
  } catch(e){
    typing.remove(); addBot({reply: "Error: " + e.message, tools: []});
  }
  btn.disabled = false;
}

btn.onclick = send;
addBot({reply: "Hi! Paste any suspicious SMS and I will check it for you. You can write in English, Urdu or Pashto.", tools: []});
</script>
</body></html>
"""


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = []


app = FastAPI(title="Scam Shield Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatIn):
    try:
        reply, tools = run_agent(body.message, body.history)
        return {"reply": reply, "tools": tools}
    except Exception as e:
        return {"reply": f"Agent error: {e}", "tools": []}

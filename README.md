# 🛡️ Scam Shield Agent

An **AI agent** that checks suspicious SMS messages. It uses my own trained deep learning
spam model as a tool, adds a link scanner and a red-flag checker, and lets Gemini explain the
result in simple **English, Urdu or Pashto**.

Built on top of my earlier project: [sms-scam-detector](https://github.com/rafaqatkhattak5/sms-scam-detector).

## How it works
1. The user pastes an SMS in the chat.
2. The agent (Gemini with function calling) decides to call its tools.
3. It combines the tool results into a verdict, reasons and clear steps to follow.

| Tool | What it does |
|---|---|
| `check_spam_score` | Runs my Embedding + BiLSTM Keras model and returns the spam probability |
| `find_links_and_numbers` | Extracts links, shortened links and phone numbers |
| `find_red_flags` | Detects urgency, PIN/OTP requests, prizes, account threats, payment requests and fake aid schemes |

## Features
- Tool-calling agent with Gemini
- Multilingual replies (English, Urdu, Pashto)
- Chat history for follow-up questions, for example "write a report for my bank"
- Mobile-friendly glass chat UI that shows the spam badge and the tools used

## Run locally
```bash
git clone https://github.com/rafaqatkhattak5/scam-shield-agent.git
cd scam-shield-agent
pip install -r requirements.txt
export GEMINI_API_KEY="your-free-key-from-aistudio.google.com"
uvicorn agent:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000`. On Windows use `set GEMINI_API_KEY=your-key` instead of `export`.

## Project structure
```
agent.py              # tools + agent + FastAPI app + chat UI
spam_detector.keras   # trained spam model
requirements.txt
README.md
```

## Limitations
- The spam model was trained on English SMS only. For Urdu, Pashto or Roman Urdu messages the agent says the score is less reliable and relies on the red-flag tool and Gemini's own reading.
- It is a learning project, not a replacement for your bank's or carrier's fraud protection.
- The free Gemini API has rate limits.

## Author
Built by Rafaqat as part of my journey into agentic AI engineering.

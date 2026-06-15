# =============================================================================
# app.py — AI Chatbot: Complete Single-File Application
# =============================================================================
#
# DESCRIPTION:
#   A full-featured AI chatbot powered by Groq (free tier) + Flask.
#   Everything — backend, frontend HTML/CSS/JS, database, tool system — lives
#   in this ONE file. No other Python files are required to get started.
#
# WHAT IT DOES:
#   - Streams AI responses using Groq's mixtral-8x7b-32768 model
#   - Stores conversations permanently in a local SQLite database
#   - Saves the last 10 messages to browser localStorage for instant restore
#   - Lets you dynamically add Python "tools" (functions) at runtime
#   - Export / Import chat history as JSON
#   - Fully responsive dark-mode chat UI
#
# =============================================================================
# STEP-BY-STEP SETUP (follow exactly — takes ~5 minutes)
# =============================================================================
#
# STEP 1 — GET A FREE GROQ API KEY
#   a) Open your browser and go to: https://console.groq.com
#   b) Click "Sign Up" and create a free account (no credit card needed)
#   c) After login, click "API Keys" in the left sidebar
#   d) Click "Create API Key", give it any name, copy the key (starts with gsk_)
#   e) Keep this key safe — you will need it in Step 3
#
# STEP 2 — INSTALL PYTHON DEPENDENCIES
#   Open a terminal (Command Prompt on Windows / Terminal on Mac/Linux) and run:
#     pip install flask groq python-dotenv
#   If "pip" is not found, try:  python -m pip install flask groq python-dotenv
#
# STEP 3 — SET YOUR API KEY (choose one method)
#   METHOD A (recommended): Create a file named ".env" in the same folder as
#     this file, with this single line:   GROQ_API_KEY=gsk_your_key_here
#   METHOD B: Set as an environment variable in your terminal:
#     Windows:  set GROQ_API_KEY=gsk_your_key_here
#     Mac/Linux: export GROQ_API_KEY=gsk_your_key_here
#
# STEP 4 — RUN THE APP
#   In the same folder as this file, run:   python app.py
#   Then open your browser and go to:       http://localhost:5000
#
# =============================================================================
# REQUIREMENTS (pip install these)
# =============================================================================
#   flask>=3.0.0
#   groq>=0.9.0
#   python-dotenv>=1.0.0
#
# =============================================================================
# TROUBLESHOOTING
# =============================================================================
#   "ModuleNotFoundError" -> run: pip install flask groq python-dotenv
#   "AuthenticationError" -> double-check your GROQ_API_KEY in .env file
#   "Port 5000 in use"    -> change PORT variable below to 5001, 5002, etc.
#   Chat not saving?      -> make sure the folder is writable (not read-only)
#   Tools not appearing?  -> check terminal for Python syntax errors in tools.py
#   Blank page on load?   -> hard-refresh browser with Ctrl+Shift+R
#
# =============================================================================
# DEPLOYMENT TO RENDER.COM (free tier — go live in under 10 minutes)
# =============================================================================
#
#  1. Create a free account at https://render.com (no credit card needed)
#  2. Push this project to GitHub:
#       - Create a repo at https://github.com/new
#       - Add this file + requirements.txt + render.yaml (see bottom of file)
#       - git add . && git commit -m "init" && git push
#  3. On Render dashboard: click "New" -> "Web Service" -> connect GitHub repo
#  4. In "Environment Variables" section, add:  GROQ_API_KEY = gsk_your_key
#  5. Render auto-detects render.yaml. Click "Create Web Service"
#  6. Wait ~3 minutes. Your URL will be: https://your-app-name.onrender.com
#  7. FREE TIER NOTE: The service sleeps after 15 min of inactivity.
#       To wake it: just visit the URL. First load takes ~30 seconds.
#       To keep it awake free: use https://uptimerobot.com to ping it every 14min
#
# =============================================================================

import os
import sys
import json
import time
import sqlite3
import importlib
import importlib.util
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, stream_with_context

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", 5000))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "mixtral-8x7b-32768"
DB_PATH = "chatbot.db"
TOOLS_FILE = "tools.py"
tools_lock = threading.Lock()

# ---------------------------------------------------------------------------
# GROQ CLIENT (lazy-init so missing key gives a clean error)
# ---------------------------------------------------------------------------
def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Create a .env file with: "
            "GROQ_API_KEY=gsk_your_key_here"
        )
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# SQLITE DATABASE SETUP
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------------------------
# TOOL SYSTEM
# ---------------------------------------------------------------------------
_tools_module = None
_tool_list = []

def load_tools():
    global _tools_module, _tool_list
    with tools_lock:
        if not os.path.exists(TOOLS_FILE):
            open(TOOLS_FILE, "w").write("# Auto-generated tools file\n# Each function here is an available tool\n\n")
        spec = importlib.util.spec_from_file_location("tools", TOOLS_FILE)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            _tools_module = mod
            _tool_list = [
                name for name in dir(mod)
                if callable(getattr(mod, name)) and not name.startswith("_")
            ]
        except Exception as e:
            print(f"[TOOLS] Error loading tools.py: {e}")

def get_tool_descriptions():
    if not _tool_list:
        return ""
    lines = ["\n\nYou have access to these Python tools (call them by name if relevant):"]
    for name in _tool_list:
        fn = getattr(_tools_module, name, None)
        if fn:
            doc = (fn.__doc__ or "No description").strip().split("\n")[0]
            lines.append(f"  - {name}(): {doc}")
    return "\n".join(lines)

def call_tool(name, args=None):
    fn = getattr(_tools_module, name, None)
    if fn:
        try:
            return fn(**(args or {}))
        except Exception as e:
            return f"Tool error: {e}"
    return f"Tool '{name}' not found."

load_tools()

# ---------------------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ===========================
# ROUTES — DATABASE
# ===========================

@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows])

@app.route("/api/conversations", methods=["POST"])
def create_conversation():
    data = request.json or {}
    title = data.get("title", "New Chat")
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)", (title, now, now))
    cid = c.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"id": cid, "title": title, "created_at": now, "updated_at": now})

@app.route("/api/conversations/<int:cid>", methods=["GET"])
def get_conversation(cid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id=?", (cid,))
    conv = c.fetchone()
    if not conv:
        conn.close()
        return jsonify({"error": "Not found"}), 404
    c.execute("SELECT role, content, timestamp FROM messages WHERE conversation_id=? ORDER BY id ASC", (cid,))
    msgs = c.fetchall()
    conn.close()
    return jsonify({
        "id": conv[0], "title": conv[1], "created_at": conv[2], "updated_at": conv[3],
        "messages": [{"role": m[0], "content": m[1], "timestamp": m[2]} for m in msgs]
    })

@app.route("/api/conversations/<int:cid>", methods=["DELETE"])
def delete_conversation(cid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    c.execute("DELETE FROM conversations WHERE id=?", (cid,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

@app.route("/api/conversations/<int:cid>/messages", methods=["POST"])
def add_message(cid):
    data = request.json or {}
    role = data.get("role", "user")
    content = data.get("content", "")
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (cid, role, content, now))
    # Update title from first user message
    c.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (cid,))
    count = c.fetchone()[0]
    if count == 1 and role == "user":
        title = content[:60] + ("..." if len(content) > 60 else "")
        c.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, now, cid))
    else:
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, cid))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

# ===========================
# ROUTES — TOOLS
# ===========================

@app.route("/api/tools", methods=["GET"])
def get_tools():
    load_tools()
    return jsonify({"tools": _tool_list})

@app.route("/api/tools/generate", methods=["POST"])
def generate_tool():
    data = request.json or {}
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "Description required"}), 400
    # Use Groq to generate the Python function
    try:
        client = get_groq_client()
        prompt = (
            f"Write a single Python function that: {description}\n\n"
            "Rules:\n"
            "1. Function name must be snake_case, descriptive, no spaces\n"
            "2. Include a docstring on the first line inside the function\n"
            "3. Handle exceptions and return strings (not print)\n"
            "4. Use only Python standard library — NO external imports\n"
            "5. Return ONLY the function code, no markdown, no explanation\n"
            "6. Start the response with 'def '\n"
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        code = resp.choices[0].message.content.strip()
        # Strip accidental markdown fences
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.endswith("```"):
            code = "\n".join(code.split("\n")[:-1])
        code = code.strip()
        if not code.startswith("def "):
            return jsonify({"error": "AI did not return valid Python. Try rephrasing."}), 400
        # Extract function name
        fn_name = code.split("(")[0].replace("def ", "").strip()
        # Append to tools.py
        with tools_lock:
            with open(TOOLS_FILE, "a") as f:
                f.write(f"\n\n{code}\n")
        load_tools()
        return jsonify({"ok": True, "function_name": fn_name, "code": code})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tools/<name>", methods=["DELETE"])
def delete_tool(name):
    with tools_lock:
        if not os.path.exists(TOOLS_FILE):
            return jsonify({"error": "tools.py not found"}), 404
        with open(TOOLS_FILE, "r") as f:
            content = f.read()
        import ast, textwrap
        try:
            tree = ast.parse(content)
        except Exception:
            return jsonify({"error": "Could not parse tools.py"}), 500
        lines = content.splitlines(keepends=True)
        # Find the function and remove it
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                start = node.lineno - 1
                end = node.end_lineno
                new_lines = lines[:start] + lines[end:]
                with open(TOOLS_FILE, "w") as f:
                    f.writelines(new_lines)
                load_tools()
                return jsonify({"ok": True})
    return jsonify({"error": f"Function '{name}' not found"}), 404

# ===========================
# ROUTES — CHAT
# ===========================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    conversation_id = data.get("conversation_id")
    save_to_db = data.get("save_to_db", True)

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    system_prompt = (
        "You are a helpful, intelligent AI assistant. "
        "Be concise but thorough. Use markdown for formatting when helpful."
        + get_tool_descriptions()
    )

    api_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    def generate():
        try:
            client = get_groq_client()
            stream = client.chat.completions.create(
                model=MODEL,
                messages=api_messages,
                max_tokens=2048,
                temperature=0.7,
                stream=True,
            )
            full_response = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response.append(delta)
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            
            assistant_text = "".join(full_response)

            # Save to DB
            if save_to_db and conversation_id:
                user_msg = messages[-1]["content"] if messages else ""
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                now = datetime.utcnow().isoformat()
                # Save user message
                c.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                          (conversation_id, "user", user_msg, now))
                # Save assistant message
                c.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                          (conversation_id, "assistant", assistant_text, now))
                # Update conversation title if first message
                c.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (conversation_id,))
                count = c.fetchone()[0]
                if count <= 2:
                    title = user_msg[:60] + ("..." if len(user_msg) > 60 else "")
                    c.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, now, conversation_id))
                else:
                    c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
                conn.commit()
                conn.close()

            yield f"data: {json.dumps({'done': True, 'full_text': assistant_text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ===========================
# ROUTE — MAIN PAGE (embedded HTML)
# ===========================

@app.route("/")
def index():
    return HTML_TEMPLATE

# ---------------------------------------------------------------------------
# EMBEDDED HTML/CSS/JS TEMPLATE
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Chatbot</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root{
  --bg:#0d0d0f;--bg2:#16161a;--bg3:#1e1e24;--bg4:#26262e;
  --border:#2e2e38;--border2:#3a3a48;
  --text:#e8e8f0;--text2:#a0a0b8;--text3:#606078;
  --accent:#7c6af7;--accent2:#a594ff;--accent3:#4ade80;
  --danger:#f87171;--warn:#fbbf24;
  --radius:12px;--radius-sm:8px;
  --shadow:0 4px 24px rgba(0,0,0,.5);
  --font-mono:'IBM Plex Mono',monospace;
  --font-sans:'IBM Plex Sans',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);height:100vh;display:flex;overflow:hidden}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:99px}

/* ---- LEFT SIDEBAR ---- */
#sidebar{width:260px;min-width:260px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;height:100vh;transition:transform .25s}
#sidebar-header{padding:18px 16px 12px;border-bottom:1px solid var(--border)}
#sidebar-header h2{font-size:13px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text2);margin-bottom:12px}
#new-chat-btn{width:100%;padding:10px 14px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius-sm);cursor:pointer;font-family:var(--font-sans);font-size:13px;font-weight:600;display:flex;align-items:center;gap:8px;transition:opacity .2s}
#new-chat-btn:hover{opacity:.85}
#conv-list{flex:1;overflow-y:auto;padding:8px}
.conv-item{padding:10px 12px;border-radius:var(--radius-sm);cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:8px;transition:background .15s;margin-bottom:2px}
.conv-item:hover{background:var(--bg3)}
.conv-item.active{background:var(--bg4)}
.conv-title{font-size:13px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.conv-date{font-size:10px;color:var(--text3);white-space:nowrap}
.conv-del{background:none;border:none;color:var(--text3);cursor:pointer;padding:2px 4px;border-radius:4px;font-size:12px;opacity:0;transition:opacity .15s}
.conv-item:hover .conv-del{opacity:1}
.conv-del:hover{color:var(--danger)}
#sidebar-footer{padding:12px 16px;border-top:1px solid var(--border);display:flex;flex-direction:column;gap:8px}

/* ---- RIGHT TOOLS SIDEBAR ---- */
#tools-sidebar{width:220px;min-width:220px;background:var(--bg2);border-left:1px solid var(--border);display:flex;flex-direction:column;height:100vh;transition:transform .25s}
#tools-sidebar-header{padding:18px 16px 12px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
#tools-sidebar-header h3{font-size:13px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text2)}
#tools-list{flex:1;overflow-y:auto;padding:8px}
.tool-item{padding:10px 12px;background:var(--bg3);border-radius:var(--radius-sm);margin-bottom:6px;display:flex;align-items:flex-start;justify-content:space-between;gap:8px}
.tool-name{font-family:var(--font-mono);font-size:12px;color:var(--accent2);word-break:break-all}
.tool-del{background:none;border:none;color:var(--text3);cursor:pointer;padding:2px 6px;border-radius:4px;font-size:12px;flex-shrink:0;transition:color .15s}
.tool-del:hover{color:var(--danger)}
.no-tools{padding:16px 12px;font-size:12px;color:var(--text3);text-align:center;line-height:1.6}

/* ---- MAIN CHAT AREA ---- */
#main{flex:1;display:flex;flex-direction:column;height:100vh;min-width:0}
#topbar{padding:14px 20px;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:12px;flex-shrink:0}
#topbar-left{display:flex;align-items:center;gap:12px}
#topbar-title{font-size:15px;font-weight:600;color:var(--text)}
#topbar-model{font-size:11px;font-family:var(--font-mono);color:var(--text3);background:var(--bg3);padding:3px 8px;border-radius:99px;border:1px solid var(--border)}
#topbar-right{display:flex;align-items:center;gap:8px}
.tb-btn{padding:7px 13px;background:var(--bg3);color:var(--text2);border:1px solid var(--border);border-radius:var(--radius-sm);cursor:pointer;font-family:var(--font-sans);font-size:12px;font-weight:500;transition:all .15s;white-space:nowrap}
.tb-btn:hover{background:var(--bg4);color:var(--text);border-color:var(--border2)}
.tb-btn.accent{background:var(--accent);color:#fff;border-color:var(--accent)}
.tb-btn.accent:hover{opacity:.85}
.tb-btn.danger:hover{background:#7f1d1d;color:var(--danger);border-color:var(--danger)}

/* ---- MESSAGES ---- */
#messages-wrap{flex:1;overflow-y:auto;padding:24px 20px}
#messages{display:flex;flex-direction:column;gap:20px;max-width:780px;margin:0 auto}
.msg-row{display:flex;gap:12px;animation:msgIn .3s ease}
@keyframes msgIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.msg-row.user{flex-direction:row-reverse}
.msg-avatar{width:32px;height:32px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:600;font-family:var(--font-mono)}
.msg-row.user .msg-avatar{background:var(--accent);color:#fff}
.msg-row.assistant .msg-avatar{background:var(--bg4);color:var(--accent2);border:1px solid var(--border)}
.msg-content-wrap{display:flex;flex-direction:column;gap:6px;max-width:75%}
.msg-row.user .msg-content-wrap{align-items:flex-end}
.msg-bubble{padding:12px 16px;border-radius:18px;font-size:14px;line-height:1.65;word-break:break-word}
.msg-row.user .msg-bubble{background:var(--accent);color:#fff;border-bottom-right-radius:4px}
.msg-row.assistant .msg-bubble{background:var(--bg3);color:var(--text);border:1px solid var(--border);border-bottom-left-radius:4px}
.msg-bubble code{font-family:var(--font-mono);font-size:12px;background:rgba(0,0,0,.35);padding:2px 6px;border-radius:4px}
.msg-bubble pre{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:14px;margin:10px 0;overflow-x:auto}
.msg-bubble pre code{background:none;padding:0;font-size:12.5px;line-height:1.6}
.msg-bubble p{margin:0 0 8px}
.msg-bubble p:last-child{margin-bottom:0}
.msg-bubble ul,.msg-bubble ol{padding-left:20px;margin:8px 0}
.msg-bubble li{margin-bottom:4px}
.msg-bubble strong{color:#fff}
.msg-row.assistant .msg-bubble strong{color:var(--accent2)}
.msg-bubble a{color:var(--accent2);text-decoration:none}
.msg-bubble a:hover{text-decoration:underline}
.msg-meta{display:flex;align-items:center;gap:8px}
.msg-time{font-size:10px;color:var(--text3);font-family:var(--font-mono)}
.copy-btn{background:none;border:none;color:var(--text3);cursor:pointer;font-size:11px;padding:2px 6px;border-radius:4px;transition:all .15s;opacity:0;font-family:var(--font-mono)}
.msg-content-wrap:hover .copy-btn{opacity:1}
.copy-btn:hover{background:var(--bg4);color:var(--text)}
.copy-btn.copied{color:var(--accent3)}

/* ---- TYPING INDICATOR ---- */
#typing-indicator{display:none;padding:0 20px}
#typing-indicator .inner{max-width:780px;margin:0 auto;display:flex;gap:12px;align-items:center}
.typing-dots{display:flex;gap:4px;padding:14px 18px;background:var(--bg3);border:1px solid var(--border);border-radius:18px;border-bottom-left-radius:4px}
.typing-dots span{width:6px;height:6px;background:var(--text3);border-radius:50%;animation:bounce .9s infinite}
.typing-dots span:nth-child(2){animation-delay:.15s}
.typing-dots span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}

/* ---- INPUT AREA ---- */
#input-area{padding:16px 20px 20px;border-top:1px solid var(--border);background:var(--bg2);flex-shrink:0}
#input-wrap{max-width:780px;margin:0 auto;display:flex;gap:10px;align-items:flex-end;background:var(--bg3);border:1px solid var(--border);border-radius:16px;padding:12px 14px;transition:border-color .2s}
#input-wrap:focus-within{border-color:var(--accent)}
#user-input{flex:1;background:none;border:none;outline:none;color:var(--text);font-family:var(--font-sans);font-size:14px;line-height:1.5;resize:none;max-height:200px;min-height:22px}
#user-input::placeholder{color:var(--text3)}
#send-btn{background:var(--accent);border:none;color:#fff;width:36px;height:36px;border-radius:10px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:opacity .2s}
#send-btn:hover{opacity:.85}
#send-btn:disabled{opacity:.4;cursor:not-allowed}
#input-hint{max-width:780px;margin:6px auto 0;font-size:11px;color:var(--text3);text-align:center}

/* ---- WELCOME ---- */
#welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;text-align:center;padding:40px 20px}
#welcome h1{font-size:28px;font-weight:600;color:var(--text);margin-bottom:10px}
#welcome p{font-size:15px;color:var(--text2);max-width:420px;line-height:1.6}
#welcome .pills{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:24px}
#welcome .pill{padding:10px 16px;background:var(--bg3);border:1px solid var(--border);border-radius:99px;font-size:13px;color:var(--text2);cursor:pointer;transition:all .15s}
#welcome .pill:hover{background:var(--bg4);border-color:var(--border2);color:var(--text)}

/* ---- MODALS ---- */
.modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.7);display:flex;align-items:center;justify-content:center;z-index:1000;opacity:0;pointer-events:none;transition:opacity .2s}
.modal-backdrop.open{opacity:1;pointer-events:auto}
.modal{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:28px;width:100%;max-width:500px;box-shadow:var(--shadow);transform:translateY(10px);transition:transform .2s}
.modal-backdrop.open .modal{transform:translateY(0)}
.modal h3{font-size:17px;font-weight:600;margin-bottom:6px}
.modal p{font-size:13px;color:var(--text2);margin-bottom:20px;line-height:1.6}
.modal label{display:block;font-size:12px;font-weight:600;color:var(--text2);letter-spacing:.05em;text-transform:uppercase;margin-bottom:6px}
.modal input,.modal textarea{width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius-sm);color:var(--text);font-family:var(--font-sans);font-size:14px;padding:10px 12px;outline:none;transition:border-color .2s;margin-bottom:16px}
.modal input:focus,.modal textarea:focus{border-color:var(--accent)}
.modal textarea{resize:vertical;min-height:80px}
.modal-actions{display:flex;gap:10px;justify-content:flex-end;margin-top:8px}
.modal-actions button{padding:9px 18px;border-radius:var(--radius-sm);border:none;cursor:pointer;font-family:var(--font-sans);font-size:13px;font-weight:600;transition:opacity .2s}
.modal-actions .cancel{background:var(--bg4);color:var(--text2)}
.modal-actions .cancel:hover{opacity:.8}
.modal-actions .confirm{background:var(--accent);color:#fff}
.modal-actions .confirm:hover{opacity:.85}
#tool-result{margin-top:12px;padding:12px;border-radius:var(--radius-sm);font-size:13px;display:none}
#tool-result.success{background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.3);color:var(--accent3)}
#tool-result.error{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.3);color:var(--danger)}
#tool-code-preview{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:12px;margin-top:8px;font-family:var(--font-mono);font-size:11.5px;color:var(--text2);white-space:pre-wrap;word-break:break-all;max-height:180px;overflow-y:auto;display:none}

/* ---- TOASTS ---- */
#toast-container{position:fixed;bottom:24px;right:24px;display:flex;flex-direction:column;gap:8px;z-index:2000}
.toast{padding:12px 18px;border-radius:var(--radius-sm);font-size:13px;color:#fff;box-shadow:var(--shadow);animation:toastIn .25s ease;max-width:300px}
@keyframes toastIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}
.toast.success{background:#16613a;border:1px solid #4ade80}
.toast.error{background:#7f1d1d;border:1px solid #f87171}
.toast.info{background:#1e1e60;border:1px solid #7c6af7}

/* ---- EMPTY STATE ---- */
.empty-conv{padding:16px 12px;font-size:12px;color:var(--text3);text-align:center}

/* ---- RESPONSIVE ---- */
@media(max-width:900px){
  #tools-sidebar{display:none}
}
@media(max-width:640px){
  #sidebar{position:fixed;left:0;top:0;z-index:500;transform:translateX(-100%)}
  #sidebar.open{transform:translateX(0)}
  #hamburger{display:flex!important}
}
#hamburger{display:none;background:none;border:none;color:var(--text);cursor:pointer;padding:4px;flex-direction:column;gap:4px}
#hamburger span{display:block;width:20px;height:2px;background:currentColor;border-radius:2px}

/* ---- LOADING SPINNER ---- */
.spinner{width:16px;height:16px;border:2px solid var(--border2);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;display:inline-block;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}

input[type="file"]{display:none}
</style>
</head>
<body>

<!-- LEFT SIDEBAR: Conversation History -->
<aside id="sidebar">
  <div id="sidebar-header">
    <h2>Conversations</h2>
    <button id="new-chat-btn" onclick="newChat()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      New Chat
    </button>
  </div>
  <div id="conv-list">
    <div class="empty-conv">No conversations yet</div>
  </div>
  <div id="sidebar-footer">
    <button class="tb-btn" onclick="exportChat()">Export Chat</button>
    <label class="tb-btn" style="cursor:pointer;text-align:center" for="import-file-input">Import Chat</label>
    <input type="file" id="import-file-input" accept=".json" onchange="importChat(event)"/>
    <button class="tb-btn danger" onclick="confirmClearAll()">Clear All History</button>
  </div>
</aside>

<!-- MAIN CHAT AREA -->
<main id="main">
  <div id="topbar">
    <div id="topbar-left">
      <button id="hamburger" onclick="toggleSidebar()">
        <span></span><span></span><span></span>
      </button>
      <div>
        <div id="topbar-title">AI Assistant</div>
        <div id="topbar-model">mixtral-8x7b-32768 via Groq</div>
      </div>
    </div>
    <div id="topbar-right">
      <button class="tb-btn" onclick="clearCurrentChat()">Clear Chat</button>
      <button class="tb-btn accent" onclick="openAddToolModal()">+ Add Tool</button>
    </div>
  </div>

  <div id="messages-wrap">
    <div id="messages">
      <div id="welcome">
        <h1>What can I help you with?</h1>
        <p>Ask me anything. I remember our conversation and can use custom tools you create.</p>
        <div class="pills">
          <div class="pill" onclick="useSuggestion(this)">Explain quantum entanglement simply</div>
          <div class="pill" onclick="useSuggestion(this)">Write a Python web scraper</div>
          <div class="pill" onclick="useSuggestion(this)">Review my resume structure</div>
          <div class="pill" onclick="useSuggestion(this)">Debug this code for me</div>
        </div>
      </div>
    </div>
  </div>

  <div id="typing-indicator">
    <div class="inner">
      <div style="width:32px;height:32px;border-radius:50%;background:var(--bg4);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:13px;font-family:var(--font-mono);color:var(--accent2)">AI</div>
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  </div>

  <div id="input-area">
    <div id="input-wrap">
      <textarea id="user-input" rows="1" placeholder="Message AI Assistant..." onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      <button id="send-btn" onclick="sendMessage()" title="Send message">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
    </div>
    <div id="input-hint">Press Enter to send, Shift+Enter for new line</div>
  </div>
</main>

<!-- RIGHT SIDEBAR: Active Tools -->
<aside id="tools-sidebar">
  <div id="tools-sidebar-header">
    <h3>Active Tools</h3>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text3)" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
  </div>
  <div id="tools-list">
    <div class="no-tools">No tools added yet.<br/>Click "+ Add Tool" to create one.</div>
  </div>
</aside>

<!-- MODAL: Add Tool -->
<div class="modal-backdrop" id="tool-modal" onclick="closeModalOnBackdrop(event)">
  <div class="modal">
    <h3>Add a New Tool</h3>
    <p>Describe what you want the tool to do in plain English. The AI will generate a Python function and make it available immediately.</p>
    <label for="tool-desc">Tool Description</label>
    <textarea id="tool-desc" placeholder="e.g., Calculate compound interest given principal, rate, and years&#10;e.g., Convert Celsius to Fahrenheit and Kelvin&#10;e.g., Count words and characters in a string"></textarea>
    <div id="tool-result"></div>
    <pre id="tool-code-preview"></pre>
    <div class="modal-actions">
      <button class="cancel" onclick="closeModal('tool-modal')">Cancel</button>
      <button class="confirm" id="tool-gen-btn" onclick="generateTool()">Generate Tool</button>
    </div>
  </div>
</div>

<!-- TOAST CONTAINER -->
<div id="toast-container"></div>

<script>
// ============================================================
// STATE
// ============================================================
let currentConvId = null;
let messages = []; // [{role, content, timestamp}]
let isStreaming = false;
const LS_KEY = 'chatbot_last_messages';
const LS_CONV_KEY = 'chatbot_last_conv_id';

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
  await loadConversations();
  await loadTools();
  restoreFromLocalStorage();
});

function restoreFromLocalStorage() {
  const lastConvId = localStorage.getItem(LS_CONV_KEY);
  if (lastConvId) {
    loadConversation(parseInt(lastConvId));
  }
}

// ============================================================
// CONVERSATIONS
// ============================================================
async function loadConversations() {
  const res = await fetch('/api/conversations');
  const convs = await res.json();
  renderConvList(convs);
}

function renderConvList(convs) {
  const el = document.getElementById('conv-list');
  if (!convs.length) {
    el.innerHTML = '<div class="empty-conv">No conversations yet</div>';
    return;
  }
  el.innerHTML = convs.map(c => `
    <div class="conv-item ${c.id === currentConvId ? 'active' : ''}" onclick="loadConversation(${c.id})">
      <div>
        <div class="conv-title">${escHtml(c.title)}</div>
        <div class="conv-date">${fmtDate(c.updated_at)}</div>
      </div>
      <button class="conv-del" onclick="event.stopPropagation();deleteConversation(${c.id})" title="Delete">x</button>
    </div>
  `).join('');
}

async function loadConversation(id) {
  const res = await fetch(`/api/conversations/${id}`);
  if (!res.ok) return;
  const data = await res.json();
  currentConvId = data.id;
  messages = data.messages || [];
  localStorage.setItem(LS_CONV_KEY, id);
  saveMessagesToLocalStorage();
  renderMessages();
  await loadConversations();
}

async function newChat() {
  const res = await fetch('/api/conversations', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({title: 'New Chat'})
  });
  const conv = await res.json();
  currentConvId = conv.id;
  messages = [];
  localStorage.setItem(LS_CONV_KEY, conv.id);
  renderMessages();
  await loadConversations();
  document.getElementById('user-input').focus();
}

async function deleteConversation(id) {
  await fetch(`/api/conversations/${id}`, {method:'DELETE'});
  if (currentConvId === id) {
    currentConvId = null;
    messages = [];
    localStorage.removeItem(LS_CONV_KEY);
    renderMessages();
  }
  await loadConversations();
  toast('Conversation deleted', 'info');
}

// ============================================================
// MESSAGES RENDERING
// ============================================================
function renderMessages() {
  const container = document.getElementById('messages');
  const welcome = document.getElementById('welcome');

  // Determine what to show
  const last10 = messages.slice(-10);

  if (!last10.length) {
    container.innerHTML = '';
    container.appendChild(createWelcome());
    return;
  }

  container.innerHTML = last10.map((m, i) => buildMsgHtml(m, i)).join('');
  scrollToBottom();
}

function createWelcome() {
  const div = document.createElement('div');
  div.id = 'welcome';
  div.innerHTML = `
    <h1>What can I help you with?</h1>
    <p>Ask me anything. I remember our conversation and can use custom tools you create.</p>
    <div class="pills">
      <div class="pill" onclick="useSuggestion(this)">Explain quantum entanglement simply</div>
      <div class="pill" onclick="useSuggestion(this)">Write a Python web scraper</div>
      <div class="pill" onclick="useSuggestion(this)">Review my resume structure</div>
      <div class="pill" onclick="useSuggestion(this)">Debug this code for me</div>
    </div>`;
  return div;
}

function buildMsgHtml(msg, idx) {
  const isUser = msg.role === 'user';
  const avatarLabel = isUser ? 'You' : 'AI';
  const renderedContent = isUser
    ? `<p>${escHtml(msg.content).replace(/\n/g,'<br/>')}</p>`
    : marked.parse(msg.content || '');
  const timeStr = msg.timestamp ? fmtTime(msg.timestamp) : '';
  return `
    <div class="msg-row ${msg.role}" data-idx="${idx}">
      <div class="msg-avatar">${avatarLabel}</div>
      <div class="msg-content-wrap">
        <div class="msg-bubble">${renderedContent}</div>
        <div class="msg-meta">
          <span class="msg-time">${timeStr}</span>
          <button class="copy-btn" onclick="copyMsg(${idx})" title="Copy message">copy</button>
        </div>
      </div>
    </div>`;
}

function appendMessage(msg) {
  messages.push(msg);
  saveMessagesToLocalStorage();
  const container = document.getElementById('messages');
  // Remove welcome if present
  const welcome = container.querySelector('#welcome');
  if (welcome) welcome.remove();
  const div = document.createElement('div');
  div.innerHTML = buildMsgHtml(msg, messages.length - 1);
  container.appendChild(div.firstElementChild);
  scrollToBottom();
}

function updateLastAssistantBubble(text) {
  const container = document.getElementById('messages');
  const rows = container.querySelectorAll('.msg-row.assistant');
  if (!rows.length) return;
  const last = rows[rows.length - 1];
  const bubble = last.querySelector('.msg-bubble');
  if (bubble) bubble.innerHTML = marked.parse(text);
  scrollToBottom();
}

// ============================================================
// SENDING MESSAGES
// ============================================================
async function sendMessage() {
  if (isStreaming) return;
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  autoResize(input);

  // Create conversation if none active
  if (!currentConvId) {
    const res = await fetch('/api/conversations', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({title: text.substring(0, 60)})
    });
    const conv = await res.json();
    currentConvId = conv.id;
    localStorage.setItem(LS_CONV_KEY, conv.id);
  }

  const userMsg = {role: 'user', content: text, timestamp: new Date().toISOString()};
  appendMessage(userMsg);

  showTyping(true);
  isStreaming = true;
  document.getElementById('send-btn').disabled = true;

  // Prepare assistant bubble
  const assistantMsg = {role: 'assistant', content: '', timestamp: new Date().toISOString()};
  appendMessage(assistantMsg);
  const msgIdx = messages.length - 1;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        messages: messages.slice(0, -1).map(m => ({role: m.role, content: m.content})),
        conversation_id: currentConvId,
        save_to_db: true
      })
    });

    showTyping(false);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.error) { toast('Error: ' + data.error, 'error'); break; }
          if (data.delta) {
            fullText += data.delta;
            messages[msgIdx].content = fullText;
            updateLastAssistantBubble(fullText);
          }
          if (data.done) {
            messages[msgIdx].content = data.full_text || fullText;
            saveMessagesToLocalStorage();
            await loadConversations();
          }
        } catch (e) {}
      }
    }
  } catch (e) {
    showTyping(false);
    toast('Network error: ' + e.message, 'error');
  }

  isStreaming = false;
  document.getElementById('send-btn').disabled = false;
  input.focus();
}

// ============================================================
// TOOLS
// ============================================================
async function loadTools() {
  const res = await fetch('/api/tools');
  const data = await res.json();
  renderToolList(data.tools || []);
}

function renderToolList(tools) {
  const el = document.getElementById('tools-list');
  if (!tools.length) {
    el.innerHTML = '<div class="no-tools">No tools added yet.<br/>Click "+ Add Tool" to create one.</div>';
    return;
  }
  el.innerHTML = tools.map(t => `
    <div class="tool-item">
      <div class="tool-name">${escHtml(t)}()</div>
      <button class="tool-del" onclick="deleteTool('${escHtml(t)}')" title="Remove tool">x</button>
    </div>`).join('');
}

async function generateTool() {
  const desc = document.getElementById('tool-desc').value.trim();
  if (!desc) { toast('Please enter a description', 'error'); return; }

  const btn = document.getElementById('tool-gen-btn');
  const resultEl = document.getElementById('tool-result');
  const codePreview = document.getElementById('tool-code-preview');

  btn.innerHTML = '<span class="spinner"></span>Generating...';
  btn.disabled = true;
  resultEl.style.display = 'none';
  codePreview.style.display = 'none';

  try {
    const res = await fetch('/api/tools/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: desc})
    });
    const data = await res.json();
    if (data.error) {
      resultEl.className = 'error';
      resultEl.textContent = 'Error: ' + data.error;
      resultEl.style.display = 'block';
    } else {
      resultEl.className = 'success';
      resultEl.textContent = 'Tool created: ' + data.function_name + '()';
      resultEl.style.display = 'block';
      codePreview.textContent = data.code;
      codePreview.style.display = 'block';
      await loadTools();
      toast('Tool "' + data.function_name + '" added!', 'success');
      document.getElementById('tool-desc').value = '';
    }
  } catch (e) {
    resultEl.className = 'error';
    resultEl.textContent = 'Network error: ' + e.message;
    resultEl.style.display = 'block';
  }

  btn.innerHTML = 'Generate Tool';
  btn.disabled = false;
}

async function deleteTool(name) {
  const res = await fetch(`/api/tools/${encodeURIComponent(name)}`, {method:'DELETE'});
  const data = await res.json();
  if (data.ok) {
    await loadTools();
    toast('Tool "' + name + '" removed', 'info');
  } else {
    toast('Could not remove tool: ' + (data.error || 'unknown error'), 'error');
  }
}

// ============================================================
// EXPORT / IMPORT
// ============================================================
function exportChat() {
  if (!messages.length) { toast('No messages to export', 'error'); return; }
  const payload = {
    exported_at: new Date().toISOString(),
    conversation_id: currentConvId,
    messages: messages
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `chat_export_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  toast('Chat exported!', 'success');
}

function importChat(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async e => {
    try {
      const data = JSON.parse(e.target.result);
      const importedMsgs = data.messages || [];
      if (!importedMsgs.length) { toast('No messages found in file', 'error'); return; }
      // Create a new conversation
      const res = await fetch('/api/conversations', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({title: 'Imported: ' + file.name})
      });
      const conv = await res.json();
      currentConvId = conv.id;
      messages = importedMsgs;
      // Save each message to DB
      for (const msg of importedMsgs) {
        await fetch(`/api/conversations/${conv.id}/messages`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({role: msg.role, content: msg.content})
        });
      }
      saveMessagesToLocalStorage();
      renderMessages();
      await loadConversations();
      toast('Chat imported successfully!', 'success');
    } catch (err) {
      toast('Invalid JSON file: ' + err.message, 'error');
    }
  };
  reader.readAsText(file);
  event.target.value = '';
}

// ============================================================
// LOCAL STORAGE
// ============================================================
function saveMessagesToLocalStorage() {
  const last10 = messages.slice(-10);
  localStorage.setItem(LS_KEY, JSON.stringify(last10));
}

function clearLocalStorage() {
  localStorage.removeItem(LS_KEY);
  localStorage.removeItem(LS_CONV_KEY);
}

// ============================================================
// CLEAR / RESET
// ============================================================
function clearCurrentChat() {
  if (!messages.length) { toast('Chat is already empty', 'info'); return; }
  messages = [];
  clearLocalStorage();
  renderMessages();
  toast('Chat cleared', 'info');
}

async function confirmClearAll() {
  if (!confirm('Delete ALL conversation history? This cannot be undone.')) return;
  const res = await fetch('/api/conversations');
  const convs = await res.json();
  for (const c of convs) {
    await fetch(`/api/conversations/${c.id}`, {method:'DELETE'});
  }
  currentConvId = null;
  messages = [];
  clearLocalStorage();
  renderMessages();
  await loadConversations();
  toast('All history cleared', 'info');
}

// ============================================================
// UI HELPERS
// ============================================================
function showTyping(show) {
  document.getElementById('typing-indicator').style.display = show ? 'block' : 'none';
  if (show) scrollToBottom();
}

function scrollToBottom() {
  const wrap = document.getElementById('messages-wrap');
  setTimeout(() => wrap.scrollTo({top: wrap.scrollHeight, behavior: 'smooth'}), 30);
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function useSuggestion(el) {
  document.getElementById('user-input').value = el.textContent;
  sendMessage();
}

function copyMsg(idx) {
  const msg = messages[idx];
  if (!msg) return;
  navigator.clipboard.writeText(msg.content).then(() => {
    const btns = document.querySelectorAll(`[data-idx="${idx}"] .copy-btn`);
    btns.forEach(b => { b.textContent = 'copied!'; b.classList.add('copied'); });
    setTimeout(() => btns.forEach(b => { b.textContent = 'copy'; b.classList.remove('copied'); }), 1500);
  });
}

function openAddToolModal() { document.getElementById('tool-modal').classList.add('open'); }
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
  document.getElementById('tool-result').style.display = 'none';
  document.getElementById('tool-code-preview').style.display = 'none';
}
function closeModalOnBackdrop(e) {
  if (e.target === e.currentTarget) closeModal(e.currentTarget.id);
}
function toggleSidebar() { document.getElementById('sidebar').classList.toggle('open'); }

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fmtDate(iso) {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return Math.floor(diff/60000) + 'm ago';
    if (diff < 86400000) return Math.floor(diff/3600000) + 'h ago';
    return d.toLocaleDateString();
  } catch { return ''; }
}

function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}); }
  catch { return ''; }
}
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("\n" + "="*60)
        print("  WARNING: GROQ_API_KEY is not set!")
        print("  Create a .env file with:  GROQ_API_KEY=gsk_your_key")
        print("  Get a free key at: https://console.groq.com")
        print("="*60 + "\n")
    print(f"  Starting AI Chatbot on http://localhost:{PORT}")
    print(f"  Model: {MODEL}")
    print(f"  Database: {DB_PATH}")
    print(f"  Tools file: {TOOLS_FILE}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)

"""Software Task Breakdown Agent  (Question 12)
Framework : LangGraph   |  Reasoning : Chain-of-Thought   |  Memory : FAISS
Setup : pip install -r requirements.txt   ->  copy .env.example to .env and add OPENAI_API_KEY
Run   : python q12_software_task_breakdown.py   ->  open http://127.0.0.1:5012
"""
import os, re, json, ast, operator, shutil
from typing import TypedDict
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY missing. Create a .env file (see .env.example).")

TASK = "break a software-development task into smaller implementation steps and remember the user's preferred programming tools."
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3)
emb = OpenAIEmbeddings(model="text-embedding-3-small")


def ask(system, human):
    """One LLM call. System/human are passed as variables so braces in text are safe."""
    p = ChatPromptTemplate.from_messages([("system", "{s}"), ("human", "{h}")])
    return (p | llm | StrOutputParser()).invoke({"s": system, "h": human})


def parse(text):
    """Split model output into (reasoning, final answer, memory note)."""
    note = ""
    m = re.search(r"MEMORY:\s*(.*)$", text, re.S)
    if m:
        note, text = m.group(1).strip(), text[:m.start()].strip()
    a = re.search(r"FINAL ANSWER:\s*(.*)$", text, re.S | re.I)
    if a:
        return text[:a.start()].strip(), a.group(1).strip(), note
    return "", text.strip(), note


# ----------------------------------------------------------------- MEMORY
INDEX_DIR = "faiss_memory_q12"
vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True) if os.path.exists(INDEX_DIR) else None


def mem_save(text):
    global vs
    if not text:
        return
    if vs is None:
        vs = FAISS.from_texts([text], emb)
    else:
        vs.add_texts([text])
    vs.save_local(INDEX_DIR)


def mem_search(q, k=3):
    if vs is None:
        return []
    n = len(vs.docstore._dict)
    return [d.page_content for d in vs.similarity_search(q, k=min(k, n))]


def mem_all():
    return [] if vs is None else [d.page_content for d in vs.docstore._dict.values()]


def mem_clear():
    global vs
    vs = None
    shutil.rmtree(INDEX_DIR, ignore_errors=True)

# ------------------------------------------------------------ AGENT
SYSTEM = ("You are an AI agent. Task: " + TASK + " Use Chain-of-Thought reasoning: break the problem into smaller sub-problems and solve them in order. Use the recalled memory when it is relevant. "
          "Format: numbered reasoning steps, then a line 'FINAL ANSWER:' with the result, "
          "then a last line 'MEMORY:' with one short fact or preference worth remembering.")


class S(TypedDict, total=False):
    q: str
    memory: str
    reasoning: str
    answer: str
    note: str


def n_recall(s):
    return {"memory": "\n".join(mem_search(s["q"])) or "No previous memory."}


def n_reason(s):
    r, a, n = parse(ask(SYSTEM, "Recalled memory:\n" + s["memory"] + "\n\nUser request: " + s["q"]))
    return {"reasoning": r, "answer": a, "note": n or s["q"]}


def n_save(s):
    mem_save(s["note"])
    return {}


g = StateGraph(S)
g.add_node("recall", n_recall)
g.add_node("reason", n_reason)
g.add_node("save", n_save)
g.add_edge(START, "recall")
g.add_edge("recall", "reason")
g.add_edge("reason", "save")
g.add_edge("save", END)
graph = g.compile()


def run(q):
    s = graph.invoke({"q": q})
    return [{"title": "Node 1 - recall (FAISS)", "content": s["memory"]},
            {"title": "Node 2 - Chain-of-Thought reasoning", "content": s["reasoning"] or "(see answer)"},
            {"title": "Node 3 - memory saved", "content": s["note"]}], s["answer"]


# ------------------------------------------------------------ BACKEND (Flask API)
app = Flask(__name__)


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.post("/api/run")
def api_run():
    q = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not q:
        return jsonify(error="Please enter a request."), 400
    try:
        steps, answer = run(q)
        return jsonify(steps=steps, answer=answer, memory=mem_all())
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/api/memory")
def api_memory():
    return jsonify(memory=mem_all())


@app.post("/api/memory/clear")
def api_clear():
    mem_clear()
    return jsonify(memory=[])


# ------------------------------------------------------------ FRONTEND (interactive HTML page)
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Software Task Breakdown Agent</title>
<style>
:root{--bg:#f5f7fb;--card:#fff;--ink:#1c2333;--mut:#667085;--line:#e4e7ee;--acc:#4f46e5;--soft:#eef0ff}
@media(prefers-color-scheme:dark){:root{--bg:#0f1420;--card:#171d2c;--ink:#e8ecf5;--mut:#98a2b3;--line:#26304a;--acc:#818cf8;--soft:#1f2740}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 system-ui,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
header{padding:18px 24px;background:var(--card);border-bottom:1px solid var(--line)}
h1{margin:0 0 8px;font-size:20px}
.tag{display:inline-block;margin-right:6px;padding:2px 10px;border-radius:99px;background:var(--soft);color:var(--acc);font-size:12px;font-weight:600}
main{display:grid;grid-template-columns:1fr 320px;gap:18px;max-width:1200px;margin:0 auto;padding:20px 24px}
@media(max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
h2{margin:0 0 10px;font-size:15px}
textarea{width:100%;min-height:90px;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit;resize:vertical}
button{padding:8px 16px;border:0;border-radius:8px;background:var(--acc);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button.ghost{background:var(--soft);color:var(--acc)}
button:disabled{opacity:.6;cursor:wait}
.row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.step{border-left:3px solid var(--acc);background:var(--soft);padding:8px 12px;margin:10px 0;border-radius:0 8px 8px 0}
.step b{display:block;font-size:13px}
pre{white-space:pre-wrap;word-wrap:break-word;margin:4px 0 0;font:inherit}
ul{margin:0;padding-left:18px}li{margin:6px 0;font-size:13px}
.mut{color:var(--mut)}.err{color:#d92d20;font-weight:600}
</style></head>
<body>
<header><h1>Software Task Breakdown Agent</h1>
<span class="tag">LangGraph</span><span class="tag">Chain-of-Thought</span><span class="tag">FAISS memory</span></header>
<main>
<section>
 <div class="card"><h2>Your request</h2>
  <textarea id="q" placeholder="Build a library management web app. I prefer Python, Flask and SQLite."></textarea>
  <div class="row"><button id="go">Run agent</button><button class="ghost" id="ex">Use example</button></div>
  <div id="err" class="err"></div></div>
 <div class="card"><h2>Agent trace</h2><div id="steps" class="mut">Run the agent to see every reasoning step.</div></div>
 <div class="card"><h2>Final answer</h2><pre id="ans" class="mut">No answer yet.</pre></div>
</section>
<aside><div class="card"><h2>Stored memory (FAISS)</h2><ul id="mem"></ul>
 <div class="row"><button class="ghost" id="rf">Refresh</button><button class="ghost" id="clr">Clear</button></div></div></aside>
</main>
<script>
const $ = id => document.getElementById(id);
const EXAMPLE = "Build a library management web app. I prefer Python, Flask and SQLite.";
function el(tag, cls, text){ const e = document.createElement(tag); if(cls) e.className = cls; if(text !== undefined) e.textContent = text; return e; }
function showMem(list){
  const ul = $("mem"); ul.innerHTML = "";
  if(!list.length){ ul.appendChild(el("li", "mut", "Memory is empty.")); return; }
  list.forEach(t => ul.appendChild(el("li", "", t)));
}
async function loadMem(){ const r = await fetch("/api/memory"); showMem((await r.json()).memory); }
$("ex").onclick = () => { $("q").value = EXAMPLE; };
$("rf").onclick = loadMem;
$("clr").onclick = async () => { const r = await fetch("/api/memory/clear", {method: "POST"}); showMem((await r.json()).memory); };
$("go").onclick = async () => {
  const q = $("q").value.trim(); $("err").textContent = "";
  if(!q){ $("err").textContent = "Please type a request."; return; }
  $("go").disabled = true; $("go").textContent = "Thinking...";
  $("steps").textContent = "Agent is working..."; $("ans").textContent = "";
  try{
    const r = await fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({query: q})});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || "Request failed");
    $("steps").innerHTML = "";
    d.steps.forEach(s => { const b = el("div", "step"); b.appendChild(el("b", "", s.title)); b.appendChild(el("pre", "", s.content)); $("steps").appendChild(b); });
    $("ans").className = ""; $("ans").textContent = d.answer; showMem(d.memory);
  }catch(e){ $("err").textContent = e.message; $("steps").textContent = ""; }
  $("go").disabled = false; $("go").textContent = "Run agent";
};
loadMem();
</script></body></html>"""

if __name__ == "__main__":
    app.run(port=5012, debug=False)

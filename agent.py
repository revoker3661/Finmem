"""Finance Companion Agent — Priya Sharma. See ARCHITECTURE.md."""
import os, sys, json, argparse, time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import tools as mock_tools

# Force UTF-8 on Windows so ₹ prints correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

# ── SECTION 1: Constants ──────────────────────────────────────────────────────

MEMORY_FILE = "memory.json"

USER_PROFILE = {
    "name": "Priya Sharma", "age": 28, "city": "Bangalore",
    "monthly_income_inr": 120000,
    "stated_goal": "Save ₹15 lakh in 2 years for a house down payment in Bangalore",
}

SESSION_DATES = {1: "Monday, November 3, 2025", 2: "Thursday, November 6, 2025"}

SESSIONS = {
    1: [
        "I just got my salary credited. Help me figure out how much I can realistically save this month.",
        "I feel like I'm spending too much on food delivery. How much did I actually spend on it last month?",
        "Okay that's worse than I thought. Let's say I want to cut that in half AND put aside ₹30,000 for my house fund this month — is that realistic given my upcoming bills?",
        "Got it. Remind me to actually transfer the ₹30,000 to my house fund on the 25th.",
    ],
    2: [
        "Hey, my colleague is selling his MacBook for ₹80,000, barely used. I've been wanting to upgrade. Should I buy it?",
    ],
}

# Tool schemas: descriptions engineered to enforce data-freshness discipline
_TOOLS = genai.protos.Tool(function_declarations=[
    genai.protos.FunctionDeclaration(
        name="get_account_balance",
        description="Get CURRENT balances. ALWAYS call this — never use a balance from a previous session, it will be stale.",
        parameters=genai.protos.Schema(type=genai.protos.Type.OBJECT, properties={}),
    ),
    genai.protos.FunctionDeclaration(
        name="get_upcoming_bills",
        description="Get bills due in the next N days. ALWAYS call before assessing savings capacity — bills may have cleared since last session.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={"days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Days ahead. Default 30.")},
        ),
    ),
    genai.protos.FunctionDeclaration(
        name="get_recent_transactions",
        description="Fetch transactions from the last N days. Use days=35 for a full previous month. Returns debits (negative) and credits (positive) in INR.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={"days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Days to look back.")},
            required=["days"],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name="set_reminder",
        description="Schedule a reminder. Call when: (1) user requests one, (2) a time-sensitive action needs flagging, OR (3) you recommend deferring a financial decision — always set a concrete date to revisit it.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "date":    genai.protos.Schema(type=genai.protos.Type.STRING, description="YYYY-MM-DD"),
                "content": genai.protos.Schema(type=genai.protos.Type.STRING, description="Reminder text. Be specific."),
            },
            required=["date", "content"],
        ),
    ),
])

# ── SECTION 2: Tool Dispatcher (zero LLM calls) ───────────────────────────────

def _inr(amount: int) -> str:
    return f"₹{abs(amount):,}"

def execute_tool(name: str, args: dict, log: list) -> str:
    """Execute tool in pure Python. Pre-processes results so LLM never does arithmetic."""
    # Normalize floats Gemini sometimes sends (e.g. days=35.0 → 35)
    args = {k: int(v) if isinstance(v, float) and v == int(v) else v for k, v in args.items()}
    log.append(name)
    print(f"\n  [TOOL] {name}({args})")

    if name == "get_account_balance":
        b = mock_tools.get_account_balance()
        out = (f"Checking: {_inr(b['checking'])} | Savings: {_inr(b['savings'])} | "
               f"House Fund: {_inr(b['house_fund'])} | Mutual Funds: {_inr(b['mutual_funds'])}")

    elif name == "get_upcoming_bills":
        days = int(args.get("days", 30))
        bills = mock_tools.get_upcoming_bills(days)
        total = sum(b["amount"] for b in bills)          # arithmetic in code
        lines = " | ".join(f"{b['description']} {_inr(b['amount'])} (due {b['date']})" for b in bills)
        out = f"Bills due (next {days}d): {lines} — TOTAL: {_inr(total)}"

    elif name == "get_recent_transactions":
        days = int(args.get("days", 30))
        session_dt = datetime(2025, 11, 3 if mock_tools.CURRENT_SESSION == 1 else 6)
        cutoff = (session_dt - timedelta(days=days)).strftime("%Y-%m-%d")
        txns = [t for t in mock_tools.get_recent_transactions(days) if t["date"] > cutoff]

        by_cat: dict = {}
        for t in txns:
            by_cat[t["category"]] = by_cat.get(t["category"], 0) + t["amount"]
        cat_summary = " | ".join(f"{c}: {_inr(v)}" for c, v in sorted(by_cat.items()))

        food = [t for t in txns if t["category"] == "food_delivery"]
        food_total = sum(t["amount"] for t in food)      # arithmetic in code
        food_detail = ", ".join(f"{t['date']} {t['merchant']} {_inr(t['amount'])}" for t in food)
        out = (f"Transactions (last {days}d) by category: {cat_summary}\n"
               f"  Food delivery total: {_inr(food_total)} — {food_detail or 'none'}")

    elif name == "set_reminder":
        r = mock_tools.set_reminder(args["date"], args["content"])
        out = f"Reminder set — {r['date']}: '{r['content']}' (id: {r['reminder_id']})"

    else:
        out = f"Unknown tool: {name}"

    print(f"  [RESULT] {out}\n")
    return out

# ── SECTION 3: Memory Layer ───────────────────────────────────────────────────

def load_memory() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {"episodic": {}, "semantic": {}}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data: dict) -> None:
    """Atomic write — prevents corruption if process dies mid-write."""
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MEMORY_FILE)

# ── SECTION 4: System Prompt Builder ─────────────────────────────────────────

def build_system_prompt(memory: dict, session: int) -> str:
    """
    Injects only SEMANTIC memory (decisions, not numbers) into the prompt.
    Raw balances / bill amounts from prior sessions are structurally absent.
    """
    s = memory.get("semantic", {})
    prompt = f"""You are Priya's personal finance companion. She trusts you to help her make smart, grounded money decisions.

TODAY: {SESSION_DATES[session]}.
USER: {USER_PROFILE['name']}, {USER_PROFILE['age']}, {USER_PROFILE['city']} | Income: ₹1,20,000/month | Goal: {USER_PROFILE['stated_goal']}

TOOL RULES (mandatory):
- ALWAYS call get_account_balance() for current balance — never estimate or reuse old figures.
- ALWAYS call get_upcoming_bills() before any savings assessment.
- ALWAYS call get_recent_transactions() to analyze spending — do not ask her to recall.
- Call set_reminder() when she asks, when a time-sensitive action needs flagging, or when you recommend deferring a decision — always anchor it to a specific date.

STYLE: Direct. Lead with the number. Use ₹ with exact figures. No excessive hedging."""

    if not s:
        return prompt

    # Session 2+: inject structured decisions — decisions don't go stale, numbers do
    mem_lines = ["\n\nMEMORY FROM PREVIOUS SESSION:"]
    if s.get("commitments"):
        mem_lines += ["  Commitments:"] + [f"    • {c}" for c in s["commitments"]]
    if s.get("goals"):
        mem_lines += ["  Goals:"] + [f"    • {g}" for g in s["goals"]]
    if s.get("behavioral_patterns"):
        mem_lines += ["  Spending patterns:"] + [f"    • {k}: {v}" for k, v in s["behavioral_patterns"].items()]
    if s.get("reminders_set"):
        mem_lines += ["  Reminders set:"] + [f"    • {r['date']}: {r['content']}" for r in s["reminders_set"]]
    if s.get("session_summary"):
        mem_lines.append(f"  Summary: {s['session_summary']}")

    mem_lines += [
        "\nDATA FRESHNESS (critical): The memory above contains DECISIONS, not current financials.",
        "Call get_account_balance() and get_upcoming_bills() fresh — those numbers have changed.",
        "NEVER quote a balance or bill amount from a previous session.",
        "\nPROACTIVE: If Priya's new question touches a prior commitment or goal, surface that link unprompted.",
    ]
    return prompt + "\n".join(mem_lines)

# ── SECTION 5: Memory Consolidation ──────────────────────────────────────────

def _call(model, *args, **kwargs):
    """Exponential backoff on rate-limit (free tier: 5–20 RPM/RPD depending on model)."""
    for wait in [15, 45, 90]:
        try:
            return model.generate_content(*args, **kwargs)
        except ResourceExhausted as e:
            print(f"  [RATE LIMIT] Waiting {wait}s... ({str(e)[:60]})")
            time.sleep(wait)
    return model.generate_content(*args, **kwargs)  # final attempt, let it raise

def consolidate(history: list, memory: dict, session: int, tools_log: list) -> dict:
    """One LLM call at session end. Judgment needed: distinguish commitments from hypotheticals."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['parts'][0]}"
        for m in history
        if isinstance(m.get("parts", [None])[0], str) and m["role"] in ("user", "model")
    )
    prompt = f"""Extract structured memory from this finance conversation. Return ONLY valid JSON.

Fields to extract:
  commitments: list of strings (concrete actions with amounts/dates if stated)
  goals: list of strings (financial goals, short and long term)
  behavioral_patterns: dict (spending category → observed pattern)
  reminders_set: list of {{date, content}} dicts
  session_summary: one-sentence string

DO NOT include: account balances, transaction lists, bill amounts — those go stale.

CONVERSATION:
{transcript}

JSON:"""

    model = genai.GenerativeModel(MODEL)
    raw = _call(model, prompt).text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    extracted = json.loads(raw)
    memory["episodic"][f"session_{session}"] = {"date": SESSION_DATES[session],
        "turns": len([m for m in history if m["role"] == "user"]) - 1, "tools_called": tools_log}
    memory["semantic"].update(extracted)
    return memory


# ── SECTION 6: Agent Loop ─────────────────────────────────────────────────────

def run_session(session: int, turns: list) -> None:
    """ReAct loop: send message → handle tool calls → get response → repeat."""
    memory = load_memory()
    model  = genai.GenerativeModel(MODEL)
    history = [
        {"role": "user",  "parts": [build_system_prompt(memory, session)]},
        {"role": "model", "parts": ["Understood. Ready to help Priya."]},
    ]
    log: list = []  # tracks tool calls for episodic memory

    print(f"\n{'='*60}\n  SESSION {session}  —  {SESSION_DATES[session]}\n{'='*60}")

    for i, msg in enumerate(turns, 1):
        print(f"\n{'─'*60}\n  PRIYA [{i}]: {msg}\n{'─'*60}")
        history.append({"role": "user", "parts": [msg]})

        while True:
            resp  = _call(model, history, tools=[_TOOLS],
                          generation_config=genai.GenerationConfig(temperature=0.3))
            parts = resp.candidates[0].content.parts
            calls = [p for p in parts if hasattr(p, "function_call") and p.function_call.name]

            if calls:
                history.append({"role": "model", "parts": parts})
                results = [
                    genai.protos.Part(function_response=genai.protos.FunctionResponse(
                        name=p.function_call.name,
                        response={"result": execute_tool(p.function_call.name, dict(p.function_call.args), log)},
                    )) for p in calls
                ]
                history.append({"role": "user", "parts": results})
            else:
                text = "".join(p.text for p in parts if hasattr(p, "text"))
                history.append({"role": "model", "parts": [text]})
                print(f"\n  AGENT: {text}")
                break

    print(f"\n{'─'*60}\n  [CONSOLIDATING MEMORY...]\n{'─'*60}")
    updated = consolidate(history, memory, session, log)
    save_memory(updated)
    s = updated["semantic"]
    print(f"  [MEMORY SAVED] commitments={len(s.get('commitments',[]))} | "
          f"goals={len(s.get('goals',[]))} | reminders={len(s.get('reminders_set',[]))} | "
          f"tools={', '.join(dict.fromkeys(log))}")
    print(f"  [SUMMARY] {s.get('session_summary','')}")
    print(f"{'='*60}\n")


# ── SECTION 7: Entry Point ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Finance Companion Agent")
    p.add_argument("--session", type=int, choices=[1, 2], required=True,
                   help="Session to run: 1=Monday Nov 3, 2=Thursday Nov 6")
    args = p.parse_args()
    mock_tools.CURRENT_SESSION = args.session
    run_session(args.session, SESSIONS[args.session])

if __name__ == "__main__":
    main()

"""Finance Companion Agent — Priya Sharma. See ARCHITECTURE.md."""
import os, sys, json, argparse, time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from groq import Groq
import tools as mock_tools

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

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

_TOOLS = [
    {"type": "function", "function": {
        "name": "get_account_balance",
        "description": "Get CURRENT balances. ALWAYS call this — never use a balance from a previous session, it will be stale.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_upcoming_bills",
        "description": "Get bills due in the next N days. ALWAYS call before assessing savings capacity — bills may have cleared since last session.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "Days ahead. Default 30."}
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_recent_transactions",
        "description": "Fetch transactions from the last N days. Use days=35 for a full previous month. Returns debits (negative) and credits (positive) in INR.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "Days to look back."}
        }, "required": ["days"]},
    }},
    {"type": "function", "function": {
        "name": "set_reminder",
        "description": "Schedule a reminder. Call when: (1) user requests one, (2) a time-sensitive action needs flagging, OR (3) you recommend deferring a financial decision — always set a concrete date to revisit it.",
        "parameters": {"type": "object", "properties": {
            "date":    {"type": "string", "description": "YYYY-MM-DD"},
            "content": {"type": "string", "description": "Reminder text. Be specific."},
        }, "required": ["date", "content"]},
    }},
]

# ── SECTION 2: Tool Dispatcher (zero LLM calls) ───────────────────────────────

def _inr(amount: int) -> str:
    return f"₹{abs(amount):,}"

def execute_tool(name: str, args: dict, log: list) -> str:
    """Execute tool in pure Python. Pre-processes results so LLM never does arithmetic."""
    args = args or {}
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
    """Injects only SEMANTIC memory (decisions, not numbers) into the prompt."""
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

    mem_lines = ["\n\nMEMORY FROM PREVIOUS SESSION:"]
    if s.get("commitments"):
        mem_lines += ["  Commitments:"] + [f"    * {c}" for c in s["commitments"]]
    if s.get("goals"):
        mem_lines += ["  Goals:"] + [f"    * {g}" for g in s["goals"]]
    if s.get("behavioral_patterns"):
        mem_lines += ["  Spending patterns:"] + [f"    * {k}: {v}" for k, v in s["behavioral_patterns"].items()]
    if s.get("reminders_set"):
        mem_lines += ["  Reminders set:"] + [f"    * {r['date']}: {r['content']}" for r in s["reminders_set"]]
    if s.get("session_summary"):
        mem_lines.append(f"  Summary: {s['session_summary']}")
    mem_lines += [
        "\nDATA FRESHNESS (critical): The memory above contains DECISIONS, not current financials.",
        "Call get_account_balance() and get_upcoming_bills() fresh — those numbers have changed.",
        "NEVER quote a balance or bill amount from a previous session.",
        "\nPROACTIVE: If Priya's new question touches a prior commitment or goal, surface that link unprompted.",
    ]
    return prompt + "\n".join(mem_lines)

# ── SECTION 5: LLM Call + Memory Consolidation ───────────────────────────────

def _call(messages: list, tools=None) -> object:
    time.sleep(2)
    kwargs = {"model": MODEL, "messages": messages, "temperature": 0.3}
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)

def consolidate(history: list, memory: dict, session: int, tools_log: list) -> dict:
    """One LLM call at session end. Extracts decisions, not numbers."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history
        if isinstance(m.get("content"), str) and m["role"] in ("user", "assistant")
    )
    prompt = f"""Extract structured memory from this finance conversation. Return ONLY valid JSON.

Fields to extract:
  commitments: list of strings (concrete actions with amounts/dates if stated)
  goals: list of strings (financial goals, short and long term)
  behavioral_patterns: dict (spending category -> observed pattern)
  reminders_set: list of {{date, content}} dicts
  session_summary: one-sentence string

DO NOT include: account balances, transaction lists, bill amounts — those go stale.

CONVERSATION:
{transcript}

JSON:"""

    raw = _call([{"role": "user", "content": prompt}]).choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    extracted = json.loads(raw)
    memory["episodic"][f"session_{session}"] = {"date": SESSION_DATES[session],
        "turns": len([m for m in history if m["role"] == "user"]) - 1, "tools_called": tools_log}
    memory["semantic"].update(extracted)
    return memory

# ── SECTION 6: Agent Loop ─────────────────────────────────────────────────────

def run_session(session: int, turns: list) -> None:
    """ReAct loop: send message -> handle tool calls -> get response -> repeat."""
    memory  = load_memory()
    history = [{"role": "system", "content": build_system_prompt(memory, session)}]
    log: list = []

    print(f"\n{'='*60}\n  SESSION {session}  —  {SESSION_DATES[session]}\n{'='*60}")

    for i, msg in enumerate(turns, 1):
        print(f"\n{'─'*60}\n  PRIYA [{i}]: {msg}\n{'─'*60}")
        history.append({"role": "user", "content": msg})

        while True:
            resp     = _call(history, tools=_TOOLS)
            msg_obj  = resp.choices[0].message
            tool_calls = msg_obj.tool_calls or []

            if tool_calls:
                history.append({"role": "assistant", "content": msg_obj.content, "tool_calls": tool_calls})
                for tc in tool_calls:
                    args   = json.loads(tc.function.arguments or "{}")
                    result = execute_tool(tc.function.name, args, log)
                    history.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                text = msg_obj.content or ""
                history.append({"role": "assistant", "content": text})
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

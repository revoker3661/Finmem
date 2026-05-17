# Finance Companion Agent

A stateful, two-session AI finance agent built from scratch — no LangChain, no LlamaIndex, no agent frameworks. The agent holds a conversation with a user on Monday, persists structured memory to disk, and picks up three days later with full context of what was decided — not just what was said.

Built as a submission for an AI Engineer assignment. The core challenge: knowing **what to remember, what to forget, and what to always verify live**.

---

## What It Does

**Session 1 — Monday, Nov 3:** Priya just got her salary. The agent helps her plan savings, analyzes her food delivery overspend, assesses whether a ₹30,000 house fund commitment is realistic given her bills, and sets a reminder.

**Session 2 — Thursday, Nov 6:** Priya asks about buying a ₹80,000 MacBook. The agent — without being told — connects this to her savings plan from Monday, fetches fresh financial data (balance has changed since Monday), and gives a grounded recommendation tied to her prior commitment.

---

## Architecture at a Glance

```
python agent.py --session 1   →   4 turns   →   memory.json written
python agent.py --session 2   →   1 turn    →   memory.json updated
```

```
┌─────────────────────────────────────────────────────────────┐
│                      agent.py                               │
│                                                             │
│  build_system_prompt()  ←  load_memory()                   │
│         │                       │                          │
│         │              semantic layer only                  │
│         │              (decisions, not numbers)             │
│         ▼                                                   │
│    ReAct Loop  ←────────────────────────────────┐          │
│         │                                       │          │
│         ▼                                       │          │
│    Gemini 2.5 Flash                             │          │
│         │                                       │          │
│    function_call?  ──yes──▶  execute_tool()     │          │
│         │                   (pure Python)       │          │
│         │                        │              │          │
│         │                   append result  ─────┘          │
│         │                                                   │
│    text response  ──▶  print to stdout                     │
│                                                             │
│  [end of session]  ──▶  consolidate()  ──▶  save_memory()  │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
FinMem/
│
├── agent.py          ← everything: loop, memory, tool dispatch, prompts (~299 lines)
├── tools.py          ← provided mock banking API (do not modify)
├── memory.json       ← created at runtime after Session 1; persists between sessions
├── sessions.md       ← provided reference for exact user messages
├── requirements.txt  ← google-generativeai, python-dotenv
├── ARCHITECTURE.md   ← full design doc written before implementation
└── .env              ← GEMINI_API_KEY=your_key (not committed)
```

**Why one file for all agent code:** the assignment is under 300 lines total. A single `agent.py` is fully auditable top-to-bottom in 5 minutes. No "where does this get called?" — you see it all in one scroll.

---

## Setup

**1. Clone and install dependencies**

```bash
git clone <your-repo-url>
cd FinMem
pip install -r requirements.txt
```

**2. Get a Gemini API key**

- Go to [Google AI Studio](https://aistudio.google.com)
- Create an API key (free tier: 1,500 requests/day)

**3. Create your `.env` file**

```
GEMINI_API_KEY=your_key_here
```

---

## Running the Agent

### Session 1 — Monday, Nov 3, 2025

```bash
python agent.py --session 1
```

This runs 4 turns of conversation. At the end, `memory.json` is written to disk with structured memory extracted from the conversation.

**What you'll see in the terminal:**

```
============================================================
  SESSION 1  —  Monday, November 3, 2025
============================================================

────────────────────────────────────────────────────────────
  PRIYA [1]: I just got my salary credited. Help me figure out...
────────────────────────────────────────────────────────────

  [TOOL] get_account_balance({})
  [RESULT] Checking: ₹1,28,000 | Savings: ₹1,45,000 | ...

  [TOOL] get_upcoming_bills({'days': 30})
  [RESULT] Bills due (next 30d): Rent ₹25,000 ... — TOTAL: ₹46,500

  AGENT: After your bills, you have ₹81,500 in headroom this month...
```

### Session 2 — Thursday, Nov 6, 2025

```bash
python agent.py --session 2
```

This runs 1 turn. The agent loads `memory.json` from Session 1, injects structured context into the system prompt, then fetches fresh balances and bills before responding.

**The agent will surface the ₹30,000 savings commitment unprompted** when Priya asks about the MacBook.

---

## How the Memory Layer Works

Memory is a single `memory.json` file with two tiers:

```json
{
  "episodic": {
    "session_1": {
      "date": "Monday, November 3, 2025",
      "turns": 6,
      "tools_called": ["get_account_balance", "get_upcoming_bills", ...]
    }
  },
  "semantic": {
    "commitments": [
      "Transfer ₹30,000 to house fund by November 25, 2025."
    ],
    "goals": [
      "Save ₹15 lakh in 2 years for a house down payment in Bangalore."
    ],
    "behavioral_patterns": {
      "food_delivery": "High spending (₹12,890 last month)"
    },
    "reminders_set": [
      {"date": "2025-11-25", "content": "Transfer ₹30,000 to your house fund"}
    ],
    "session_summary": "..."
  }
}
```

**What is stored:** commitments, goals, behavioral patterns, reminders — things that don't expire.

**What is deliberately NOT stored:**

| Not stored | Reason |
|---|---|
| Account balances | Stale within hours — always fetched live |
| Transaction lists | Grow unboundedly, stale, always fetched live |
| Bill amounts | Bills get paid; stale within days |
| Full conversation text | Unstructured, too large, not useful for inference |

**Atomic write:** Memory is written via `tmp → os.replace()` to prevent corruption if the process dies mid-write.

**Consolidation:** After all turns complete, one final LLM call extracts structured memory from the full transcript. One call at the end, not mid-conversation — this gives it full context and produces clean output.

---

## How Tool Dispatch Works

All tool execution is pure Python — zero LLM involvement in computation.

```
Gemini says: call get_recent_transactions(days=35)
                          │
                          ▼
         execute_tool("get_recent_transactions", {"days": 35})
                          │
              ┌───────────┴─────────────┐
         filter by date             filter by category
         (Python comparison)        (Python string match)
                          │
                     sum amounts
                     (Python sum())
                          │
                          ▼
         "Food delivery total: ₹12,890 — Oct 3 Swiggy ₹1,200, ..."
                          │
                          ▼
                 returned to Gemini as a string
```

**Operations done in code, never LLM:**

| Operation | Method |
|---|---|
| Sum food delivery spend | `sum(t['amount'] for t in food)` |
| Filter by category | `if t['category'] == 'food_delivery'` |
| Filter by date range | `t['date'] > cutoff` string comparison |
| Calculate savings headroom | `balance - total_bills` |
| Format ₹ currency | f-string with `abs()` |

---

## Tool Schema Design

Tool descriptions are engineered to reinforce data-freshness discipline at the schema level — a second enforcement layer on top of the system prompt:

```python
"Get CURRENT balances. ALWAYS call this — never use a balance 
 from a previous session, it will be stale."
```

```python
"Schedule a reminder. Call when: (1) user requests one, 
 (2) a time-sensitive action needs flagging, OR 
 (3) you recommend deferring a financial decision — always 
 set a concrete date to revisit it."
```

The third condition on `set_reminder` is what causes the agent to proactively set a Dec 1 "revisit MacBook" reminder in Session 2 without being asked.

---

## System Prompt Design

**Session 1 system prompt includes:**
- Session date ("TODAY: Monday, November 3, 2025")
- User profile
- 4 mandatory tool rules (each starts with "ALWAYS")
- Style: "Direct. Lead with the number."

**Session 2 adds (injected from `memory.json` semantic layer):**
```
MEMORY FROM PREVIOUS SESSION:
  Commitments:
    • Transfer ₹30,000 to house fund by November 25, 2025.
    • Reduce food delivery spending to ₹6,445 this month.
  Goals:
    • Save ₹15 lakh in 2 years for a house down payment in Bangalore.
  ...

DATA FRESHNESS (critical): The memory above contains DECISIONS, not 
current financials. Call get_account_balance() and get_upcoming_bills() 
fresh — those numbers have changed.
NEVER quote a balance or bill amount from a previous session.

PROACTIVE: If Priya's new question touches a prior commitment or goal, 
surface that link unprompted.
```

**Why balances are absent from the prompt:** They're structurally absent from `memory.json` — the data simply doesn't exist to quote. System prompt instruction is one layer of defense; missing data is the other.

---

## Session 2 — Full Internal Flow

When Priya asks *"Should I buy the MacBook for ₹80,000?"*:

```
Step 1 — Memory already in system prompt:
  "Transfer ₹30,000 to house fund by Nov 25"  ← Gemini sees this before responding

Step 2 — Tool: get_account_balance()
  Thursday balance: ₹99,820  (Monday was ₹1,28,000 — rent paid since then)

Step 3 — Tool: get_upcoming_bills()
  Remaining bills: ₹21,500  (rent cleared; SIP + internet + credit card remain)

Step 4 — Tool: get_recent_transactions(7)
  Food delivery (3 days): ₹3,180 — on track for ₹21,000/month (target was ₹6,445)

Step 5 — Python synthesis in execute_tool():
  available after bills = ₹99,820 − ₹21,500 = ₹78,320
  MacBook cost          = ₹80,000
  House fund commitment = ₹30,000 (due Nov 25)
  Real available        = ₹78,320 − ₹30,000 = ₹48,320
  → MacBook is not feasible without breaking the savings commitment

Step 6 — Tool: set_reminder("2025-12-01", "Revisit MacBook purchase...")
  Proactively set — agent deferred a decision, per tool schema rule 3

Step 7 — Gemini synthesizes final response:
  "₹80,000 MacBook is technically buyable, but only by skipping your
   ₹30,000 house fund transfer due Nov 25. Food delivery is also tracking
   worse than your ₹6,445 target. Recommend waiting until December — 
   reminder set for Dec 1 to revisit."
```

---

## LLM vs. Code — Decision Map

| Decision | Owner | Why |
|---|---|---|
| Is buying the MacBook wise given the savings plan? | LLM | Multi-factor judgment, tone calibration |
| What tools to call for each user question? | LLM | Requires understanding intent |
| Which Session 1 commitments are relevant to Session 2? | LLM | Relevance matching, inference |
| What to extract into memory after session ends? | LLM | Distinguishing signal from noise |
| Sum of food delivery transactions | Code | Pure arithmetic — `sum()` |
| Filter transactions by category | Code | Deterministic string match |
| Filter transactions by date range | Code | Date comparison |
| Calculate remaining savings capacity | Code | Subtraction |
| Format ₹ currency values | Code | String formatting |
| Atomic memory file write | Code | OS-level operation |

---

## Model

**Gemini 2.5 Flash** via Google AI Studio.

- Free tier: 1,500 requests/day
- Native function calling with schema validation
- 1M token context window
- Session 1 uses ~8–12 API calls; Session 2 uses ~4–6 — well within free tier

Rate-limit handling: exponential backoff at 15s → 45s → 90s before raising.

---

## Constraints Met

- No agent frameworks (no LangChain, LlamaIndex, CrewAI)
- All arithmetic done in Python — LLM never sums or parses dates
- Memory persists to disk between sessions (`memory.json`)
- 299 lines of agent code total
- Full tool-call trace visible in stdout for both sessions

---

## File the Assignment Asked For

| Deliverable | Location |
|---|---|
| Code | This repo |
| Session transcripts | Run `python agent.py --session 1` and `--session 2` |
| Architecture writeup | `ARCHITECTURE.md` |
| Loom walkthrough | See submission email |
| One-page writeup | See submission email |

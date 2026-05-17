# Finance Companion Agent — Architecture & Implementation Guide

> **Purpose of this document:** Single source of truth for architecture, design decisions,
> implementation plan, and reasoning. Written before a line of code is touched. Read this
> before reading any code.

---

## Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [LLM Selection & Rationale](#2-llm-selection--rationale)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Project Structure](#4-project-structure)
5. [Layer-by-Layer Deep Dive](#5-layer-by-layer-deep-dive)
   - 5.1 [Entry Point & Session Management](#51-entry-point--session-management)
   - 5.2 [Context Builder](#52-context-builder)
   - 5.3 [Agent Core Loop](#53-agent-core-loop)
   - 5.4 [Tool Dispatcher](#54-tool-dispatcher)
   - 5.5 [Memory Layer](#55-memory-layer)
   - 5.6 [Memory Consolidation](#56-memory-consolidation)
6. [Memory Architecture (Deep Dive)](#6-memory-architecture-deep-dive)
7. [System Prompt Design](#7-system-prompt-design)
8. [Data Freshness Policy](#8-data-freshness-policy)
9. [Session Flow Walk-Through](#9-session-flow-walk-through)
   - 9.1 [Session 1 — Monday Nov 3](#91-session-1--monday-nov-3)
   - 9.2 [Session 2 — Thursday Nov 6](#92-session-2--thursday-nov-6)
10. [LLM vs. Code Decision Map](#10-llm-vs-code-decision-map)
11. [Tool Schema Definitions](#11-tool-schema-definitions)
12. [Implementation Plan (Step-by-Step)](#12-implementation-plan-step-by-step)
13. [Key Design Decisions & Trade-offs](#13-key-design-decisions--trade-offs)
14. [What Makes This Stand Out](#14-what-makes-this-stand-out)
15. [Constraints Checklist](#15-constraints-checklist)

---

## 1. What We Are Building

A **stateful, two-session finance companion agent** for a user named Priya Sharma. The agent must:

- Hold an intelligent conversation across **two separate process runs**, days apart
- **Remember decisions**, not data — the savings commitment from Session 1 survives to Session 2; the account balance from Session 1 does not
- **Proactively connect** new questions (MacBook purchase) to prior context (savings plan) without being asked to
- **Call live tools** instead of quoting stale memory whenever numbers could have changed
- Demonstrate clean judgment: which decisions belong to the LLM, which belong to code

The core challenge is not building a chatbot. It is building a system that knows **what to remember, what to forget, and what to verify**.

### User Profile

```python
USER_PROFILE = {
    "name": "Priya Sharma",
    "age": 28,
    "city": "Bangalore",
    "monthly_income_inr": 120000,   # post-tax, credited on the 1st
    "stated_goal": "Save ₹15 lakh in 2 years for a house down payment in Bangalore",
}
```

### Sessions at a Glance

| | Session 1 | Session 2 |
|---|---|---|
| Date | Monday, Nov 3, 2025 | Thursday, Nov 6, 2025 |
| Opener | Salary just credited, plan savings | Colleague selling MacBook for ₹80,000 |
| Turns | 4 | 1 |
| Key outcome | ₹30k savings commitment + reminder set | Nuanced purchase advice tied to savings plan |
| Memory state at start | Empty | Loaded from Session 1 |

---

## 2. LLM Selection & Rationale

### Decision: Google Gemini 2.0 Flash

```
Provider   : Google AI Studio
Model      : gemini-2.0-flash
SDK        : google-generativeai (Python)
Cost       : Free tier — 1,500 requests/day, 1M tokens/minute
```

### Comparison Table

| Criterion | Gemini 2.0 Flash | Groq / Llama 3.3 70B | Mistral Free | Claude (Anthropic) |
|---|---|---|---|---|
| Free tier generosity | 1,500 req/day | ~14,400 req/day | 1 req/sec | No free API tier |
| Native function calling | Robust, schema-based | Inconsistent | Basic | Excellent |
| Context window | 1,000,000 tokens | 128,000 tokens | 32,000 tokens | 200,000 tokens |
| Financial reasoning quality | Excellent | Good | Average | Excellent |
| Structured JSON output | Native (`response_schema`) | Prompt-dependent | Prompt-dependent | Excellent |
| Multi-turn reliability | Excellent | Good | Average | Excellent |
| Latency | ~1–2s | <1s | ~2s | ~2–3s |

### Why Not Groq Despite More Requests

Groq's Llama models have **unreliable tool calling** in multi-step chains. In Session 2, the agent needs to call three tools sequentially and synthesize results. Gemini 2.0 Flash handles this natively and reliably.

### Why Not Mistral Free

Context window (32K) is too small once we add system prompt + memory + conversation history + tool results.

### Free Tier Adequacy

Session 1: ~8–12 API calls (4 turns × ~2 calls per turn for tool use + final response)
Session 2: ~4–6 API calls (1 turn with multiple tools + consolidation)
Testing (20 full runs): ~400 calls
**Total: well under 1,500/day limit.**

---

## 3. High-Level Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PRIYA'S FINANCE COMPANION                                ║
║                      Complete System Architecture                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  INVOCATION                                                              │
  │  python agent.py --session 1   →   Session 1 (Monday)                   │
  │  python agent.py --session 2   →   Session 2 (Thursday)                 │
  └──────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  CONTEXT BUILDER                                                         │
  │                                                                          │
  │  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐   │
  │  │ USER_PROFILE │    │  memory.json    │    │  Session date        │   │
  │  │ (static)     │    │  semantic layer │    │  (injected as today) │   │
  │  └──────┬───────┘    └────────┬────────┘    └──────────┬───────────┘   │
  │         │                     │                         │               │
  │         └─────────────────────┼─────────────────────────┘               │
  │                               │                                          │
  │                               ▼                                          │
  │                    ┌─────────────────────┐                               │
  │                    │   SYSTEM PROMPT     │                               │
  │                    │   (assembled here)  │                               │
  │                    └──────────┬──────────┘                               │
  └───────────────────────────────┼──────────────────────────────────────────┘
                                  │
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  AGENT CORE LOOP  —  Gemini 2.0 Flash                                   │
  │                                                                          │
  │  message_history = [system_prompt]                                       │
  │                                                                          │
  │  for each user_turn:                                                     │
  │    message_history.append(user_turn)                                     │
  │                                                                          │
  │    while True:                                                           │
  │      response = gemini.generate(message_history, tools=TOOL_SCHEMAS)    │
  │                                                                          │
  │      if response has tool_calls:                                         │
  │        results = tool_dispatcher.execute(tool_calls)  ← pure code       │
  │        message_history.append(tool_results)                              │
  │        continue   ← loop: Gemini reasons again with results              │
  │                                                                          │
  │      else:                                                               │
  │        print(response.text)   ← final answer                            │
  │        message_history.append(response)                                  │
  │        break                                                             │
  │                                                                          │
  │  [after all turns complete]                                              │
  │  memory_consolidator.run(message_history)                                │
  └──────────────────────────────────────────────────────────────────────────┘
                     │                        │
         ┌───────────┘                        └───────────────┐
         ▼                                                    ▼
  ┌─────────────────────────────┐    ┌──────────────────────────────────────┐
  │  TOOL DISPATCHER            │    │  MEMORY LAYER                        │
  │  (zero LLM calls)           │    │                                      │
  │                             │    │  ┌─────────────┐  ┌───────────────┐ │
  │  get_account_balance()      │    │  │  EPISODIC   │  │   SEMANTIC    │ │
  │  get_upcoming_bills(n)      │    │  │             │  │               │ │
  │  get_recent_transactions(n) │    │  │  session    │  │  commitments  │ │
  │  set_reminder(date, text)   │    │  │  date       │  │  goals        │ │
  │                             │    │  │  tools      │  │  patterns     │ │
  │  + pre/post processing:     │    │  │  called     │  │  reminders    │ │
  │  • filter by category       │    │  │  summary    │  │               │ │
  │  • sum amounts              │    │  └─────────────┘  └───────────────┘ │
  │  • date range filtering     │    │                                      │
  │  • format INR values        │    │  Persists to: memory.json            │
  └─────────────────────────────┘    └──────────────────────────────────────┘
```

---

## 4. Project Structure

```
submission/
│
├── agent.py              ← Main file. Contains everything: loop, memory,
│                           tool dispatch, prompt builder. ~220 lines.
│
├── tools.py              ← PROVIDED. Do not modify. Four mock tools +
│                           CURRENT_SESSION flag.
│
├── memory.json           ← Created at runtime after Session 1 ends.
│                           Persists between process runs. This is the
│                           entire memory layer — no database needed.
│
├── .env                  ← GEMINI_API_KEY=your_key_here
│                           Never committed to git.
│
├── requirements.txt      ← google-generativeai
│                           python-dotenv
│
├── sessions.md           ← PROVIDED. Exact user messages for both sessions.
│
├── README (1).md         ← PROVIDED. Assignment specification.
│
└── ARCHITECTURE.md       ← THIS FILE. Full documentation.
```

### Why One File for All Agent Code

The assignment asks for under 300 lines *total*. More importantly, a single `agent.py` is:
- **Auditable in 5 minutes** — the evaluator reads top to bottom
- **No abstraction debt** — no "where does this actually get called?"
- **Easier to walk through on Loom** — one file, one scroll

The code is organized internally with clear section comments as dividers, not artificial file splits.

### Internal Layout of agent.py

```
agent.py
│
├── [SECTION 1] Imports & constants
│   ├── USER_PROFILE dict
│   ├── SESSION_DATES dict (maps session number to date string)
│   └── TOOL_SCHEMAS list (Gemini function calling format)
│
├── [SECTION 2] Tool dispatcher
│   ├── _preprocess(tool_name, args) → modified args
│   ├── execute_tool(tool_name, args) → result dict
│   └── _postprocess(tool_name, raw_result) → formatted result
│
├── [SECTION 3] Memory layer
│   ├── load_memory() → dict
│   └── save_memory(data: dict)
│
├── [SECTION 4] Prompt builder
│   └── build_system_prompt(memory: dict, session: int) → str
│
├── [SECTION 5] Memory consolidator
│   └── consolidate(conversation_history: list, memory: dict) → dict
│
├── [SECTION 6] Agent loop
│   └── run_session(session: int, turns: list[str])
│
└── [SECTION 7] Entry point
    └── main()
```

---

## 5. Layer-by-Layer Deep Dive

### 5.1 Entry Point & Session Management

```python
# Usage
python agent.py --session 1
python agent.py --session 2
```

The entry point does four things:
1. Parse `--session` argument
2. Load the correct user turns from the hardcoded `SESSIONS` dict (mirrors sessions.md exactly)
3. Set `tools.CURRENT_SESSION` dynamically so tool mock data matches
4. Call `run_session(session_number, turns)`

```python
SESSION_DATES = {
    1: "Monday, November 3, 2025",
    2: "Thursday, November 6, 2025",
}

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
```

**Why hardcode turns here instead of reading sessions.md?**
Avoids file I/O dependency and keeps the agent fully self-contained. The evaluator runs one command; it works.

---

### 5.2 Context Builder

Assembles the system prompt from three sources:

```
USER_PROFILE (static, always included)
       +
SEMANTIC MEMORY (from memory.json, empty in Session 1)
       +
SESSION DATE (injected as "today's date")
       +
BEHAVIORAL INSTRUCTIONS (tool discipline rules, tone)
       ↓
  SYSTEM PROMPT STRING
```

**Critical rule:** Only the **semantic layer** of memory is injected. Raw numbers, transaction lists, and balances from Session 1 are never included. Only structured decisions:

```
Session 1 context injected into Session 2:
  ✓ "Priya committed to transferring ₹30,000 to her house fund on Nov 25"
  ✓ "Goal: reduce food delivery spend by 50%"
  ✓ "Long-term: ₹15L house fund in 2 years"
  ✗ "Checking balance was ₹128,000 on Monday"   ← NEVER injected
  ✗ "Upcoming bills totaled ₹46,500"             ← NEVER injected
```

---

### 5.3 Agent Core Loop

The loop is a standard **ReAct-style** (Reason + Act) loop:

```
┌─────────────────────────────────────────────────────────┐
│                   AGENT TURN LOOP                       │
│                                                         │
│  user message                                           │
│      │                                                  │
│      ▼                                                  │
│  append to history                                      │
│      │                                                  │
│      ▼                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  gemini.generate_content(history, tools)        │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                   │
│          ┌──────────┴──────────┐                        │
│    function_call?         text response                 │
│          │                    │                         │
│          ▼                    ▼                         │
│   execute_tool()         print to user                  │
│          │                    │                         │
│          ▼                    ▼                         │
│   append tool result     append to history              │
│          │                    │                         │
│          └──────────┐         └──── end of turn         │
│                     ▼                                   │
│              loop again ←── Gemini sees tool result     │
│              (can call more tools or give final answer) │
└─────────────────────────────────────────────────────────┘
```

**Key property:** Gemini can call tools **multiple times in a single user turn**. In Session 2, one user message triggers three sequential tool calls before the final answer. The loop handles this naturally — it keeps cycling until Gemini produces a text response with no tool calls.

**Message history structure:**

```python
history = [
    {"role": "user", "parts": [system_prompt]},          # injected as first user msg
    {"role": "model", "parts": ["Understood."]},          # model acknowledges
    {"role": "user", "parts": [user_turn_1]},
    {"role": "model", "parts": [function_call_1]},
    {"role": "user", "parts": [function_result_1]},
    {"role": "model", "parts": [function_call_2]},
    {"role": "user", "parts": [function_result_2]},
    {"role": "model", "parts": [final_response]},
    {"role": "user", "parts": [user_turn_2]},
    ...
]
```

**Logging:** Every tool call and result is printed to stdout with clear separators so transcripts show the full trace.

---

### 5.4 Tool Dispatcher

Executes tool calls from Gemini using pure Python. **No LLM involved.**

```
Gemini says: call get_recent_transactions(days=30)
                           │
                           ▼
            execute_tool("get_recent_transactions", {"days": 30})
                           │
                  ┌────────┴────────┐
                  │                 │
           _preprocess()      calls tools.get_recent_transactions(30)
           (validate args)          │
                                    ▼
                            raw list of transactions
                                    │
                                    ▼
                          _postprocess() if needed
                          (e.g., sum food delivery = code, not LLM)
                                    │
                                    ▼
                          return result dict to agent loop
```

**Pre/post processing that happens in code:**

| Operation | Where | Why not LLM |
|---|---|---|
| Filter transactions by category | Code (`if t['category'] == 'food_delivery'`) | Deterministic logic |
| Sum filtered amounts | Code (`sum(t['amount'] for t in filtered)`) | Arithmetic |
| Filter by date range | Code (string comparison on date field) | Deterministic |
| Format as ₹ currency | Code (f-string) | String formatting |
| Calculate remaining savings capacity | Code (balance - sum of bills) | Arithmetic |

The tool result returned to Gemini is **already processed** — Gemini sees "Food delivery spend last month: ₹11,790" not a raw list of 18 transactions.

---

### 5.5 Memory Layer

Two functions. Simple interface.

```python
def load_memory() -> dict:
    """Read memory.json. Return empty structure if file doesn't exist."""

def save_memory(data: dict) -> None:
    """Atomically write memory.json."""
```

**Atomic write pattern** (prevents corruption if process dies mid-write):

```python
def save_memory(data: dict) -> None:
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MEMORY_FILE)   # atomic on all platforms
```

**Empty memory structure (Session 1 start):**

```json
{
  "episodic": {},
  "semantic": {}
}
```

**Populated memory structure (after Session 1):**

```json
{
  "episodic": {
    "session_1": {
      "date": "2025-11-03",
      "tools_called": [
        "get_account_balance",
        "get_recent_transactions",
        "get_upcoming_bills",
        "get_recent_transactions",
        "get_upcoming_bills",
        "set_reminder"
      ],
      "turns_count": 4
    }
  },
  "semantic": {
    "commitments": [
      {
        "action": "transfer to house fund",
        "amount_inr": 30000,
        "date": "2025-11-25",
        "status": "pending"
      }
    ],
    "goals": [
      "Save ₹15 lakh in 2 years for a house down payment in Bangalore",
      "Reduce food delivery spending by 50% starting November"
    ],
    "behavioral_patterns": {
      "food_delivery": "historically overspending; ₹11,790 in October vs. ~₹6,000 target"
    },
    "reminders_set": [
      {
        "date": "2025-11-25",
        "content": "Transfer ₹30,000 to house fund",
        "reminder_id": "rem_XXXX"
      }
    ],
    "session_1_summary": "Priya reviewed her salary, identified food delivery overspend (₹11,790/month), committed to saving ₹30,000 this month for her house fund, and set a reminder for Nov 25."
  }
}
```

---

### 5.6 Memory Consolidation

After all turns in a session complete, one final LLM call extracts structured memory:

```
Input:  Full conversation transcript (message history as text)
Prompt: Structured extraction request
Output: JSON with commitments, goals, patterns, reminders
Action: Merge into memory.json semantic layer, save to disk
```

**Why one LLM call at the end, not rule-based triggers?**

Rule-based: "if `set_reminder` was called, save the reminder" — brittle. Misses implicit commitments like "let's cut food delivery in half" which are never explicitly triggered.

LLM-based mid-conversation: wasteful, adds latency to every turn.

LLM-based at end: one call, full context, extracts everything coherently. This is the right trade-off.

**Consolidation prompt structure:**

```
You are extracting structured memory from a finance conversation.
Given the conversation below, extract ONLY:

1. Explicit commitments (actions with amounts and dates)
2. Stated goals (short and long term)
3. Behavioral patterns observed (spending habits)
4. Any reminders that were set

Return ONLY valid JSON matching this schema:
{
  "commitments": [...],
  "goals": [...],
  "behavioral_patterns": {...},
  "reminders_set": [...],
  "session_summary": "one sentence"
}

Do NOT include:
- Account balances
- Transaction lists
- Upcoming bill amounts
- Any numbers that could be stale

CONVERSATION:
{transcript}
```

---

## 6. Memory Architecture (Deep Dive)

### The Core Principle

> **Store decisions. Not data.**

Data goes stale. A balance from Monday is wrong by Thursday. A commitment made on Monday is still valid on Thursday. Memory is for the latter.

### Two-Tier Design

```
┌─────────────────────────────────────────────────────────────┐
│                      memory.json                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  EPISODIC LAYER          │  SEMANTIC LAYER                 │
│  "What happened"         │  "What it means"                │
│                          │                                  │
│  • Session dates         │  • Commitments made             │
│  • Tools called          │  • Goals stated                 │
│  • Turn count            │  • Behavioral patterns          │
│  • Audit trail           │  • Reminders set                │
│                          │  • Session summary              │
│  NOT injected into       │                                  │
│  Session 2 prompt        │  INJECTED into Session 2 prompt │
│                          │                                  │
│  Purpose: Loom video,    │  Purpose: Agent has context     │
│  debugging, auditability │  without stale numbers          │
└─────────────────────────────────────────────────────────────┘
```

### What Is Deliberately NOT Stored

| Data | Reason Not Stored |
|---|---|
| Account balances | Stale within hours. Always call `get_account_balance()` |
| Transaction lists | Grows unboundedly, stale, always call `get_recent_transactions()` |
| Upcoming bill amounts | Bills get paid. Stale within days. Always call `get_upcoming_bills()` |
| Full conversation text | Too large, not structured, not useful for inference |
| LLM response text | Computations embedded in prose are untrustworthy for future use |

### Memory Evolution Across Sessions

```
Before Session 1:        After Session 1:         After Session 2 (future):
memory.json              memory.json              memory.json
──────────────           ──────────────────────   ─────────────────────────
{                        {                        {
  "episodic": {},          "episodic": {            "episodic": {
  "semantic": {}             "session_1": {..}        "session_1": {..},
}                          },                         "session_2": {..}
                           "semantic": {            },
                             "commitments":[..],    "semantic": {
                             "goals": [...],          (merged/updated)
                             "patterns": {..},      }
                             "reminders": [..]    }
                           }
                         }
```

---

## 7. System Prompt Design

### Design Philosophy

The system prompt must do three things simultaneously:
1. **Give the agent personality and role** (who it is)
2. **Supply context** (who Priya is, what was decided before)
3. **Enforce discipline** (what to fetch live vs. what to trust from memory)

The third is the most critical and least obvious. Without explicit instructions, LLMs will happily quote stale memory numbers — because they don't know the numbers are stale.

### Session 1 System Prompt

```
You are Priya's personal finance companion. She trusts you to help her
make smart, grounded money decisions.

TODAY: Monday, November 3, 2025.

USER PROFILE:
  Name: Priya Sharma, 28, Bangalore
  Monthly income: ₹1,20,000 (post-tax, credited on the 1st)
  Goal: Save ₹15 lakh in 2 years for a house down payment

TOOL USE RULES (follow strictly):
  - Always call get_account_balance() to get her current balance.
    Never estimate or assume a balance.
  - Always call get_upcoming_bills() before assessing savings capacity.
  - Always call get_recent_transactions() to analyze spending.
    Filter and sum in your head — don't ask her to count.
  - Call set_reminder() when she explicitly requests a reminder.

RESPONSE STYLE:
  - Be direct. Lead with the number or decision, then explain.
  - Use ₹ with exact figures from tool results.
  - Don't hedge excessively. She can handle real numbers.
  - Never apologize for giving honest assessments.
```

### Session 2 System Prompt (additions)

Everything above, plus:

```
MEMORY FROM YOUR PREVIOUS CONVERSATION (Monday, Nov 3):
  Commitments:
    • Priya committed to transferring ₹30,000 to her house fund by Nov 25, 2025
    • Priya aimed to cut food delivery spending by 50% this month
  Long-term goal: Save ₹15 lakh in 2 years for Bangalore house
  Reminders set: Transfer ₹30,000 to house fund on 2025-11-25

CRITICAL — DATA FRESHNESS:
  The numbers above are DECISIONS, not current financial state.
  Before giving any financial assessment today (Nov 6):
    • Call get_account_balance() — rent has likely been paid since Monday
    • Call get_upcoming_bills() — some bills may have cleared
    • Check get_recent_transactions() if spending patterns are relevant
  NEVER quote account balances or bill totals from Monday's session.
  Those numbers are stale.

PROACTIVE JUDGMENT:
  When Priya raises a new financial question, check whether it
  intersects with any commitment or goal from memory before responding.
  If it does, surface that connection explicitly — don't wait for her
  to ask.
```

---

## 8. Data Freshness Policy

This is the most important architectural decision in the system. Stated explicitly:

```
╔═══════════════════════════════════════════════════════════════╗
║                   DATA FRESHNESS POLICY                       ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ALWAYS FETCH LIVE            │  ALWAYS READ FROM MEMORY      ║
║  (call tool every time)       │  (never stale)                ║
║  ─────────────────────        │  ─────────────────────        ║
║  Account balances             │  User goals                   ║
║  Upcoming bills               │  Commitments made             ║
║  Recent transactions          │  Reminders set                ║
║  Any number that changes      │  Behavioral context           ║
║  with time                    │  Session history metadata     ║
║                               │                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### Why This Policy Exists

Between Session 1 (Monday) and Session 2 (Thursday), the following changed:
- Checking balance: ₹128,000 → ₹99,820 (rent paid, food orders added)
- Upcoming bills: 4 items → 3 items (rent cleared)
- Recent transactions: new entries added

If Session 2 quoted Monday's balance of ₹128,000, the advice would be catastrophically wrong. The user would think she could afford the MacBook; she cannot.

This policy is enforced in two places:
1. **System prompt:** explicit instruction to never quote stale numbers
2. **Memory schema:** balances and transaction lists are structurally absent from memory.json — the data simply doesn't exist to quote

---

## 9. Session Flow Walk-Through

### 9.1 Session 1 — Monday Nov 3

#### Turn 1: "I just got my salary credited. Help me figure out how much I can realistically save this month."

```
Agent internal flow:
  → Tool: get_account_balance()
    Result: checking=128,000 | savings=145,000 | house_fund=95,000
  → Tool: get_upcoming_bills(30)
    Result: rent=25,000 | SIP=10,000 | internet=3,500 | credit_card=8,000
    Code: sum = 46,500
  → Code: 128,000 - 46,500 = 81,500 (realistic savings headroom)
  → Gemini: contextualize with ₹15L goal (needs ₹6,250/month for 24 months)
  → Response: "After bills, you have about ₹81,500 in headroom this month..."
```

#### Turn 2: "I feel like I'm spending too much on food delivery. How much did I actually spend on it last month?"

```
Agent internal flow:
  → Tool: get_recent_transactions(days=35)   ← 35 to capture all of October
    Result: full transaction list
  → Code: filter where category == "food_delivery"
    Filtered: Swiggy×7, Zomato×3 entries (October only)
  → Code: sum amounts = -11,790
  → Response: "You spent ₹11,790 on food delivery in October..."
```

Note: The summation is done in code, not by the LLM. The LLM receives "food delivery total: ₹11,790" not a list of transactions to add up.

#### Turn 3: "Cut food delivery in half AND put aside ₹30,000 for house fund — is that realistic?"

```
Agent internal flow:
  → Tool: get_upcoming_bills(30)   ← called again, current state
  → Tool: get_account_balance()    ← called again, current state
  → Code: 
      food_delivery_target = 11,790 / 2 = 5,895  (savings of ~5,895)
      bills_remaining = 46,500
      savings_goal = 30,000
      total_committed = 46,500 + 30,000 = 76,500
      available = 128,000 - 76,500 = 51,500 discretionary remaining
  → Gemini: assess feasibility, note it's tight but doable
  → Response: "Yes, it's realistic — here's how the math works..."
```

#### Turn 4: "Remind me to actually transfer the ₹30,000 to my house fund on the 25th."

```
Agent internal flow:
  → Tool: set_reminder(date="2025-11-25", content="Transfer ₹30,000 to house fund")
    Result: {status: "set", reminder_id: "rem_XXXX", ...}
  → Response: "Done — reminder set for November 25th..."

[Session ends]
  → Memory consolidation call:
      Input: full 4-turn conversation
      Output: structured JSON with commitments, goals, patterns
  → save_memory(consolidated_data)
  → memory.json written to disk
```

---

### 9.2 Session 2 — Thursday Nov 6

#### Turn 1: "Hey, my colleague is selling his MacBook for ₹80,000, barely used. I've been wanting to upgrade. Should I buy it?"

```
Agent internal flow (this is the showcase):

  Step 1 — Memory check (happens before any tool call):
    System prompt already contains:
    • "Priya committed to ₹30,000 house fund transfer on Nov 25"
    • "Goal: reduce food delivery by 50%"
    Gemini recognizes: new purchase question → affects savings plan

  Step 2 — Fetch live financial state:
  → Tool: get_account_balance()
    Result: checking=99,820 | savings=145,000 | house_fund=95,000
    (Monday's ₹128,000 checking → ₹99,820 after rent + food orders)

  Step 3 — Check remaining obligations:
  → Tool: get_upcoming_bills()
    Result: SIP=10,000 | internet+mobile=3,500 | credit_card=8,000
    Code: remaining bills = 21,500

  Step 4 — Check food delivery progress:
  → Tool: get_recent_transactions(7)
    Result: 4 food delivery entries since Monday = ₹3,180 in 3 days
    Code: on track for ~₹21,000/month (worse than before, not the 50% cut goal)

  Step 5 — Code-level synthesis:
    available_after_bills = 99,820 - 21,500 = 78,320
    macbook_cost = 80,000
    shortfall = 80,000 - 78,320 = 1,680   ← can barely afford it
    but: house_fund_transfer = 30,000 due Nov 25
    real_available = 78,320 - 30,000 = 48,320
    macbook_is_feasible = False (not without skipping house fund transfer)

  Step 6 — Set a forward-looking reminder:
  → Tool: set_reminder(
        date="2025-12-01",
        content="Revisit MacBook purchase — Nov savings goal completed or not?"
    )

  Step 7 — Gemini synthesizes final response:
    • MacBook costs ₹80,000
    • After bills, you have ₹78,320 — barely enough, but
    • You committed to ₹30,000 house fund transfer on Nov 25
    • That leaves ₹48,320 — buying the MacBook breaks the commitment
    • Food delivery: ₹3,180 in 3 days — you're overspending vs. the 50% cut goal
    • Recommendation: don't buy now; revisit in December after Nov 25 transfer
    • Reminder set for Dec 1 to revisit

  → Response: nuanced, data-backed, connected to prior plan
```

This response demonstrates all four requirements:
1. Memory: references savings plan from Monday
2. Judgment: connects MacBook question to savings plan unprompted
3. Tool discipline: called get_account_balance() fresh, didn't use Monday's ₹128,000
4. Tool calling: set_reminder() called with a future-oriented reminder

---

## 10. LLM vs. Code Decision Map

| Decision | Owner | Reason |
|---|---|---|
| Is buying the MacBook wise given the savings plan? | LLM | Requires synthesizing balance, commitments, patterns, judgment |
| Is ₹30k savings realistic given bills? | LLM | Multi-factor assessment, risk judgment, tone calibration |
| What tools should be called for this user question? | LLM | Requires understanding intent and financial context |
| Which commitments from Session 1 are relevant to Session 2? | LLM | Relevance matching, inference |
| What to store in memory after session ends? | LLM (consolidation call) | Distinguishing signal from noise requires judgment |
| Sum of food delivery transactions | Code | Pure arithmetic |
| Filter transactions by category | Code | Deterministic string match |
| Filter transactions by date range | Code | Date comparison |
| Calculate remaining savings capacity | Code | Subtraction |
| Format ₹ currency values | Code | String formatting |
| Validate tool arguments | Code | Type/range checking |
| Atomic memory file write | Code | OS-level operation |
| Parse `--session` CLI argument | Code | CLI parsing |

### The Key Test (From the Assignment)

> "If you're using an LLM to do arithmetic, parse a date, or sum a column, that's a flag."

Every single arithmetic operation in this system is done in Python before the result is shown to the LLM. The LLM receives pre-computed summaries, not raw data to process.

---

## 11. Tool Schema Definitions

Gemini function calling uses this format:

```python
TOOL_SCHEMAS = [
    {
        "name": "get_recent_transactions",
        "description": (
            "Fetch transactions from the last N days. "
            "Returns debits (negative) and credits (positive) in INR. "
            "Use this to analyze spending by category or identify patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back. Use 35 for full previous month.",
                }
            },
            "required": ["days"],
        },
    },
    {
        "name": "get_account_balance",
        "description": (
            "Get current account balances across checking, savings, house fund, "
            "and mutual funds. Always call this for current balance — never use "
            "a balance from a previous session."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_upcoming_bills",
        "description": (
            "Get scheduled bills and auto-debits due in the next N days. "
            "Always call this before assessing savings capacity — bills may "
            "have been paid since the last session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Days to look ahead. Default 30.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "set_reminder",
        "description": (
            "Schedule a reminder for the user on a specific date. "
            "Call when the user explicitly requests a reminder, or when "
            "a time-sensitive financial action should be flagged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Reminder date in YYYY-MM-DD format.",
                },
                "content": {
                    "type": "string",
                    "description": "What the reminder should say. Be specific.",
                },
            },
            "required": ["date", "content"],
        },
    },
]
```

### Tool Description Engineering

Tool descriptions are critical for LLM judgment. Each description tells Gemini:
- **When** to call the tool
- **What** it returns
- **Why** it should prefer it over stale data

The phrase "Always call this" in `get_account_balance` and `get_upcoming_bills` descriptions reinforces the data freshness policy at the tool schema level — a second layer of enforcement beyond the system prompt.

---

## 12. Implementation Plan (Step-by-Step)

### Phase 1: Setup (30 minutes)

- [ ] Create project folder, initialize git
- [ ] Create `.env` with `GEMINI_API_KEY`
- [ ] Create `requirements.txt`: `google-generativeai`, `python-dotenv`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test Gemini API with a minimal call (confirm key works, free tier active)

### Phase 2: Tool Dispatcher (45 minutes)

- [ ] Write `execute_tool(name, args)` function
- [ ] Add pre-processing for `get_recent_transactions`:
  - Date range filtering
  - Category filtering
  - Amount summation (returns pre-computed dict, not raw list)
- [ ] Log every tool call and result with clear stdout formatting
- [ ] Test each tool independently with mock calls

### Phase 3: Memory Layer (30 minutes)

- [ ] Write `load_memory()` with empty-structure default
- [ ] Write `save_memory()` with atomic write pattern
- [ ] Define memory.json schema (episodic + semantic)
- [ ] Test: write → read → verify round-trip

### Phase 4: System Prompt Builder (45 minutes)

- [ ] Write `build_system_prompt(memory, session)` function
- [ ] Session 1 version: profile + tools + style only
- [ ] Session 2 version: adds memory section with semantic data
- [ ] Explicitly test that balances/bills are structurally absent from Session 2 prompt

### Phase 5: Memory Consolidation (45 minutes)

- [ ] Write consolidation prompt
- [ ] Write `consolidate(history, existing_memory)` function
- [ ] Test against a sample Session 1 transcript
- [ ] Verify output JSON matches expected schema
- [ ] Verify balances are NOT extracted (the LLM should not include them)

### Phase 6: Agent Loop (60 minutes)

- [ ] Write the message history management
- [ ] Write the tool-use loop (while True, break on text response)
- [ ] Wire in tool dispatcher
- [ ] Wire in memory consolidation at end of session
- [ ] Add conversation logging (visible tool calls in output)

### Phase 7: Entry Point (20 minutes)

- [ ] Parse `--session` argument
- [ ] Set `tools.CURRENT_SESSION` dynamically
- [ ] Wire in `SESSIONS` dict with exact user messages
- [ ] Add `SESSION_DATES` for system prompt injection

### Phase 8: End-to-End Testing (60 minutes)

- [ ] Run Session 1 → verify memory.json is written correctly
- [ ] Read memory.json → verify semantic layer contains commitments, not balances
- [ ] Run Session 2 → verify agent mentions savings plan unprompted
- [ ] Verify Session 2 calls `get_account_balance()` (not quoting Monday's balance)
- [ ] Verify Session 2 calls `set_reminder()`
- [ ] Count total lines of code → must be under 300

### Phase 9: Polish (30 minutes)

- [ ] Clean up log output to look professional for transcript submission
- [ ] Verify all code comments are meaningful (architecture decisions, not narration)
- [ ] Write `requirements.txt` with pinned versions
- [ ] Add one-line `--help` text to CLI
- [ ] Final line count check

---

## 13. Key Design Decisions & Trade-offs

### Decision 1: Single file vs. multiple files

**Chosen:** Single `agent.py`

**Alternative:** `agent.py` + `memory.py` + `prompts.py`

**Why single file:** The 300-line constraint makes multi-file splitting artificial. More importantly, a single file is walkable top-to-bottom in the Loom video without jumping between files. The internal structure (clear section comments) provides the same separation of concerns without filesystem overhead.

---

### Decision 2: JSON file vs. SQLite for memory

**Chosen:** `memory.json`

**Alternative:** `memory.db` (SQLite)

**Why JSON:** The memory structure is a single document that is read and written atomically per session. There are no queries, no joins, no concurrent writes. SQLite adds ~30 lines of boilerplate (schema, CREATE TABLE, INSERT, SELECT) for zero benefit at this scale. JSON is also directly readable in the Loom video.

---

### Decision 3: Consolidation at session end vs. real-time memory writes

**Chosen:** One LLM consolidation call at session end

**Alternative:** Save memory after each turn / rule-based triggers

**Why consolidation:** Real-time writes create noise and partial state. Rule-based triggers miss implicit commitments. One consolidation call at session end has full context and produces clean, coherent memory. The cost is one extra API call per session, well within free tier.

---

### Decision 4: Pre-process tool results in code before sending to LLM

**Chosen:** Code computes sums, filters, formats before LLM sees results

**Alternative:** Send raw transaction list, let LLM sum/filter

**Why code:** The assignment explicitly flags LLM arithmetic as a red flag. More importantly, it's wrong — LLMs can miscalculate on lists. Python `sum()` is exact. Pre-processing also reduces token count significantly (one number vs. 18 transaction objects).

---

### Decision 5: Tool descriptions that enforce freshness

**Chosen:** Tool descriptions include "Always call this, never use stale data"

**Alternative:** Rely solely on system prompt instructions

**Why both:** Defense in depth. The system prompt is the primary enforcement mechanism. Tool descriptions serve as a secondary reminder visible at the moment the LLM is deciding whether to call the tool. Redundant but cheap.

---

## 14. What Makes This Stand Out

Most submissions will have:
- A flat memory dict with whatever happened to be saved
- A system prompt that says "remember the user's preferences"
- Tool calling that works but isn't principled
- LLM doing arithmetic somewhere

This submission has:

| Differentiator | What Others Do | What We Do |
|---|---|---|
| Memory design | Flat JSON dump | Two-tier episodic + semantic, decisions not numbers |
| Data freshness | Hope the LLM doesn't quote stale data | Structurally prevent stale numbers from existing in memory |
| Tool dispatch | Pass raw results to LLM | Pre-process in code, LLM sees summaries |
| Consolidation | Save mid-conversation or not at all | Dedicated end-of-session extraction pass |
| Session 2 proactivity | Requires being told to check the plan | System prompt design makes it inevitable |
| Code discipline | Mix of LLM and code for computation | All arithmetic in Python, explicit decision map |

The **Loom video** is where this architecture pays off most. You can walk through:
1. `memory.json` — show the two-tier structure, explain why balances are absent
2. The tool dispatcher — show one tool call with pre-processing
3. The system prompt — show the freshness rules and why they're there
4. The Session 2 response — show it connecting to Session 1 without being told

---

## 15. Constraints Checklist

```
[ ] No agent frameworks (LangChain, LlamaIndex, CrewAI, etc.)
[ ] LLM calls only where judgment is needed
    [ ] Arithmetic done in code
    [ ] Date parsing done in code
    [ ] Filtering done in code
[ ] Memory persists to disk between sessions
[ ] Under 300 lines of code total
[ ] Session 1 uses CURRENT_SESSION = 1
[ ] Session 2 uses CURRENT_SESSION = 2
[ ] Full transcripts with tool calls visible
[ ] Deliverable: repo link
[ ] Deliverable: transcripts
[ ] Deliverable: 10-minute Loom
    [ ] Memory layer shown
    [ ] One tool-call decision walked through
    [ ] One prompt explained
[ ] Deliverable: one-page writeup
    [ ] What was stored / not stored after Session 1
    [ ] One LLM decision + one code decision, with reasons
    [ ] AI usage disclosure + one specific rejection example
    [ ] One-week redesign answer
```

---

*Document version: 1.0 — written before implementation begins.*
*Update this document if architecture decisions change during implementation.*

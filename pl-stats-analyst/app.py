"""
app.py
======
Streamlit UI for the Basic Stats Analyst agent.

Run:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from verbal_model import SYSTEM_PROMPT, VERBAL_MODEL_PAIRS, QUALITY_DEFINITIONS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Basic Stats Analyst · PL 2024/25",
    page_icon="⚽",
    layout="centered",
)

# ── Load data (cached) ─────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    BASE = Path(__file__).parent
    return pd.read_parquet(BASE / "output" / "player_qualities.parquet")

df = load_data()

QUALITY_SCORE_COLS = sorted(c for c in df.columns if c.endswith("_score") and c.startswith("q_"))
QUALITY_LABEL_COLS = sorted(c for c in df.columns if c.endswith("_label") and c.startswith("q_"))

# ── Tool description ───────────────────────────────────────────────────────────
TOOL_DESCRIPTION = f"""Execute Python/pandas code against the Premier League 2024/25 player statistics DataFrame (variable: df).
ALWAYS assign your final answer to a variable called 'result'.

DataFrame has {len(df)} players and {len(df.columns)} columns. Key columns:

BIOGRAPHICAL: short_name, team_name, main_position, total_minutes, matches_played, birth_date, height, foot

OUTCOMES: total_goals, total_assists, total_yellow_cards, total_red_cards

PER-90 METRICS: xg_total_p90, shots_p90, shots_on_target_p90, passes_attempted_p90, pass_accuracy_pct,
  progressive_passes_p90, key_passes_p90, smart_passes_p90, shot_assists_p90, crosses_p90,
  carries_p90, progressive_carries_p90, dribbles_attempted_p90, interceptions_p90,
  recoveries_p90, counterpressing_recoveries_p90, defensive_duels_p90, aerial_duels_won_p90,
  aerial_duel_won_pct, ball_losses_p90, fouls_committed_p90,
  pct_actions_z1, pct_actions_z2, pct_actions_z3, pct_actions_z4, pct_actions_z5

COMPOSITE QUALITY SCORES (z-score based):
  {chr(10).join('  ' + c for c in QUALITY_SCORE_COLS)}

COMPOSITE QUALITY LABELS (outstanding/excellent/good/average/below average/poor):
  {chr(10).join('  ' + c for c in QUALITY_LABEL_COLS)}

POSITIONS: {", ".join(sorted(df["main_position"].dropna().unique()))}
"""

TOOLS = [
    {
        "name": "execute_pandas",
        "description": TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python/pandas code. Must assign the final answer to a variable named 'result'."
                }
            },
            "required": ["code"]
        }
    }
]

# ── Pandas executor ────────────────────────────────────────────────────────────
def execute_pandas(code: str) -> str:
    local_vars = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(compile(code, "<string>", "exec"), local_vars)
        result = local_vars.get("result", None)
        if result is None:
            return "Code executed but no 'result' variable was set."
        if isinstance(result, pd.DataFrame):
            return result.to_string(index=False)
        if isinstance(result, pd.Series):
            return result.to_string()
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# ── Agent call ─────────────────────────────────────────────────────────────────
def ask(question: str, api_key: str) -> str:
    client  = anthropic.Anthropic(api_key=api_key)
    messages = list(VERBAL_MODEL_PAIRS) + [
        {"role": "user", "content": question}
    ]

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            )

        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": execute_pandas(b.input["code"])}
                for b in tool_blocks
            ]
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return "No response generated."

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Basic Stats Analyst")
    st.caption("Premier League 2024/25 · Wyscout Event Data")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-...",
    )

    st.divider()
    st.markdown(f"**{len(df)} players** · **{len(QUALITY_DEFINITIONS)} qualities**")
    st.markdown("**Qualities:**")
    for q in QUALITY_DEFINITIONS:
        st.markdown(f"- {q}")

    st.divider()
    st.markdown("**Example questions**")
    examples = [
        "Who scored the most goals?",
        "Top 5 midfielders for Creativity",
        "What is Breaking the Lines?",
        "Who has the highest xG per 90?",
        "Best Ball Winning Specialists among centre-backs",
        "How does De Bruyne compare to Ødegaard?",
        "Which team has the most Wing Wizards?",
        "What is Possession Anchor?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex

# ── Main chat area ─────────────────────────────────────────────────────────────
st.header("Ask me anything about PL 2024/25")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle sidebar button question
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not api_key:
        with st.chat_message("assistant"):
            st.warning("Please enter your Anthropic API key in the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(question, api_key)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# Handle typed question
if question := st.chat_input("Ask about players, teams, or qualities…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not api_key:
        with st.chat_message("assistant"):
            st.warning("Please enter your Anthropic API key in the sidebar.")
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ No API key provided."})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(question, api_key)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Basic Stats Analyst · Premier League 2024/25

An LLM-powered football analysis tool built on Wyscout event data from the full 2024/25 Premier League season (380 matches, 331 players). Ask questions in plain English — the agent retrieves data, computes stats, and answers like a senior scout.

---

## What it does

- Answers definitional questions about football qualities and what they measure
- Retrieves player rankings, comparisons, and profiles from the data
- Contextualises numbers within team style, system effects, and positional role
- Covers all outfield positions and goalkeepers across the full PL season

The agent uses Claude (Anthropic) as its backbone and a pandas tool to query the player statistics database. It writes and executes code against the data on the fly — no pre-computed queries.

---

## How to use it

Open the app and enter your **Anthropic API key** in the sidebar. Then ask anything:

- *"Who are the top 5 centre-backs for aerial dominance?"*
- *"What is Breaking the Lines?"*
- *"How does Salah compare to Saka as a winger?"*
- *"Find me midfielders who score well in both Possession Anchor and Breaking the Lines"*
- *"Which striker has the best hold-up game?"*
- *"What do you look for in a Defensive Midfielder?"*
- *"Top wingers under 21 for the Winger profile"*

You can also click the example questions in the sidebar to get started.

---

## The 14 Composite Qualities

Every player is scored across 14 qualities built from per-90 metrics, z-score normalised within their position group. Each quality is rated: **outstanding / excellent / good / average / below average / poor**.

| Quality | What it measures | Positions |
|---|---|---|
| **Breaking the Lines** | Vertical progression from deep — passes and carries that bypass the midfield block | CB, FB, MF |
| **Vertical Box Threat** | Carries and runs that penetrate the penalty area | Winger, Striker, MF |
| **The Zone Skipper** | Long-range distribution — switching the ball across zones | CB, FB, MF, GK |
| **Creative Playmaker** | Chance creation — key passes, smart passes, shot assists | MF |
| **High-Intensity Presser** | Counterpressing and defensive work rate | MF, Winger, Striker |
| **The Box Invader** | Delivery into the box — crosses and passes to the penalty area | Winger, FB, MF |
| **Possession Anchor** | Passing accuracy and reliability under pressure | MF, CB |
| **Ball Winning Specialist** | Defensive duels, interceptions, recoveries | MF, CB |
| **Transitional Engine** | Win possession and immediately drive forward | MF, Winger |
| **Wing Wizard** | 1v1 dribbling and progressive carries in wide areas | Winger |
| **Aerial Powerhouse** | Aerial duel dominance | All positions |
| **Final Third Catalyst** | Activity and output in the attacking third | MF, Winger, Striker |
| **Link Play** | Hold-up game — connecting midfield to attack outside the box | Striker only |
| **Between the Lines** | Operating in the space just outside the box and penetrating into it | Striker, Winger, MF |

---

## Pitch zones

The data uses five pitch zones based on x-coordinate (0 = own goal, 100 = opponent's goal):

| Zone | Area | X range |
|---|---|---|
| Z1 | Own goal area | 0–20 |
| Z2 | Low block / defensive half | 20–40 |
| Z3 | Creation zone / central midfield | 40–60 |
| Z4 | Final third (outside the box) | 60–80 |
| Z5 | Opponent's penalty area | 80–100 |

The agent always translates these into plain football language — it never says "Z3" in a response.

---

## Scouting profiles

The agent is trained on positional profiles for squad building across 8 roles:

- Full Back
- Centre Back — Positional
- Centre Back — Agile
- Defensive Midfielder — Positional
- Defensive Midfielder — Dynamic
- Wide / Attacking Midfielder
- Winger (and Winger U21)
- Striker

Ask *"What do you look for in a [position]?"* to get a full scouting brief grounded in the data.

---

## A note on contextualisation

Stats reflect both individual quality and team/system effects. A centre-back at a possession team will face fewer aerial duels; a striker at a low-block team will get fewer transitions. The agent is trained to flag these effects — a low score in an area with low volume is a question to investigate, not a verdict.

---

## Data

- **Competition:** Premier League 2024/25
- **Source:** Wyscout event data (380 matches)
- **Coverage:** 331 players with ≥ 450 minutes played
- **Season:** Full season (all 38 gameweeks)

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) and enter your Anthropic API key in the sidebar.

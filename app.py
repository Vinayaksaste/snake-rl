

import os

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from streamlit_keypress import key_press_events

from agent import QLearningAgent
from game import SnakeGame
from leaderboard import add_score, load_leaderboard
from train import train_agent
from utils import draw_board_html, load_stats, plot_rewards

MODEL_PATH = "snake_agent.pkl"
STATS_PATH = "training_stats.pkl"
BOARD_SIZE = 12

st.set_page_config(
    page_title="Snake RL Demo",
    page_icon="🐍",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    h1, h2, h3, p, label, span, div {
        color: #e2e8f0 !important;
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: linear-gradient(135deg, #1e293b, #334155);
        color: white;
        padding: 0.7rem 1rem;
        font-weight: 600;
    }

    .stMetric {
        background: rgba(15, 23, 42, 0.72);
        padding: 12px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# SESSION STATE
# -------------------------
if "game" not in st.session_state:
    st.session_state.game = SnakeGame(board_size=BOARD_SIZE)

if "agent" not in st.session_state:
    if os.path.exists(MODEL_PATH):
        st.session_state.agent = QLearningAgent.load(MODEL_PATH)
    else:
        st.session_state.agent = QLearningAgent()

if "playing" not in st.session_state:
    st.session_state.playing = False

if "paused" not in st.session_state:
    st.session_state.paused = False

if "mode" not in st.session_state:
    st.session_state.mode = "AI"

if "human_direction" not in st.session_state:
    st.session_state.human_direction = 1

if "rewards" not in st.session_state:
    st.session_state.rewards = []

if "scores" not in st.session_state:
    st.session_state.scores = []

if "avg_rewards" not in st.session_state:
    st.session_state.avg_rewards = []

if "best_score" not in st.session_state:
    st.session_state.best_score = 0

if "total_episodes" not in st.session_state:
    st.session_state.total_episodes = 0

if "saved_epsilon" not in st.session_state:
    st.session_state.saved_epsilon = 0.0

if "last_reward" not in st.session_state:
    st.session_state.last_reward = 0.0

if "speed" not in st.session_state:
    st.session_state.speed = 5

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = load_leaderboard()

# -------------------------
# LOAD TRAINING STATS
# -------------------------
if os.path.exists(STATS_PATH):
    try:
        stats = load_stats(STATS_PATH)
        st.session_state.rewards = stats.get("rewards", [])
        st.session_state.scores = stats.get("scores", [])
        st.session_state.avg_rewards = stats.get("avg_rewards", [])
        st.session_state.best_score = stats.get("best_score", 0)
        st.session_state.total_episodes = stats.get("episodes", 0)
        st.session_state.saved_epsilon = stats.get("epsilon", 0.0)
    except Exception:
        pass

# -------------------------
# HELPERS
# -------------------------
def reset_game():
    st.session_state.game = SnakeGame(board_size=BOARD_SIZE)
    st.session_state.playing = False
    st.session_state.paused = False
    st.session_state.last_reward = 0.0


def update_direction_from_key(key):
    if not key:
        return

    key = str(key).lower()

    key_map = {
        "arrowup": 0,
        "up": 0,
        "w": 0,
        "arrowright": 1,
        "right": 1,
        "d": 1,
        "arrowdown": 2,
        "down": 2,
        "s": 2,
        "arrowleft": 3,
        "left": 3,
        "a": 3,
    }

    if key in key_map:
        st.session_state.human_direction = key_map[key]


def info_card(title, content):
    st.markdown(
        f"""
        <div style="
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.18);
        ">
            <h3 style="margin-top:0; color:#f8fafc;">{title}</h3>
            <p style="color:#cbd5e1; line-height:1.8; margin-bottom:0;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


pressed_key = key_press_events()
update_direction_from_key(pressed_key)

# -------------------------
# HEADER
# -------------------------
st.title("🐍 Snake Reinforcement Learning")
st.caption("Interactive Reinforcement Learning Snake AI")

st.markdown(
    """
    <div style="
        padding: 18px 20px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        margin-bottom: 18px;
    ">
        <h3 style="margin:0 0 10px 0; color:#f8fafc;">About this project</h3>
        <p style="margin:0; line-height:1.7; color:#cbd5e1;">
            This is a Reinforcement Learning Snake game built with Python and Streamlit.
            The AI learns by trial and error using Q-learning, and the app lets you train,
            play, pause, reset, save models, load models, and view a leaderboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How it works"):
    st.write(
        """
        - **AI Mode**: the snake plays automatically using the trained Q-learning agent.
        - **Human Mode**: you control the snake with keyboard keys or WASD.
        - **Training**: the agent learns from rewards such as eating food and avoiding collisions.
        - **Leaderboard**: best scores are saved so you can track progress.
        """
    )
    st.write("**Tech stack:** Python, Streamlit, NumPy, Matplotlib, custom RL logic.")

# -------------------------
# CONTROLS
# -------------------------
c1, c2, c3 = st.columns(3)

with c1:
    episodes = st.number_input(
        "Training Episodes",
        min_value=10,
        max_value=5000,
        value=300,
        step=50,
    )

with c2:
    st.session_state.speed = st.slider(
        "Game Speed",
        min_value=1,
        max_value=10,
        value=st.session_state.speed,
    )

with c3:
    st.session_state.mode = st.radio(
        "Mode",
        ["AI", "Human"],
        horizontal=True,
    )

model_name = st.text_input(
    "Model Name",
    value="snake_agent_custom.pkl"
)

# -------------------------
# BUTTONS
# -------------------------
b1, b2, b3 = st.columns(3)

with b1:
    if st.button("▶ Start Game"):
        if st.session_state.game.game_over:
            st.session_state.game = SnakeGame(board_size=BOARD_SIZE)
        st.session_state.playing = True
        st.session_state.paused = False
        st.rerun()

with b2:
    if st.session_state.playing and not st.session_state.paused:
        if st.button("⏸ Pause Game"):
            st.session_state.paused = True
            st.rerun()
    elif st.session_state.playing and st.session_state.paused:
        if st.button("▶ Resume Game"):
            st.session_state.paused = False
            st.rerun()
    else:
        st.button("⏸ Pause Game", disabled=True)

with b3:
    if st.button("Reset Game"):
        reset_game()
        st.rerun()

# -------------------------
# TRAIN MODEL
# -------------------------
if st.button("Train Model"):
    with st.spinner("Training AI..."):
        agent, rewards, scores = train_agent(
            episodes=int(episodes),
            board_size=BOARD_SIZE,
            model_path=MODEL_PATH,
            stats_path=STATS_PATH,
        )

        st.session_state.agent = agent
        st.session_state.rewards = rewards
        st.session_state.scores = scores

        stats = load_stats(STATS_PATH)
        st.session_state.avg_rewards = stats.get("avg_rewards", [])
        st.session_state.best_score = stats.get("best_score", 0)
        st.session_state.total_episodes = stats.get("episodes", 0)
        st.session_state.saved_epsilon = stats.get("epsilon", 0.0)

        st.success("Training Complete!")

# -------------------------
# SAVE / LOAD MODEL
# -------------------------
save_col, load_col = st.columns(2)

with save_col:
    if st.button("💾 Save Model"):
        try:
            st.session_state.agent.save_custom(model_name)
            st.success(f"Model saved as {model_name}")
        except Exception as e:
            st.error(str(e))

with load_col:
    if st.button("📂 Load Model"):
        try:
            st.session_state.agent = QLearningAgent.load(model_name)
            st.success(f"Loaded model {model_name}")
        except Exception as e:
            st.error(str(e))

# -------------------------
# GAME LOOP
# -------------------------
interval_ms = max(120, 800 - (st.session_state.speed * 60))

if (
    st.session_state.playing
    and not st.session_state.paused
    and not st.session_state.game.game_over
):
    st_autorefresh(interval=interval_ms, key=f"snake_refresh_{interval_ms}")

    if st.session_state.mode == "AI":
        state = st.session_state.game.get_state()
        action = st.session_state.agent.choose_action(state, training=False)
        _, reward, done, _ = st.session_state.game.step(action)
        st.session_state.last_reward = reward

        if done:
            st.session_state.playing = False
            st.session_state.paused = False
            st.session_state.best_score = max(st.session_state.best_score, st.session_state.game.score)
            st.session_state.leaderboard = add_score(st.session_state.game.score, st.session_state.mode)

    else:
        st.session_state.game.set_direction(st.session_state.human_direction)
        _, reward, done, _ = st.session_state.game.step(0)
        st.session_state.last_reward = reward

        if done:
            st.session_state.playing = False
            st.session_state.paused = False
            st.session_state.best_score = max(st.session_state.best_score, st.session_state.game.score)
            st.session_state.leaderboard = add_score(st.session_state.game.score, st.session_state.mode)

# -------------------------
# METRICS
# -------------------------
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Current Score", st.session_state.game.score)

with m2:
    st.metric("Best Score", st.session_state.best_score)

with m3:
    st.metric("Episodes", st.session_state.total_episodes)

with m4:
    st.metric("Epsilon", f"{st.session_state.saved_epsilon:.3f}")

with m5:
    avg = 0
    if len(st.session_state.rewards) > 0:
        avg = round(
            sum(st.session_state.rewards[-20:]) / min(len(st.session_state.rewards), 20),
            2,
        )
    st.metric("Avg Reward", avg)

# -------------------------
# GAME BOARD
# -------------------------
st.subheader("Game Board")

components.html(
    draw_board_html(st.session_state.game),
    height=680,
)

# -------------------------
# HUMAN INFO
# -------------------------
if st.session_state.mode == "Human":
    st.info("Use Arrow Keys or WASD keys to control the snake.")

# -------------------------
# GAME OVER
# -------------------------
if st.session_state.game.game_over:
    st.error(f"Game Over — Final Score: {st.session_state.game.score}")

# -------------------------
# ANALYTICS
# -------------------------
st.subheader("Training Analytics")

if st.session_state.playing:
    st.info("Pause the game to view training charts.")
else:
    if len(st.session_state.rewards) > 0:
        fig = plot_rewards(st.session_state.rewards, st.session_state.avg_rewards)
        st.pyplot(fig, clear_figure=True, width="stretch")
        plt.close(fig)
    else:
        st.info("Train the model to see analytics.")

# -------------------------
# LEADERBOARD
# -------------------------
st.subheader("🏆 Leaderboard")

leaderboard_data = st.session_state.leaderboard

if len(leaderboard_data) > 0:
    for idx, entry in enumerate(leaderboard_data, start=1):
        st.markdown(
            f"""
            <div style="
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 14px 18px;
                margin-bottom: 12px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.12);
            ">
                <strong>#{idx}</strong><br>
                Score: <strong>{entry['score']}</strong><br>
                Mode: <strong>{entry['mode']}</strong><br>
                Time: {entry['time']}
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("No leaderboard scores yet.")

# -------------------------
# PROJECT EXPLANATION
# -------------------------
st.subheader("Project Explanation")

info_card(
    "What is Reinforcement Learning?",
    """
    Reinforcement Learning is a machine learning technique where an agent learns by interacting with an environment using rewards and penalties.
    """
)

info_card(
    " How the Snake Learns",
    """
    The snake learns by trying actions repeatedly. Good actions receive rewards while bad actions receive penalties. Over time the AI discovers better movement strategies.
    """
)
info_card(
    " How the Snake Decides Which Direction to Move",
    """
    The snake does not guess randomly all the time. It looks at the current state of the game and checks:
    - whether there is danger straight ahead
    - whether there is danger on the left
    - whether there is danger on the right
    - where the food is
    - which direction it is currently facing

    After that, the AI compares the possible actions:
    - move straight
    - turn left
    - turn right

    For each action, it uses the learned Q-values and chooses the action with the highest expected long-term reward.

    In simple words:
    the snake asks, "Which move will help me survive longer and reach food faster?"
    """
)

info_card(
    " Mathematical Idea Behind the Decision",
    """
    The Q-learning update rule is:

    Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]

    Where:
    - Q(s,a) = how good it is to take action a in state s
    - α = learning rate
    - r = reward after taking the action
    - γ = discount factor for future reward
    - max Q(s',a') = best future value after moving to the next state

    Example:

    Suppose:
    - current Q(s,a) = 4.0
    - reward r = 10
    - learning rate α = 0.1
    - discount factor γ = 0.9
    - best future Q value = 8

    Then:

    Q(s,a) = 4 + 0.1 [10 + 0.9×8 − 4]
           = 4 + 0.1 [10 + 7.2 − 4]
           = 4 + 0.1 [13.2]
           = 5.32

    So the AI increases the value of that action because it led to a good result.
    """
)

info_card(
    " Reinforcement Learning in Easy Language",
    """
    Reinforcement Learning means learning by doing.

    The snake tries an action, sees what happened, gets a reward or penalty, and then learns from that result.

    Easy way to think about it:
    - good move → reward
    - bad move → penalty
    - repeat many times → learn better strategy

    It is similar to teaching a child:
    - if the child does something right, you encourage it
    - if the child makes a mistake, you correct it
    - over time, the child improves

    In the same way, the snake improves through experience.
    """
)

info_card(
    " When the Snake Fails",
    """
    The snake fails when it makes a move that causes death.

    Main failure cases:
    - hitting a wall
    - hitting its own body
    - moving into a trapped position with no escape

    Why this happens:
    - the agent has not learned enough yet
    - exploration sometimes makes random bad moves
    - the state representation is simplified, so the AI does not know everything
    - the Q-values for that situation may still be incorrect

    In the beginning, failure is normal because the agent is still learning.
    Over time, repeated failures teach it which actions are dangerous.
    """
)

info_card(
    " How Food Is Generated",
    """
    Food is generated randomly in an empty cell on the board.

    The game checks all cells and removes the ones already occupied by the snake.
    Then it randomly chooses one of the remaining free cells.

    This means:
    - food never appears inside the snake
    - food can appear at different positions each time
    - every free cell has equal chance of getting food

    Example:
    If there are 100 empty cells, each one has a 1/100 chance of being selected.

    This makes the game dynamic and prevents the snake from predicting food location.
    """
)

info_card(
    " Exploration vs Exploitation",
    """
    During training, the AI balances two behaviors:

    Exploration:
    - try random moves
    - discover new strategies

    Exploitation:
    - use what it already learned
    - choose the move with the best Q-value

    Epsilon controls this balance:
    - high epsilon = more exploration
    - low epsilon = more exploitation

    At the start, the snake explores more.
    Later, it relies more on learned knowledge.
    """
)

st.markdown(
    """
    ## A Small Full Example of Decision Making

    Imagine the snake is in a state where:

    - danger straight = 0
    - danger left = 1
    - danger right = 0
    - food is on the right
    - current direction is up

    The AI checks the Q-values for:
    - straight
    - left
    - right

    Suppose the learned values are:

    - Q(s, straight) = 2.1
    - Q(s, left) = -1.4
    - Q(s, right) = 3.8

    Then the snake chooses **right** because 3.8 is the highest value.

    That is how the AI "decides" the correct path.
    """
)

info_card(
    "How the Snake Moves and Decides",
    """
    The snake does not see the whole future. It only observes the current state: danger ahead, left, right, current direction, and food direction. Based on this state, the agent chooses the action with the highest expected long-term reward.
    """
)

info_card(
    "Q-Learning Formula",
    """
    Q(s,a) ← Q(s,a) + α[r + γ maxQ(s',a') − Q(s,a)]

    This updates the value of a state-action pair using the reward now and the best future reward expected later.
    """
)

st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max Q(s',a') - Q(s,a)]")

info_card(
    "What is Save Model?",
    """
    Save Model stores the learned Q-values and training progress into a file so the AI can be reused later without losing what it learned.
    """
)

info_card(
    "What is Load Model?",
    """
    Load Model restores a previously trained AI model from a file. This lets you continue from an old trained brain instead of starting from zero.
    """
)

info_card(
    "What is Game Speed?",
    """
    Game speed controls how quickly the snake moves during gameplay. Higher speed means faster movement and more difficulty.
    """
)

info_card(
    "What are Training Episodes?",
    """
    One training episode is one complete game session. Training for more episodes usually makes the AI smarter because it gets more chances to learn from rewards and mistakes.
    """
)

info_card(
    " What is Current Score?",
    """
    Current Score is the number of food items eaten in the active game session.
    """
)

info_card(
    "What is Best Score?",
    """
    Best Score is the highest score achieved so far across your saved play sessions and training runs.
    """
)

info_card(
    " What are Episodes, Epsilon, and Avg Reward?",
    """
    Episodes show how many training games were played. Epsilon controls exploration: high epsilon means more random actions, low epsilon means more learned actions. Avg Reward is the average reward from recent episodes and helps show whether training is improving.
    """
)

info_card(
    "What is Training Analytics?",
    """
    Training Analytics is the graph that shows how rewards change across episodes. If the curve rises over time, the agent is usually learning better behavior.
    """
)

info_card(
    " What is the Leaderboard?",
    """
    The leaderboard stores the highest scores achieved in AI mode and Human mode, along with the time and mode used for each run.
    """
)

info_card(
    "Technologies Used",
    """
    Python, Streamlit, NumPy, Matplotlib, Q-learning, and custom game logic.
    """
)

info_card(
    "Project Purpose",
    """
    This project demonstrates reinforcement learning concepts, AI training visualization, interactive gameplay, and browser-based deployment. It is useful for learning, portfolios, demos, and academic showcases.
    """
)
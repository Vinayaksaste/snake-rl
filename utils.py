
import base64
import mimetypes
import pickle
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt



@lru_cache(maxsize=64)
def image_to_data_uri(image_path: str):
    if not image_path:
        return None

    normalized_path = image_path.replace("\\", "/")
    path = Path(normalized_path)

    if not path.exists():
        return None

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/png"

    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def draw_board_html(game):
    board_size = game.board_size
    snake = set(game.snake)
    head = game.snake[0]
    food = game.food

    food_image_uri = image_to_data_uri(getattr(game, "food_face", None))

    html = f"""
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        align-items:center;
        padding:18px 0;
        box-sizing:border-box;
        background: radial-gradient(circle at top, #1e293b 0%, #0f172a 55%, #020617 100%);
        border-radius:24px;
        border:1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    ">
        <div style="
            display:grid;
            grid-template-columns: repeat({board_size}, 1fr);
            gap:5px;
            padding:18px;
            background: rgba(15, 23, 42, 0.55);
            border-radius:22px;
            backdrop-filter: blur(8px);
            width:min(100%, 540px);
            box-sizing:border-box;
        ">
    """

    for y in range(board_size):
        for x in range(board_size):
            pos = (x, y)

            cell_style = """
                aspect-ratio: 1 / 1;
                width: 100%;
                border-radius: 10px;
                background: #1e293b;
                border: 1px solid rgba(255,255,255,0.05);
            """

            if pos == food:
                if food_image_uri:
                    cell_style = f"""
                        aspect-ratio: 1 / 1;
                        width: 100%;
                        border-radius: 10px;
                        background-image: url('{food_image_uri}');
                        background-size: cover;
                        background-position: center;
                        background-repeat: no-repeat;
                        border: 1px solid rgba(255,255,255,0.05);
                        box-shadow: 0 0 12px rgba(251, 113, 133, 0.55);
                    """
                else:
                    cell_style = """
                        aspect-ratio: 1 / 1;
                        width: 100%;
                        border-radius: 10px;
                        background: #fb7185;
                        border: 1px solid rgba(255,255,255,0.05);
                        box-shadow: 0 0 12px rgba(251, 113, 133, 0.55);
                    """
            elif pos == head:
                cell_style = """
                    aspect-ratio: 1 / 1;
                    width: 100%;
                    border-radius: 10px;
                    background: #22c55e;
                    border: 1px solid rgba(255,255,255,0.05);
                    box-shadow: 0 0 12px rgba(34, 197, 94, 0.7);
                """
            elif pos in snake:
                cell_style = """
                    aspect-ratio: 1 / 1;
                    width: 100%;
                    border-radius: 10px;
                    background: #16a34a;
                    border: 1px solid rgba(255,255,255,0.05);
                    box-shadow: 0 0 8px rgba(22, 163, 74, 0.45);
                """

            html += f"""
            <div style="{cell_style}"></div>
            """

    html += """
        </div>
    </div>
    """

    return html


def plot_rewards(rewards, avg_rewards=None):
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(rewards, linewidth=1.8, label="Reward")

    if avg_rewards is not None and len(avg_rewards) > 0:
        ax.plot(avg_rewards, linewidth=2.5, label="Average Reward")

    ax.set_title("Training Performance")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend()
    fig.tight_layout()
    return fig


def load_stats(path):
    with open(path, "rb") as f:
        return pickle.load(f)
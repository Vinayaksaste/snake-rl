import json
import os
from datetime import datetime

LEADERBOARD_PATH = "leaderboard.json"


def load_leaderboard():

    if not os.path.exists(LEADERBOARD_PATH):
        return []

    try:

        with open(LEADERBOARD_PATH, "r") as f:
            return json.load(f)

    except Exception:
        return []


def save_leaderboard(data):

    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(data, f, indent=4)


def add_score(score, mode):

    data = load_leaderboard()

    data.append(
        {
            "score": score,
            "mode": mode,
            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
    )

    data = sorted(
        data,
        key=lambda x: x["score"],
        reverse=True
    )

    data = data[:10]

    save_leaderboard(data)

    return data
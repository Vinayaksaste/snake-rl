
import argparse
import pickle

from agent import QLearningAgent
from game import SnakeGame


def train_agent(
    episodes=300,
    board_size=12,
    model_path="snake_agent.pkl",
    stats_path="training_stats.pkl",
    max_steps_per_episode=500,
):
    env = SnakeGame(board_size=board_size)
    agent = QLearningAgent()

    rewards = []
    scores = []
    avg_rewards = []

    best_score = 0

    for episode in range(1, episodes + 1):
        state = env.reset()

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = agent.choose_action(state, training=True)

            next_state, reward, done, score = env.step(action)

            agent.learn(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state
            total_reward += reward
            steps += 1

        agent.end_episode()

        rewards.append(total_reward)
        scores.append(score)

        best_score = max(best_score, score)

        avg_reward = sum(rewards[-20:]) / min(len(rewards), 20)
        avg_rewards.append(avg_reward)

        if episode % 25 == 0 or episode == 1:
            print(
                f"Episode {episode}/{episodes} | "
                f"Score: {score} | "
                f"Best: {best_score} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save(model_path)

    stats = {
        "rewards": rewards,
        "scores": scores,
        "avg_rewards": avg_rewards,
        "best_score": best_score,
        "episodes": episodes,
        "epsilon": agent.epsilon,
    }

    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    return agent, rewards, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--board_size", type=int, default=12)
    parser.add_argument("--model_path", type=str, default="snake_agent.pkl")
    parser.add_argument("--stats_path", type=str, default="training_stats.pkl")
    args = parser.parse_args()

    train_agent(
        episodes=args.episodes,
        board_size=args.board_size,
        model_path=args.model_path,
        stats_path=args.stats_path,
    )
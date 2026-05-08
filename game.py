
import random
from pathlib import Path
from typing import Tuple


class SnakeGame:
    # 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
    DIRECTIONS = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    ]

    def __init__(self, board_size: int = 12):
        self.board_size = board_size
        self.friend_face_images = self._load_friend_faces()
        self.reset()

    def _load_friend_faces(self):
        folder = Path("assets/friends")
        if not folder.exists():
            return []

        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        return [
            str(p)
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

    def _random_food_face(self):
        if not self.friend_face_images:
            return None
        return random.choice(self.friend_face_images)

    def reset(self):
        mid = self.board_size // 2
        self.snake = [
            (mid, mid),
            (mid - 1, mid),
            (mid - 2, mid),
        ]
        self.direction = 1  # start moving RIGHT
        self.score = 0
        self.game_over = False
        self.food, self.food_face = self._place_food()
        return self.get_state()

    def _place_food(self):
        free_cells = [
            (x, y)
            for x in range(self.board_size)
            for y in range(self.board_size)
            if (x, y) not in self.snake
        ]
        food_pos = random.choice(free_cells)
        food_face = self._random_food_face()
        return food_pos, food_face

    def _collision(self, point: Tuple[int, int]) -> bool:
        x, y = point
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return True
        return point in self.snake[1:]

    def _next_position(self, direction_idx: int) -> Tuple[int, int]:
        dx, dy = self.DIRECTIONS[direction_idx]
        head_x, head_y = self.snake[0]
        return head_x + dx, head_y + dy

    def set_direction(self, new_direction: int):
        """
        Used for Human mode.
        Prevents instant 180-degree reversal.
        """
        if new_direction not in (0, 1, 2, 3):
            return

        if len(self.snake) > 1 and (self.direction + 2) % 4 == new_direction:
            return

        self.direction = new_direction

    def get_state(self):
        head = self.snake[0]
        dir_idx = self.direction

        straight = self._next_position(dir_idx)
        right = self._next_position((dir_idx + 1) % 4)
        left = self._next_position((dir_idx - 1) % 4)

        danger_straight = self._collision(straight)
        danger_right = self._collision(right)
        danger_left = self._collision(left)

        dir_up = dir_idx == 0
        dir_right = dir_idx == 1
        dir_down = dir_idx == 2
        dir_left = dir_idx == 3

        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]

        return (
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_up),
            int(dir_right),
            int(dir_down),
            int(dir_left),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
        )

    def step(self, action: int):
        """
        action:
            0 -> straight
            1 -> turn right
            2 -> turn left
        returns: next_state, reward, done, score
        """
        if self.game_over:
            return self.get_state(), 0.0, True, self.score

        old_food = self.food
        old_head = self.snake[0]
        old_distance = abs(old_head[0] - old_food[0]) + abs(old_head[1] - old_food[1])

        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        dx, dy = self.DIRECTIONS[self.direction]
        new_head = (old_head[0] + dx, old_head[1] + dy)

        if self._collision(new_head):
            self.game_over = True
            return self.get_state(), -10.0, True, self.score

        self.snake.insert(0, new_head)

        reward = -0.05

        if new_head == old_food:
            self.score += 1
            reward = 10.0
            self.food, self.food_face = self._place_food()
        else:
            new_distance = abs(new_head[0] - old_food[0]) + abs(new_head[1] - old_food[1])
            if new_distance < old_distance:
                reward += 0.15
            else:
                reward -= 0.15
            self.snake.pop()

        return self.get_state(), float(reward), False, self.score
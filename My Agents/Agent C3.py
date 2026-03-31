from utils import State, Action
import math
import time
import numpy as np

class StudentAgent:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.transpo_table = {}

    def choose_action(self, state: State) -> Action:
        start_time = time.time()
        time_limit = 2.5

        valid_actions = state.get_all_valid_actions()
        if len(valid_actions) == 0:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]

        best_value = -math.inf
        best_action = valid_actions[0]

        for depth in range(1, self.max_depth + 1):
            if (time.time() - start_time) > time_limit:
                break

            value, action_found = self._alpha_beta_value(
                state=state.clone(),
                depth=depth,
                alpha=-math.inf,
                beta=math.inf,
                start_time=start_time,
                time_limit=time_limit
            )
            if action_found is None:
                break

            best_value, best_action = value, action_found

            if best_value > 900000:
                break

        return best_action

    def _alpha_beta_value(self, state: State, depth: int, alpha: float, beta: float,
                          start_time: float, time_limit: float) -> tuple[float, Action|None]:
        if (time.time() - start_time) > time_limit:
            return 0, None

        if state.is_terminal() or depth == 0:
            return self.evaluate_state(state), None

        actions = state.get_all_valid_actions()
        if not actions:
            val = self.evaluate_state(state)
            return val, None

        key = self._build_transpo_key(state, depth, alpha, beta)

        if key in self.transpo_table:
            return self.transpo_table[key]

        is_maximizing = (state.fill_num == 1)
        best_value = -math.inf if is_maximizing else math.inf
        best_action = None

        for act in actions:
            if (time.time() - start_time) > time_limit:
                self.transpo_table[key] = (best_value, None)
                return best_value, None

            next_state = state.change_state(act, check_valid_action=False)
            val, chosen_act = self._alpha_beta_value(next_state, depth - 1, alpha, beta,
                                                     start_time, time_limit)

            if chosen_act is None and (time.time() - start_time) > time_limit:
                self.transpo_table[key] = (best_value, None)
                return best_value, None

            if is_maximizing:
                if val > best_value:
                    best_value = val
                    best_action = act
                alpha = max(alpha, best_value)
                if best_value >= beta:
                    break
            else:
                if val < best_value:
                    best_value = val
                    best_action = act
                beta = min(beta, best_value)
                if best_value <= alpha:
                    break

        self.transpo_table[key] = (best_value, best_action)
        return best_value, best_action

    def _build_transpo_key(self, state: State, depth: int, alpha: float, beta: float):
        fill_num = state.fill_num
        pla = state.prev_local_action
        board_4d = state.board
        board_9x9 = self.to_nine_by_nine(board_4d)

        if pla is None:
            all_syms = self.all_symmetries_9x9(board_9x9)
            candidates = []
            for arr_9x9 in all_syms:
                arr_bytes = arr_9x9.tobytes()
                candidate = (arr_bytes, None, fill_num, depth, alpha, beta)
                candidates.append(candidate)
            best_key = min(candidates)
            return best_key
        else:
            arr_bytes = board_9x9.tobytes()
            key = (arr_bytes, pla, fill_num, depth, alpha, beta)
            return key

    @staticmethod
    def to_nine_by_nine(board_4d: np.ndarray) -> np.ndarray:
        arr_9x9 = np.zeros((9, 9), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        row = i*3 + k
                        col = j*3 + l
                        arr_9x9[row, col] = board_4d[i, j, k, l]
        return arr_9x9

    @staticmethod
    def rotate_90(arr_9x9: np.ndarray) -> np.ndarray:
        n = 9
        rotated = np.zeros_like(arr_9x9)
        for r in range(n):
            for c in range(n):
                rotated[c, n-1-r] = arr_9x9[r, c]
        return rotated

    @staticmethod
    def flip_horizontal(arr_9x9: np.ndarray) -> np.ndarray:
        n = 9
        flipped = np.zeros_like(arr_9x9)
        for r in range(n):
            for c in range(n):
                flipped[r, n-1-c] = arr_9x9[r, c]
        return flipped

    def all_symmetries_9x9(self, arr_9x9: np.ndarray) -> list[np.ndarray]:
        syms = []
        current = arr_9x9

        for _ in range(4):
            syms.append(current)
            current = self.rotate_90(current)

        flipped = self.flip_horizontal(arr_9x9)
        current = flipped
        for _ in range(4):
            syms.append(current)
            current = self.rotate_90(current)

        return syms

    def evaluate_state(self, state: State) -> float:
        if state.is_terminal():
            util = state.terminal_utility()
            if util == 1.0:
                return 999999
            elif util == 0.0:
                return -999999
            else:
                return 0

        score = 0.0
        lbs = state.local_board_status
        center_weight_meta = 1.3

        for i in range(3):
            for j in range(3):
                val = lbs[i][j]
                base = 80.0
                if (i, j) == (1, 1):
                    base *= center_weight_meta
                if val == 1:
                    score += base
                elif val == 2:
                    score -= base

        for meta_r in range(3):
            for meta_c in range(3):
                if lbs[meta_r][meta_c] == 0:
                    local_score = self.analyze_local_board(state.board[meta_r][meta_c])
                    score += local_score

        score += self.analyze_meta_board(lbs)

        if state.prev_local_action is not None:
            (r, c) = state.prev_local_action
            if lbs[r][c] != 0:
                if state.fill_num == 1:
                    score += 15
                else:
                    score -= 15

        return score

    def analyze_local_board(self, board_3x3: np.ndarray) -> float:
        score = 0.0
        center_cell = board_3x3[1][1]
        if center_cell == 1:
            score += 2
        elif center_cell == 2:
            score -= 2

        lines = []
        for r in range(3):
            lines.append(list(board_3x3[r]))
        for c in range(3):
            col = [board_3x3[r][c] for r in range(3)]
            lines.append(col)
        diag1 = [board_3x3[i][i] for i in range(3)]
        diag2 = [board_3x3[i][2 - i] for i in range(3)]
        lines.append(diag1)
        lines.append(diag2)

        for line in lines:
            if 2 not in line:
                if line.count(1) == 2 and line.count(0) == 1:
                    score += 10
                else:
                    score += 1.5 * line.count(1)
            elif 1 not in line:
                if line.count(2) == 2 and line.count(0) == 1:
                    score -= 10
                else:
                    score -= 1.5 * line.count(2)

        return score

    def analyze_meta_board(self, meta_3x3: np.ndarray) -> float:
        score = 0.0
        def cv(x):
            return x if x != 3 else -1

        lines = []
        for r in range(3):
            row = [cv(meta_3x3[r][c]) for c in range(3)]
            lines.append(row)
        for c in range(3):
            col = [cv(meta_3x3[r][c]) for r in range(3)]
            lines.append(col)
        diag1 = [cv(meta_3x3[i][i]) for i in range(3)]
        diag2 = [cv(meta_3x3[i][2 - i]) for i in range(3)]
        lines.append(diag1)
        lines.append(diag2)

        for line in lines:
            if (2 not in line) and (-1 not in line):
                if line.count(1) == 2 and line.count(0) == 1:
                    score += 50
                else:
                    score += 2.0 * line.count(1)
            elif (1 not in line) and (-1 not in line):
                if line.count(2) == 2 and line.count(0) == 1:
                    score -= 50
                else:
                    score -= 2.0 * line.count(2)

        return score

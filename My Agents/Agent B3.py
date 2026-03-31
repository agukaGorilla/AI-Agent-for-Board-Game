from utils import State, Action
import math
import time

class StudentAgent:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth

    def choose_action(self, state: State) -> Action:
        start_time = time.time()
        time_limit = 2.5
        valid_actions = state.get_all_valid_actions()
        if len(valid_actions) == 0:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]

        best_action = valid_actions[0]
        best_value = -math.inf

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
            return self.evaluate_state(state), None

        if state.fill_num == 1:
            best_value = -math.inf
            best_action = None
            for action in actions:
                if (time.time() - start_time) > time_limit:
                    return best_value, None
                next_state = state.change_state(action, check_valid_action=False)
                val, _ = self._alpha_beta_value(
                    next_state, depth-1, alpha, beta, start_time, time_limit
                )
                if _ is None and (time.time() - start_time) > time_limit:
                    return best_value, None
                if val > best_value:
                    best_value = val
                    best_action = action
                alpha = max(alpha, best_value)
                if best_value >= beta:
                    break
            return best_value, best_action
        else:
            best_value = math.inf
            best_action = None
            for action in actions:
                if (time.time() - start_time) > time_limit:
                    return best_value, None
                next_state = state.change_state(action, check_valid_action=False)
                val, _ = self._alpha_beta_value(
                    next_state, depth-1, alpha, beta, start_time, time_limit
                )
                if _ is None and (time.time() - start_time) > time_limit:
                    return best_value, None
                if val < best_value:
                    best_value = val
                    best_action = action
                beta = min(beta, best_value)
                if best_value <= alpha:
                    break
            return best_value, best_action

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

        meta_lines_score = self.analyze_meta_board(lbs)
        score += meta_lines_score

        if state.prev_local_action is not None:
            (r, c) = state.prev_local_action
            if lbs[r][c] != 0:
                if state.fill_num == 1:
                    score += 15
                else:
                    score -= 15

        return score

    def analyze_local_board(self, board_3x3) -> float:
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
                elif line.count(1) == 1 and line.count(0) == 2:
                    score += 3
                elif line.count(0) == 3:
                    score += 1
            if 1 not in line:
                if line.count(2) == 2 and line.count(0) == 1:
                    score -= 10
                elif line.count(2) == 1 and line.count(0) == 2:
                    score -= 3
                elif line.count(0) == 3:
                    score -= 1
        return score

    def analyze_meta_board(self, meta_board_3x3) -> float:
        def cv(x):
            return x if x != 3 else -1

        score = 0.0
        lines = []
        for r in range(3):
            lines.append([cv(meta_board_3x3[r][c]) for c in range(3)])
        for c in range(3):
            lines.append([cv(meta_board_3x3[r][c]) for r in range(3)])
        lines.append([cv(meta_board_3x3[i][i]) for i in range(3)])
        lines.append([cv(meta_board_3x3[i][2 - i]) for i in range(3)])

        for line in lines:
            if 2 not in line:
                if line.count(1) == 2 and line.count(0) == 1:
                    score += 50
                elif line.count(1) == 1 and line.count(0) == 2:
                    score += 15
                elif line.count(0) == 3:
                    score += 3
            if 1 not in line:
                if line.count(2) == 2 and line.count(0) == 1:
                    score -= 50
                elif line.count(2) == 1 and line.count(0) == 2:
                    score -= 15
                elif line.count(0) == 3:
                    score -= 3
        return score

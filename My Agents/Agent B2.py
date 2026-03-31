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
            if time.time() - start_time > time_limit:
                break

            value, action_found = self._max_value(
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

        return best_action

    def _max_value(self, state: State, depth: int, alpha: float, beta: float,
                   start_time: float, time_limit: float) -> tuple[float, Action|None]:
        if time.time() - start_time > time_limit:
            return 0, None

        if state.is_terminal() or depth == 0:
            return self.evaluate_state(state), None

        actions = state.get_all_valid_actions()
        if not actions:
            return self.evaluate_state(state), None

        best_value = -math.inf
        best_action = None

        for action in actions:
            if time.time() - start_time > time_limit:
                return best_value, None

            next_state = state.change_state(action, check_valid_action=False)
            val, _ = self._min_value(next_state, depth - 1, alpha, beta,
                                     start_time, time_limit)

            if _ is None and (time.time() - start_time > time_limit):
                return best_value, None

            if val > best_value:
                best_value = val
                best_action = action

            alpha = max(alpha, best_value)
            if best_value >= beta:
                break

        return best_value, best_action

    def _min_value(self, state: State, depth: int, alpha: float, beta: float,
                   start_time: float, time_limit: float) -> tuple[float, Action|None]:
        if time.time() - start_time > time_limit:
            return 0, None

        if state.is_terminal() or depth == 0:
            return self.evaluate_state(state), None

        actions = state.get_all_valid_actions()
        if not actions:
            return self.evaluate_state(state), None

        best_value = math.inf
        best_action = None

        for action in actions:
            if time.time() - start_time > time_limit:
                return best_value, None

            next_state = state.change_state(action, check_valid_action=False)
            val, _ = self._max_value(next_state, depth - 1, alpha, beta,
                                     start_time, time_limit)

            if _ is None and (time.time() - start_time > time_limit):
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
        center_weight = 1.3
        lbs = state.local_board_status

        for i in range(3):
            for j in range(3):
                val = lbs[i][j]
                base = 50.0
                if (i, j) == (1, 1):
                    base *= center_weight
                if val == 1:
                    score += base
                elif val == 2:
                    score -= base

        for meta_row in range(3):
            for meta_col in range(3):
                if lbs[meta_row][meta_col] == 0:
                    local_board = state.board[meta_row][meta_col]
                    p1_threats = self.local_two_in_a_row(local_board, 1)
                    p2_threats = self.local_two_in_a_row(local_board, 2)
                    score += 5 * p1_threats
                    score -= 5 * p2_threats

        meta_p1_threats = self.meta_two_in_a_row(lbs, 1)
        meta_p2_threats = self.meta_two_in_a_row(lbs, 2)
        score += 30 * meta_p1_threats
        score -= 30 * meta_p2_threats

        return score

    def local_two_in_a_row(self, board_3x3, player: int) -> int:
        count = 0
        for r in range(3):
            row = list(board_3x3[r])
            if row.count(player) == 2 and row.count(0) == 1:
                count += 1

        for c in range(3):
            col = [board_3x3[r][c] for r in range(3)]
            if col.count(player) == 2 and col.count(0) == 1:
                count += 1

        diag1 = [board_3x3[i][i] for i in range(3)]
        diag2 = [board_3x3[i][2 - i] for i in range(3)]
        if diag1.count(player) == 2 and diag1.count(0) == 1:
            count += 1
        if diag2.count(player) == 2 and diag2.count(0) == 1:
            count += 1

        return count

    def meta_two_in_a_row(self, meta_board_3x3, player: int) -> int:
        count = 0

        def cell_value(v):
            return v if v != 3 else -1

        for r in range(3):
            row = [cell_value(meta_board_3x3[r][c]) for c in range(3)]
            if row.count(player) == 2 and row.count(0) == 1:
                count += 1

        for c in range(3):
            col = [cell_value(meta_board_3x3[r][c]) for r in range(3)]
            if col.count(player) == 2 and col.count(0) == 1:
                count += 1

        diag1 = [cell_value(meta_board_3x3[i][i]) for i in range(3)]
        diag2 = [cell_value(meta_board_3x3[i][2 - i]) for i in range(3)]
        if diag1.count(player) == 2 and diag1.count(0) == 1:
            count += 1
        if diag2.count(player) == 2 and diag2.count(0) == 1:
            count += 1

        return count

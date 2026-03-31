from utils import State, Action
import math
import time

class StudentAgent:
    def __init__(self, depth=3):
        self.depth = depth

    def choose_action(self, state: State) -> Action:
        start_time = time.time()
        time_limit = 2.5
        valid_actions = state.get_all_valid_actions()
        if len(valid_actions) == 1:
            return valid_actions[0]

        best_action = valid_actions[0]
        best_value = -math.inf

        for d in range(1, self.depth + 1):
            if time.time() - start_time > time_limit:
                break

            value, action_found = self._max_value(
                state=state.clone(),
                depth=d,
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

        v = -math.inf
        best_action = None
        actions = state.get_all_valid_actions()

        if not actions:
            return self.evaluate_state(state), None

        for action in actions:
            if time.time() - start_time > time_limit:
                return v, None

            next_state = state.change_state(action, check_valid_action=False)
            val, _ = self._min_value(next_state, depth - 1, alpha, beta, start_time, time_limit)

            if _ is None and time.time() - start_time > time_limit:
                return v, None

            if val > v:
                v = val
                best_action = action

            alpha = max(alpha, v)
            if v >= beta:
                break

        return v, best_action

    def _min_value(self, state: State, depth: int, alpha: float, beta: float,
                   start_time: float, time_limit: float) -> tuple[float, Action|None]:
        if time.time() - start_time > time_limit:
            return 0, None

        if state.is_terminal() or depth == 0:
            return self.evaluate_state(state), None

        v = math.inf
        best_action = None
        actions = state.get_all_valid_actions()

        if not actions:
            return self.evaluate_state(state), None

        for action in actions:
            if time.time() - start_time > time_limit:
                return v, None

            next_state = state.change_state(action, check_valid_action=False)
            val, _ = self._max_value(next_state, depth - 1, alpha, beta, start_time, time_limit)

            if _ is None and time.time() - start_time > time_limit:
                return v, None

            if val < v:
                v = val
                best_action = action

            beta = min(beta, v)
            if v <= alpha:
                break

        return v, best_action

    def evaluate_state(self, state: State) -> float:
        if state.is_terminal():
            util = state.terminal_utility()
            if util == 1.0:
                return 999999
            elif util == 0.0:
                return -999999
            else:
                return 0
        else:
            score = 0
            lbs = state.local_board_status
            for i in range(3):
                for j in range(3):
                    if lbs[i][j] == 1:
                        score += 50
                    elif lbs[i][j] == 2:
                        score -= 50
                    elif lbs[i][j] == 3:
                        score += 0
            return score

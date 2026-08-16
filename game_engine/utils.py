import numpy as np

class Action:
    def __init__(self, meta_row: int, meta_col: int, local_row: int, local_col: int):
        self.meta_row = meta_row
        self.meta_col = meta_col
        self.local_row = local_row
        self.local_col = local_col

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.meta_row, self.meta_col, self.local_row, self.local_col) == \
               (other.meta_row, other.meta_col, other.local_row, other.local_col)

    def __hash__(self):
        return hash((self.meta_row, self.meta_col, self.local_row, self.local_col))

    def __repr__(self):
        return f"Action(({self.meta_row},{self.meta_col}) -> ({self.local_row},{self.local_col}))"


class State:
    def __init__(self):
        self.board = np.zeros((3, 3, 3, 3), dtype=np.int8)
        self.local_board_status = np.zeros((3, 3), dtype=np.int8)
        self.prev_local_action = None 
        self.fill_num = 1              

    def clone(self):
        new_state = State()
        new_state.board = self.board.copy()
        new_state.local_board_status = self.local_board_status.copy()
        new_state.prev_local_action = self.prev_local_action
        new_state.fill_num = self.fill_num
        return new_state

    def get_all_valid_actions(self) -> list[Action]:
        if self.is_terminal():
            return []

        valid_actions = []
        target_meta = self.prev_local_action

        if target_meta is not None:
            r, c = target_meta
            if self.local_board_status[r][c] == 0:
                meta_targets = [(r, c)]
            else:
                meta_targets = [(mr, mc) for mr in range(3) for mc in range(3) if self.local_board_status[mr][mc] == 0]
        else:
            meta_targets = [(mr, mc) for mr in range(3) for mc in range(3) if self.local_board_status[mr][mc] == 0]

        for (mr, mc) in meta_targets:
            for lr in range(3):
                for lc in range(3):
                    if self.board[mr][mc][lr][lc] == 0:
                        valid_actions.append(Action(mr, mc, lr, lc))

        return valid_actions

    def change_state(self, action: Action, check_valid_action: bool = False) -> 'State':
        next_state = self.clone()
        mr, mc, lr, lc = action.meta_row, action.meta_col, action.local_row, action.local_col
        
        next_state.board[mr][mc][lr][lc] = self.fill_num
        next_state.prev_local_action = (lr, lc)

        local_board = next_state.board[mr][mc]
        if next_state._check_win(local_board, self.fill_num):
            next_state.local_board_status[mr][mc] = self.fill_num
        elif np.all(local_board != 0):
            next_state.local_board_status[mr][mc] = 3

        next_state.fill_num = 2 if self.fill_num == 1 else 1
        return next_state

    def is_terminal(self) -> bool:
        lbs = self.local_board_status
        return self._check_win(lbs, 1) or self._check_win(lbs, 2) or np.all(lbs != 0)

    def terminal_utility(self) -> float:
        lbs = self.local_board_status
        if self._check_win(lbs, 1):
            return 1.0
        elif self._check_win(lbs, 2):
            return 0.0
        return 0.5

    def _check_win(self, grid, player: int) -> bool:
        for i in range(3):
            if np.all(grid[i, :] == player) or np.all(grid[:, i] == player):
                return True
        if grid[0, 0] == player and grid[1, 1] == player and grid[2, 2] == player:
            return True
        if grid[0, 2] == player and grid[1, 1] == player and grid[2, 0] == player:
            return True
        return False

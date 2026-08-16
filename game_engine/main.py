import importlib
import os
import sys
import time
from utils import State, Action

def render_board(state: State):
    symbols = {0: '.', 1: 'X', 2: 'O'}
    print("\n" + "=" * 33)
    for mr in range(3):
        for lr in range(3):
            row_str = ""
            for mc in range(3):
                status = state.local_board_status[mr][mc]
                if status == 1:
                    row_str += "  X  X  X  "
                elif status == 2:
                    row_str += "  O  O  O  "
                elif status == 3:
                    row_str += "  -  -  -  "
                else:
                    cells = [symbols[state.board[mr][mc][lr][lc]] for lc in range(3)]
                    row_str += f" {' '.join(cells)} "
                if mc < 2:
                    row_str += "|"
            print(row_str)
        if mr < 2:
            print("-" * 33)
    print("=" * 33 + "\n")

def load_agents():
    # Dynamically find the agent directory one level up from game_engine
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agent_dir = os.path.abspath(os.path.join(current_dir, '..', 'agent'))
    sys.path.append(agent_dir)
    
    agents = {}
    if not os.path.exists(agent_dir):
        print(f"Error: Agent directory not found at {agent_dir}")
        return agents

    for filename in sorted(os.listdir(agent_dir)):
        if filename.endswith(".py") and not filename.startswith("__"):
            mod_name = filename[:-3]
            try:
                module = importlib.import_module(mod_name)
                if hasattr(module, "StudentAgent"):
                    agents[mod_name] = module.StudentAgent
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    return agents

def get_human_action(state: State) -> Action:
    valid_actions = state.get_all_valid_actions()
    while True:
        try:
            print("Format: meta_row meta_col local_row local_col (0-indexed)")
            inp = input("Enter move (e.g. 0 1 2 2): ").strip().split()
            if len(inp) != 4:
                continue
            mr, mc, lr, lc = map(int, inp)
            act = Action(mr, mc, lr, lc)
            if act in valid_actions:
                return act
            print("Invalid action for current board state. Try again.")
        except ValueError:
            print("Invalid input. Enter 4 numbers separated by spaces.")

def run_game(player1_agent, player2_agent):
    state = State()
    agents = {1: player1_agent, 2: player2_agent}

    while not state.is_terminal():
        render_board(state)
        curr_player = state.fill_num
        agent = agents[curr_player]

        print(f"Turn: Player {curr_player} ({'Human' if agent is None else agent.__class__.__name__})")
        
        if agent is None:
            action = get_human_action(state)
        else:
            start_t = time.time()
            action = agent.choose_action(state.clone())
            elapsed = time.time() - start_t
            print(f"Agent played {action} in {elapsed:.2f}s")

        if action is None:
            print(f"Player {curr_player} returned no action. Game over.")
            break

        state = state.change_state(action)

    render_board(state)
    util = state.terminal_utility()
    if util == 1.0:
        print("Winner: Player 1 (X)")
    elif util == 0.0:
        print("Winner: Player 2 (O)")
    else:
        print("Game ended in a Draw!")

if __name__ == "__main__":
    available_agents = load_agents()
    print("Available agents found:", list(available_agents.keys()))
    
    print("\nSelect Player 1 (X):")
    print("0: Human")
    agent_names = list(available_agents.keys())
    for i, name in enumerate(agent_names, 1):
        print(f"{i}: {name}")
    p1_idx = int(input("Choice: "))
    p1 = None if p1_idx == 0 else available_agents[agent_names[p1_idx - 1]]()

    print("\nSelect Player 2 (O):")
    print("0: Human")
    for i, name in enumerate(agent_names, 1):
        print(f"{i}: {name}")
    p2_idx = int(input("Choice: "))
    p2 = None if p2_idx == 0 else available_agents[agent_names[p2_idx - 1]]()

    run_game(p1, p2)

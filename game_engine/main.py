import importlib
import os
import sys
import time
from utils import State, Action

# ANSI Color Codes for the terminal
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def render_board(state: State):
    """
    Renders the 9x9 grid. Overlays a giant colored X, O, or - on captured local boards.
    """
    print("\n" + "=" * 29)
    for mr in range(3):
        for lr in range(3):
            row_parts = []
            for mc in range(3):
                status = state.local_board_status[mr][mc]
                
                # Render Giant Red X for Player 1 win
                if status == 1:
                    text = [" \\   / ", "   X   ", " /   \\ "][lr]
                    text = f"{RED}{text}{RESET}"
                
                # Render Giant Blue O for Player 2 win
                elif status == 2:
                    text = [" / - \\ ", " |   | ", " \\ - / "][lr]
                    text = f"{BLUE}{text}{RESET}"
                
                # Render Giant - for a Draw
                elif status == 3:
                    text = ["       ", " - - - ", "       "][lr]
                
                # Render normal 3x3 cells if still active
                else:
                    cells = []
                    for lc in range(3):
                        val = state.board[mr][mc][lr][lc]
                        if val == 1:
                            cells.append(f"{RED}X{RESET}")
                        elif val == 2:
                            cells.append(f"{BLUE}O{RESET}")
                        else:
                            cells.append(".")
                    
                    text = f" {cells[0]} {cells[1]} {cells[2]} "
                
                row_parts.append(text)
            
            # Join the local columns with vertical separators
            print(" | ".join(row_parts))
        
        # Add horizontal separators between meta-rows
        if mr < 2:
            print("-" * 29)
    print("=" * 29 + "\n")

def load_agents():
    # Dynamically find the "My Agents" directory one level up from game_engine
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agent_dir = os.path.abspath(os.path.join(current_dir, '..', 'My Agents'))
    
    agents = {}
    if not os.path.exists(agent_dir):
        print(f"Error: Agent directory not found at {agent_dir}")
        return agents

    sys.path.append(agent_dir)
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
            
            # Updated error message with rule clarification
            print(f"{RED}Invalid action for the current board state.{RESET}")
            print("Remember the 'send rule': your opponent's last move determines which meta-board you must play in.")
            print("Please read the rules here to find valid moves: https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe")
            print("Try again.\n")
            
        except ValueError:
            print("Invalid input. Enter 4 numbers separated by spaces.")

def run_game(player1_agent, player2_agent):
    state = State()
    agents = {1: player1_agent, 2: player2_agent}
    
    # Check if this is an AI vs AI match to apply longer delays
    is_auto_match = (player1_agent is not None) and (player2_agent is not None)

    while not state.is_terminal():
        render_board(state)
        curr_player = state.fill_num
        agent = agents[curr_player]

        player_color = RED if curr_player == 1 else BLUE
        print(f"Turn: {player_color}Player {curr_player}{RESET} ({'Human' if agent is None else agent.__class__.__name__})")
        
        if agent is None:
            action = get_human_action(state)
        else:
            start_t = time.time()
            action = agent.choose_action(state.clone())
            elapsed = time.time() - start_t
            print(f"Agent played {action} in {elapsed:.2f}s")
            
            # Cinematic delay for AI moves
            if is_auto_match:
                time.sleep(1.2)  # Longer delay so you can watch them fight
            else:
                time.sleep(0.4)  # Shorter delay against humans

        if action is None:
            print(f"Player {curr_player} returned no action. Game over.")
            break

        state = state.change_state(action)

    render_board(state)
    util = state.terminal_utility()
    if util == 1.0:
        print(f"Winner: {RED}Player 1 (X){RESET}")
    elif util == 0.0:
        print(f"Winner: {BLUE}Player 2 (O){RESET}")
    else:
        print("Game ended in a Draw!")

if __name__ == "__main__":
    available_agents = load_agents()
    
    if not available_agents:
        print("\nNo valid agents found. Exiting.")
        sys.exit(1)
        
    print("Available agents found:", list(available_agents.keys()))
    
    print(f"\nSelect {RED}Player 1 (X){RESET}:")
    print("0: Human")
    agent_names = list(available_agents.keys())
    for i, name in enumerate(agent_names, 1):
        print(f"{i}: {name}")
    p1_idx = int(input("Choice: "))
    p1 = None if p1_idx == 0 else available_agents[agent_names[p1_idx - 1]]()

    print(f"\nSelect {BLUE}Player 2 (O){RESET}:")
    print("0: Human")
    for i, name in enumerate(agent_names, 1):
        print(f"{i}: {name}")
    p2_idx = int(input("Choice: "))
    p2 = None if p2_idx == 0 else available_agents[agent_names[p2_idx - 1]]()

    run_game(p1, p2)

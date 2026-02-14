import copy
import heapq
import time

goal_state = [[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 0]]

very_easy = [[1, 2, 3],
             [4, 5, 6], 
             [0, 7, 8]]

easy = [[8, 1, 3],
        [4, 0, 2], 
        [7, 6, 5]]

normal = [[5, 2, 3],
          [7, 8, 1], 
          [4, 0, 6]]

hard = [[5, 2, 8],
        [7, 4, 3], 
        [6, 1, 0]]

very_hard = [[4, 3, 5],
             [0, 1, 2], 
             [6, 8, 7]]

extreme = [[0, 8, 7],
           [2, 5, 1], 
           [3, 6, 4]]

class Node:
    def __init__(self, board, g, h, parent=None):
        self.board = board
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def board_to_tuple(board):
    tuple_board = []
    for row in board:
        for num in row:
            tuple_board.append(num)
    return tuple(tuple_board)

def find_empty_space(board):
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] == 0:
                return (r, c)
    return None

def is_goal(board):
    return board == goal_state

def generate_moves(board):
    n = len(board)
    possible_moves = []
    r, c = find_empty_space(board)

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row_pos, col_pos in moves:
        nr, nc = r + row_pos, c + col_pos
        if 0 <= nr < n and 0 <= nc < n:
            new_board = copy.deepcopy(board)
            new_board[r][c], new_board[nr][nc] = new_board[nr][nc], new_board[r][c]
            possible_moves.append(new_board)
    
    return possible_moves

def build_goal_state(n):
    nums = list(range(1, n * n)) + [0]
    return [nums[i * n:(i + 1) * n] for i in range(n)]

def set_goal_state(n):
    global goal_state, goal_pos
    goal_state = build_goal_state(n)
    goal_pos = position_coordinates(goal_state)

def h_misplaced(board):
    count = 0
    for r in range(len(board)):
        for c in range(len(board[r])):
            val = board[r][c]
            if val != 0 and val != goal_state[r][c]:
                count += 1
    return count

def position_coordinates(goal):
    pos = {}
    n = len(goal)
    for r in range(n):
        for c in range(n):
            pos[goal[r][c]] = (r, c)
    return pos

def h_manhattan(board):
    n = len(board)
    dist = 0
    for r in range(n):
        for c in range(n):
            val = board[r][c]
            if val != 0:
                gr, gc = goal_pos[val]
                dist += abs(r - gr) + abs(c - gc)
    return dist

def uniform_cost(initial_board):
    start_node = Node(initial_board, g=0, h=0, parent=None)

    frontier = []
    heapq.heappush(frontier, start_node)

    visited = set()
    nodes_expanded = 0
    max_queue_size = 1

    while frontier:
        max_queue_size = max(max_queue_size, len(frontier))
        node = heapq.heappop(frontier)
        state = board_to_tuple(node.board)

        if state in visited:
            continue
        visited.add(state)

        if is_goal(node.board):
            return node, nodes_expanded, max_queue_size

        nodes_expanded += 1

        # debug for nxn expansion
        '''
        if nodes_expanded % 5000 == 0:
            print("expanded:", nodes_expanded, "max queue:", max_queue_size)
        '''

        for neighbor in generate_moves(node.board): 
            neighbor_state = board_to_tuple(neighbor)
            if neighbor_state in visited:
                continue
            child = Node(neighbor, g=node.g + 1, h=0, parent=node)
            heapq.heappush(frontier, child)
    
    return None, nodes_expanded, max_queue_size

def a_star(initial_board, heuristic_fn):
    start_node = Node(initial_board, g=0, h=heuristic_fn(initial_board), parent=None)

    frontier = []
    heapq.heappush(frontier, start_node)

    visited = set()
    nodes_expanded = 0
    max_queue_size = 1

    while frontier:
        max_queue_size = max(max_queue_size, len(frontier))
        node = heapq.heappop(frontier)
        state = board_to_tuple(node.board)

        if state in visited:
            continue
        visited.add(state)

        if is_goal(node.board):
            return node, nodes_expanded, max_queue_size

        nodes_expanded += 1
        
        # debug for nxn expansion
        '''
        if nodes_expanded % 5000 == 0:
            print("expanded:", nodes_expanded, "max queue:", max_queue_size)
        '''

        for neighbor in generate_moves(node.board): 
            neighbor_state = board_to_tuple(neighbor)
            if neighbor_state in visited:
                continue
            child = Node(neighbor, g=node.g + 1, h=heuristic_fn(neighbor), parent=node)
            heapq.heappush(frontier, child)
    
    return None, nodes_expanded, max_queue_size

def a_star_misplaced(start_board):
    return a_star(start_board, h_misplaced)

def a_star_manhattan(start_board):
    return a_star(start_board, h_manhattan)

def enter_custom_puzzle(n):
    print(f"Enter a {n}x{n} puzzle. For each row, enter {n} numbers separated by spaces. Use 0 to represent the empty space.")

    while True:
        board = []
        for i in range(n):
            while True:
                raw_string = input(f"Enter row {i+1}: ").strip()
                parts = raw_string.split()

                if len(parts) != n or not all(p.isdigit() for p in parts):
                    print(f"Invalid row. Enter {n} numbers within range 0 to {n * n - 1} separated by spaces. Use 0 for the empty space.")
                    continue

                row = [int(x) for x in parts]
                if any(x < 0 or x >= n*n for x in row):
                    print(f"Invalid row. Enter {n} numbers within range 0 to {n*n-1}.")
                    continue

                board.append(row)
                break
                
        flat = list(board_to_tuple(board))
        if sorted(flat) != list(range(n * n)):
            print(f"Invalid puzzle. Use each number 0 to {n * n - 1} exactly once.")
            continue

        return board

def select_algorithm():
    while True:
        print("Select algorithm:")
        print("1) Uniform Cost Search")
        print("2) A* with Misplaced Tile")
        print("3) A* with Manhattan Distance")
        choice = input("Enter 1, 2, or 3: ").strip()
        
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice, try again.")

def main():
    print("Welcome to the n-Puzzle Solver!")
    set_goal_state(3)
    while True:
        mode = input("Type '1' to use a default puzzle, '2' to create your own, or 'q' to quit: ").strip()
        if mode == "q":
            return
        if mode not in {"1", "2"}:
            print("Invalid choice, try again.")
            continue
        else:
            break

    if mode == "1":
        print("Default puzzles: ")
        print("1. Very Easy")
        print("2. Easy")
        print("3. Normal")
        print("4. Hard")
        print("5. Very Hard")
        print("6. Extreme")
        
        while True:
            pick = input("Enter your choice by digit, or 'q' to quit: ").strip()
            if pick in {"1","2","3","4","5","6"}:
                break
            if pick == "q":
                return
            print("Invalid choice. Please enter 1-6.")
        
        if pick == "1":
            board = very_easy
        elif pick == "2":
            board = easy
        elif pick == "3":
            board = normal
        elif pick == "4":
            board = hard
        elif pick == "5":
            board = very_hard
        elif pick == "6":
            board = extreme
    else:
        while True:
            raw_numbers = input("Enter puzzle size n (e.g., 3 for a 3x3 puzzle) or 'q' to quit: ").strip()
            if raw_numbers.isdigit() and int(raw_numbers) >= 2:
                n = int(raw_numbers)
                break
            elif raw_numbers == "q":
                return
            print("Invalid input. Please enter an integer >= 2 or 'q' to quit.")
        
        set_goal_state(n)
        board = enter_custom_puzzle(n)
    
    algorithm = select_algorithm()

    start = time.time()
    if algorithm == "1":
        node, expanded, max_q = uniform_cost(board)
    elif algorithm == "2":
        node, expanded, max_q = a_star_misplaced(board)
    elif algorithm == "3":
        node, expanded, max_q = a_star_manhattan(board)
    else:
        print("Invalid algorithm choice. Please enter 1, 2, or 3.")
        return
    elapsed = time.time() - start
    
    if node is None:
        print("No solution found.")
        return
    
    path = reconstruct_path(node)

    for i, n in enumerate(path):
        print(f"\nStep {i}")
        print_board(n.board)
        print(f"g={n.g}, h={n.h}, f={n.f}")
    
    print("\nTotal nodes expanded:", expanded)
    print("Depth:", node.g)
    print("Max queue size:", max_q)
    print(f"Elapsed time: {elapsed:.4f}s")

def run_tests():
    set_goal_state(3)
    assert board_to_tuple(goal_state) == (1,2,3,4,5,6,7,8,0)
    print("board_to_tuple successful")

    assert find_empty_space(goal_state) == (2, 2)
    print("find_empty_space successful")

    assert is_goal(goal_state) is True
    assert is_goal(very_easy) is False
    print("is_goal successful")

    corner_gap = [[0, 1, 2], 
                  [3, 4, 5], 
                  [6, 7, 8]]

    center_gap = [[1, 2, 3], 
                  [4, 0, 5], 
                  [6, 7, 8]]

    edge_gap = [[1, 0, 2], 
                [3, 4, 5], 
                [6, 7, 8]]

    assert len(generate_moves(corner_gap)) == 2
    assert len(generate_moves(center_gap)) == 4
    assert len(generate_moves(edge_gap)) == 3
    print("generate_moves successful")

    assert h_misplaced(goal_state) == 0
    assert h_misplaced(very_easy) == 2
    assert h_misplaced(easy) == 5
    print("h_misplaced successful")

    assert h_manhattan(goal_state) == 0
    assert h_manhattan(very_easy) == 2
    assert h_manhattan(easy) == 10
    print("h_manhattan successful")

    test = [[1,2,3], [4,5,6], [7,0,8]]
    goal_node, expanded, max_queue = uniform_cost(test)
    assert goal_node is not None
    assert is_goal(goal_node.board)
    assert goal_node.g == 1
    print("uniform_cost successful")

    node, expanded, max_queue = a_star_misplaced(test)
    assert node is not None
    assert is_goal(node.board)
    assert node.g == 1
    print("a_star_misplaced successful")

    node, expanded, max_queue = a_star_manhattan(test)
    assert node is not None
    assert is_goal(node.board)
    assert node.g == 1
    print("a_star_manhattan successful")

    node_u, exp_u, max_u = uniform_cost(easy)
    node_m, exp_m, max_m = a_star_manhattan(easy)
    node_mi, exp_mi, max_mi = a_star_misplaced(easy)

    print("UCS expanded:", exp_u, "max queue:", max_u)
    print("A* Manhattan expanded:", exp_m, "max queue:", max_m)
    print("A* Misplaced expanded:", exp_mi, "max queue:", max_mi)

def run_data(label, board, algo_fn):
    set_goal_state(len(board))
    start = time.time()
    node, expanded, max_q = algo_fn(board)
    elapsed = time.time() - start
    depth = node.g if node else -1
    print(f"{label}: depth = {depth}, expanded = {expanded}, maxQ = {max_q}, time = {elapsed}s")

def run_all_data():
    print("--------------------------------")
    run_data("UCS very easy", very_easy, uniform_cost)
    run_data("Misplaced very easy", very_easy, a_star_misplaced)
    run_data("Manhattan very easy", very_easy, a_star_manhattan)
    print("--------------------------------")

    run_data("UCS easy", easy, uniform_cost)
    run_data("Misplaced easy", easy, a_star_misplaced)
    run_data("Manhattan easy", easy, a_star_manhattan)
    print("--------------------------------")

    run_data("UCS normal", normal, uniform_cost)
    run_data("Misplaced normal", normal, a_star_misplaced)
    run_data("Manhattan normal", normal, a_star_manhattan)
    print("--------------------------------")

    run_data("UCS hard", hard, uniform_cost)
    run_data("Misplaced hard", hard, a_star_misplaced)
    run_data("Manhattan hard", hard, a_star_manhattan)
    print("--------------------------------")

    run_data("UCS very hard", very_hard, uniform_cost)
    run_data("Misplaced very hard", very_hard, a_star_misplaced)
    run_data("Manhattan very hard", very_hard, a_star_manhattan)
    print("--------------------------------")

    run_data("UCS extreme", extreme, uniform_cost)
    run_data("Misplaced extreme", extreme, a_star_misplaced)
    run_data("Manhattan extreme", extreme, a_star_manhattan)

def reconstruct_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    return path

def print_board(board):
    for row in board:
        print(tuple(row))

if __name__ == "__main__":
    # run_tests()
    # run_all_data()
    main()
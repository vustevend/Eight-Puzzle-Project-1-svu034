import copy  # used to deep-copy board states
import heapq  # used for priority-queue behavior
import time  # used to measure runtime

# default solved 3x3 goal state
goal_state = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]

# very easy preset puzzle
very_easy = [[1, 2, 3],
             [4, 5, 6],
             [0, 7, 8]]

# easy preset puzzle
easy = [[8, 1, 3],
        [4, 0, 2],
        [7, 6, 5]]

# normal preset puzzle
normal = [[5, 2, 3],
          [7, 8, 1],
          [4, 0, 6]]

# hard preset puzzle
hard = [[5, 2, 8],
        [7, 4, 3],
        [6, 1, 0]]

# very hard preset puzzle
very_hard = [[4, 3, 5],
             [0, 1, 2],
             [6, 8, 7]]

# extreme preset puzzle
extreme = [[0, 8, 7],
           [2, 5, 1],
           [3, 6, 4]]

class Node:
    # constructor stores board state and costs for one search node
    def __init__(self, board, g, h, parent=None):
        self.board = board  # board at this node
        self.g = g  # path cost from start to this node
        self.h = h  # heuristic estimate to goal
        self.f = g + h  # total score used by priority queue
        self.parent = parent  # points to previous node in the path

    # allows heapq to compare Node objects using f-cost
    def __lt__(self, other):
        return self.f < other.f

# converts a 2D board list into one hashable tuple
def board_to_tuple(board):
    tuple_board = []  # temporary list holds flattened values
    for row in board:
        for num in row:
            tuple_board.append(num)
    return tuple(tuple_board)

# finds the coordinates of the blank tile (0)
def find_empty_space(board):
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] == 0:
                return (r, c)
    return None

# checks if the current board matches the global goal
def is_goal(board):
    return board == goal_state

# generates all legal next boards by moving the blank one step
def generate_moves(board):
    n = len(board)  # puzzle dimension
    possible_moves = []  # collects all neighbors
    r, c = find_empty_space(board)  # gets blank position

    # direction offsets: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row_pos, col_pos in moves:
        nr, nc = r + row_pos, c + col_pos
        if 0 <= nr < n and 0 <= nc < n:
            new_board = copy.deepcopy(board)  # avoids mutating original board
            new_board[r][c], new_board[nr][nc] = new_board[nr][nc], new_board[r][c]
            possible_moves.append(new_board)
    
    return possible_moves

# builds the solved goal board for any n x n puzzle
def build_goal_state(n):
    nums = list(range(1, n * n)) + [0]
    return [nums[i * n:(i + 1) * n] for i in range(n)]

# updates global goal board and tile-position lookup
def set_goal_state(n):
    global goal_state, goal_pos
    goal_state = build_goal_state(n)
    goal_pos = position_coordinates(goal_state)

# heuristic counts tiles that are not in their goal position
def h_misplaced(board):
    count = 0
    for r in range(len(board)):
        for c in range(len(board[r])):
            val = board[r][c]
            if val != 0 and val != goal_state[r][c]:
                count += 1
    return count

# builds a map from tile value to its goal coordinates
def position_coordinates(goal):
    pos = {}  # dict will store value -> (row, col)
    n = len(goal)  # puzzle dimension
    for r in range(n):
        for c in range(n):
            pos[goal[r][c]] = (r, c)
    return pos

# heuristic sums Manhattan distances for all non-blank tiles
def h_manhattan(board):
    n = len(board)  # puzzle dimension
    dist = 0  # accumulates total Manhattan distance
    for r in range(n):
        for c in range(n):
            val = board[r][c]
            if val != 0:
                gr, gc = goal_pos[val]
                dist += abs(r - gr) + abs(c - gc)
    return dist

# runs Uniform Cost Search over puzzle states
def uniform_cost(initial_board):
    # creates the start node with zero path and heuristic cost
    start_node = Node(initial_board, g=0, h=0, parent=None)

    frontier = []  # min-heap frontier
    heapq.heappush(frontier, start_node)

    visited = set()  # tracks already-expanded states
    nodes_expanded = 0  # counts expanded nodes
    max_queue_size = 1  # tracks largest frontier length

    while frontier:
        # updates the max frontier size seen so far
        max_queue_size = max(max_queue_size, len(frontier))
        # pops the node with smallest f from the heap
        node = heapq.heappop(frontier)
        # converts board to tuple so it can go into visited
        state = board_to_tuple(node.board)

        if state in visited:
            continue
        visited.add(state)

        # exits as soon as a goal node is popped
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
            # creates neighbor with cost one step deeper
            child = Node(neighbor, g=node.g + 1, h=0, parent=node)
            heapq.heappush(frontier, child)

    # returns failure plus stats if search space is exhausted
    return None, nodes_expanded, max_queue_size

# runs generic A* with a selectable heuristic function
def a_star(initial_board, heuristic_fn):
    # start node gets its heuristic from the chosen function
    start_node = Node(initial_board, g=0, h=heuristic_fn(initial_board), parent=None)

    frontier = []  # min-heap frontier
    heapq.heappush(frontier, start_node)

    visited = set()  # tracks expanded states
    nodes_expanded = 0  # counts expanded nodes
    max_queue_size = 1  # tracks peak frontier size

    while frontier:
        # updates frontier-size metric
        max_queue_size = max(max_queue_size, len(frontier))
        # pops the currently best node by f = g + h
        node = heapq.heappop(frontier)
        # hashes board form for visited lookup
        state = board_to_tuple(node.board)

        if state in visited:
            continue
        visited.add(state)

        # returns immediately when goal is found
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
            # computes heuristic for each neighbor before pushing
            child = Node(neighbor, g=node.g + 1, h=heuristic_fn(neighbor), parent=node)
            heapq.heappush(frontier, child)

    # returns failure plus stats if no solution is found
    return None, nodes_expanded, max_queue_size

# wrapper runs A* with misplaced-tile heuristic
def a_star_misplaced(start_board):
    return a_star(start_board, h_misplaced)

# wrapper runs A* with Manhattan-distance heuristic
def a_star_manhattan(start_board):
    return a_star(start_board, h_manhattan)

# prompts for a custom n x n puzzle and validates input hard
def enter_custom_puzzle(n):
    # tells the user exactly how to enter each row
    print(f"Enter a {n}x{n} puzzle. For each row, enter {n} numbers separated by spaces. Use 0 to represent the empty space.")

    while True:
        board = []  # starts a fresh board entry attempt
        for i in range(n):
            while True:
                # reads one row as raw text input
                raw_string = input(f"Enter row {i+1}: ").strip()
                # splits row text into tokens by spaces
                parts = raw_string.split()

                # enforces correct row length and integer-only tokens
                if len(parts) != n or not all(p.isdigit() for p in parts):
                    print(f"Invalid row. Enter {n} numbers within range 0 to {n * n - 1} separated by spaces. Use 0 for the empty space.")
                    continue

                # converts row text tokens into ints
                row = [int(x) for x in parts]
                # enforces each value is inside legal puzzle range
                if any(x < 0 or x >= n * n for x in row):
                    print(f"Invalid row. Enter {n} numbers within range 0 to {n * n - 1}.")
                    continue

                # keeps this row and moves to next row
                board.append(row)
                break

        # flattens board so we can validate number uniqueness
        flat = list(board_to_tuple(board))
        # checks the board contains each number 0..n^2-1 exactly once
        if sorted(flat) != list(range(n * n)):
            print(f"Invalid puzzle. Use each number 0 to {n * n - 1} exactly once.")
            continue

        # returns once full board is valid
        return board

# shows algorithm options and returns valid selection
def select_algorithm():
    while True:
        print("Select algorithm:")
        print("1) Uniform Cost Search")
        print("2) A* with Misplaced Tile")
        print("3) A* with Manhattan Distance")
        # reads user selection as a trimmed string
        choice = input("Enter 1, 2, or 3: ").strip()

        # accepts only menu values 1/2/3
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice, try again.")

# interactive program entry point
def main():
    # greets the user when the program starts
    print("Welcome to the n-Puzzle Solver!")
    # initializes default goal data for 3x3 mode
    set_goal_state(3)
    while True:
        # asks whether to use preset puzzle, custom puzzle, or quit
        mode = input("Type '1' to use a default puzzle, '2' to create your own, or 'q' to quit: ").strip()
        # exits program on q
        if mode == "q":
            return
        # rejects anything except menu options 1/2
        if mode not in {"1", "2"}:
            print("Invalid choice, try again.")
            continue
        else:
            # breaks once mode is valid
            break

    # branch handles preset puzzle selection
    if mode == "1":
        print("Default puzzles: ")
        print("1. Very Easy")
        print("2. Easy")
        print("3. Normal")
        print("4. Hard")
        print("5. Very Hard")
        print("6. Extreme")

        while True:
            # asks which preset puzzle to run
            pick = input("Enter your choice by digit, or 'q' to quit: ").strip()
            # accepts the 6 preset choices
            if pick in {"1", "2", "3", "4", "5", "6"}:
                break
            # allows quitting from preset menu
            if pick == "q":
                return
            print("Invalid choice. Please enter 1-6.")

        # branches map menu choice to the actual preset board
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
    # branch handles custom puzzle setup
    else:
        while True:
            # asks for puzzle dimension n
            raw_numbers = input("Enter puzzle size n (e.g., 3 for a 3x3 puzzle) or 'q' to quit: ").strip()
            # accepts integer sizes >= 2
            if raw_numbers.isdigit() and int(raw_numbers) >= 2:
                n = int(raw_numbers)
                break
            # allows quitting custom setup
            elif raw_numbers == "q":
                return
            print("Invalid input. Please enter an integer >= 2 or 'q' to quit.")

        # updates goal for selected puzzle size
        set_goal_state(n)
        # reads and validates custom board input
        board = enter_custom_puzzle(n)

    # gets algorithm selection from menu
    algorithm = select_algorithm()

    # starts runtime timer
    start = time.time()
    # branches run the selected algorithm
    if algorithm == "1":
        node, expanded, max_q = uniform_cost(board)
    elif algorithm == "2":
        node, expanded, max_q = a_star_misplaced(board)
    elif algorithm == "3":
        node, expanded, max_q = a_star_manhattan(board)
    else:
        print("Invalid algorithm choice. Please enter 1, 2, or 3.")
        return
    # computes elapsed runtime
    elapsed = time.time() - start

    # handles unsolved cases
    if node is None:
        print("No solution found.")
        return

    # reconstructs the full path from start to goal
    path = reconstruct_path(node)

    # prints each step in the solution path
    for i, n in enumerate(path):
        print(f"\nStep {i}")
        print_board(n.board)
        print(f"g={n.g}, h={n.h}, f={n.f}")

    # lines print summary statistics
    print("\nTotal nodes expanded:", expanded)
    print("Depth:", node.g)
    print("Max queue size:", max_q)
    print(f"Elapsed time: {elapsed:.4f}s")

# function runs quick assertions to verify core behavior
def run_tests():
    # ensures all tests use a 3x3 goal
    set_goal_state(3)
    # checks that flattening the goal returns expected tuple order
    assert board_to_tuple(goal_state) == (1, 2, 3, 4, 5, 6, 7, 8, 0)
    print("board_to_tuple successful")

    # verifies blank-tile location for goal board
    assert find_empty_space(goal_state) == (2, 2)
    print("find_empty_space successful")

    # verifies goal detection for goal and non-goal boards
    assert is_goal(goal_state) is True
    assert is_goal(very_easy) is False
    print("is_goal successful")

    # board has blank in a corner (2 legal moves)
    corner_gap = [[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]]

    # board has blank in center (4 legal moves)
    center_gap = [[1, 2, 3],
                  [4, 0, 5],
                  [6, 7, 8]]

    # board has blank on an edge (3 legal moves)
    edge_gap = [[1, 0, 2],
                [3, 4, 5], 
                [6, 7, 8]]

    # assert expected branching factors for different blank positions
    assert len(generate_moves(corner_gap)) == 2
    assert len(generate_moves(center_gap)) == 4
    assert len(generate_moves(edge_gap)) == 3
    print("generate_moves successful")

    # validate misplaced heuristic values
    assert h_misplaced(goal_state) == 0
    assert h_misplaced(very_easy) == 2
    assert h_misplaced(easy) == 5
    print("h_misplaced successful")

    # validate Manhattan heuristic values
    assert h_manhattan(goal_state) == 0
    assert h_manhattan(very_easy) == 2
    assert h_manhattan(easy) == 10
    print("h_manhattan successful")

    # test board is one move from goal
    test = [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
    # solves test board with UCS
    goal_node, expanded, max_queue = uniform_cost(test)
    assert goal_node is not None
    assert is_goal(goal_node.board)
    assert goal_node.g == 1
    print("uniform_cost successful")

    # solves test board with A* misplaced
    node, expanded, max_queue = a_star_misplaced(test)
    assert node is not None
    assert is_goal(node.board)
    assert node.g == 1
    print("a_star_misplaced successful")

    # solves test board with A* Manhattan
    node, expanded, max_queue = a_star_manhattan(test)
    assert node is not None
    assert is_goal(node.board)
    assert node.g == 1
    print("a_star_manhattan successful")

    # run all three methods on easy preset to compare work
    node_u, exp_u, max_u = uniform_cost(easy)
    node_m, exp_m, max_m = a_star_manhattan(easy)
    node_mi, exp_mi, max_mi = a_star_misplaced(easy)

    print("UCS expanded:", exp_u, "max queue:", max_u)
    print("A* Manhattan expanded:", exp_m, "max queue:", max_m)
    print("A* Misplaced expanded:", exp_mi, "max queue:", max_mi)

# runs one algorithm on one board and prints performance data
def run_data(label, board, algo_fn):
    # sets goal size to match the input board size
    set_goal_state(len(board))
    # starts timing
    start = time.time()
    # runs the selected algorithm function
    node, expanded, max_q = algo_fn(board)
    # computes elapsed time
    elapsed = time.time() - start
    # gets solution depth or -1 if unsolved
    depth = node.g if node else -1
    # prints one benchmark row
    print(f"{label}: depth = {depth}, expanded = {expanded}, maxQ = {max_q}, time = {elapsed}s")

# runs benchmark rows for all preset boards and algorithms
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

# rebuilds path by walking parent links from goal to start
def reconstruct_path(goal_node):
    path = []  # list will hold nodes in reverse first
    node = goal_node  # starts from goal node
    while node is not None:
        path.append(node)
        node = node.parent  # steps backward one parent at a time
    path.reverse()  # flips order to start -> goal
    return path

# prints board rows as tuples
def print_board(board):
    for row in board:
        print(tuple(row))

# block runs only when this file is executed directly
if __name__ == "__main__":
    # run_tests()  # execute built-in tests
    # run_all_data()  # run algorithm comparison dataset
    main()

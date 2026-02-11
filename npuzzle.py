import copy
import heapq
from typing import List, Tuple, Optional

goal_state = [[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 0]]

depth_2 = [[1, 2, 3], 
           [4, 5, 6], 
           [0, 7, 8]]

challenge_state = [[8, 1, 3], 
                   [4, 0, 2], 
                   [7, 6, 5]]

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

goal_pos = position_coordinates(goal_state)

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

## TESTS
if __name__ == "__main__":
    assert board_to_tuple(goal_state) == (1,2,3,4,5,6,7,8,0)
    print("board_to_tuple successful")

    assert find_empty_space(goal_state) == (2, 2)
    print("find_empty_space successful")

    assert is_goal(goal_state) is True
    assert is_goal(depth_2) is False
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
    assert h_misplaced(depth_2) == 2
    assert h_misplaced(challenge_state) == 5
    print("h_misplaced successful")

    assert h_manhattan(goal_state) == 0
    assert h_manhattan(depth_2) == 2
    assert h_manhattan(challenge_state) == 10
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

    node_u, exp_u, max_u = uniform_cost(challenge_state)
    node_m, exp_m, max_m = a_star_manhattan(challenge_state)
    node_mi, exp_mi, max_mi = a_star_misplaced(challenge_state)

    print("UCS expanded:", exp_u, "max queue:", max_u)
    print("A* Manhattan expanded:", exp_m, "max queue:", max_m)
    print("A* Misplaced expanded:", exp_mi, "max queue:", max_mi)
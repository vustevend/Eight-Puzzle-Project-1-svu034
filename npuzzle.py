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
        for c in range(len(board)):
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
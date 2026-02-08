import heapq
from typing import List, Tuple, Optional

goal_state = [[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 0]]

depth_1 = [[1, 2, 3], 
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

if __name__ == "__main__":
    b = goal_state
    assert board_to_tuple(b) == (1,2,3,4,5,6,7,8,0)
    print("board_to_tuple successful")


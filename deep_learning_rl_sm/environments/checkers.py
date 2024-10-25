from gymnasium import spaces
import numpy as np
from our_gym2 import OurEnv as gym

class CheckersEnv(gym.Env):
    def __init__(self):
        super(CheckersEnv, self).__init__()
        # Define an 8x8 board space
        self.observation_space = spaces.Box(low=-2, high=2, shape=(8, 8), dtype=np.int8)
        
        # Define the action space as a tuple (start_row, start_col, end_row, end_col)
        self.action_space = spaces.Tuple((
            spaces.Discrete(8), spaces.Discrete(8),  # start position
            spaces.Discrete(8), spaces.Discrete(8)   # end position
        ))
        
        # Initialize the board
        self.reset()
        self.action_mask = self.generate_action_mask()

    def reset(self):
        # Setup initial positions for the checkers pieces
        self.board = np.zeros((8, 8), dtype=np.int8)
        # Player 1's pieces
        self.board[0:3:2, 1::2] = 1
        self.board[1:3:2, 0::2] = 1
        # Player 2's pieces
        self.board[5:8:2, 0::2] = -1
        self.board[6:8:2, 1::2] = -1
        self.current_player = 1  # 1 for Player 1, -1 for Player 2

        self.done = False
        return self.board

    def step(self, action):
        start_row, start_col, end_row, end_col = action
        # Validate the move and make sure it's the current player's turn
        if not self.is_valid_move(start_row, start_col, end_row, end_col):
            return self.board, -10, self.done, {}  # Invalid move penalty
        
        # Move the piece by moving the number to the end position, leaving the start positions as 0's
        self.board[end_row, end_col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0
        
        # Check if the move is a capture (because of jumping we are moving 2 diagonally)
        if abs(start_row - end_row) == 2 and abs(start_col - end_col) == 2:
            # Remove the captured piece
            captured_row = (start_row + end_row) // 2
            captured_col = (start_col + end_col) // 2
            self.board[captured_row, captured_col] = 0
            
            reward = 1  # Reward for capturing a piece
        else:
            reward = 0  # Normal move, no reward
        
        # Promote to king if the piece reaches the opposite side of the board
        if end_row == 0 and self.board[end_row, end_col] == -1:
            self.board[end_row, end_col] = -2  # Player 2 gets a king
        elif end_row == 7 and self.board[end_row, end_col] == 1:
            self.board[end_row, end_col] = 2  # Player 1 gets a king

        # Check if the game is over
        self.done = self.is_game_over()
        
        # Switch turns
        self.current_player *= -1
        self.action_mask = self._generate_action_mask()
        
        return self.board, reward, self.done, {}

    def is_valid_move(self, start_row, start_col, end_row, end_col):
        # Basic bounds checking
        print("Bounds")
        if not (0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8):
            return False
        
        # Check if the move is starting from the current player's piece
        print(self.board[start_row, start_col])
        print(self.board)
        piece = self.board[start_row, start_col]
        if piece == 0 or np.sign(piece) != self.current_player:
            return False

        # Check if the destination is empty
        print("empty")
        if self.board[end_row, end_col] != 0:
            return False
        
        # Normal piece can only move diagonally by one row (or two if jumping)
        print("daigonal")
        row_diff = end_row - start_row
        col_diff = abs(end_col - start_col)

        # Check valid move distance for normal pieces
        if piece == 1 or piece == -1:  # Normal pieces
            if self.current_player == 1 and row_diff != 1 and row_diff != 2:
                return False  # Player 1 moves up the board
            if self.current_player == -1 and row_diff != -1 and row_diff != -2:
                return False  # Player 2 moves down the board
        else:  # Kings
            if abs(row_diff) != 1 and abs(row_diff) != 2:
                return False  # Kings can move both ways
        
        # Check for valid capture (jumping over an opponent's piece)
        if abs(row_diff) == 2 and col_diff == 2:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            mid_piece = self.board[mid_row, mid_col]
            if np.sign(mid_piece) == self.current_player or mid_piece == 0:
                return False  # Can't jump over own piece or empty space
        
        return True
    
    def _generate_action_mask(self):
        # TODO: brute force implementation, can be made more efficient
        action_mask = np.zeros((8, 8, 8, 8), dtype=np.int8)
        for start_row in range(8):
            for start_col in range(8):
                for end_row in range(8):
                    for end_col in range(8):
                        if self.is_valid_move(start_row, start_col, end_row, end_col):
                            action_mask[start_row, start_col, end_row, end_col] = 1
        return action_mask

    def is_game_over(self):
        # Check if either player has no pieces left
        if not (1 in self.board or 2 in self.board):  # Player 1 lost
            return True
        if not (-1 in self.board or -2 in self.board):  # Player 2 lost
            return True
        
        return False

    
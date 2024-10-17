from deep_learning_rl_sm.environments.checkers import CheckersEnv
import unittest
import numpy as np

def test_dummy():
    return True

class TestCheckersEnv(unittest.TestCase):

    def setUp(self):
        """Initialize a new CheckersEnv before each test."""
        self.env = CheckersEnv()

    def test_reset(self):
        """Test that the environment resets correctly and initializes the board."""
        board = self.env.reset()
        expected_board = np.zeros((8, 8), dtype=np.int8)
        expected_board[0:3:2, 1::2] = 1
        expected_board[1:3:2, 0::2] = 1
        expected_board[5:8:2, 0::2] = -1
        expected_board[6:8:2, 1::2] = -1
        np.testing.assert_array_equal(board, expected_board)
    
    def test_valid_move(self):
        """Test that a normal valid move works correctly for Player 1."""
        self.env.reset()
        action = (2, 1, 3, 0)  # A normal move for Player 1
        self.assertTrue(self.env.is_valid_move(*action))

    def test_invalid_move(self):
        """Test that an invalid move (moving to an occupied space) is correctly rejected."""
        self.env.reset()
        action = (2, 1, 1, 2)  # Player 1 tries to move onto another piece
        self.assertFalse(self.env.is_valid_move(*action))

    def test_capture_move(self):
        """Test that a valid capture (jumping over opponent) works."""
        self.env.reset()
        # Make a capture possible by Player 1
        self.env.board[3, 2] = -1  # Opponent piece for Player 1 to jump over
        action = (2, 1, 4, 3)  # Player 1 jumps over opponent
        self.assertTrue(self.env.is_valid_move(*action))

    def test_invalid_capture(self):
        """Test that an invalid capture (jumping over an empty space) is rejected."""
        self.env.reset()
        action = (2, 1, 4, 3)  # Player 1 tries to jump over nothing
        self.assertFalse(self.env.is_valid_move(*action))
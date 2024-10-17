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

    def test_normal_move(self):
        """Test executing a valid normal move and updating the board."""
        self.env.reset()
        action = (2, 1, 3, 2)  # A normal move for Player 1
        board, reward, done, _ = self.env.step(action)
        self.assertEqual(board[3, 2], 1)  # The piece should now be at (3, 2)
        self.assertEqual(board[2, 1], 0)  # The original spot should be empty
        self.assertEqual(reward, 0)  # No reward for a normal move

    def test_capture_move_execution(self):
        """Test executing a capture move and updating the board."""
        self.env.reset()
        # Set up a capture scenario
        self.env.board[3, 2] = -1  # Opponent piece
        action = (2, 1, 4, 3)  # Player 1 captures opponent
        board, reward, done, _ = self.env.step(action)
        self.assertEqual(board[4, 3], 1)  # Player 1 should be at (4, 3)
        self.assertEqual(board[3, 2], 0)  # The opponent's piece should be removed
        self.assertEqual(reward, 1)  # Reward for the capture
    
    def test_king_promotion(self):
        """Test that a piece is promoted to king when it reaches the opposite side."""
        self.env.reset()
        # Move a Player 1 piece to the last row
        self.env.board[6, 1] = 1  # Clear space
        self.env.board[7, 2] = 0  # Clear space
        action = (6, 1, 7, 2)  # Player 1 moves to last row
        board, reward, done, _ = self.env.step(action)
        self.assertEqual(board[7, 2], 2)  # The piece should now be a king (2)
    
    def test_game_over(self):
        """Test that the game ends when one player has no pieces left."""
        self.env.reset()
        # Clear Player 2's pieces
        self.env.board[5:8, :] = 0
        self.env.board[6:8, :] = 0
        self.assertTrue(self.env.is_game_over())
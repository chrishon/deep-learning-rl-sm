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
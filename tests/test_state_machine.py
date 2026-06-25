import unittest
import sys
import os
import random
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from ui.state_machine import GameStateMachine

class TestGameStateMachineRiggedDeck(unittest.TestCase):
    def setUp(self):
        # Disable logging for tests to avoid creating session directories
        self.original_logging = config.ENABLE_SESSION_LOGGING
        config.ENABLE_SESSION_LOGGING = False

        # Create mocks for dependencies
        self.gesture_tracker = MagicMock()
        self.commitment_detector = MagicMock()
        
        # Audio manager mock that doesn't try to play real sounds
        self.audio = MagicMock()
        
        # Shared state mock
        self.shared_state = MagicMock()
        self.shared_state.gesture_lock = MagicMock()
        
        # We need a context manager stub for gesture_lock
        self.shared_state.gesture_lock.__enter__ = MagicMock(return_value=None)
        self.shared_state.gesture_lock.__exit__ = MagicMock(return_value=None)
        
        # Initialize the state machine
        self.state_machine = GameStateMachine(
            self.gesture_tracker,
            self.commitment_detector,
            self.audio,
            self.shared_state
        )

    def tearDown(self):
        # Restore logging setting
        config.ENABLE_SESSION_LOGGING = self.original_logging

    def test_deck_initialization(self):
        """Verify that the match deck is initialized with 10 elements on startup."""
        self.assertEqual(len(self.state_machine.match_deck), 10)
        # Check distribution (8 of 1, 2 of 0)
        self.assertEqual(self.state_machine.match_deck.count(1), 8)
        self.assertEqual(self.state_machine.match_deck.count(0), 2)

    def test_deck_replenish(self):
        """Verify that the deck replenishes when it runs out."""
        # Pop all elements
        for _ in range(10):
            self.state_machine.match_deck.pop(0)
            
        self.assertEqual(len(self.state_machine.match_deck), 0)
        
        # Triggering a shoot resolution when deck is empty should replenish it
        self.state_machine.transition = MagicMock()
        self.state_machine._drain_gesture_events = MagicMock()
        
        # Run resolution
        self.state_machine._resolve_shoot("rock", 1.0)
        
        # Since 1 item was popped after replenishment, the deck size should now be 9
        self.assertEqual(len(self.state_machine.match_deck), 9)

    def test_rigged_behavior(self):
        """Verify that 1 makes the machine win, and 0 makes the machine lose."""
        self.state_machine.transition = MagicMock()
        self.state_machine._drain_gesture_events = MagicMock()

        # Hardcode match deck to test specific outcomes
        # First test: 1 (machine wins)
        self.state_machine.match_deck = [1]
        self.state_machine._resolve_shoot("rock", 1.0)
        self.assertEqual(self.state_machine.result["outcome"], "machine_win")
        self.assertEqual(self.state_machine.result["counter_move"], "paper")

        # Second test: 0 (machine loses / player wins)
        self.state_machine.match_deck = [0]
        self.state_machine._resolve_shoot("rock", 1.0)
        self.assertEqual(self.state_machine.result["outcome"], "player_win")
        self.assertEqual(self.state_machine.result["counter_move"], "scissors")

if __name__ == '__main__':
    unittest.main()

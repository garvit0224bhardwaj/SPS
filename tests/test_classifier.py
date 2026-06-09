import unittest
import sys
import os
import random
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.classifier import classify_gesture
from core.bootstrap_dataset import make_hand_landmarks, random_rotation_matrix

class TestMLPClassifier(unittest.TestCase):
    def setUp(self):
        # Seed for reproducible random tests
        random.seed(42)
        np.random.seed(42)

    def test_rock_predictions(self):
        """Verify that various sizes and rotations of Rock are correctly classified."""
        for _ in range(20):
            scale = random.uniform(0.12, 0.40)
            translation = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(-0.1, 0.1)])
            rot = random_rotation_matrix(max_angle_deg=45.0)
            is_left = (random.random() < 0.5)
            handedness = "Left" if is_left else "Right"
            
            lm = make_hand_landmarks("rock", rot, scale, translation, is_left=is_left)
            result = classify_gesture(lm, handedness)
            
            self.assertEqual(result["gesture"], "rock", f"Failed rock prediction: {result}")
            self.assertTrue(result["valid"])

    def test_paper_predictions(self):
        """Verify that various sizes and rotations of Paper are correctly classified."""
        for _ in range(20):
            scale = random.uniform(0.12, 0.40)
            translation = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(-0.1, 0.1)])
            rot = random_rotation_matrix(max_angle_deg=45.0)
            is_left = (random.random() < 0.5)
            handedness = "Left" if is_left else "Right"
            
            lm = make_hand_landmarks("paper", rot, scale, translation, is_left=is_left)
            result = classify_gesture(lm, handedness)
            
            self.assertEqual(result["gesture"], "paper", f"Failed paper prediction: {result}")
            self.assertTrue(result["valid"])

    def test_scissors_predictions(self):
        """Verify that various sizes and rotations of Scissors are correctly classified."""
        for _ in range(20):
            scale = random.uniform(0.12, 0.40)
            translation = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(-0.1, 0.1)])
            rot = random_rotation_matrix(max_angle_deg=45.0)
            is_left = (random.random() < 0.5)
            handedness = "Left" if is_left else "Right"
            
            lm = make_hand_landmarks("scissors", rot, scale, translation, is_left=is_left)
            result = classify_gesture(lm, handedness)
            
            self.assertEqual(result["gesture"], "scissors", f"Failed scissors prediction: {result}")
            self.assertTrue(result["valid"])

    def test_unknown_predictions(self):
        """Verify that Thumbs-up and random hands are correctly classified as unknown."""
        # Test thumbs_up
        for _ in range(10):
            scale = random.uniform(0.12, 0.40)
            translation = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(-0.1, 0.1)])
            rot = random_rotation_matrix(max_angle_deg=45.0)
            is_left = (random.random() < 0.5)
            handedness = "Left" if is_left else "Right"
            
            lm = make_hand_landmarks("thumbs_up", rot, scale, translation, is_left=is_left)
            result = classify_gesture(lm, handedness)
            
            self.assertEqual(result["gesture"], "unknown", f"Expected thumbs-up to be unknown, got {result}")
            self.assertFalse(result["valid"])

        # Test random garbage hands
        for _ in range(20):
            scale = random.uniform(0.12, 0.40)
            translation = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(-0.1, 0.1)])
            rot = random_rotation_matrix(max_angle_deg=180.0) # Full 3D rotation for noise
            is_left = (random.random() < 0.5)
            handedness = "Left" if is_left else "Right"
            
            lm = make_hand_landmarks("random", rot, scale, translation, is_left=is_left)
            result = classify_gesture(lm, handedness)
            
            # Since "random" can occasionally look like a valid shape, we don't assert it strictly
            # but check it works without raising exceptions.
            self.assertIn(result["gesture"], ["rock", "paper", "scissors", "unknown"])

if __name__ == '__main__':
    unittest.main()

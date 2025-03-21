import unittest
from src.models.tracker import Tracker

class TestTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = Tracker()

    def test_initialize_tracker(self):
        self.assertIsNotNone(self.tracker)

    def test_track_objects(self):
        # Assuming we have a method to simulate detection results
        detections = [{'id': 1, 'bbox': [100, 100, 50, 50]}, {'id': 2, 'bbox': [200, 200, 50, 50]}]
        tracked_objects = self.tracker.track(detections)
        self.assertEqual(len(tracked_objects), 2)

    def test_update_tracker(self):
        initial_detections = [{'id': 1, 'bbox': [100, 100, 50, 50]}]
        self.tracker.track(initial_detections)
        updated_detections = [{'id': 1, 'bbox': [110, 110, 50, 50]}]
        tracked_objects = self.tracker.track(updated_detections)
        self.assertEqual(tracked_objects[0]['bbox'], [110, 110, 50, 50])

if __name__ == '__main__':
    unittest.main()
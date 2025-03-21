import unittest
from src.models.detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ObjectDetector(model_path='data/models/pretrained_model.h5')

    def test_load_model(self):
        self.assertIsNotNone(self.detector.model)

    def test_detect_objects(self):
        test_frame = ...  # Load or create a test frame
        detections = self.detector.detect(test_frame)
        self.assertIsInstance(detections, list)
        self.assertGreater(len(detections), 0)

    def test_detect_objects_empty_frame(self):
        empty_frame = ...  # Create an empty frame
        detections = self.detector.detect(empty_frame)
        self.assertEqual(len(detections), 0)

if __name__ == '__main__':
    unittest.main()
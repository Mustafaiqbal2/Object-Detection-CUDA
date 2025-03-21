import unittest
from src.camera.webcam import Webcam

class TestWebcam(unittest.TestCase):

    def setUp(self):
        self.webcam = Webcam()

    def test_start_webcam(self):
        self.assertTrue(self.webcam.start(), "Webcam should start successfully")

    def test_get_frame(self):
        self.webcam.start()
        frame = self.webcam.get_frame()
        self.assertIsNotNone(frame, "Frame should not be None")
        self.assertGreater(frame.shape[0], 0, "Frame height should be greater than 0")
        self.assertGreater(frame.shape[1], 0, "Frame width should be greater than 0")

    def test_release_webcam(self):
        self.webcam.start()
        self.webcam.release()
        self.assertFalse(self.webcam.is_running(), "Webcam should be released and not running")

if __name__ == '__main__':
    unittest.main()
import unittest
import cv2

class TestAuxiliaryUnit(unittest.TestCase):
    def setUp(self):
        self.image_path = 'images/1.png'
        self.cv_image = cv2.imread(self.image_path)

if __name__ == '__main__':
    unittest.main()
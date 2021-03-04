import unittest
import cv2

import src.exercise1 as exercise


class TestAuxiliaryUnit(unittest.TestCase):
    def setUp(self):
        self.image_path = 'tests/fixtures/test_img.jpg'
        self.image = cv2.imread(self.image_path)

    def test_img_read(self):
        image = exercise.read_img(self.image_path)
        self.assertIsNotNone(image)

    def test_resize_image_width(self):
        image = self.image
        image_shape = image.shape
        image_returned = exercise.change_size(image, 1200, 1200, 0.1)
        image_returned_shape = image_returned.shape
        self.assertNotEqual(image_shape, image_returned_shape)


if __name__ == '__main__':
    unittest.main()

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

    def test_change_image_color(self):
        image = exercise.change_img_color(self.image, cv2.COLOR_BGR2GRAY)
        self.assertEqual(len(image.shape), 2)

    def test_square_crop(self):
        image = exercise.apply_square_crop(self.image, 200, 200, 700)
        self.assertEqual(image.shape[0], image.shape[1])


if __name__ == '__main__':
    unittest.main()

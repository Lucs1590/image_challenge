import cv2
import random
import numpy as np


def main():
    raw_img = read_img('resources/images/image.jpg')
    show_and_save(raw_img, 1)
    gray_img = change_img_color(raw_img, cv2.COLOR_BGR2GRAY)
    show_and_save(gray_img, 2)
    noise_img = add_salt_pepper_noise(gray_img, 0.1)
    show_and_save(noise_img, 3)
    restored_img = apply_median(noise_img)
    show_and_save(restored_img, 4)
    rotated_img = rotate_image(restored_img, cv2.ROTATE_180)
    show_and_save(rotated_img, 5)
    (original_height, original_width) = raw_img.shape[:2]
    squared_image = apply_square_crop(rotated_img, 300, 100, 500)
    show_and_save(squared_image, 6)
    resized_img = change_size(
        squared_image, original_height, original_width, 0.75)
    show_and_save(resized_img, 7)
    colored_img = change_img_color(resized_img, cv2.COLOR_GRAY2BGR)
    show_and_save(colored_img, 8)


def read_img(path):
    """
    # Read Image
    This function make the image reading.
    """
    return cv2.imread(path)


def show_and_save(img, img_number):
    """
    # Show Image
    This function show and save an image.
    """
    cv2.imwrite("resources/results/{0}.png".format(img_number), img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def change_img_color(img, color_transformation):
    """
    # Change Color Image
    This function change the color of image.
    """
    return cv2.cvtColor(img, color_transformation)


def add_salt_pepper_noise(img, prob):
    """
    # Add Salt Pepper Noise
    This function adds salt and pepper noise to an image by
    considering a probability of this happening.
    """
    aux = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.random() > 0.5:
                if random.random() < prob:
                    aux[i][j] = 255
            else:
                if random.random() < prob:
                    aux[i][j] = 0
    return aux


def apply_median(img):
    """
    # Apply Median
    This function applies the average to an image considering
    a 3x3 kernel.
    """
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            img[i, j] = np.sort(img[i-1:i+2, j-1:j+2], axis=None)[5] if img[i,
                                                                            j] == 0 or img[i, j] == 255 else img[i, j]
    return img  # I could use cv2.medianBlur(img, 3)


def rotate_image(img, rotation):
    """
    # Rotate Image
    This function apply rotation on image.
    """
    return cv2.rotate(img, rotation)


def apply_square_crop(img, y, x, size):
    """
    # Rotate Image
    This function apply square crop with a size.
    """
    return img[y:y+size, x:x+size]


def change_size(img, original_height, original_width, percent):
    """
    # Change Size
    This function resize an image based on a percentage.
    """
    return cv2.resize(img, (int(original_width * percent), int(original_height * percent)))


if __name__ == "__main__":
    main()

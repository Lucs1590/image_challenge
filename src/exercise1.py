import cv2
# from matplotlib import pyplot as plt
import random
import numpy as np


def main():
    raw_img = read_img('resources/images/1.jpg')
    show_img(raw_img)
    gray_img = change_img_color(raw_img, cv2.COLOR_BGR2GRAY)
    show_img(gray_img)
    noise_img = add_salt_pepper_noise(gray_img, 0.1)
    show_img(noise_img)
    restored_img = apply_median(noise_img)
    show_img(restored_img)
    rotated_img = rotate_image(restored_img, cv2.ROTATE_180)
    show_img(rotated_img)



def read_img(path):
    """
    # Read Image
    This function make the image reading.
    """
    return cv2.imread(path)


def show_img(img):
    """
    # Show Image
    This function show image.
    """
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


if __name__ == "__main__":
    main()

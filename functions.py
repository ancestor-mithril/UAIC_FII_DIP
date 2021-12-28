import os
import pathlib
import random
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
from numpy import ndarray
from image_format import RGBImage, HSVImage, YCbCrImage


def identify_face(im_path: str = "data/5.jpg", wait: bool = True) -> None:
    print(im_path)
    image = RGBImage(im_path).image
    skin_image = RGBImage(im_path).skin_image
    top_row = None
    top_left = None
    top_right = None
    top_down = None
    for i in range(skin_image.shape[0]):
        top_left_2 = None

        for j in range(skin_image.shape[1]):
            if skin_image[i, j] == 1:
                if top_row is None:
                    top_row = i
                    top_right = top_left = j
                    continue

                top_right_2 = j
                if top_left_2 is None:
                    top_left_2 = j

                if j > top_right:
                    top_right = j
                if j < top_left:
                    top_left = j

                if top_left_2 is not None and top_right_2 is not None and top_left + 5 < top_left_2 and top_right - 5 > top_right_2:
                    top_down = i
                    break
        if top_down is not None:
            break
    if top_down is None:
        top_down = skin_image.shape[0] - 1
    top = top_row
    down = min(skin_image.shape[0] - 2, top_down + (top_down - top_row) // 2)
    left = top_left
    right = top_right

    for i in range(top, down):
        image[i, left, 0] = image[i, left, 1] = image[i, left, 2] = 0
        image[i, right, 0] = image[i, right, 1] = image[i, right, 2] = 0

    for j in range(left, right):
        image[top, j, 0] = image[top, j, 1] = image[top, j, 2] = 0
        image[down, j, 0] = image[down, j, 1] = image[down, j, 2] = 0

    cv2.imshow(im_path, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_face_recognition():
    for im_path in tqdm([x for x in pathlib.Path("data/Pratheepan/Pratheepan_Dataset/FacePhoto").glob('*.jpg')]):
        identify_face(str(im_path), wait=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_skin_image(im_path: str = "data/5.jpg") -> None:
    RGBImage(im_path).show()
    HSVImage(im_path).show()
    YCbCrImage(im_path).show()


def test() -> None:
    pratheepan_face = "data/Pratheepan/Pratheepan_Dataset/FacePhoto"
    pratheepan_face_truth = "data/Pratheepan/Ground_Truth/GroundT_FacePhoto"
    pratheepan_family = "data/Pratheepan/Pratheepan_Dataset/FamilyPhoto"
    pratheepan_family_truth = "data/Pratheepan/Ground_Truth/GroundT_FamilyPhoto"
    evaluate(pratheepan_face, pratheepan_face_truth)
    evaluate(pratheepan_family, pratheepan_family_truth)


def get_accuracy(true_positive: int, true_negative: int, false_negative: int, false_positive: int) -> float:
    assert true_positive + true_negative + false_negative + false_positive > 0
    return (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)


def evaluate(path: str, truth_path: str) -> None:
    confusion_matrix_rgb = np.array([0, 0, 0, 0])
    confusion_matrix_hsv = np.array([0, 0, 0, 0])
    confusion_matrix_ycbcr = np.array([0, 0, 0, 0])
    for im_path in tqdm([x for x in pathlib.Path(path).glob('*.jpg')]):
        # print(im_path)
        if im_path.name == "PatCassidyFamily.jpg":
            break
        truth = process_truth(cv2.imread(os.path.join(truth_path, ".png".join(im_path.name.rsplit(".jpg", 1)))))
        im_path = str(im_path)
        confusion_matrix_rgb += np.array(diff(truth, RGBImage(im_path).skin_image))
        confusion_matrix_hsv += np.array(diff(truth, HSVImage(im_path).skin_image))
        confusion_matrix_ycbcr += np.array(diff(truth, YCbCrImage(im_path).skin_image))
    print(f"Accuracy:\nRGB: {get_accuracy(*confusion_matrix_rgb)}\n"
          f"HSV: {get_accuracy(*confusion_matrix_hsv)}\n"
          f"YCbCr: {get_accuracy(*confusion_matrix_ycbcr)}")


def verify():
    # PIC_0111
    truth_path = "data/Pratheepan/Ground_Truth/GroundT_FamilyPhoto/PatCassidyFamily.png"
    im_path = "data/Pratheepan/Pratheepan_Dataset/FamilyPhoto/PatCassidyFamily.jpg"
    truth = process_truth(cv2.imread(truth_path))
    diff(truth, RGBImage(im_path).skin_image)
    diff(truth, HSVImage(im_path).skin_image)
    diff(truth, YCbCrImage(im_path).skin_image)
    RGBImage(im_path).show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def diff(truth: ndarray, result: ndarray) -> (int, int, int, int):
    true_positive, true_negative, false_negative, false_positive = 0, 0, 0, 0
    for i in range(truth.shape[0]):
        for j in range(truth.shape[1]):
            if truth[i, j] == result[i, j] == 1:
                true_positive += 1
            if truth[i, j] == result[i, j] == 0:
                true_negative += 1
            if truth[i, j] == 1 and result[i, j] == 0:
                false_negative += 1
            if truth[i, j] == 0 and result[i, j] == 1:
                false_positive += 1
    return true_positive, true_negative, false_negative, false_positive


def process_truth(truth: ndarray) -> ndarray:
    bidimensional_truth = np.zeros(truth.shape[:2])
    for i in range(truth.shape[0]):
        for j in range(truth.shape[1]):
            if truth[i, j, 0] == 255:
                bidimensional_truth[i, j] = 1
    return bidimensional_truth


def draw_point(img: ndarray, point: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1) -> ndarray:
    x, y = point
    b, g, r = color
    for i in range(x - thickness // 2, x + thickness // 2 + 1):
        img[i, y, 0] = b
        img[i, y, 1] = g
        img[i, y, 2] = r
    return img


def create_emoticon() -> None:
    emoticon = np.full((224, 224, 3), 255, dtype=np.uint8)
    emoticon = cv2.circle(img=emoticon, center=(112, 112), radius=100, color=(30, 174, 252), thickness=cv2.FILLED)
    emoticon = cv2.circle(img=emoticon, center=(80, 80), radius=10, color=(0, 0, 0), thickness=cv2.FILLED)
    emoticon = cv2.circle(img=emoticon, center=(144, 80), radius=10, color=(0, 0, 0), thickness=cv2.FILLED)
    emoticon = cv2.circle(img=emoticon, center=(60, 140), radius=15, color=(23, 113, 237), thickness=cv2.FILLED)
    emoticon = cv2.circle(img=emoticon, center=(164, 140), radius=15, color=(23, 113, 237), thickness=cv2.FILLED)
    for y in range(85, 113):
        draw_point(emoticon, point=(155 - int(0.02 * ((y - 112) ** 2)), y), color=(0, 0, 0), thickness=5)
    for y in range(113, 140):
        draw_point(emoticon, point=(155 - int(0.02 * ((112 - y) ** 2)), y), color=(0, 0, 0), thickness=5)
    cv2.imshow("Image", emoticon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_grayscale(im_path: str = "data/5.jpg") -> None:
    im = RGBImage(im_path)
    im.show_image()
    im.show_greyscale(strategy="simple_averaging")
    im.show_greyscale(strategy="weighted_average", param=1)
    im.show_greyscale(strategy="weighted_average", param=2)
    im.show_greyscale(strategy="weighted_average", param=3)
    im.show_greyscale(strategy="desaturation")
    im.show_greyscale(strategy="single_colour_channel", param="r")
    im.show_greyscale(strategy="single_colour_channel", param="g")
    im.show_greyscale(strategy="single_colour_channel", param="b")
    im.show_greyscale(strategy="custom_number_of_grey_shades", param=32)
    im.show_greyscale(strategy="custom_number_of_grey_shades", param=16)
    im.show_greyscale(strategy="custom_number_of_grey_shades", param=8)
    im.show_greyscale(strategy="custom_number_of_grey_shades", param=4)
    im.show_greyscale(strategy="custom_number_of_grey_shades_with_error_diffusion_dithering", param=32)
    im.show_greyscale(strategy="custom_number_of_grey_shades_with_error_diffusion_dithering", param=16)
    im.show_greyscale(strategy="custom_number_of_grey_shades_with_error_diffusion_dithering", param=8)
    im.show_greyscale(strategy="custom_number_of_grey_shades_with_error_diffusion_dithering", param=4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def coloring_grayscale(im_path: str = "data/5.jpg") -> None:
    def get_color(x) -> (int, int, int):
        while True:
            a = random.uniform(0, 255)
            b = random.uniform(0, 255)
            c = random.uniform(0, 255)
            if abs((a + b + c) / 3 / 255 - x) < 10e-2:
                break
        return int(a), int(b), int(c)

    im = RGBImage(im_path)
    im.show_image()
    grayscale = im.grayscale
    unique_shades = np.unique(grayscale)
    color_map = {
        str(x): get_color(x) for x in unique_shades
    }
    image = np.zeros(list(grayscale.shape) + [3], dtype=np.uint8)
    for i in range(grayscale.shape[0]):
        for j in range(grayscale.shape[1]):
            image[i, j, 0], image[i, j, 1], image[i, j, 2] = color_map[str(grayscale[i, j])]

    cv2.imshow("Grayscale image", grayscale)
    cv2.imshow("Colored image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

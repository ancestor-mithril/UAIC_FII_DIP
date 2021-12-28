import cv2

from functions import show_skin_image, test, identify_face, test_face_recognition, verify, create_emoticon, \
    get_grayscale, coloring_grayscale

if __name__ == "__main__":
    test()
    show_skin_image()
    identify_face("data/2.jpg")
    create_emoticon()

    get_grayscale("data/2.jpg")

    coloring_grayscale("data/2.jpg")
    # verify()
    print("OK")

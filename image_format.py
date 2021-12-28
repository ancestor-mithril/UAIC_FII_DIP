from abc import ABC, abstractmethod

import cv2
import numpy as np
from numpy import ndarray

from grayscale_convertors import SimpleAveragingGrayscaleConvertor, WeightedAverageGrayscaleConvertor, \
    DesaturationGrayscaleConvertor, SingleColorChannelGrayscaleConvertor, CustomNumberOfGrayShadesGrayscaleConvertor, \
    CustomNumberOfGrayShadesWithErrorDiffusionDitheringGrayscaleConvertor


class AbstractImage(ABC):
    def __init__(self, im_path: str):
        self.image = cv2.imread(im_path)
        if self.image is None:
            raise Exception("No image")
        self.convert_image_to_format()
        self.skin_image = self.get_skin_image(self.image)

    def convert_image_to_format(self) -> None:
        pass

    @abstractmethod
    def is_skin(self, pixel: ndarray) -> bool:
        pass

    def show_image(self) -> None:
        cv2.imshow("Image", self.image)

    def show(self) -> None:
        self.show_image()
        cv2.imshow("Skin image", self.skin_image)
        cv2.waitKey(0)

    def get_skin_image(self, image: ndarray) -> ndarray:
        skin_image = np.zeros(image.shape[:2])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if self.is_skin(image[i, j]):
                    skin_image[i, j] = 1
        return skin_image


class RGBImage(AbstractImage, ABC):
    def __init__(self, im_path: str):
        super().__init__(im_path)
        self.grayscale = CustomNumberOfGrayShadesGrayscaleConvertor(self.image, 32).get_grayscale()

    def is_skin(self, pixel: ndarray) -> bool:
        b, g, r = pixel
        return r > 95 and g > 40 and b > 20 and (int(max(r, g, b)) - int(min(r, g, b))) > 15 and abs(
            int(r) - int(g)) > 15 and r > g and r > b

    def show_greyscale(self, strategy: str, param=None) -> None:
        all_strategies = {
            "simple_averaging": SimpleAveragingGrayscaleConvertor,
            "weighted_average": WeightedAverageGrayscaleConvertor,
            "desaturation": DesaturationGrayscaleConvertor,
            "single_colour_channel": SingleColorChannelGrayscaleConvertor,
            "custom_number_of_grey_shades": CustomNumberOfGrayShadesGrayscaleConvertor,
            "custom_number_of_grey_shades_with_error_diffusion_dithering":
                CustomNumberOfGrayShadesWithErrorDiffusionDitheringGrayscaleConvertor,
        }
        if strategy not in all_strategies:
            raise Exception(f"Unknown strategy {strategy}. Accepted strategies: {', '.join(all_strategies)}")

        image = all_strategies[strategy](self.image, param).get_grayscale()
        cv2.imshow(f"{strategy}_{str(param)}", image)


class HSVImage(AbstractImage, ABC):
    def __init__(self, im_path: str):
        super().__init__(im_path)

    def is_skin(self, pixel: ndarray) -> bool:
        h, s, v = pixel
        return 0 <= h <= 50 and 0.23 <= s / 255 <= 0.68 and 0.35 <= v / 255 <= 1

    def show_image(self) -> None:
        im2 = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
        cv2.imshow("Image", im2)

    def convert_image_to_format(self) -> None:
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)


class YCbCrImage(AbstractImage, ABC):
    def __init__(self, im_path: str):
        super().__init__(im_path)

    def is_skin(self, pixel: ndarray) -> bool:
        y, cr, cb = pixel
        return y > 80 and 85 < cb < 135 < cr < 180

    def show_image(self) -> None:
        im2 = cv2.cvtColor(self.image, cv2.COLOR_YCrCb2RGB)
        cv2.imshow("Image", im2)

    def convert_image_to_format(self) -> None:
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                b, g, r = self.image[i, j]
                self.image[i, j, 0] = 0.299 * r + 0.587 * g + 0.114 * b
                self.image[i, j, 2] = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
                self.image[i, j, 1] = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

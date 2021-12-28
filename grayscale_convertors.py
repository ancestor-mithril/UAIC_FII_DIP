from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray


class AbstractGrayscaleConvertor(ABC):
    def __init__(self, image: ndarray, param=None):
        self.param = param
        self.image = image
        self.grayscale = np.zeros(image.shape[:2])
        self.error_message = "Strategy does not use param"

    def get_grayscale(self) -> ndarray:
        if not self.check_param():
            self.wrong_param()
        self.do_convert()
        return self.grayscale

    @abstractmethod
    def do_convert(self) -> None:
        pass

    def check_param(self) -> bool:
        return self.param is None

    def wrong_param(self) -> None:
        raise Exception(self.error_message)

    def simple_mutation(self, mutation: callable) -> None:
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                self.grayscale[i, j] = mutation(*self.image[i, j])


class SimpleAveragingGrayscaleConvertor(AbstractGrayscaleConvertor, ABC):
    def __init__(self, image: ndarray, param=None):
        super().__init__(image, param)

    def do_convert(self) -> None:
        def mutation(b: int, g: int, r: int) -> float:
            return (int(g) + int(b) + int(r)) / 3 / 255

        self.simple_mutation(mutation)


class WeightedAverageGrayscaleConvertor(AbstractGrayscaleConvertor, ABC):
    def __init__(self, image: ndarray, param=None):
        super().__init__(image, param)
        self.error_message = "Param must be 1, 2 or 3"

    def check_param(self) -> bool:
        return self.param in [1, 2, 3]

    def do_convert(self) -> None:
        def mutation_1(b: int, g: int, r: int) -> float:
            return (0.3 * r + 0.59 * g + 0.11 * b) / 255

        def mutation_2(b: int, g: int, r: int) -> float:
            return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255

        def mutation_3(b: int, g: int, r: int) -> float:
            return (0.299 * r + 0.587 * g + 0.114 * b) / 255

        mutations = [mutation_1, mutation_2, mutation_3]
        self.simple_mutation(mutations[self.param - 1])


class DesaturationGrayscaleConvertor(AbstractGrayscaleConvertor, ABC):
    def __init__(self, image: ndarray, param=None):
        super().__init__(image, param)

    def do_convert(self) -> None:
        def mutation(b: int, g: int, r: int) -> float:
            return (int(min(r, g, b)) + int(max(r, g, b))) / 2 / 255

        self.simple_mutation(mutation)


class SingleColorChannelGrayscaleConvertor(AbstractGrayscaleConvertor, ABC):
    def __init__(self, image: ndarray, param=None):
        super().__init__(image, param)
        self.error_message = "Param must be r, g or b"

    def check_param(self) -> bool:
        return self.param in ["r", "g", "b"]

    def do_convert(self) -> None:
        def mutation_1(b: int, g: int, r: int) -> float:
            return r / 255

        def mutation_2(b: int, g: int, r: int) -> float:
            return g / 255

        def mutation_3(b: int, g: int, r: int) -> float:
            return b / 255

        mutations = {
            "r": mutation_1,
            "g": mutation_2,
            "b": mutation_3
        }
        self.simple_mutation(mutations[self.param])


class CustomNumberOfGrayShadesGrayscaleConvertor(AbstractGrayscaleConvertor, ABC):
    def __init__(self, image: ndarray, param=None):
        super().__init__(image, param)
        self.error_message = "0 <= param <= 255"

    def check_param(self) -> bool:
        self.param = int(self.param)
        return 0 <= self.param <= 255

    def do_convert(self) -> None:
        shades_of_gray = [i / 255 for i in range(0, 256, 255 // self.param) for _ in range(255 // self.param)]

        def mutation(b: int, g: int, r: int) -> float:
            return shades_of_gray[(int(g) + int(b) + int(r)) // 3]

        self.simple_mutation(mutation)


class CustomNumberOfGrayShadesWithErrorDiffusionDitheringGrayscaleConvertor(AbstractGrayscaleConvertor, ABC):
    def __init__(self, image: ndarray, param=None):
        super().__init__(image, param)
        self.error_message = "0 <= param <= 255"

    def check_param(self) -> bool:
        self.param = int(self.param)
        return 0 <= self.param <= 255

    def do_convert(self) -> None:
        shades_of_gray = [i / 255 for i in range(0, 256, 255 // self.param) for _ in range(255 // self.param)]

        def mutation(b: int, g: int, r: int) -> float:
            return shades_of_gray[(int(g) + int(b) + int(r)) // 3]

        def add_error(x: int, y: int, err: float):
            if x >= self.image.shape[0] or y >= self.image.shape[1] or x < 0 or y < 0:
                return
            for channel in range(3):
                value = int(int(self.image[x, y, channel]) + err)
                if value < 0:
                    self.image[x, y, channel] = 0
                else:
                    if value > 255:
                        self.image[x, y, channel] = 255
                    else:
                        self.image[x, y, channel] = value

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                self.grayscale[i, j] = mutation(*self.image[i, j])
                error = (
                                int(self.image[i, j, 0]) + int(self.image[i, j, 1]) + int(self.image[i, j, 2])
                        ) / 3 - self.grayscale[i, j] * 255
                add_error(i, j + 1, error * 7 / 16)
                add_error(i + 1, j - 1, error * 3 / 16)
                add_error(i + 1, j, error * 5 / 16)
                add_error(i + 1, j + 1, error * 1 / 16)

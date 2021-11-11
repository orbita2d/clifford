import numpy as np


class GeneratorSettings:
    """ ABC for generator object. Generators choose the path through parameter space that our animation takes."""
    def __init__(self, data: dict):
        self.type: str = data["type"]
        self.data: dict = data["data"]

    def get_p(self, i: int, n: int, delta: float):
        # Get the parameter p(i / n) for the generator, with some 'nudge' delta
        raise SyntaxError('GeneratorSettings is an abstract base class. get_p() should be overwritten')


class GeneratorFrame(GeneratorSettings):
    """ Just return a single point in space. Useful for single frames."""
    # Generate a single frame at p0
    def __init__(self, data: dict):
        super(GeneratorFrame, self).__init__(data)
        self.p0: np.ndarray = np.array(self.data["p0"])

    def get_p(self, i: int, n: int, delta: float) -> np.ndarray:
        # Get the parameter p(i / n) for the generator
        p0 = self.p0
        return self.p0


class GeneratorTwoVector(GeneratorSettings):
    """ Trace an ellipse in parameter space. p(t) = p0 + px cos t + py sin t"""
    def __init__(self, data: dict):
        super(GeneratorTwoVector, self).__init__(data)
        self.p0: np.ndarray = np.array(self.data["p0"])
        self.px: np.ndarray = np.array(self.data["px"])
        self.py: np.ndarray = np.array(self.data["py"])

    def get_p(self, i: int, n: int, delta: float) -> np.ndarray:
        # Get the parameter p(i / n) for the generator
        path_theta = 2 * np.pi * i / n
        p0 = self.p0
        px = self.px
        py = self.py
        return p0 + (1 - delta) * (np.cos(path_theta) * px + np.sin(path_theta) * py)


def get_generator(data: dict) -> GeneratorSettings:
    if data["type"] == "two-vector":
        return GeneratorTwoVector(data)
    if data["type"] == "frame":
        return GeneratorFrame(data)
    else:
        raise AttributeError(f'Do not support type: {["type"]}')

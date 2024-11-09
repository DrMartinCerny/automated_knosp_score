import numpy as np

class Vessel:
    """Custom representation of the an artery.
    """
    def __init__(self, pts, binary_mask, com) -> None:
        self.pts = pts
        self.mask = binary_mask
        self.com = com


class Line:
    """Object for the lines.
    """

    R = np.array([[0, 1], [-1, 0]])

    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b
        self.v = self.R @ (a-b)

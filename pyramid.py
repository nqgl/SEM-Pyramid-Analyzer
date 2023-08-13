import analysis
import numpy as np
import cv2
from t1cv import line, arrow 



class Pyramid:
    def __init__(self, IA :analysis.ImageAnalysis, pos :np.ndarray, id = None):
        self.positions = [pos]
        self.IA = IA
        self.cross_lengths = [5,5,5,5]
        self.id = id
        self.cross_mask = None
        self.mean_position = pos


    def set_cross_lengths(self, lengths):
        self.cross_lengths = lengths

    def get_center(self):
        return self.mean_position

    def absorb(self, other):
        if other is None:
            return
        if other is self:
            return
        self.positions += other.positions
        self.mean_position = np.mean(self.positions, axis=0)

    def get_mask(self, canvas = None, color=None, meanonly = False, lengths = None):
        if meanonly:
            return self.get_cross_mask(self.mean_position[0], self.mean_position[1], color=color, canvas=canvas, lengths = lengths)
        for pos in self.positions + [self.mean_position]:
            canvas = self.get_cross_mask(pos[0], pos[1], canvas = canvas, color=color, lengths = lengths)
        return canvas

    def get_line_endpoint(self, length, i, rotation = None, x = None, y = None):
        if length < 0:
            raise ValueError("Length must be positive")
        _x, _y = self.get_center()
        x = _x if x is None else x
        y = _y if y is None else y
        rotation = self.IA.rotation if rotation is None else rotation

        endpoint = np.float32((x + length * np.cos(rotation + i * np.pi / 2), y + length * np.sin(rotation + i * np.pi / 2)))
        return endpoint


    def get_cross_mask(self, x = None, y = None, color=None, lengths = None, rotation = None, thickness = 1, canvas = None):
        _x, _y = self.get_center()
        x = _x if x is None else x
        y = _y if y is None else y
        x_int = int(x)
        y_int = int(y)
        rotation = self.IA.rotation if rotation is None else rotation
        if canvas is None:
            canvas = np.zeros_like(self.IA.gray, dtype=np.int16) - 1
        if color is None:
            color = self.id if color is None else self.color
        lengths = self.cross_lengths if lengths is None else lengths
        for i in range(4):
            endpoint = self.get_line_endpoint(lengths[i], i, rotation)
            line(canvas, (x_int, y_int), endpoint,
                    color=color, thickness=thickness)
        return canvas

    def is_inside_box(self, box):
        x, y = self.get_center()
        box = [np.min(box[:,0]), np.min(box[:,1]), np.max(box[:,0]), np.max(box[:,1])]
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]



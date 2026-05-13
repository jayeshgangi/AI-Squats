import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

class AngleSmoother:
    def __init__(self,window=7):
        self.window = window
        self.values = []

    def smooth(self,value):
        self.values.append(value)
        if len(self.values) > self.window:
            self.values.pop(0)
        return sum(self.values) / len (self.values)

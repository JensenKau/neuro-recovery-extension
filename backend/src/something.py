from __future__ import annotations
from scipy.interpolate import CubicSpline
import numpy as np

if __name__ == "__main__":
    x = np.arange(10)
    y = np.sin(x)
    cs = CubicSpline(x, y)
    xs = np.arange(0, 9.6, 0.1)
    ys = cs(xs)
    
    print(y)
    print(ys)
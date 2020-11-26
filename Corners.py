import numpy as np

def getCorners(points):
    points = points.reshape((4,2))
    corners = np.zeros((4,2), dtype = "float32")
    
    s = points.sum(axis = 1)
    corners[0] = points[np.argmin(s)]
    corners[2] = points[np.argmax(s)]
    
    d = np.diff(points,axis = 1)
    corners[1] = points[np.argmin(d)]
    corners[3] = points[np.argmax(d)]
    
    return corners
import cv2
import numpy as np
from enum import Enum
import sys

class RECT_OPTION(Enum):
    X       = 0
    Y       = 1
    WIDTH   = 2
    HEIGHT  = 3
    AREA    = 4

def calcAveColorInRect(frame, rect):
    aveColors = [0, 0, 0]
    x, y, w, h = rect
    submat = frame[y:y+h,x:x+w,:]
    channels = [submat[:,:,i] for i in range(3)]
    aveColors = [aveColors[i] + np.mean(channels[i]) for i in range(3)]
    return aveColors

def mergeRects(rects):

    # If there is one contour, return first
    x, y, w, h = rects[0]

    # If there are two contours, extrapolate
    if len(rects) == 2:

        # If we have two boxes or more, retrieve overarching rectanlge
        xl, yl, wl, hl = 0, 0, 0, 0 
        xr, yr, wr, hr = 0, 0, 0, 0

        # Get bounding rectangles of both
        x1, y1, w1, h1 = rects[0]
        x2, y2, w2, h2 = rects[1]

        # Check if two boxes are within feasible distance
        diff = abs(x1 - x2)
        if diff > 500:
            return x, y, w, h

        # Check which side rectangles are on, and calculate surrounding box
        if x1 < x2:
            xl = x1
            yl = y1
            xr = x2
            yr = y2
            wr = w2
            hr = h2
        else:
            xl = x2
            yl = y2
            xr = x1
            yr = y1
            wr = w1
            hr = h1

        x = xl
        y = yl
        w = (abs(xr - xl) + wr)
        h = (abs(yr - yl) + hr)

    return x, y, w, h


def sortRectsByMaxOption(n, option, rects):
    sortedRects = []
    for i in range(n):
        alphaIndex = getMaxIndex(rects, option)
        sortedRects.append(rects[alphaIndex])
        rects.pop(alphaIndex)
        if len(rects) == 0:
            break
    return sortedRects

def sortRectsByMinOption(n, option, rects):
    sortedRects = []
    for i in range(n):
        betaIndex = getMinIndex(rects, option)
        sortedRects.append(rects[betaIndex])
        rects.pop(betaIndex)
        if len(rects) == 0:
            break
    return sortedRects

def getMaxIndex(rects, option):
    alpha_index = 0;
    maxV = -sys.maxsize - 1;
    curV = 0;
    for i in range(len(rects)):

        switcher = {
            RECT_OPTION.X:      rects[i][0],
            RECT_OPTION.Y:      rects[i][1],
            RECT_OPTION.WIDTH:  rects[i][2],
            RECT_OPTION.HEIGHT: rects[i][3],
            RECT_OPTION.AREA:   rects[i][2] * rects[i][3]
        }

        curV = switcher.get(option, "Invalid Option")
        if curV > maxV:
            maxV = curV
            alpha_index = i
    
    return alpha_index;

def getMinIndex(rects, option):
    beta_index = 0;
    maxV = sys.maxsize;
    curV = 0;
    for i in range(len(rects)):

        switcher = {
            RECT_OPTION.X:      rects[i][0],
            RECT_OPTION.Y:      rects[i][1],
            RECT_OPTION.WIDTH:  rects[i][2],
            RECT_OPTION.HEIGHT: rects[i][3],
            RECT_OPTION.AREA:   rects[i][2] * rects[i][3]
        }

        curV = switcher.get(option, "Invalid Option")
        if curV < maxV:
            maxV = curV
            beta_index = i
    
    return beta_index;
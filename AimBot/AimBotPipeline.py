import cv2
import time
import numpy as np
import VisionUtils
from VisionUtils import RECT_OPTION
from VisionUtils import sortRectsByMaxOption
from VisionUtils import mergeRects
from VisionUtils import calcAveColorInRect
from MathUtils import clip


class AimBotPipeline:
    def __init__(self, MIN_HSV, MAX_HSV):
        self.MIN_HSV        = MIN_HSV
        self.MAX_HSV        = MAX_HSV
        self.IMG_HEIGHT     = 0
        self.IMG_WIDTH      = 0
        self.marginMatrix   = [25, 90, 50]


        self.red            = (0, 0, 255)
        self.green          = (0, 255, 0)
        self.thickness      = 2
        self.font           = cv2.FONT_HERSHEY_COMPLEX

        self.goalFound      = False
        self.goalDegreeError = 0
        self.goalDistance   = 0
        self.goalRect       = [0, 0, 0, 0]

        self.DEBUG_MODE_ON  = True
        self.INIT_COMPLETED = False
        self.START_SECONDS  = time.time()
        self.INIT_SECONDS   = 5



    def processFrame(self, input):

        self.INIT_COMPLETED = (time.time() - self.START_SECONDS) > self.INIT_SECONDS
        output = self.regPipe(input) if self.INIT_COMPLETED else self.initPipe(input)
        return output


    def regPipe(self, input):
                
        output = input

        # Get sizing of image
        self.IMG_HEIGHT, self.IMG_WIDTH, _ = input.shape

        # Convert & Blur
        HSVInput = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(HSVInput, (35, 35), 0)

        # Threshold
        thresh = cv2.inRange(blur, self.MIN_HSV, self.MAX_HSV)

        # Erode & Dilate
        eroded = cv2.erode(thresh, (5, 5))
        dilated = cv2.dilate(eroded, (5, 5))

        # Get contours and error check
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) == 0):
            self.goalFound = False
            self.goalDegreeError = 0
            self.goalDistance = 0
            return output
        self.goalFound = True

        # Convert to rects
        rects = [cv2.boundingRect(contour) for contour in contours]

        # Auto-Update HSV midgame
        #self.updateHSV(output, rects)

        # Heuristics to get goalRect
        largestRects = sortRectsByMaxOption(2, RECT_OPTION.AREA, rects)
        self.goalRect = mergeRects(largestRects)

        # Get center
        gx, gy, gw, gh = self.goalRect
        cx = gx + (gw // 2)
        cy = gy + (gh // 2)
        center = (cx, cy)

        # Calculate Error
        pixelError = (self.IMG_WIDTH // 2) - cx
        self.goalDegreeError = pixelError
        self.goalDistance = pixelError * 10

        # Logging Shapes & Degree & Pixel Data
        cv2.rectangle(output, (gx, gy), (gx + gw, gy + gh), self.green, self.thickness)
        cv2.line(output, (cx, cy), (cx + pixelError, cy),   self.red,   self.thickness)
        coords = str("(" + str(cx) + ", " + str(cy) + ")")
        output = cv2.putText(output, coords, (cx, cy), self.font, 0.5, (0, 255, 0)) 


        return thresh if self.DEBUG_MODE_ON else output

        

    def initPipe(self, input):
        # Run for 5s

        output = input

        # Get sizing of image
        self.IMG_HEIGHT, self.IMG_WIDTH, _ = input.shape

        # Convert & Blur
        HSVInput = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(HSVInput, (35, 35), 0)

        # Get center rect
        sideLength = 50
        x = (self.IMG_WIDTH//2) - sideLength//2
        y = (self.IMG_HEIGHT//2) - sideLength//2
        initRect = [x, y, x+sideLength, y+sideLength]

        # Calculate average HSV within initRect     Note: updateHSV expects an array of rectangles
        self.updateHSV(blur, [initRect])

        # Log rect for driver-placement
        output = cv2.rectangle(output, (x,y), (x+sideLength, y+sideLength), self.green, self.thickness)

        # Log time left
        timeLeft = str(round(time.time() - self.START_SECONDS))
        mx = self.IMG_WIDTH - 100
        my = self.IMG_HEIGHT - 100
        output = cv2.putText(output, timeLeft, (mx, my), self.font, 0.5, (0, 255, 0)) 

        return output


    def updateHSV(self, frame, rects):
        aveHSV = [0, 0, 0]
        for rect in rects:
            tmpAveHSV = calcAveColorInRect(frame, rect)
            aveHSV = [round(aveHSV[i] + tmpAveHSV[i]) for i in range(3)] 
        aveHSV = [aveHSV[i] // 2 for i in range(3)] if len(rects) == 2 else aveHSV

        # Make thresholds and clip if necessary
        self.MAX_HSV = tuple([clip(aveHSV[i] + self.marginMatrix[i], 0, 255) for i in range(3)])
        self.MIN_HSV = tuple([clip(aveHSV[i] - self.marginMatrix[i], 0, 255) for i in range(3)])

        print("Ave HSV: " + str(aveHSV[0]) + "   " + str(aveHSV[1]) + "   " + str(aveHSV[2]))
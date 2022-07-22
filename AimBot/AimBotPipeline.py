import cv2
import time
import numpy as np
import VisionUtils
from VisionUtils import RECT_OPTION
from VisionUtils import sortRectsByMaxOption
from VisionUtils import mergeRects
from VisionUtils import calcAveColorInRect
from VisionUtils import submat
from MathUtils import clip
from copy import deepcopy


class AimBotPipeline:
    def __init__(self, MIN_HSV, MAX_HSV):
        self.MIN_HSV1           = MIN_HSV
        self.MAX_HSV1           = MAX_HSV        
        self.MIN_HSV2           = MIN_HSV
        self.MAX_HSV2           = MAX_HSV
        self.IMG_HEIGHT         = 0
        self.IMG_WIDTH          = 0
        self.marginMatrix1       = [10, 10, 10]
        self.marginMatrix2       = [10, 10, 10]
        self.CONVERSION         = cv2.COLOR_BGR2HSV #cv2.COLOR_BGR2YCbCr
        self.sideLength         = 20
        self.EXTEND_MARGIN      = 50
        self.HSV2_RATE          = 1
        self.aveHSV             = [0, 0, 0]

        self.red                = (0, 0, 255)
        self.green              = (0, 255, 0)
        self.thickness          = 2
        self.font               = cv2.FONT_HERSHEY_COMPLEX

        self.goalFound          = False
        self.goalDegreeError    = 0
        self.goalDistance       = 0
        self.goalRect           = [0, 0, 0, 0]

        self.DEBUG_MODE_ON      = False
        self.INIT_COMPLETED     = False
        self.START_SECONDS      = time.time()
        self.INIT_SECONDS       = 5



    def processFrame(self, input):

        self.INIT_COMPLETED = (time.time() - self.START_SECONDS) > self.INIT_SECONDS
        output = self.regPipe(input) if self.INIT_COMPLETED else self.initPipe(input)
        return output


    def regPipe(self, input):
                
        output = deepcopy(input)

        # Get sizing of image
        self.IMG_HEIGHT, self.IMG_WIDTH, _ = input.shape

        # Convert & Blur
        converted = cv2.cvtColor(input, self.CONVERSION)
        blur = cv2.GaussianBlur(converted, (35, 35), 0)

        # Threshold
        thresh = cv2.inRange(blur, self.MIN_HSV1, self.MAX_HSV2)

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
        #self.updateThresh(output, rects)

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
        cv2.putText(output, coords, (cx, cy), self.font, 0.5, (0, 255, 0)) 


        return thresh if self.DEBUG_MODE_ON else output



    def regPipeV2(self, input):
                
        output = deepcopy(input)

        # Get sizing of image
        self.IMG_HEIGHT, self.IMG_WIDTH, _ = input.shape

        # Convert & Blur
        converted = cv2.cvtColor(input, self.CONVERSION)
        blur = cv2.GaussianBlur(converted, (35, 35), 0)

        # Save copy
        input2 = deepcopy(blur)

        # Threshold
        thresh = cv2.inRange(blur, self.MIN_HSV1, self.MAX_HSV1)

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
        #self.updateThresh(output, rects)

        # Heuristics to get goalRect
        largestRect = sortRectsByMaxOption(1, RECT_OPTION.AREA, rects)[0]
        largestRect = np.asarray(largestRect)
        # x, y, w, h
        largestRect[0] = clip(largestRect[0] - self.EXTEND_MARGIN//2, 0, 300)
        largestRect[1] = clip(largestRect[0] - self.EXTEND_MARGIN//2, 0, 300)
        largestRect[2] = clip(largestRect[2] + self.EXTEND_MARGIN, 0, 300)
        largestRect[3] = clip(largestRect[3] + self.EXTEND_MARGIN, 0, 300)



        input2 = submat(input2, largestRect)
        output = submat(output, largestRect)

        # Threshold2
        thresh2 = cv2.inRange(input2, self.MIN_HSV2, self.MAX_HSV2)

        # Erode2 & Dilate2
        eroded2 = cv2.erode(thresh2, (5, 5))
        dilated2 = cv2.dilate(eroded2, (5, 5))

        # Get contours
        contours2, hierarchy2 = cv2.findContours(dilated2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) == 0):
            print("Cannot find contours")
            return output


        # If we haven't properly thresholded
        #if (len(contours2) > 1):

        # Expand margins
        self.marginMatrix2 = [val + self.HSV2_RATE for val in self.marginMatrix2]

        # Make thresholds and clip if necessary
        self.MAX_HSV2 = tuple([clip(aveHSV[i] + self.marginMatrix1[i], 0, 255) for i in range(3)])
        self.MIN_HSV2 = tuple([clip(aveHSV[i] - self.marginMatrix1[i], 0, 255) for i in range(3)])

        print("MIN HSV2: " + str(self.MIN_HSV2[0]) + "   " + str(self.MIN_HSV2[1]) + "   " + str(self.MIN_HSV2[2]))
        print("MAX HSV2: " + str(self.MAX_HSV2[0]) + "   " + str(self.MAX_HSV2[1]) + "   " + str(self.MAX_HSV2[2]))



        """ 
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
        cv2.putText(output, coords, (cx, cy), self.font, 0.5, (0, 255, 0))  """


        return thresh if self.DEBUG_MODE_ON else output



        

    def initPipe(self, input):
        # Run for 5s

        output = deepcopy(input)

        # Get sizing of image
        self.IMG_HEIGHT, self.IMG_WIDTH, _ = input.shape

        # Convert & Blur
        converted = cv2.cvtColor(input, self.CONVERSION)
        blur = cv2.GaussianBlur(converted, (35, 35), 0)

        # Get center rect
        x = (self.IMG_WIDTH//2) - self.sideLength//2 # (Center of image - half a sidelength)
        #y = (self.IMG_HEIGHT//2) - self.sideLength//2

        y = 120
        h = 50
        initRect = [x, y, self.sideLength, h]

        # Calculate average HSV within initRect     Note: updateHSV expects an array of rectangles
        self.updateThresh(blur, [initRect])

        # Log rect for driver-placement
        cv2.rectangle(output, (x,y), (x+self.sideLength, y+self.sideLength), self.green, self.thickness)

        # Log time left
        timeLeft = str(round(time.time() - self.START_SECONDS))
        mx = self.IMG_WIDTH - 100
        my = self.IMG_HEIGHT - 100
        cv2.putText(output, timeLeft, (mx, my), self.font, 0.5, (0, 255, 0)) 

        return output


    def updateThresh(self, frame, rects):
        rectLength = len(rects)
        if (rectLength > 0):
            aveHSV = [0, 0, 0]
            for rect in rects:
                # Get average HSV of frame
                tmpAveHSV = calcAveColorInRect(frame, rect)

                # Set aveHSV[0] to the sum of aveHSV[0] + tmpAveHSV[0] 
                aveHSV = [round(aveHSV[i] + tmpAveHSV[i]) for i in range(3)] 

            # We summed up all respective averages for H, S, and V, or Y, Cb, Cr
            # We need to divide by however many rectangles we summed together, assuming not dividing by 0
            aveHSV = [aveHSV[i] // rectLength for i in range(3)]

            # Make thresholds and clip if necessary
            self.MAX_HSV1 = tuple([clip(aveHSV[i] + self.marginMatrix1[i], 0, 255) for i in range(3)])
            self.MIN_HSV1 = tuple([clip(aveHSV[i] - self.marginMatrix1[i], 0, 255) for i in range(3)])

            # Make thresholds and clip if necessary
            self.MAX_HSV2 = tuple([clip(aveHSV[i] + self.marginMatrix1[i], 0, 255) for i in range(3)])
            self.MIN_HSV2 = tuple([clip(aveHSV[i] - self.marginMatrix1[i], 0, 255) for i in range(3)])

            print("Ave HSV: " + str(aveHSV[0]) + "   " + str(aveHSV[1]) + "   " + str(aveHSV[2]))
        

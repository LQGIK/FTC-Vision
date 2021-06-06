import numpy as np
import cv2

# 25 margin
MAX_HUE = 125
MIN_HUE = 100

# 90 margin
MAX_SAT = 255
MIN_SAT = 165

# 110 margin
MAX_VAL = 200
MIN_VAL = 90 

MIN_HSV = (MIN_HUE, MIN_SAT, MIN_VAL)
MAX_HSV = (MAX_HUE, MAX_SAT, MAX_VAL)
FOV = 1.2
IMG_WIDTH, IMG_HEIGHT, error = 0, 0, 0
color = (0, 255, 0)
thickness = 2
font = cv2.FONT_HERSHEY_COMPLEX

def contourPipeline(input, contours):

    # Going through every contours found in the image. 
    for cnt in contours : 
    
        # Epsilon decides how many inner lines there are, how interconnected it should appear
        epsilonPercent = 0.05
        approx = cv2.approxPolyDP(cnt, epsilonPercent * cv2.arcLength(cnt, True), True) 
    
        # draws boundary of contours. 
        cv2.drawContours(input, [approx], 0, (0, 0, 255), 5)  
    
        # Used to flatted the array containing 
        # the co-ordinates of the vertices. 
        n = approx.ravel()  
        i = 0
    
        for j in n : 
            if (i % 2 == 0): 
                x = n[i] 
                y = n[i + 1] 
    
                # String containing the co-ordinates. 
                string = str(x) + " " + str(y)  
    
                if (i == 0): 
                    # text on topmost co-ordinate. 
                    cv2.putText(input, "Arrow tip", (x, y), 
                                    font, 0.5, (255, 0, 0))  
                else: 
                    # text on remaining co-ordinates. 
                    cv2.putText(input, string, (x, y),  
                            font, 0.5, (0, 255, 0))  
            i = i + 1
    return input


def addTargetBox(input, side_length):
    center_x = IMG_WIDTH / 2
    center_y = IMG_HEIGHT / 2
    half_side_length = side_length / 2
    upper_left = (int(center_x - half_side_length), int(center_y - half_side_length))
    bottom_right = (int(center_x + half_side_length), int(center_y + half_side_length))
    output = cv2.rectangle(input, upper_left, bottom_right, (0, 0, 255), 2)

    return output

def pixels2Deg(pixels):
    deg = pixels * (FOV / IMG_WIDTH)
    return deg

def drawBoundingBoxes(input, contours):
    output = input
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input, (x,y), (x+w, y+h), (0, 255, 0), 2)

    return output

def findLargestContourIndex(contours):
    maxArea = 0
    maxIndex = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxIndex = i
    return maxIndex

def findNLargestContours(n, contours):
    new_contours = []
    for i in range(n):
        li = findLargestContourIndex(contours)
        new_contours.append(contours[li])
        
        contours.pop(li)
        if len(contours) == 0:
            break 
    return new_contours

def getGoalRect(new_contours):

    # If there is one contour, return first
    x, y, w, h = cv2.boundingRect(new_contours[0])

    # If there are two contours, extrapolate
    if len(new_contours) == 2:

        # If we have two boxes or more, retrieve overarching rectanlge
        xl, yl, wl, hl = 0, 0, 0, 0 
        xr, yr, wr, hr = 0, 0, 0, 0

        # Get bounding rectangles of both
        x1, y1, w1, h1 = cv2.boundingRect(new_contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(new_contours[1])

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
 


def goal_pipeline(input):

    global MIN_HSV, MAX_HSV

    # Set output to input
    output = input

    # Convert to HSV
    hsv_frame = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)

    # Blurring
    blur = cv2.GaussianBlur(hsv_frame, (35, 35), 0)

    # Thresholding
    thresh = cv2.inRange(blur, MIN_HSV, MAX_HSV)

    # Erosion and Dilation
    eroded = cv2.erode(thresh, (5, 5))
    dilated = cv2.dilate(eroded, (5, 5))

    # Get contours and error check
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 0):
        return input

    # Get two largest contours (might return less than 2 contours)
    new_contours = findNLargestContours(2, contours)
    
    # Get rects
    rects = []
    for contour in new_contours:
        rects.append(cv2.boundingRect(contour))



    aveH = 0
    aveS = 0
    aveV = 0
    for rect in rects:
        x, y, w, h = rect
        submat = hsv_frame[y:y+h,x:x+w,:]
        h_frame = submat[:,:,0]
        s_frame = submat[:,:,1]
        v_frame = submat[:,:,2]

        tmpAveH = np.mean(h_frame)
        tmpAveS = np.mean(s_frame)
        tmpAveV = np.mean(v_frame)

        aveH += tmpAveH
        aveS += tmpAveS
        aveV += tmpAveV

    aveH /= 2
    aveS /= 2
    aveV /= 2

    print("Ave HSV: " + str(aveH) + "   " + str(aveS) + "   " + str(aveV))

    '''
    # 25 margin
    MAX_HUE = 125
    MIN_HUE = 100

    # 90 margin
    MAX_SAT = 255
    MIN_SAT = 165

    # 110 margin
    MAX_VAL = 200
    MIN_VAL = 90 
    '''

    hMOE = 25
    sMOE = 90
    vMOE = 110

    hMin = aveH - hMOE
    hMax = aveH + hMOE

    sMin = aveS - sMOE
    sMax = aveS + sMOE

    vMin = aveV - vMOE
    vMax = aveV + vMOE

    hMin = 0 if hMin < 0 else hMin
    sMin = 0 if sMin < 0 else sMin
    vMin = 0 if vMin < 0 else vMin

    hMax = 255 if hMax > 255 else hMax
    sMax = 255 if sMax > 255 else sMax
    vMax = 255 if vMax > 255 else vMax

    MAX_HSV = [hMax, sMax, vMax]
    MIN_HSV = [hMin, sMin, vMin]

    # Get coords of goal rectangle, if one, return the first contour, if two, extrapolate
    x, y, w, h = getGoalRect(new_contours) 

    # Draw goal rectangle
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Calculate error
    global error
    center_x = x + (w//2)
    center_y = y + (h//2)
    pixel_error = (IMG_WIDTH//2) - center_x
    error = pixels2Deg(pixel_error)
    cv2.line(output, (center_x, center_y), (center_x + pixel_error, center_y), (0, 0, 255), thickness)

    # Log center
    global font
    coords = str("(" + str(center_x) + ", " + str(center_y) + ")")
    output = cv2.putText(output, coords, (center_x, center_y),  
                            font, 0.5, (0, 255, 0))  

    # Draw two box contours
    output = drawBoundingBoxes(output, new_contours)

    
    return output


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_EXPOSURE,-5)

    # Get image dimensions
    global IMG_HEIGHT, IMG_WIDTH, error
    _, frame = cap.read()
    IMG_HEIGHT, IMG_WIDTH, _ = frame.shape

    while (True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Run processing pipeline on image
        output = goal_pipeline(frame)

        # Flip the frame just so it's nice
        flipHorizontal = cv2.flip(output, 1)

        # Log the error on screen
        flipHorizontal = cv2.putText(flipHorizontal, "Error: " + str(error), (IMG_WIDTH - 200, IMG_HEIGHT - 100), font, 0.5, (0, 255, 0))

        # Display the resulting frame
        cv2.imshow('frame', flipHorizontal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
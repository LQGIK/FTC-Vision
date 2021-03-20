import numpy as np
import cv2

MAX_HUE = 120
MIN_HUE = 100

MAX_SAT = 255
MIN_SAT = 165

MAX_VAL = 200
MIN_VAL = 90 

MIN_HSV = (MIN_HUE, MIN_SAT, MIN_VAL)
MAX_HSV = (MAX_HUE, MAX_SAT, MAX_VAL)

IMG_WIDTH = 0
IMG_HEIGHT = 0

font = cv2.FONT_HERSHEY_COMPLEX
error = 0

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


def getTwoLargestContours(contours):
    contourAreas = [cv2.contourArea(cnt) for cnt in contours]
    outer_i = contourAreas.index(max(contourAreas))
    contourAreas.pop(outer_i)
    inner_i = contourAreas.index(max(contourAreas))
    if (outer_i <= inner_i):
        inner_i += 1
    goalContours = [contours[outer_i], contours[inner_i]]

    return goalContours


def drawBoundingBoxes(input, contours):
    output = input
    goalContours = list(contours)
    if len(goalContours) > 2:
        goalContours = getTwoLargestContours(contours)

    for cnt in goalContours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        output = cv2.rectangle(input, (x,y), (x+w, y+h), (0, 255 ,0), 2)

    return output

def hsv_pipeline(input):

    # Convert to HSV
    hsv_frame = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)

    # Blurring
    blur = cv2.GaussianBlur(hsv_frame, (35, 35), 0)

    # Thresholding
    thresh = cv2.inRange(blur, (100, 0, 0), (120, 255, 255))

    eroded = cv2.erode(thresh, (5, 5))
    dilated = cv2.dilate(eroded, (5, 5))

    # Get contours and error check
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 0):
        return input

    # Draw contours
    output = drawBoundingBoxes(input, contours)

    # Get error
    global error
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x + (w//2)
        center_y = y + (h//2)
        center = (center_x, center_y)
        error += (IMG_WIDTH//2) - center_x
    error /= len(contours)

    return output


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_EXPOSURE,-1)

    # Get image dimensions
    global IMG_HEIGHT, IMG_WIDTH, error
    _, frame = cap.read()
    IMG_HEIGHT, IMG_WIDTH, _ = frame.shape

    while (True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Run processing pipeline on image
        output = hsv_pipeline(frame)

        # Flip the frame just so it's nice
        flipHorizontal = cv2.flip(output, 1)

        # Log the error on screen
        flipHorizontal = cv2.putText(flipHorizontal, str(error), (IMG_WIDTH - 100, IMG_HEIGHT - 100), font, 1, (0, 255, 0))

        # Display the resulting frame
        cv2.imshow('frame', flipHorizontal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
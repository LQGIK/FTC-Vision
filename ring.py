import numpy as np
import cv2

MAX_HUE = 130
MIN_HUE = 90

MAX_SAT = 220
MIN_SAT = 180

MAX_VAL = 80
MIN_VAL = 45

MIN_HSV = (MIN_HUE, MIN_SAT, MIN_VAL)
MAX_HSV = (MAX_HUE, MAX_SAT, MAX_VAL)

IMG_WIDTH = 0
IMG_HEIGHT = 0

font = cv2.FONT_HERSHEY_COMPLEX
ring_count = 0
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
    output = cv2.rectangle(input, upper_left, bottom_right, (0, 255, 0), 2)

    return output


def drawBoundingBoxes(input, contours):
    output = input
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        output = cv2.rectangle(input, (x,y), (x+w, y+h), (0, 255, 0), 2)

    return output


def findWidestContourIndex(contours):
    maxWidth = 0
    maxIndex = 0
    for i in range(len(contours)):
        cnt = contours[i]
        _, _, w, _ = cv2.boundingRect(cnt)
        if w > maxWidth:
            maxWidth = w
            maxIndex = i
    return maxIndex

def findNWidestContours(n, contours):
    new_contours = []
    for i in range(n):
        li = findWidestContourIndex(contours)
        new_contours.append(contours[li])
        
        contours.pop(li)
        if len(contours) == 0:
            break 
    return new_contours


def ring_pipeline(input):

    global IMG_HEIGHT, IMG_WIDTH
    IMG_HEIGHT, IMG_WIDTH, _ = input.shape


    # Set output to input
    output = input

    # Convert to HSV
    hsv_frame = cv2.cvtColor(input, cv2.COLOR_BGR2YCrCb)

    # Blurring
    blur = cv2.GaussianBlur(hsv_frame, (35, 35), 0)

    # Thresholding
    thresh = cv2.inRange(blur, MIN_HSV, MAX_HSV)

    # Erosion and Dilation
    eroded = cv2.erode(thresh, (5, 5))
    dilated = cv2.dilate(eroded, (5, 5))

    # Get contours and error check
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # We have at least one ring
    global ring_count
    if (len(contours) > 0):
        # Get two largest contours (might return less than 2 contours)
        new_contours = findNWidestContours(1, contours)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(new_contours[0])

        # Calculate error
        global error
        center_x = x + (w//2)
        center_y = y + (h//2)
        error = (IMG_WIDTH//2) - center_x

        # Log center
        global font
        coords = str("(" + str(center_x) + ", " + str(center_y) + ")")

        output = cv2.putText(output, coords, (center_x, center_y),  
                                font, 0.5, (0, 255, 0))  


        # Draw two box contours
        output = drawBoundingBoxes(output, new_contours)

    


        if h < (0.5 * w):
            ring_count = 1
        else:
            ring_count = 4

    else:
        ring_count = 0


    addTargetBox(output, 2)



    return output


def channel_pipeline(input, n):

    # Convert to HSV
    hsv_frame = cv2.cvtColor(input, cv2.COLOR_BGR2YCrCb)

    # Extract channel
    channel = hsv_frame[:,:,n]

    return channel 


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_EXPOSURE,-1)

    # Get image dimensions
    global IMG_HEIGHT, IMG_WIDTH, error
    _, frame = cap.read()

    while (True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Run processing pipeline on image
        frame = cv2.imread("./images/4.png")
        output = ring_pipeline(frame)

        # Flip the frame just so it's nice
        flipHorizontal = cv2.flip(output, 1)

        # Log the error on screen
        flipHorizontal = cv2.putText(flipHorizontal, "Error: " + str(error), (IMG_WIDTH - 100, IMG_HEIGHT - 50), font, 0.5, (0, 255, 0))
        flipHorizontal = cv2.putText(flipHorizontal, "Rings: " + str(ring_count), (IMG_WIDTH - 100, IMG_HEIGHT - 30), font, 0.5, (0, 255, 0))


        cv2.imshow('frame', flipHorizontal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
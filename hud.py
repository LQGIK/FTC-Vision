import numpy as np
import cv2

FOV = 1.2 
IMG_WIDTH, IMG_HEIGHT, error = 0, 0, 0
color = (0, 255, 0)
thickness = 2

def pixel2Deg(x):
    global FOV, IMG_WIDTH
    deg = x * (FOV / IMG_WIDTH)
    return deg

def deg2Pixels(deg):
    global FOV, IMG_WIDTH
    pixels = deg * (IMG_WIDTH / FOV)
    return pixels 

def hud_pipeline(input):
    output = input

    # Get width and height of image
    global IMG_WIDTH, IMG_HEIGHT
    IMG_HEIGHT, IMG_WIDTH, _ = input.shape

    # Calculate center
    center_x = IMG_WIDTH//2
    center_y = IMG_HEIGHT//2

    # Make Lines
    global color, thickness
    cv2.line(output, (center_x, 0), (center_x, IMG_HEIGHT), color, thickness)
    cv2.line(output, (0+1, 0), (0+1, IMG_HEIGHT), color, thickness)
    cv2.line(output, (IMG_WIDTH-1, 0), (IMG_WIDTH-1, IMG_HEIGHT), color, thickness)


    return output


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
        output = hud_pipeline(frame)

        # Flip the frame just so it's nice
        flipHorizontal = cv2.flip(output, 1)

        # Show screen
        cv2.imshow('HUD', flipHorizontal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
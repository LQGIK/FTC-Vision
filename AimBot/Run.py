import cv2
from AimBotPipeline import AimBotPipeline

def main():
    # Initialize cap object
    cap = cv2.VideoCapture(0)

    # Only for LGG-Laptop
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    # Read in initial frame
    _, frame = cap.read()

    # Initialize pipeline
    MAX_HSV = (125, 255, 200)
    MIN_HSV = (100, 165, 90)
    # H: 25     S: 90      V:110

    pipe = AimBotPipeline(MIN_HSV, MAX_HSV)

    while (True):

        # Capture frame-by-frame
        ret, frame = cap.read()


        '''

            PIPELINE HERE

                          '''
        output = pipe.processFrame(frame)


        # Display the resulting frame
        cv2.imshow('frame', cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
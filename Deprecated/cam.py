import cv2

def main():
    cap = cv2.VideoCapture(0)

    # Only for LGG-Laptop
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    _, frame = cap.read()

    while (True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
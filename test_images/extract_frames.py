import numpy as np
import cv2

if __name__=="__main__":
    cap = cv2.VideoCapture('../challenge_video.mp4')
    nframe = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print("Screenshot saved.")
                cv2.imwrite('screenshot_ch{}.jpg'.format(nframe), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        nframe += 1

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
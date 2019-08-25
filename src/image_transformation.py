import glob

import cv2
import numpy as np

CALIBRATION_IMAGES_PATTERN = '../camera_cal/calibration*.jpg'

# Camera calibration with chess images
def camera_calibration(show_process=True, img_size=(1280, 720)):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(CALIBRATION_IMAGES_PATTERN)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if show_process:
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    if show_process:
        cv2.destroyAllWindows()

    if len(objpoints) > 0:
        retval, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        return camera_matrix, dist_coeffs
    else:
        return None, None


def correct_distortion(img, camera_matrix, dist_coeffs):
    undist = cv2.undistort(img, camera_matrix, dist_coeffs)
    return undist


def warp_image(img, src, xmargin = 250, ymargin=15):
    imgshape = (img.shape[1], img.shape[0])
    dst = np.float32([[xmargin, ymargin],
                      [imgshape[0] - xmargin, ymargin],
                      [imgshape[0] - xmargin, imgshape[1] - ymargin],
                      [xmargin, imgshape[1] - ymargin]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (imgshape[0], imgshape[1]), flags=cv2.INTER_LINEAR)
    return M, warped

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from image_transformation import camera_calibration, correct_distortion, warp_image
from thresholds import apply_thresholds
from lane_pixels import find_lane_pixels, search_around_poly
from curvature import measure_curvature_real


def pipeline(image, camera_matrix, dist_coeffs, previous_data=False, previous_left_fit=None, previous_right_fit=None):
    # 2- Apply a distortion correction to raw images.
    undist = correct_distortion(image, camera_matrix, dist_coeffs)

    # 3- Binary image (thresholding w/ color transforms, gradients, etc.)
    binary_image = apply_thresholds(undist)

    # 4- Perspective transform (birds-eye view)
    src = np.float32([[560, 475], [725, 475], [1010, 660], [295, 660]])
    M, binary_warped = warp_image(binary_image, src)
    # 5- Detect lane pixels.
    left_fitx, right_fitx, ploty, img_res, _, _ = find_lane_pixels(binary_warped)

    # 6- Determine curvature and position with respect to center.
    curvature = measure_curvature_real(left_fitx, right_fitx, ploty)

    # 7- Warp lane boundaries into original image.
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # 8- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # Curvature and offset
    center_x = np.mean(pts[:, :, 0])
    offset = (image.shape[1] // 2)-center_x
    offset_dir = "left" if offset<0 else "right"
    result = cv2.putText(result, "Curvature: {}".format(curvature), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    result = cv2.putText(result, "Offset: {} ({}})".format(np.abs(offset), offset_dir), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # plt.imshow(result)

    return result


if __name__=="__main__":
    # img = mpimg.imread("../test_images/straight_lines1.jpg")
    img = mpimg.imread("../test_images/test1.jpg")
    # 1- Camera calibration matrix and distorsion coefficeint give a chess of chessboard images
    camera_matrix, dist_coeffs = camera_calibration(show_process=False)
    res = pipeline(img, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    # plt.imshow(res)

    print("Finished")



import numpy as np
import cv2


# Define a class to receive the characteristics of each line detection
from curvature import measure_curvature_real
from image_transformation import camera_calibration, correct_distortion, warp_image
from lane_pixels import find_lane_pixels, search_around_poly
from thresholds import apply_thresholds


class Line():
    def __init__(self, n, detected, n_undetected, n_last_fits, current_fit):
        self.n = n

        # was the line detected in the last iteration?
        self.detected = detected
        self.n_undetected = n_undetected
        if not detected:
            self.n_undetected += 1

        # Custom
        self.n_last_fits = n_last_fits[-n:]

        if detected and current_fit is not None:
            self.n_last_fits.append(current_fit)

        self.best_fit = np.median(self.n_last_fits, axis=0)


def binarize_image(image, camera_matrix, dist_coeffs):
    # 2- Apply a distortion correction to raw images.
    undist = correct_distortion(image, camera_matrix, dist_coeffs)

    # 3- Binary image (thresholding w/ color transforms, gradients, etc.)
    binary_image = apply_thresholds(undist)
    return binary_image*255


def pipeline_video(image, camera_matrix, dist_coeffs, prev_left, prev_right):
    # 2- Apply a distortion correction to raw images.
    undist = correct_distortion(image, camera_matrix, dist_coeffs)

    # 3- Binary image (thresholding w/ color transforms, gradients, etc.)
    binary_image = apply_thresholds(undist)

    # 4- Perspective transform (birds-eye view)
    src = np.float32([[560, 475], [725, 475], [1010, 660], [295, 660]])
    M, binary_warped = warp_image(binary_image, src)
    # 5- Detect lane pixels.
    if prev_left.detected and prev_right.detected:
        left_fitx, right_fitx, ploty, img_res, left_fit, right_fit = search_around_poly(binary_warped,
                                                                                        prev_left.best_fit,
                                                                                        prev_right.best_fit)
        detected = True
    else:
        detected, left_fitx, right_fitx, ploty, img_res, left_fit, right_fit = find_lane_pixels(binary_warped)

    left_line = Line(prev_left.n, detected, prev_left.n_undetected, prev_left.n_last_fits, left_fit)
    right_line = Line(prev_right.n, detected, prev_right.n_undetected, prev_right.n_last_fits, right_fit)

    if detected:
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
        offset = np.abs((image.shape[1] // 2)-center_x)
        result = cv2.putText(result, "Curvature: {}".format(curvature), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        result = cv2.putText(result, "Offset: {}".format(offset), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # plt.imshow(result)
    else:
        result = undist
        result = cv2.putText(result, "Lines not detected", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return result, left_line, right_line


if __name__=="__main__":
    camera_matrix, dist_coeffs = camera_calibration(show_process=False)
    n = 5

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('../project_video.mp4')

    # Define the codec and create VideoWriter object
    #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output2.mp4  ',fourcc, 20.0, (1280, 720))
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280,720))

    left_line = Line(n, False, 0, [], None)
    right_line = Line(n, False, 0, [], None)

    process_frame = True
    nframe = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # frame = cv2.flip(frame,0)
            if process_frame:
                processed_frame, left_line, right_line = pipeline_video(frame, camera_matrix, dist_coeffs, left_line, right_line)
                # processed_frame = binarize_image(frame, camera_matrix, dist_coeffs)
            else:
                processed_frame = frame

            # write the processed frame
            out.write(processed_frame)

            cv2.imshow('frame', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                print("Triggering processing")
                process_frame = not process_frame
                left_line = Line(n, 0, False, [], None)
                right_line = Line(n, 0, False, [], None)
                continue

        else:
            break
        nframe += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

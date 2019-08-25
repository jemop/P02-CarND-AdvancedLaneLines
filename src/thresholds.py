import cv2
import numpy as np


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh = (0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1*(orient == "x"), 1*(orient == "y"), ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.abs(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude = np.sqrt(sobel_x**2+sobel_y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(255*magnitude/np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    sobel_x_abs = np.abs(sobel_x)
    sobel_y_abs = np.abs(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(sobel_y_abs, sobel_x_abs)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir>=thresh[0]) & (grad_dir<=thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def hls_extract(img, channel):
    # 1) Convert to HLS color space
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls_image[:, :, channel]


def hsv_extract(img, channel):
    # 1) Convert to HSV color space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv_image[:, :, channel]


def hls_threshold(img, channel=2, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros(hls_image[:, :, 2].shape)
    binary_output[(hls_image[:, :, channel]>thresh[0]) & (hls_image[:, :, channel] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def hsv_threshold(img, channel=2, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros(hls_image[:, :, 2].shape)
    binary_output[(hls_image[:, :, channel]>thresh[0]) & (hls_image[:, :, channel] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def apply_single_threshold(img, thresh=(0,255)):
    binary_output = np.zeros(img.shape)
    binary_output[(img[:, :] > thresh[0]) & (img[:, :] <= thresh[1])] = 1
    return binary_output


# Edit this function to create your own pipeline.
def apply_thresholds(image):
    sobel_kernel = 15
    mag_kernel = 9
    dir_kernel = 15
    abs_threshold = (10, 70)
    mag_threshold = (10, 70)
    dir_threshold = (0.7, 1.3)
    s_threshold = (170, 255)
    h_threshold = (0, 20)

    img = np.copy(image)

    # Gradient thresholding
    sobelx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=sobel_kernel, thresh=abs_threshold)
    sobely_binary = abs_sobel_thresh(img, orient='y', sobel_kernel=sobel_kernel, thresh=abs_threshold)
    mag_binary = mag_thresh(img, sobel_kernel=mag_kernel, mag_thresh=mag_threshold)
    dir_binary = dir_thresh(img, sobel_kernel=dir_kernel, thresh=dir_threshold)
    gradient_binary = np.zeros_like(dir_binary)
    # gradient_binary[((sobelx_binary == 1) & (sobely_binary == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1  # Example
    # gradient_binary[(sobelx_binary == 1) & (sobely_binary == 1)] = 1  # Example
    gradient_binary[((mag_binary == 1) & (dir_binary == 1))] = 1  # Example
    # combined_gradient_bin[(sobelx_binary==1)] = 1  # Example


    # Threshold color channel
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_binary = apply_single_threshold(hsv_image[:,:,2], thresh=(220, 255))
    s_binary = hls_threshold(img, 2, s_threshold)
    h_binary = hls_threshold(img, 0, h_threshold)
    color_binary =np.zeros_like(s_binary)
    # color_binary = np.copy(s_binary)
    color_binary[(v_binary==1) | (s_binary==1)] = 1

    # Stack each channel
    # binary_img = np.uint8(np.dstack((np.zeros_like(combined_gradient_bin), combined_gradient_bin, color_binary)) * 255)

    # binary_img = np.copy(gradient_binary)
    binary_img = np.zeros_like(gradient_binary)
    # binary_img[(gradient_binary == 1) & (color_binary == 1)] = 1
    binary_img[(color_binary==1) & (gradient_binary==1)] = 1

    return binary_img

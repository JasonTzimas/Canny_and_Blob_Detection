import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from include.functions import *
import os
import sys
import itertools
import argparse
from scipy.signal import convolve2d



def main(input_image, out_folder):
    root = os.getcwd()
    path_name = out_folder
    full_path = os.path.join(root, path_name)
    os.makedirs(full_path, exist_ok=True)

    root = os.getcwd()
    output_path = os.path.join(root, out_folder)

    # Read and show one image
    im1 = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    axs.imshow(im1, cmap="gray")
    axs.set_axis_off()
    fig.suptitle("Original Image", fontsize=16)
    name = "original_image.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)


    # Create X,Y Sobel filters and convolve the Image with them
    SobX = np.array([[1, 0 , -1], [2, 0, -2], [1, 0, -1]])
    SobY = SobX.T

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    dX = convolve2d(im1, SobX, mode="full", fillvalue=0)
    axs[0].imshow(dX, cmap="gray")
    dY = convolve2d(im1, SobY, mode="full", fillvalue=0)
    axs[1].imshow(dY, cmap="gray")
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    fig.suptitle("Sobel Convolution Outputs in both directions", fontsize=16)
    name = "Sobel_filter.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)



    # Let's now plot the filters, the filtered Image the Gradient Magnitudes and Angles
    dX_new_noiseless, dY_new_noiseless, DGX, DGY, magn_noiseless, angle_noiseless = DG_filtering(im1, sigma=3)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].imshow(DGX)
    axs[1].imshow(DGY)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    axs[0].set_title("X Derivative of Gaussian Filter $\sigma=3$", fontsize=16)
    axs[1].set_title("Y Derivative of Gaussian Filter $\sigma=3$", fontsize=16)
    name = "derivatives.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)
    


    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].imshow(dX_new_noiseless, cmap="gray")
    axs[1].imshow(dY_new_noiseless, cmap="gray")
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    axs[0].set_title("Noiseless X Derivative", fontsize=16)
    axs[1].set_title("Noiseless Y Derivative", fontsize=16)
    name = "derivative_convs.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)


    # Now let us threshold the outputs
    thresh = 38
    thresh_noiseless = magn_noiseless.copy()
    mask = thresh_noiseless <= thresh
    thresh_noiseless[mask] = 0

    # Plot the thresholded magnitude images
    fig, axs = plt.subplots(1, 1, figsize=(14, 5))
    axs.imshow(thresh_noiseless, cmap="gray")
    axs.set_axis_off()
    axs.set_title("Noiseless Thresholded Gradient Magnitude (thresh = {})".format(thresh), fontsize=14)
    name = "thresholded_gradients.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)


    # Non Maxima Supression
    out_b = non_maxima_suppresion_b(magn_noiseless, angle_noiseless)

    fig, axs = plt.subplots(1, 1, figsize=(14, 8))
    axs.imshow(out_b, cmap="gray")
    axs.set_axis_off()
    axs.set_title("Non-maximum-supression", fontsize=16)
    name = "non_maxima_supression.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)



    # Get some Images and display the Canny Edge Detection Output
    images = []
    im = cv2.imread("data/sample_input/okeefe.jpg", cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (int(im.shape[1] * 0.3), int(im.shape[0] * 0.3)))
    images.append(im)
    im = cv2.imread("data/sample_input/kusama.jpg", cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (int(im.shape[1] * 0.85), int(im.shape[0] * 0.85)))
    images.append(im)
    im = cv2.imread("data/sample_input/khalo.jpg", cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (int(im.shape[1] * 0.7), int(im.shape[0] * 0.7)))
    images.append(im)
    im = cv2.imread("data/sample_input/lasnig.png", cv2.IMREAD_GRAYSCALE)
    images.append(im)


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    out_final, out_thresh = Canny(im1, 50, 120, 3.5)
    axs[0].imshow(out_thresh, cmap="gray")
    axs[1].imshow(out_final, cmap="gray")
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    fig.suptitle("Canny Edge Detection for all given images and $(t_1 = 50, t_2=120)$")
    name = "canny_buterrfly.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process an image and save results.")
    parser.add_argument("input_image", type=str, help="Path to the input image file")
    parser.add_argument("out_folder", type=str, help="Path to the output folder")

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.input_image, args.out_folder)


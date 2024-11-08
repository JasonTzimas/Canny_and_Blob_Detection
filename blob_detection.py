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

    # Generate some LoG kernels and visualize them
    sigmas = [2, 4, 8, 12]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, sig in enumerate(sigmas):
        axs[i%2, i//2].imshow(generate_log_kernel(sig, norm=False))
        axs[i%2, i//2].set_axis_off()
        axs[i%2, i//2].set_title("LoG filter $\sigma={}$".format(sig), fontsize=16)
    name = "log_filters.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)

    im = cv2.imread("data/sample_input/einstein.jpg", cv2.IMREAD_GRAYSCALE)
    sigma0 = 1.2
    scale_factor = 1.3
    n = 12
    filtered_images, radius = LoG_filter(im, sigma0, scale_factor, n)
    harris_responses = [harris_response_for_image(filt_image, k=0.04) for filt_image in filtered_images]
    fig, axs = plt.subplots(4, 3, figsize=(18, 25))
    for i in range(n):
        axs[i//3, i%3].imshow(filtered_images[i], cmap="jet")
        axs[i//3, i%3].set_axis_off()
    name = "einstein.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)


    im = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    sigma0 = 1.8
    scale_factor = 1.25
    n = 15
    filtered_images, radius = LoG_filter(im, sigma0, scale_factor, n)
    blob_points, harris_flats, harris_edges = scale_space_non_maxima_suppression(im, filtered_images, radius, 1, harris_thresh=-1e9)



    red = [255,0,0]
    blue = [0, 0, 255]
    green = [0, 255, 0]

    rgb_image = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)
    rgb_image_copy = rgb_image.copy()
    for (x, y) in harris_edges:
        cv2.circle(rgb_image_copy, (y, x), 4, blue, -1)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(rgb_image_copy)
    axs.set_axis_off()
    axs.set_title("Image and Harris Rejected blob points", fontsize=16)
    name = "harris_rejected_points_einsten.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)



    fig, axs = plt.subplots(5, 3, figsize=(18, 27))
    for j, filt_im in enumerate(filtered_images):
        filt_normalized = cv2.normalize(filt_im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        filt_uint8 = np.uint8(filt_normalized)
        filt_rgb = cv2.cvtColor(filt_uint8, cv2.COLOR_GRAY2BGR)
        for (x, y, r, i) in blob_points:
            if i == j:
                cv2.circle(filt_rgb, (y, x), 4, red, -1)
        axs[j //3, j%3].imshow(filt_rgb)

        axs[j //3, j%3].set_axis_off()

    fig.suptitle("Blob points in their corresponding filtered Images", fontsize=16)
    name = "blobs_filters_einstein"
    path = os.path.join(output_path, name)
    fig.savefig(path)

    for (x, y, r, i) in blob_points:
        cv2.circle(rgb_image, (y, x), int(np.ceil(r)), red, 1)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(rgb_image)
    axs.set_axis_off()
    axs.set_title("Detected Blobs", fontsize=16)
    name = "blobs_einstein.png"
    path = os.path.join(output_path, name)
    fig.savefig(path)


    resp = harris_response_for_image(im, k=0.04)
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.imshow(resp)
    axs.set_axis_off()
    axs.set_title("Harris output on initial Image", fontsize=16)
    name = "harris_corners_einstein.png"
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


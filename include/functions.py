import numpy as np
from scipy.signal import convolve2d
import cv2

def Sobel(dim):
    '''
    Return 3x3 Sobel filter of either X or Y direction depending on input dim
    '''
    if dim not in [0, 1]:
        raise Exception("dim must be either 0 or 1")
    if dim==0:
        return np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        return np.array([[1, 2 , 1], [0, 0, 0], [-1, -2, -1]])
    

def gaussian_filter(sigma):
    '''
    Generate a gaussian filter of kernel size dynamically calculated based on sigma
    Returns: filter, kernel size
    '''
    dim = int(np.ceil(3 * sigma))
    if dim % 2 == 0:
        dim += 1
    X, Y = np.mgrid[-dim//2+1:dim//2+1, -dim//2+1:dim//2+1]
    gf = np.exp(-((X**2 + Y**2) / (2 * sigma**2))) / (2 * np.pi * sigma**2)
    
    return gf, dim


def DG(sigma):
    '''
    Generate the Derivative of a Gaussian based on the given Sigma
    Returns: Both X and Y derivative filters
    '''
    gf, dim = gaussian_filter(sigma)
    if dim <= 3:
        DGX = Sobel(dim=1)
        DGY = Sobel(dim=0)
    else:
        DGX = convolve2d(gf, Sobel(dim=1),mode="valid", boundary="symm")
        DGY = convolve2d(gf, Sobel(dim=0),mode="valid", boundary="symm")
    
    return DGX, DGY

def DG_filtering(im, sigma):
    '''
    Filter input sigma with Derivative of Gaussian filters on directions X and Y. Performs appropriate replicate padding to accomodate conservation of dimensions
    Returns: Filters, Derivatives, Gradient Magnitudes, Gradient Angles in (-pi/2, pi/2)
    '''
    DGX, DGY = DG(sigma=sigma)
    padding = int(DGX.shape[0] // 2)
    new_im = np.ndarray((im.shape[0] + 2 * padding, im.shape[1] + 2 * padding))
    new_im[padding:-padding, padding:-padding] = im
    new_im[padding:-padding, :padding] = np.repeat(im[:, 0].reshape(-1, 1), padding, axis=1)
    new_im[padding:-padding, -padding:] = np.repeat(im[:, 0].reshape(-1, 1), padding, axis=1)
    new_im[:padding, padding:-padding] = np.repeat(im[0, :].reshape(1, -1), padding, axis=0)
    new_im[-padding:, padding:-padding] = np.repeat(im[0, :].reshape(1, -1), padding, axis=0)
    new_im[:padding, :padding] = im[0, 0] * np.ones((padding, padding))
    new_im[:padding, -padding:] = im[0, -1] * np.ones((padding, padding))
    new_im[-padding:, :padding] = im[-1, 0] * np.ones((padding, padding))
    new_im[-padding:, -padding:] = im[-1, -1] * np.ones((padding, padding))

    dX = convolve2d(new_im, DGX, mode="valid")
    dY = convolve2d(new_im, DGY, mode="valid")
    magn = np.sqrt(dX**2 + dY**2)
    angle = np.arctan(dX / dY) 
    return dX, dY, DGX, DGY, magn, angle



# Non-Maxima Suppresion
# Way 1: Based on 3x3 neighborhood regardless of angle

def non_maxima_suppresion_a(gradient_magnitudes):
    '''
    Perform Non-Maximaum-Suppression using Strategy A: (Suppress non maximum pixels regardless of the gradient angle)
    Returns: Suppressed filtered Image
    '''
    output = np.ndarray((gradient_magnitudes.shape[0] + 2, gradient_magnitudes.shape[1] + 2))
    output[1:-1, 1:-1] = gradient_magnitudes
    output[:, [0, -1]] = 0
    output[[0, -1], -1] = 0
    out_list = [0 if output[i, j] < np.max(output[i-1:i+2, j-1:j+2]) else output[i, j] for i in range(1, output.shape[0]-1) for j in range(1, output.shape[1] - 1)]
    out_array = np.array(out_list).reshape(gradient_magnitudes.shape[0], gradient_magnitudes.shape[1])
    output[1:-1, 1:-1] = out_array
    return output

# Way 2: Discretizing the angles in 4 bins, namely (0, 45, 90, 135) and performing non-maxima suppresion at these angles
def non_maxima_suppresion_b(gradient_magnitudes, gradient_angles):
    '''
    Perform Non-Maximaum-Suppression using Strategy B: (Suppress non maximum pixels in the direction of the quantized angle in bins of angle (0, 45, 90, 135))
    Returns: Suppressed filtered Image
    '''
    output = np.ndarray((gradient_magnitudes.shape[0], gradient_magnitudes.shape[1]))
    output[:, [0, -1]] = 0
    output[[0, -1], :] = 0
    output[1:-1, 1:-1] = gradient_magnitudes[1:-1, 1:-1]

    # Map angles
    for i in range(1, output.shape[0] - 1):
        for j in range(1, output.shape[1] - 1):
            ang = gradient_angles[i-1, j-1]
            if np.abs(ang) < np.pi / 8:
                if gradient_magnitudes[i, j] < np.max(gradient_magnitudes[i, j-1:j+2]):
                    output[i, j] = 0
            elif ang >= np.pi / 8 and ang < 3 * np.pi / 8:
                if gradient_magnitudes[i, j] < gradient_magnitudes[i-1, j+1] or gradient_magnitudes[i, j] < gradient_magnitudes[i+1, j-1]:
                    output[i, j] = 0
            elif ang <= - np.pi / 8 and ang > - 3 * np.pi / 8:
                if gradient_magnitudes[i, j] < gradient_magnitudes[i+1, j+1] or gradient_magnitudes[i, j] < gradient_magnitudes[i-1, j-1]:
                    output[i, j] = 0
            else:
                if gradient_magnitudes[i, j] < np.max(gradient_magnitudes[i-1:i+2, j]):
                    output[i, j] = 0
    
    return output


def non_maxima_suppresion_c(gradient_magnitudes, gradient_angles):
    '''
    Perform Non-Maximaum-Suppression using Strategy C: (Suppress non maximum pixels in the direction of the actual gradient angle using interpolated pixel intensities
    Returns: Suppressed filtered Image
    '''
    output = np.ndarray((gradient_magnitudes.shape[0], gradient_magnitudes.shape[1]))
    output[:, [0, -1]] = 0
    output[[0, -1], :] = 0
    output[1:-1, 1:-1] = gradient_magnitudes[1:-1, 1:-1]
    for i in range(1, output.shape[0] - 1):
        for j in range(1, output.shape[1] - 1):
            ang = gradient_angles[i-1, j-1]
            if ang > 0 and ang <= np.pi / 4:
                coef = np.tan(ang)
                p1 = (1 - coef) * gradient_magnitudes[i, j+1] + coef * gradient_magnitudes[i-1, j+1]
                p2 = (1 - coef) * gradient_magnitudes[i, j-1] + coef * gradient_magnitudes[i+1, j-1] 
                #print(coef, ang, "Case = 1")
                if gradient_magnitudes[i, j] < p1 or gradient_magnitudes[i, j] < p2:
                    output[i, j] = 0
            elif ang > np.pi / 4 and ang <= np.pi / 2:
                coef = np.tan(np.pi / 2 - ang)
                p1 = (1 - coef) * gradient_magnitudes[i-1, j] + coef * gradient_magnitudes[i-1, j+1]
                p2 = (1 - coef) * gradient_magnitudes[i+1, j] + coef * gradient_magnitudes[i+1, j-1] 
                #print(coef, ang, "Case = 2")
                if gradient_magnitudes[i, j] < p1 or gradient_magnitudes[i, j] < p2:
                    output[i, j] = 0
            elif ang <= 0 and ang > - np.pi / 4:
                coef = np.tan(np.abs(ang))
                p1 = (1 - coef) * gradient_magnitudes[i, j+1] + coef * gradient_magnitudes[i+1, j+1]
                p2 = (1 - coef) * gradient_magnitudes[i, j-1] + coef * gradient_magnitudes[i-1, j-1] 
                #print(coef, ang, "Case = 3")
                if gradient_magnitudes[i, j] < p1 or gradient_magnitudes[i, j] < p2:
                    output[i, j] = 0
            else:
                coef = np.tan(np.pi / 2 - np.abs(ang))
                p1 = (1 - coef) * gradient_magnitudes[i+1, j] + coef * gradient_magnitudes[i+1, j+1]
                p2 = (1 - coef) * gradient_magnitudes[i-1, j] + coef * gradient_magnitudes[i-1, j-1] 
                #print(coef, ang, "Case = 4")
                if gradient_magnitudes[i, j] < p1 or gradient_magnitudes[i, j] < p2:
                    output[i, j] = 0
            
           
    
    return output



def connectivity_labeling(input_image):
    '''
    Adapt connectivity labeling algorithm for Canny Edge detection Hysterisis Thresholding
    Returns: Canny Edge Detection output
    '''
    label = 1
    flag_image = np.zeros_like(input_image)
    output_image = np.zeros_like(input_image)
    for i in range(flag_image.shape[0]):
        for j in range(flag_image.shape[1]):
            if input_image[i, j] == 2:
                if flag_image[i, j] == 0: # Pixel not yet labeled
                    # Label all component-connected pixels
                    flag_image[i, j] = label
                    queue = [(i, j)]
                    while queue:
                        tail = queue[-1]
                        queue.pop()
                        new_queue = [(k, n) for k in range(tail[0]-1, tail[0]+2) for n in range(tail[1]-1, tail[1]+2) if (k != tail[0] or n != tail[1])
                                     and k >= 0 and k <= flag_image.shape[0] - 1
                                      and n >= 0 and n <= flag_image.shape[1] - 1 and (input_image[k, n] == 1 or input_image[k, n] == 2) and flag_image[k, n] == 0]
                        for pix in new_queue: 
                            flag_image[pix] = label
                            output_image[pix] = 2
                        queue = new_queue + queue

    return output_image



def Canny(img, thresh1, thresh2, sigma):
    '''
    Canny Edge Detection from scratch
    Inputs: threshold1, threshold2 (t1 < t2), sigma for Bluring
    Returns: Canny Edge Detection Output
    '''
    dX, dY, DGX, DGY, magn, angle = DG_filtering(img, sigma=sigma)
    thresh1 = np.std(img) / 255 * thresh1
    thresh2 = np.std(img) / 255 * thresh2
    out = non_maxima_suppresion_c(magn, angle)
    out_thresh = np.zeros_like(out)
    out_thresh[np.logical_and(out >= thresh1, out <= thresh2)] = 1
    out_thresh[out > thresh2] = 2
    out_final = connectivity_labeling(out_thresh)
    return out_final, out_thresh



# Blob detection
def generate_log_kernel(sigma, norm=True):
    '''
    Generates the Laplacian of a Gaussian Kernel based on the given sigma value. The kernel size is dynamically defined. If norm is set to True the Kernel is normalized 
    Returns: LoG Filter
    '''
    # Ensure size is odd
    size = int(np.ceil(4 * sigma))
    if size % 2 == 0:
        size += 1
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = -1/ (np.pi * sigma**4) * (1 - (x**2 + y**2) / (2 * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    if norm:
        g *= sigma**2
    return g

def replicate_padding(im, filter):
    '''
    Function that performs replicate padding given the input image and filter
    Returns: Padded Image
    '''
    padding = int(filter.shape[0] // 2)
    new_im = np.ndarray((im.shape[0] + 2 * padding, im.shape[1] + 2 * padding))
    new_im[padding:-padding, padding:-padding] = im
    new_im[padding:-padding, :padding] = np.repeat(im[:, 0].reshape(-1, 1), padding, axis=1)
    new_im[padding:-padding, -padding:] = np.repeat(im[:, -1].reshape(-1, 1), padding, axis=1)
    new_im[:padding, padding:-padding] = np.repeat(im[0, :].reshape(1, -1), padding, axis=0)
    new_im[-padding:, padding:-padding] = np.repeat(im[-1, :].reshape(1, -1), padding, axis=0)
    new_im[:padding, :padding] = im[0, 0] * np.ones((padding, padding))
    new_im[:padding, -padding:] = im[0, -1] * np.ones((padding, padding))
    new_im[-padding:, :padding] = im[-1, 0] * np.ones((padding, padding))
    new_im[-padding:, -padding:] = im[-1, -1] * np.ones((padding, padding))

    return new_im

def LoG_filter(im, sigma0, scale_factor, n):
    '''
    Perform LoG Filtering, of the input image using a set of filters defined by the initial sigma0 and sigmas generated by recursively scaling sigma0 n-times
    Returns: Filtered Images, Blob Radius
    '''
    sigmas = [sigma0 * (np.power(scale_factor, i)) for i in range(n)]
    radius = [np.sqrt(2) * sigma for sigma in sigmas]
    filters = [generate_log_kernel(sigma) for sigma in sigmas]
    filtered_images = []
    for i, filter in enumerate(filters):
        new_im = replicate_padding(im, filter)
        filt_im = np.power(convolve2d(new_im, filter, mode="valid"), 2)
        filt_im /= np.std(filt_im)
        filtered_images.append(filt_im)

    return filtered_images, radius

def harris_response_for_neighborhood(neighborhood, k=0.04):
    '''
    Perform the Harris Transform for Corner-Edge Detection, for the center of the given neighborhood and the k-constant
    Returns: Harris Value for center pixel
    '''
    # Compute gradients (Sobel operators applied to the neighborhood)
    # Note: Normally, we would use cv2.Sobel or similar on a larger image. Here, we approximate this for the 3x3 case.
    dx = Sobel(dim=0)
    dy = Sobel(dim=1)
    Ix = np.sum(neighborhood * dx)
    Iy = np.sum(neighborhood * dy)
    
    # Compute products of derivatives at every pixel
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2
    
    # Sum of products of derivatives for the pixels in the window
    # For a 3x3 neighborhood, it's just the value itself, no sum needed.
    
    # Compute the determinant and trace of the matrix M
    detM = Ixx * Iyy - Ixy**2
    traceM = Ixx + Iyy
    
    # Calculate Harris response
    R = detM - k * traceM**2
    
    return R


def harris_response_for_image(image, k):
    '''
    Perform the Harris Transform for Corner-Edge Detection, for the entire Image given the constant k
    Returns: Harris Values for the entire Image
    '''
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_filtered = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute gradients
    Ix = cv2.Sobel(gray_filtered, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(gray_filtered, cv2.CV_64F, 0, 1, ksize=5)

    # Compute products of derivatives
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2

    # Apply Gaussian window (w)
    gIxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    gIxy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    gIyy = cv2.GaussianBlur(Iyy, (3, 3), 0)

    # Compute determinant and trace of the matrix M
    detM = gIxx * gIyy - gIxy**2
    traceM = gIxx + gIyy
    
    # Compute Harris response
    R = detM - k * traceM**2

    # Determine if (x, y) is a corner
    return R


def scale_space_non_maxima_suppression(image, filtered_scale_space, radius, thresh, harris_thresh):
    '''
    Performs Non-Maximum-Suppression for the given scale-space 3D Tensor by sliding a 3x3x3 cube around and asserting whether the center pixel is the maximum or not for the given neighborhood
    Returns: Detected Blob coordinates, Blobs associated to flat regions given Harris value, Blobs associated with edge points given Harris value
    '''
    scale_space = np.array(filtered_scale_space)
    blobs = np.ones_like(filtered_scale_space)
    blob_points = []
    harris_edges = []
    harris_flats = []
    harris_response = harris_response_for_image(image, k=0.04)

    for i in range(1, len(filtered_scale_space)-1):
        for x in range(1, blobs.shape[1]-1):
            for y in range(1, blobs.shape[2]-1):
                if scale_space[i, x, y] < np.max(scale_space[i-1:i+2, x-1:x+2, y-1:y+2]):
                    blobs[i, x, y] = 0
                else:
                    if scale_space[i, x, y] > thresh:
                        harris = harris_response[x, y]
                        if harris > harris_thresh:
                            blob_points.append((x, y, radius[i], i))
                            harris_flats.append((x, y))
                        else:
                            harris_edges.append((x, y))

    return blob_points, harris_flats, harris_edges

def LoG_filter_b(im, sigma0, scale_factor, n):
    '''
    Performs LoG Filtering using strategy B, by iteratively downscaling the Image instead of Upscaling the LoG filter
    Returns: Scale-Space filters, BloB radius
    '''
    sigmas = [sigma0 * np.power((1 / scale_factor), i) for i in range(n)]
    radius = [3 * sigma for sigma in sigmas]
    dim0 = (im.shape[1], im.shape[0])
    filter = generate_log_kernel(sigma0, norm=False)
    image_scale_space = []
    filtered_scale_space = []
    temp = im.copy()
    for i in range(n):
        image_scale_space.append(temp)
        temp = cv2.GaussianBlur(temp,(3,3), 0)
        dim = (int(scale_factor * temp.shape[1]), int(scale_factor * temp.shape[0]))
        temp = cv2.resize(temp, (dim), interpolation = cv2.INTER_LINEAR)
        filtered_temp = convolve2d(replicate_padding(temp, filter), filter, mode="valid")
        filtered_temp = np.power(filtered_temp, 2) 
        filtered_temp /= np.std(filtered_temp)
        filtered_scale_space.append(cv2.resize(filtered_temp, dim0, interpolation = cv2.INTER_LANCZOS4 ))
    

    return filtered_scale_space, radius
import numpy as np
# import cv2
from t1cv import *
# import bottleneck as bn
# from sklearn.cluster import (OPTICS, HDBSCAN

#plot a histogram of the intensity values
import matplotlib.pyplot as plt
# plt.hist(gray1.ravel(),256,[0,256])
# plt.show()
# plt.hist(gray2.ravel(),256,[0,256])
# plt.show()


def log_normalize(gray):
# def clahe_normalize(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(img_clahe, (5,5), 0)
    kernel = np.ones((5,5),np.uint8)
    img_opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

    return img_opening
    # flatten the histogram by taking the log
    log_gray = np.log(np.float32(gray))
    # plt.hist(log_gray.ravel(),256,[0,256])
    normalized = (log_gray - np.min(log_gray)) / (np.max(log_gray) - np.min(log_gray))
    # normu8 = 
    normu8 = cv2.equalizeHist(cv2.GaussianBlur(gray, (5,5), 0))
    normu8 = np.max(np.dstack((np.uint8(normu8 * log_gray / np.max(log_gray) / 2 + normu8 / 2), gray)), axis = 2)
    # normu8 = cv2.GaussianBlur(normu8, (7,7), 0)
    return cv2.GaussianBlur(cv2.equalizeHist(gray), ksize=(5,5), sigmaX=0)
    # plt.hist(normu8.ravel(),256,[0,256])
    # plt.show()
    #plot gray1, gray2, and normu8 on the same figure
    # plt.subplot(131)
    # plt.imshow(gray1, cmap='gray')
    # plt.subplot(132)
    # plt.imshow(gray2, cmap='gray')
    # plt.subplot(133)
    # plt.imshow(normu8, cmap='gray')
    # plt.subplot(231)
    # plt.hist(gray1.ravel(),256,[0,256])
    # plt.subplot(232)
    # plt.hist(gray2.ravel(),256,[0,256])
    # plt.subplot(233)
    # plt.hist(normu8.ravel(),256,[0,256])

    # plt.show()
    # IA.show(normu8, 5000)

def brighten_normalize(gray):
    # flatten the histogram making the image brighter and uniform histogram
    gray = np.float32(gray)
    gray_padded = np.pad(gray, 1, mode="extend")
    print(gray_padded.shape, gray.shape)



def rm_lf(image, rad = 39):
    # Load the image in grayscale

    # Perform FFT with shift (move the zero frequency component to the center of the spectrum)
    dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Generate a low-pass filter
    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2

    # Create a mask with high value 1 at low frequencies and 0 at HF, circle with radius 30
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-rad:crow+rad, ccol-rad:ccol+rad] = 1

    # Apply mask to the DFT transformed image
    fshift = dft_shift*mask

    # Shift back (inverse shift)
    f_ishift = np.fft.ifftshift(fshift)

    # Inverse DFT to get the image back 
    img_back = cv2.idft(f_ishift)
    lpf = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    # cv2.imwrite('low_pass_filtered_image.png', img_back)
    h,l = np.max(lpf), np.min(lpf)
    lpf = (lpf - l) / (h - l)
    

    return lpf
    # Save the image


# log_normalize(gray2)
# log_normalize(gray1)
    

def apply_multi_low_pass(gray):
    im = rm_lf(gray)
    k = 20
    t = 100
    r = 70
    for rad in range(k, k + t):
        im += rm_lf(gray, rad)
    im = im + np.float32(gray) / 255 * r
    im = im / (t + r)
    return np.uint8(im * 255)



def main():
    from grad_like import ImageGradStats
    from analysis import ImageAnalysis
    IA = ImageAnalysis('437-1-03.tif')
    gray1 = IA.gray
    IA = ImageAnalysis('242316_01.tif')
    gray2 = IA.gray
    IA.show(apply_multi_low_pass(gray2), 5000)

if __name__ == "__main__":
    main()

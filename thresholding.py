import cv2
from skimage import filters , io
import numpy as np
from matplotlib import pyplot as plt


img = io.imread('../Thresholding/bookpage.jpg')
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

gray_scaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval2, threshold_gray = cv2.threshold(gray_scaled_img,10,255,cv2.THRESH_BINARY)

adaptive_threshold = cv2.adaptiveThreshold(gray_scaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 115, 1)


retval3, threshold_otsu = cv2.threshold(gray_scaled_img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


fig , ax = plt.subplots(nrows=5,figsize=(10,10))


ax[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Bookpage")

ax[1].imshow(cv2.cvtColor(threshold,cv2.COLOR_BGR2RGB))
ax[1].set_title("Thresholded Bookpage")

ax[2].imshow(cv2.cvtColor(threshold_gray,cv2.COLOR_BGR2RGB))
ax[2].set_title("Grayscaled and Thresholded Bookpage")


ax[3].imshow(cv2.cvtColor(adaptive_threshold,cv2.COLOR_BGR2RGB))
ax[3].set_title("Bookpage With Adaptive Threshold")

ax[4].imshow(cv2.cvtColor(threshold_otsu,cv2.COLOR_BGR2RGB))
ax[4].set_title("Bookpage With Otsu Threshold")


plt.tight_layout()
plt.show()

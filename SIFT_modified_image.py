# Import necessary pre-requisite
from SIFT import sift_maker, gaussiun_convolve
import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte

# Read the image
book_image = cv2.imread(r"C:\Users\Ajeet\Downloads\Masters\2nd Semester\Advance Image Processing\AIP2023-Assignment1\AIP2023-Assignment1\books.png")
# Now apply Rotation operations on image
rotated_book = cv2.rotate(book_image, cv2.ROTATE_90_CLOCKWISE)
plt.figure('Book image')
plt.subplot(2,3,1)
plt.imshow(book_image)
plt.title('Original image')
# Apply sift on rotated image
out_book_rotate = sift_maker(rotated_book)
plt.subplot(2,3,2)
plt.imshow(out_book_rotate)
plt.title('Rotated image SIFT output')

# Now apply Upscale
upscaled_book = cv2.resize(book_image, (720,720), interpolation = cv2.INTER_AREA)
out_upscaled_book = sift_maker(upscaled_book)
plt.subplot(2,3,3)
plt.imshow(out_upscaled_book)
plt.title('Upscaled SIFT output')

# Now apply Downscale
downscaled_book = cv2.resize(book_image, (300,300), interpolation = cv2.INTER_AREA)
out_downscaled_book = sift_maker(downscaled_book)
plt.subplot(2,3,4)
plt.imshow(out_downscaled_book)
plt.title('Downscaled SIFT output')

# Perform Gaussian blur on an image
gaussian_book = gaussiun_convolve(book_image, 4)
out_gaussian_book = sift_maker(gaussian_book)
plt.subplot(2,3, 5)
plt.imshow(out_gaussian_book)
plt.title('Gaussian blur SIFT output')

#  Add gaussian noise to image
noise = skimage.util.random_noise(book_image, mode='gaussian', seed=None, clip=True)
cv_image = img_as_ubyte(noise)
out_noisy_book = sift_maker(cv_image)
plt.subplot(2,3,6)
plt.imshow(out_noisy_book)
plt.title('Gaussian Noise image SIFT output')

# Now perform same for building image
# Read the image
building_image = cv2.imread(r"C:\Users\Ajeet\Downloads\Masters\2nd Semester\Advance Image Processing\AIP2023-Assignment1\AIP2023-Assignment1\building.png")
# Now apply Rotation operations on image
rotated_build = cv2.rotate(building_image, cv2.ROTATE_90_CLOCKWISE)
plt.figure('Building image')
plt.subplot(2,3,1)
plt.imshow(building_image)
plt.title('Original image')
# Apply sift on rotated image
out_build_rotate = sift_maker(rotated_build)
plt.subplot(2,3,2)
plt.imshow(out_build_rotate)
plt.title('Rotated image SIFT output')

# Now apply Upscale
upscaled_build = cv2.resize(building_image, (720,720), interpolation = cv2.INTER_AREA)
out_upscaled_build = sift_maker(upscaled_build)
plt.subplot(2,3,3)
plt.imshow(out_upscaled_build)
plt.title('Upscaled SIFT output')

# Now apply Downscale
downscaled_build = cv2.resize(building_image, (300,300), interpolation = cv2.INTER_AREA)
out_downscaled_build = sift_maker(downscaled_build)
plt.subplot(2,3,4)
plt.imshow(out_downscaled_build)
plt.title('Downscaled SIFT output')

# Perform Gaussian blur on an image
gaussian_build = gaussiun_convolve(building_image, 4)
out_gaussian_build = sift_maker(gaussian_build)
plt.subplot(2,3, 5)
plt.imshow(out_gaussian_build)
plt.title('Gaussian blur SIFT output')

#  Add gaussian noise to image
noise = skimage.util.random_noise(building_image, mode='gaussian', seed=None, clip=True)
cv_image = img_as_ubyte(noise)
out_noisy_build = sift_maker(cv_image)
plt.subplot(2,3,6)
plt.imshow(out_noisy_build)
plt.title('Gaussian Noise image SIFT output')
plt.show()




# Import necessary library
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Use a function which evaluate the gaussian filter and convolve input with it. 
def gaussiun_convolve(image, std_deviation):
    return gaussian_filter(image, std_deviation)

# Construct a function which perform SIFT operation
def sift_maker(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # We know that s=2 or k=2 ^ 1/s. So here I will use s = 3
    k = pow(2, (1/3))
    first_filtered_image = gaussiun_convolve(gray_image, 1)
    second_filtered_image = gaussiun_convolve(gray_image, 1*k)
    third_filtered_image = gaussiun_convolve(gray_image, 1*k*k)
    fourth_filtered_image = gaussiun_convolve(gray_image, 1*k*k*k)
    
    # Now take difference of two images
    diff_imag1 = second_filtered_image - first_filtered_image
    diff_img2 = third_filtered_image - second_filtered_image
    diff_img3 = fourth_filtered_image - third_filtered_image

    # Now iterate the elements which have 8 neighbour pixel in diff_img2 and compare with others
    M, N, _ = original_image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if diff_img2[i,j] > max(np.max(diff_imag1[i-1:i+1, j-1:j+1]), np.max(diff_img3[i-1:i+1, j-1:j+1])
            , diff_img2[i-1,j], diff_img2[i-1,j+1], diff_img2[i,j-1],diff_img2[i,j+1],diff_img2[i-1,j-1], diff_img2[i+1,j+1], diff_img2[i-1,j+1], diff_img2[i+1,j-1]):
                # Highlight the pixel
                original_image = cv2.circle(original_image,(j,i),1,(255,0,0),2)
    return original_image


# Driver code 
if __name__=='__main__':

    book_image = cv2.imread(r"C:\Users\Ajeet\Downloads\Masters\2nd Semester\Advance Image Processing\AIP2023-Assignment1\AIP2023-Assignment1\books.png")
    building_image = cv2.imread(r"C:\Users\Ajeet\Downloads\Masters\2nd Semester\Advance Image Processing\AIP2023-Assignment1\AIP2023-Assignment1\building.png") 
    # Plot the original image
    plt.figure('First step of SIFT')
    plt.subplot(2, 2, 1)
    plt.imshow(book_image)
    plt.title('Original image of book.png')
    plt.subplot(2, 2, 3)
    plt.imshow(building_image)
    plt.title('Original image of building.png')
    # Call the function
    out_book = sift_maker(book_image)
    out_build = sift_maker(building_image)
    # Plot the output
    plt.subplot(2,2, 2)
    plt.imshow(out_book)
    plt.title('Feature of book.png')
    plt.subplot(2,2, 4)
    plt.imshow(out_build)
    plt.title('Feature of building.png')
    plt.show()

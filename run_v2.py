import cv2
from processor import process_image

input_images = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
# input_images = ['1.jpg']


for image_file in input_images:
    bgr_image = cv2.imread('graded_input/' + image_file)
    result = process_image(bgr_image)
    cv2.imwrite('graded_images/' + image_file.replace('.jpg', '.png'), bgr_image)
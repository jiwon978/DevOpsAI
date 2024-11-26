import cv2

image_path = 'C:/Users/user/Desktop/assignment/DevOps_project/test.png'
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image. Please check the file path.")
else:
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
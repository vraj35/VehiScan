import cv2


# Define a mouse callback function
def onMouse(event, x, y, flag, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print("Mouse position: (", x, ", ", y, ")")


# Load an image
image = cv2.imread("image.png")

# Create a window and set the mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", onMouse)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

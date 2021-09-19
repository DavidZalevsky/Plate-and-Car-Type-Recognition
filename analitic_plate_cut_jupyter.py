import cv2
from matplotlib import pyplot as plt
import imutils


def analitic_plate_cut(image):

    img = cv2.imread(image)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray scale
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    bilateral_filter = cv2.bilateralFilter(gray, 20, 50, 50)  # Noise reduction
    #plt.imshow(cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB))
    canny_edge_detection = cv2.Canny(bilateral_filter, 50, 200)  # Edge detection
    # plt.imshow(cv2.cvtColor(canny_edge_detection, cv2.COLOR_BGR2RGB))

    vertices = cv2.findContours(canny_edge_detection.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # finding vertices of all polygons
    contours = imutils.grab_contours(vertices)  # finding exact simplified countours with given vertices
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # sorting from longest to smallest countour, saving only longest countour from immage

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            cord = approx
            break

    try:
        y1 = cord[2][0][0]
        y2 = cord[0][0][0]
        x2 = cord[2][0][1]
        x1 = cord[0][0][1]

        cropped_plate_image = gray[x1:x2, y1:y2]
        plt.imshow(cv2.cvtColor(cropped_plate_image, cv2.COLOR_BGR2RGB))

        cv2.imwrite('cropped_plate_image.png', cropped_plate_image)

    except:
        print("Plate not found!")

#analitic_plate_cut("input_plate_view.png") ###


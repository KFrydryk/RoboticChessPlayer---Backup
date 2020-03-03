import cv2
import numpy as np
from settings import SettingsChessSquare

class ChessSquare:
    # 1. wierzcholek , lewy gorny
    # 2. wierzcholek , prawy gorny
    # 3. wierzcholek , prawy dolny
    # 4. wierzcholek , lewy dolny
    ID = 0
    IDOld = 0
    nazwa = ""
    czysrodek = False
    srodek=[]
    wierzcholki =[]
    srodekFigury = []
    settings = []

    def __init__(self,name, corners , center, i):
        self.ID = i
        self.IDOld = i
        self.nazwa = name
        self.srodek = np.array([center[0],center[1]])
        self.wierzcholki = np.array([[corners[0],corners[1]],[corners[2],corners[3]],[corners[4],corners[5]],[corners[6],corners[7]] ])
        self.settings = SettingsChessSquare()

    def kolo(self, img):
        obraz_po_trans, M ,M2 = four_point_transform(img,self.wierzcholki,1)
        obraz_po_trans = cv2.blur(obraz_po_trans,(3,3))
        cv2.imwrite("krok.jpg",obraz_po_trans)
        output = obraz_po_trans.copy()
        gray = cv2.imread('krok.jpg',0)
        # detect circles in the image

        if self.settings.DebugMode:
            str = "{}.jpg".format(self.ID)
            cv2.imwrite("./Debug/Squares/"+str, gray)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.settings.h1, self.settings.h2, minRadius=self.settings.h3, maxRadius=25, param1=40, param2=30)

        # ensure at least some circles were found
        if circles is not None:
            self.czysrodek = True
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[00, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                xy2 = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
                xy2 = np.expand_dims(xy2, 0)
            powrotbazy = cv2.perspectiveTransform(xy2, M2)
            powrotbazy =np.resize(powrotbazy,2)
            self.srodekFigury = powrotbazy
        cv2.imshow("pole", output)
        return 1


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, method): #metoda 1 -  getPerspective, metoda 2 - findHomography
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    if method == 1:
        M = cv2.getPerspectiveTransform(rect, dst)
        M2 = cv2.getPerspectiveTransform(dst, rect)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped, M, M2
    elif method == 2:
        M, mask = cv2.findHomography(rect, dst, cv2.RANSAC)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped, M
    else:
        return 0,0,0
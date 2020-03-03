import cv2
import numpy as np


def colorbalance():
    cap = cv2.VideoCapture(1)
    ret, img = cap.read()
    cv2.imwrite('frame.jpg', img)
    img = cv2.imread('frame.jpg')

    b,g,r = cv2.split(img)

    bnew = np.array(b, dtype = np.uint16)
    bnew = np.array(bnew*3, dtype = np.uint16)
    bnew[bnew>255] = 255
    b=np.array(bnew, dtype = np.uint8)

    gnew = np.array(g, dtype = np.uint16)
    gnew = np.array(gnew*1.2, dtype = np.uint16)
    gnew[gnew>255] = 255
    g=np.array(gnew, dtype = np.uint8)

    rnew=r
    rnew = np.array(rnew*0.6, dtype = np.uint8)
    r = rnew

    img2 = cv2.merge([b,g,r])
    img2 = cv2.flip(img2,0)
    cv2.imwrite("Corrected.jpg", img2)

colorbalance()
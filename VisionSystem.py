from ChessBoard import ChessBoard
import cv2
import numpy as np
import copy
from ChessSquare import order_points, four_point_transform
from settings import SettingsVisionSystem


cap = cv2.VideoCapture(1)

class Vision:
    markerCenter = []
    markerx = []
    markery = []
    linex = []
    liney = []
    robotBaseCenter = []
    robotBaseSquare = []
    robotBaseMatrix = []
    coefX = 0
    coefY = 0
    settings = []

    def TakeImage(self):
        ret, img = cap.read()
        cv2.imwrite('frame.jpg', img)

        img = cv2.imread('frame.jpg')
        cv2.imshow("przed", img)
        rows, cols, _ = img.shape

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

        b, g, r = cv2.split(img)

        bnew = np.array(b, dtype=np.uint16)
        bnew = np.array(bnew * 3, dtype=np.uint16)
        bnew[bnew > 255] = 255
        b = np.array(bnew, dtype=np.uint8)
        gnew = np.array(g, dtype=np.uint16)
        gnew = np.array(gnew * 1.2, dtype=np.uint16)
        gnew[gnew > 255] = 255
        g = np.array(gnew, dtype=np.uint8)
        rnew = r
        rnew = np.array(rnew * 0.6, dtype=np.uint8)
        r = rnew
        img2 = cv2.merge([b, g, r])
        img2 = cv2.flip(img2, 1)
        cv2.imshow("Po korekcji", img2)
        cv2.imwrite("Corrected.jpg", img2)
        return img2

    def __init__(self,img):
        self.chessBoard = ChessBoard(img)
        self.settings = SettingsVisionSystem()


    def VisionInitializate(self):
        self.chessBoard.Initialize()

        #wyszukiwanie markerów bazy
        self.FindMarkersPosition(self.chessBoard.smallImg, 120, 140, 140)
        #tworzenie czwartego punktu do wyznaczenia macierzy transformacji obrazka pierwotnego do bazy robota
        dx = self.markery[0]-self.markerCenter[0]
        dy = self.markery[1]-self.markerCenter[1]
        extraMarker = (self.markerx[0]+dx, self.markerx[1]+dy)

        #zapis punktów w odpowiedniej formie
        points1 = np.array([[self.markerCenter[0],self.markerCenter[1]],[self.markerx[0],self.markerx[1]], [extraMarker[0],extraMarker[1]], [self.markery[0],self.markery[1]]])

        warped, h = four_point_transform(self.chessBoard.smallImg, points1,2)
        cv2.imwrite("ObrazWBazieRobota.jpg",warped)

        self.robotBaseMatrix = h

        #obliczanie współczynników przeliczenia współrzędnych robota na obrazku na współrzędne rzeczywiste
        realDistX = 557#557 #[mm]
        realDistY = 349#349 #[mm]

        self.coefX = realDistX/warped.shape[1]
        self.coefY = realDistY/warped.shape[0]


        #utworzenie kopii listy współrzędnych środków pól szachowych i obliczenie ich współrzędnych w układzie bazy robota
        self.robotBaseSquare = copy.deepcopy(self.chessBoard.squares)
        for i in range(len(self.robotBaseSquare)):
            self.robotBaseSquare[i].srodek = np.vstack([self.robotBaseSquare[i].srodek[0].flatten(), self.robotBaseSquare[i].srodek[1].flatten()]).T.astype(np.float32)
            #self.robotBaseSquare[i].srodek = np.append(self.robotBaseSquare[i].srodek,0)
            self.robotBaseSquare[i].srodek = np.expand_dims(self.robotBaseSquare[i].srodek, 0)
            self.robotBaseSquare[i].srodek = cv2.perspectiveTransform(self.robotBaseSquare[i].srodek, h)

            if self.settings.DebugMode:
                #rysowanie środków pól na obrazku w bazie robota
                if self.robotBaseSquare[i].srodek[0, 0, 1]>0: #self.robotBaseSquare[i].srodek[1]>0:
                    cv2.circle(warped, (self.robotBaseSquare[i].srodek[0, 0, 0], self.robotBaseSquare[i].srodek[0, 0, 1]), 1, (0, 255, 0), 3)

                # zaznaczanie pozycji wykrytego pionka na obrazie
                # Piece = self.VisionPieceCenter(28)
                # cv2.circle(warped, (Piece[0,0,0], Piece[0,0,1]), 1,(0, 0, 255), 3)
                cv2.imwrite("./Debug/ObrazWBazieRobota.jpg", warped)




    def VisionReinitializate(self,img):
        self.chessBoard.ChangeStartImage(img)

    def VisionSquareCenter(self,search):
        i = self.SearchSquares(search)
        return self.chessBoard.squares[i].srodek

    def VisionPieceCenter(self,search):
        print("Search: {}".format(search))
        self.chessBoard.squares[search].kolo(self.chessBoard.smallImg)

        if self.chessBoard.squares[search].czysrodek:
            PieceCenter = self.chessBoard.squares[search].srodekFigury
            PieceCenter = np.vstack([PieceCenter[0].flatten(), PieceCenter[1].flatten()]).T.astype(np.float32)
            # self.robotBaseSquare[i].srodek = np.append(self.robotBaseSquare[i].srodek,0)
            PieceCenter = np.expand_dims(PieceCenter, 0)
            PieceCenter = cv2.perspectiveTransform(PieceCenter, self.robotBaseMatrix)
            return PieceCenter
        else:
            return 0

    def SearchSquares(self, search):
        i = 1
        while search != self.chessBoard.squares[i].ID:
            i = i + 1
            if i == 65:
                break
        if i < 65:
            return self.chessBoard.squares[i].srodek
        else:
            return 0

    def FindMarkersPosition(self, img, p1, p2, p3):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # define range of blue color in HSV
        lower = np.array([p1 - 15, p2, p3])
        upper = np.array([p1 + 15, 255, 255])

        # Threshold the HSV image to get only blue colors

        mask = cv2.inRange(hsv, lower, upper)
        # cv2.imshow("maska",mask)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(hsv, img, mask=mask)


        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("RES.jpg",res)
        '''liczenie Houghem'''
        ret, thresh = cv2.threshold(res, 50, 255, 0)

        cv2.imwrite("Maska.jpg", thresh)

        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 6, 50, param1=140,param2=30, minRadius=4, maxRadius=15)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[00, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                if self.settings.DebugMode:
                    cv2.circle(img, (x, y), r, (255, 0, 0), 2)
                    cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

            markers = []
            s = []
            d = []
            for i in range(3):
                markers.append(circles[i][0:2])
                s.append(markers[i][0] + markers[i][1])
                d.append(markers[i][0] - markers[i][1])
            rect = np.zeros((3, 2))

            # the top-left point will have the smallest sum, whereas
            # the bottom-right point will have the largest sum
            rect[0] = markers[s.index(min(s))]
            rect[1] = markers[d.index(max(d))]
            rect[2] = markers[d.index(min(d))]
            self.markerCenter = (int(rect[0][0]), int(rect[0][1]))
            self.markerx = (int(rect[1][0]), int(rect[1][1]))
            self.markery = (int(rect[2][0]), int(rect[2][1]))
            del s
            del rect
            # now, compute the difference between the points, the
            # top-right point will have the smallest difference,
            # whereas the bottom-left will have the largest difference
        else:
            print("Nie znaleziono znaczników bazy")

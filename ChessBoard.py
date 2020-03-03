# coding=utf-8
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2  # For Sobel etc
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from ChessSquare import ChessSquare
from settings import SettingsChessBoard

np.set_printoptions(suppress=True, linewidth=200)  # Better printing of arrays
plt.rcParams['image.cmap'] = 'jet'  # Default colormap is jet


# Saddle

def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)
    S = gxx * gyy - gxy ** 2
    return S


def nonmax_sup(img, win=10):
    w, h = img.shape
    img_sup = np.zeros_like(img, dtype=np.float64)
    for i, j in np.argwhere(img):
        # Get neigborhood
        ta = max(0, i - win)
        tb = min(w, i + win + 1)
        tc = max(0, j - win)
        td = min(h, j + win + 1)
        cell = img[ta:tb, tc:td]
        val = img[i, j]
        if np.sum(cell.max() == cell) > 1:
            print("Nonmax_sup: {}".format(cell.argmax()))
        if cell.max() == val:
            img_sup[i, j] = val
    return img_sup


def pruneSaddle(s):
    thresh = 128
    score = (s > 0).sum()
    while (score > 10000):
        thresh = thresh * 2
        s[s < thresh] = 0
        score = (s > 0).sum()


def getMinSaddleDist(saddle_pts, pt):
    best_dist = None
    best_pt = pt
    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt) ** 2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt
    return best_pt, np.sqrt(best_dist)


# Contour

def simplifyContours(contours):
    for i in range(len(contours)):
        # Approximate contour and update in place
        contours[i] = cv2.approxPolyDP(contours[i], 0.04 * cv2.arcLength(contours[i], True), True)


def is_square(cnt, eps=3.0, xratio_thresh=0.5):
    # 4x2 array, rows are each point, columns are x and y
    center = cnt.sum(axis=0) / 4

    # Side lengths of rectangular contour
    dd0 = np.sqrt(((cnt[0, :] - cnt[1, :]) ** 2).sum())
    dd1 = np.sqrt(((cnt[1, :] - cnt[2, :]) ** 2).sum())
    dd2 = np.sqrt(((cnt[2, :] - cnt[3, :]) ** 2).sum())
    dd3 = np.sqrt(((cnt[3, :] - cnt[0, :]) ** 2).sum())

    # diagonal ratio
    xa = np.sqrt(((cnt[0, :] - cnt[2, :]) ** 2).sum())
    xb = np.sqrt(((cnt[1, :] - cnt[3, :]) ** 2).sum())
    xratio = xa / xb if xa < xb else xb / xa

    # Check whether all points part of convex hull
    # ie. not this http://i.stack.imgur.com/I6yJY.png
    # all corner angles, angles are less than 180 deg, so not necessarily internal angles
    ta = getAngle(dd3, dd0, xb)
    tb = getAngle(dd0, dd1, xa)
    tc = getAngle(dd1, dd2, xb)
    td = getAngle(dd2, dd3, xa)
    angle_sum = np.round(ta + tb + tc + td)

    is_convex = np.abs(angle_sum - 360) < 5

    angles = np.array([ta, tb, tc, td])
    good_angles = np.all((angles > 40) & (angles < 140))

    # side ratios
    dda = dd0 / dd1
    if dda < 1:
        dda = 1. / dda
    ddb = dd1 / dd2
    if ddb < 1:
        ddb = 1. / ddb
    ddc = dd2 / dd3
    if ddc < 1:
        ddc = 1. / ddc
    ddd = dd3 / dd0
    if ddd < 1:
        ddd = 1. / ddd
    side_ratios = np.array([dda, ddb, ddc, ddd])
    good_side_ratios = np.all(side_ratios < eps)

    # Return whether side ratios within certain ratio < epsilon
    return (
        # abs(1.0 - dda) < eps and
        # abs(1.0 - ddb) < eps and
        # xratio > xratio_thresh and
        # good_side_ratios and
        # is_convex and
        good_angles)


def getAngle(a, b, c):
    # Get angle given 3 side lengths, in degrees
    k = (a * a + b * b - c * c) / (2 * a * b)
    # Handle floating point errors
    if (k < -1):
        k = -1
    elif k > 1:
        k = 1
    return np.arccos(k) * 180.0 / np.pi


def getContourVals(cnt, img):
    cimg = np.zeros_like(img)
    cv2.drawContours(cimg, [cnt], 0, color=255, thickness=-1)
    return img[cimg != 0]


def pruneContours(contours, hierarchy, saddle):
    new_contours = []
    new_hierarchies = []
    for i in range(len(contours)):
        cnt = contours[i]
        h = hierarchy[i]

        # Must be child
        if h[2] != -1:
            continue

        # Only rectangular contours allowed
        if len(cnt) != 4:
            continue

        # Only contours that fill an area of at least 8x8 pixels
        if cv2.contourArea(cnt) < 8 * 8:
            continue

        if not is_square(cnt):
            continue

        cnt = updateCorners(cnt, saddle)
        # If not all saddle corners
        if len(cnt) != 4:
            continue

        new_contours.append(cnt)
        new_hierarchies.append(h)
    new_contours = np.array(new_contours)
    new_hierarchy = np.array(new_hierarchies)
    if len(new_contours) == 0:
        return new_contours, new_hierarchy

    # Prune contours below median area
    areas = [cv2.contourArea(c) for c in new_contours]
    mask = [areas >= np.median(areas) * 0.25] and [areas <= np.median(areas) * 2.0]
    new_contours = new_contours[mask]
    new_hierarchy = new_hierarchy[mask]
    return np.array(new_contours), np.array(new_hierarchy)


def getContours(img, edges, iters=10):
    # Morphological Gradient to get internal squares of canny edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    _, contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    simplifyContours(contours)

    return np.array(contours), hierarchy[0]


# Corners

def updateCorners(contour, saddle):
    ws = 4  # half window size (+1)
    new_contour = contour.copy()
    for i in range(len(contour)):
        cc, rr = contour[i, 0, :]
        rl = max(0, rr - ws)
        cl = max(0, cc - ws)
        window = saddle[rl:min(saddle.shape[0], rr + ws + 1), cl:min(saddle.shape[1], cc + ws + 1)]
        br, bc = np.unravel_index(window.argmax(), window.shape)
        s_score = window[br, bc]
        br -= min(ws, rl)
        bc -= min(ws, cl)
        if s_score > 0:
            new_contour[i, 0, :] = cc + bc, rr + br
        else:
            return []
    return new_contour


# Grid

def getIdentityGrid(N):
    a = np.arange(N)
    b = a.copy()
    aa, bb = np.meshgrid(a, b)
    return np.vstack([aa.flatten(), bb.flatten()]).T


def getChessGrid(quad):
    quadA = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    quadB = getIdentityGrid(4) - 1
    quadB_pad = np.pad(quadB, ((0, 0), (0, 1)), 'constant', constant_values=1)
    C_thing = (np.matrix(M) * quadB_pad.T).T
    #     bad = (C_thing[:,2] < 0.3).A.flatten()
    C_thing[:, :2] /= C_thing[:, 2]
    return C_thing


def findGoodPoints(grid, spts, max_px_dist=5):
    # Snap grid points to closest saddle point within range and return updated
    # grid = Nx2 points on grid
    new_grid = grid.copy()
    chosen_spts = set()
    N = len(new_grid)
    grid_good = np.zeros(N, dtype=np.bool)
    hash_pt = lambda pt: "%d_%d" % (pt[0], pt[1])

    for pt_i in range(N):
        pt2, d = getMinSaddleDist(spts, grid[pt_i, :2].A.flatten())
        if hash_pt(pt2) in chosen_spts:
            d = max_px_dist
        else:
            chosen_spts.add(hash_pt(pt2))
        if (d < max_px_dist):  # max dist to replace with
            new_grid[pt_i, :2] = pt2
            grid_good[pt_i] = True
    return new_grid, grid_good


def getInitChessGrid(quad):
    quadA = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    return makeChessGrid(M, 1)


def makeChessGrid(M, N=1):
    ideal_grid = getIdentityGrid(2 + 2 * N) - N
    ideal_grid_pad = np.pad(ideal_grid, ((0, 0), (0, 1)), 'constant', constant_values=1)  # Add 1's column
    # warped_pts = M*pts
    grid = (np.matrix(M) * ideal_grid_pad.T).T
    grid[:, :2] /= grid[:, 2]  # normalize by t
    grid = grid[:, :2]  # remove 3rd column
    return grid, ideal_grid, M


def generateNewBestFit(grid_ideal, grid, grid_good):
    a = np.float32(grid_ideal[grid_good])
    b = np.float32(grid[grid_good])
    M = cv2.findHomography(a, b, cv2.RANSAC)
    return M


def getGrads(img):
    img = cv2.blur(img, (5, 5))
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad_mag = gx * gx + gy * gy
    grad_phase = np.arctan2(gy, gx)  # from -pi to pi
    grad_phase_masked = grad_phase.copy()
    gradient_mask_threshold = 2 * np.mean(grad_mag.flatten())
    grad_phase_masked[grad_mag < gradient_mask_threshold] = np.nan
    return grad_mag, grad_phase_masked, grad_phase, gx, gy


def getBestLines(img_warped):
    grad_mag, grad_phase_masked, grad_phase, gx, gy = getGrads(img_warped)

    # X
    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0

    score_x = np.sum(gx_pos, axis=0) * np.sum(gx_neg, axis=0)
    # Y
    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0
    score_y = np.sum(gy_pos, axis=1) * np.sum(gy_neg, axis=1)

    # Choose best internal set of 7
    a = np.array([(offset + np.arange(7) + 1) * 32 for offset in np.arange(1, 11 - 2)])
    scores_x = np.array([np.sum(score_x[pts]) for pts in a])
    scores_y = np.array([np.sum(score_y[pts]) for pts in a])

    # 15x15 grid, so along an axis a set of 7, and an internal 7 at that, so 13x13 grid, 7x7 possibility inside
    # We're also using a 1-padded grid so 17x17 grid
    # We only want the internal choices (13-7) so 6x6 possible options in the 13x13
    # so 2,3,4,5,6,7,8 to 8,9,10,11,12,13,14 ignoring 0,1 and 15,16,17
    best_lines_x = a[scores_x.argmax()]
    best_lines_y = a[scores_y.argmax()]
    return (best_lines_x, best_lines_y)


def loadImage(filepath):


    img_orig = PIL.Image.open(filepath)
    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(1000.0 / img_width, 1000.0 / img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width, new_height), resample=PIL.Image.BILINEAR)
    imgColor = np.array(img)
    cv2.imwrite("ObrazPoPrzeskalowaniu.jpg", imgColor)
    img = img.convert('L')  # grayscale
    img = np.array(img)
    return img, imgColor


def findChessboard(img, min_pts_needed=15, max_pts_needed=25):
    blur_img = cv2.blur(img, (3, 3))  # Blur it

    saddle = getSaddle(blur_img)
    saddle = -saddle
    saddle[saddle < 0] = 0
    pruneSaddle(saddle)
    s2 = nonmax_sup(saddle)
    s2[s2 < 45000] = 0
    spts = np.argwhere(s2)


    edges = cv2.Canny(img, 10, 124)
    contours_all, hierarchy = getContours(img, edges)
    contours, hierarchy = pruneContours(contours_all, hierarchy, saddle)


    curr_num_good = 0
    curr_grid_next = None
    curr_grid_good = None
    curr_M = None

    for cnt_i in range(len(contours)):
        #print ("On Contour %d" % cnt_i)
        cnt = contours[cnt_i].squeeze()

        grid_curr, ideal_grid, M = getInitChessGrid(cnt)

        for grid_i in range(7):
            grid_curr, ideal_grid, _ = makeChessGrid(M, N=(grid_i + 1))
            grid_next, grid_good = findGoodPoints(grid_curr, spts)
            num_good = np.sum(grid_good)
            # print('I %d (N=%d), num_good: %d of %d' % (grid_i, grid_i+1, num_good, grid_good.size))
            if num_good < 4:
                M = None
                # print ("Failed to converge on this one")
                break
            M, _ = generateNewBestFit(ideal_grid, grid_next, grid_good)
            # Check that a valid and reasonable M was returned
            if M is None or np.abs(M[0, 0] / M[1, 1]) > 15 or np.abs(M[1, 1] / M[0, 0]) > 15:
                #             if M is None:
                M = None
                # print ("Failed to converge on this one")
                break
        if M is None:
            continue
        elif num_good > curr_num_good:
            curr_num_good = num_good
            curr_grid_next = grid_next
            curr_grid_good = grid_good
            curr_M = M

        # If we found something with more than max needed, good enough to stop here
        if num_good > max_pts_needed:
            break

    # If we found something
    if curr_num_good > min_pts_needed:
        final_ideal_grid = getIdentityGrid(2 + 2 * 7) - 7
        return curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts
    else:
        return None, None, None, None, None


def setName(i):
    b2 = int(i % 8)
    a2 = int((i - 1) / 8)

    a = int(i % 8)
    b = int((i - 1) / 8)
    if a == 1:
        litera = 'A'
    elif a == 2:
        litera = 'B'
    elif a == 3:
        litera = 'C'
    elif a == 4:
        litera = 'D'
    elif a == 5:
        litera = 'E'
    elif a == 6:
        litera = 'F'
    elif a == 7:
        litera = 'G'
    elif a == 0:
        litera = 'H'

    if b == 0:
        cyfra = '1'
    elif b == 1:
        cyfra = '2'
    elif b == 2:
        cyfra = '3'
    elif b == 3:
        cyfra = '4'
    elif b == 4:
        cyfra = '5'
    elif b == 5:
        cyfra = '6'
    elif b == 6:
        cyfra = '7'
    elif b == 7:
        cyfra = '8'
    nazwa = litera + cyfra
    return nazwa, a2, b2

def Informcjeopolach(best_lines_x, best_lines_y, M,M2):

    new_best_lines_x = (best_lines_x[6] - best_lines_x[0])/6
    new_best_lines_y = (best_lines_y[6] - best_lines_y[0])/6

    append_best_lines_x = np.concatenate(([int(best_lines_x[0] - new_best_lines_x)], best_lines_x, [int(best_lines_x[len(best_lines_x)-1] + new_best_lines_x)]))
    append_best_lines_y = np.concatenate(([int(best_lines_y[0] - new_best_lines_y)], best_lines_y, [int(best_lines_y[len(best_lines_y)-1] + new_best_lines_y)]))

    x, y = np.meshgrid(append_best_lines_x, append_best_lines_y)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = np.expand_dims(xy, 0)
    xy_unwarp = cv2.perspectiveTransform(xy, M)

    center_best_lines_x = np.append(best_lines_x, [int(best_lines_x[6] + new_best_lines_x)])
    center_best_lines_y = np.append(best_lines_y, [int(best_lines_y[6] + new_best_lines_y)])

    center_x = int((best_lines_x[6] - best_lines_x[0])/6/2)
    center_y = int((best_lines_y[6] - best_lines_y[0])/6/2)

    center_best_lines_x = center_best_lines_x - center_x
    center_best_lines_y = center_best_lines_y - center_y


    x2, y2 = np.meshgrid(center_best_lines_x, center_best_lines_y)
    xy2 = np.vstack([x2.flatten(), y2.flatten()]).T.astype(np.float32)
    xy2 = np.expand_dims(xy2, 0)
    xy_unwarp2 = cv2.perspectiveTransform(xy2, M2)

    lista_pol = []

    for i in range(1, 65):

        nazwa, a, b = setName(i)
        center_pola = np.array([xy_unwarp2[0, (i - 1), 0], xy_unwarp2[0, (i - 1), 1]])

        corners_pola = np.array(
            [xy_unwarp[0, 9 * a + b - 1, 0], xy_unwarp[0, 9 * a + b - 1, 1], xy_unwarp[0, (9 * a + b - 1) + 1, 0],
             xy_unwarp[0, (9 * a - 1) + b + 1, 1], xy_unwarp[0, 9 * a + b + 10 - 1, 0],
             xy_unwarp[0, 9 * a + b + 10 - 1, 1], xy_unwarp[0, 9 * a + 9 + b - 1, 0],
             xy_unwarp[0, 9 * a + 9 - 1 + b, 1]])
        if b == 0:
            corners_pola = np.array(
                [xy_unwarp[0, 9 * a + 16, 0], xy_unwarp[0, 9 * a + 16, 1], xy_unwarp[0, (9 * a) + 17, 0],
                 xy_unwarp[0, (9 * a) + 17, 1], xy_unwarp[0, 9 * a + 8, 0], xy_unwarp[0, 9 * a + 8, 1],
                 xy_unwarp[0, 9 * a + 7, 0], xy_unwarp[0, 9 * a + 7, 1]])

        pole = ChessSquare(name=nazwa, center=center_pola, corners=corners_pola, i=i)
        lista_pol.append(pole)

    return lista_pol


class ChessBoard:
    startImg = ""
    smallImg = ""
    settings = ""
    marker =[]
    centerMarker = 0    # centerMarker określa indeks narożnika, który jest najbliżej kropki-markera (białe pole)
    centerMarker2 = 0   # centerMarker2 określa indeks czarnego pola w narożniku, które jest nabliżej kropki-markera
    corners = []
    squares = []
    settings = []

    def __init__(self, img):
        self.startImg, self.smallImg = loadImage(img)# 'input/img_10.png'

        self.settings = SettingsChessBoard()


    def FindMarkerPosition(self):
        hsv = cv2.cvtColor(self.smallImg, cv2.COLOR_RGB2HSV)

        # define range of green color in HSV
        lower_green = np.array([40, 100, 80])
        upper_green = np.array([80, 255, 255])

        # Threshold the HSV image to get only green colors
        if self.settings.DebugMode:
            cv2.imwrite("./Debug/ObrazHSV.jpg", hsv)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        if self.settings.DebugMode:
            cv2.imwrite("./Debug/MaskaDlaHSV.jpg", mask)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(hsv, self.smallImg, mask=mask)

        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(res, 100, 255, 0)
        if self.settings.DebugMode:
            cv2.imwrite('wykrytakropka.jpg', thresh)
        contours = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        N = cv2.moments(cnt)
        # centroid_x = M10/M00 and centroid_y = M01/M00
        if   N != None:
            x = int(N['m10'] / N['m00'])
            y = int(N['m01'] / N['m00'])
            self.marker = (x, y)
            print("pozycja zielonej kropki:{} {}".format(x, y))
            return 1
        else:
            return 0

    def Plot(self):
        imshow(self.smallImg, cmap='Greys_r')
        grej = cv2.cvtColor(self.smallImg, cv2.COLOR_BGR2GRAY)
        Kani = cv2.Canny(grej, 80,10)
        if self.settings.DebugMode:
            cv2.imwrite("./Debug/GrejOrgObrazu.jpg", grej)
            cv2.imwrite("./Debug/KaniOrgObrazu.jpg",Kani)
        axs = plt.axis()
        for i in range(len(self.squares)):
            self.squares[i].kolo(self.startImg)

            plt.plot(self.squares[i].wierzcholki[0:4, 0], self.squares[i].wierzcholki[0:4, 1], 'g.', )  # plot kropek
            plt.text(self.squares[i].srodek[0],self.squares[i].srodek[1],self.squares[i].nazwa)
            if self.centerMarker == i:
                plt.plot(self.squares[i].srodek[0], self.squares[i].srodek[1], 'r.', )
            elif self.centerMarker2 == i:
                plt.plot(self.squares[i].srodek[0], self.squares[i].srodek[1], 'b.', )
            else:
                plt.plot(self.squares[i].srodek[0], self.squares[i].srodek[1], 'y.', )
            if self.squares[i].czysrodek:
                print("ID pola: {}".format(self.squares[i].ID))
                #plt.plot(self.squares[i].srodekFigury[0], self.squares[i].srodekFigury[1], 'r.', )

        plt.axis(axs)
        plt.savefig('result.png', bbox_inches='tight')
        plt.show()

    def ChangeStartImage(self,img):
        self.startImg = img

    def filterbyvalue(seq, value):
        for el in seq:
            if el.attribute == value: yield el


    def Initialize(self):
        M, ideal_grid, grid_next, grid_good, spts = findChessboard(self.startImg)
        if M is not None:
            M, _ = generateNewBestFit((ideal_grid + 8) * 32, grid_next, grid_good)  # generate mapping for warping image
            img_warp = cv2.warpPerspective(self.startImg, M, (17 * 32, 17 * 32), flags=cv2.WARP_INVERSE_MAP)
            isMarkerFound = self.FindMarkerPosition()
            best_lines_x, best_lines_y = getBestLines(img_warp)
            self.squares = Informcjeopolach(best_lines_x, best_lines_y, M, M)

            if isMarkerFound:
                c=[]

                c.append(np.sqrt(np.power(np.absolute(self.squares[0].srodek[0] - self.marker[0]), 2) + np.power(np.absolute(self.squares[0].srodek[1] - self.marker[1]), 2)))
                c.append(np.sqrt(np.power(np.absolute(self.squares[7].srodek[0] - self.marker[0]), 2) + np.power(np.absolute(self.squares[7].srodek[1] - self.marker[1]), 2)))
                c.append(np.sqrt(np.power(np.absolute(self.squares[56].srodek[0] - self.marker[0]), 2) + np.power(np.absolute(self.squares[56].srodek[1] - self.marker[1]), 2)))
                c.append(np.sqrt(np.power(np.absolute(self.squares[63].srodek[0] - self.marker[0]), 2) + np.power(np.absolute(self.squares[63].srodek[1] - self.marker[1]), 2)))

                self.centerMarker =  c.index(min(c))
                if self.centerMarker == 0:
                    c[self.centerMarker] = 999999
                    self.centerMarker = 0
                if self.centerMarker == 1:
                    c[self.centerMarker] = 999999
                    self.centerMarker = 7
                if self.centerMarker == 2:
                    c[self.centerMarker] = 999999
                    self.centerMarker = 56
                if self.centerMarker == 3:
                    c[self.centerMarker] = 999999
                    self.centerMarker = 63

                self.centerMarker2 = c.index(min(c))
                if self.centerMarker2 == 0:
                    self.centerMarker2 = 0
                if self.centerMarker2 == 1:
                    self.centerMarker2 = 7
                if self.centerMarker2 == 2:
                    self.centerMarker2 = 56
                if self.centerMarker2 == 3:
                    self.centerMarker2 = 63

                print("Indeks białego pola centerMarker: {}".format(self.centerMarker))
                print("Indeks czarnego pola centerMarker2: {}".format(self.centerMarker2))


                if self.centerMarker == 0 and self.centerMarker2 == 56:
                    id=1
                    for i in range(1, 9):
                        for j in range(1, 9):
                            self.squares[(j-1)*8+i-1].ID = id
                            id = id + 1
                    id=1
                    for i in range(1, 9):
                        for j in range(1, 9):
                            self.squares[(8-j)*8+i-1].ID = id
                            id = id + 1
                elif self.centerMarker == 56 and self.centerMarker2 == 63:
                    for i in range(0, 64):
                        self.squares[i].ID = 64 - i
                elif self.centerMarker == 63 and self.centerMarker2 == 7:
                    id=1
                    for i in range(1, 9):
                        for j in range(1, 9):
                            self.squares[(i-1)*8+j-1].ID = id
                            id = id + 1
                    id=1
                    for i in range(1, 65):
                        self.squares[64 - i].ID = id
                        id = id + 1


                self.squares.sort(key=lambda x: x.ID)
                self.centerMarker = 7
                self.centerMarker2 = 0
                for i in range(len(self.squares)):
                    self.squares[i].nazwa,a,b = setName(self.squares[i].ID)





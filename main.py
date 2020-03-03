from VisionSystem import Vision
from Server import Server
import numpy as np
from ChessAPP import ChessAI, GUI
from ColorBalance import colorbalance
from ChessBoard import loadImage

#Setting
zOver = -10
z0 = 113
gripperOpen = False
gripperClose = True
velocity = 0
OriginPosition = [-110 , 201 ,-95 , gripperOpen, 30]
DismissPosition = [655 , 187 ,-178 , gripperOpen, 30]

game = ChessAI()
gui = GUI()
colorbalance()
# 1 step initialization
visionSystem = Vision('Corrected.jpg')

visionSystem.VisionInitializate()

visionSystem.chessBoard.Plot()

DeltaX = (visionSystem.robotBaseSquare[56].srodek[0][0][0] * visionSystem.coefX)- (visionSystem.robotBaseSquare[0].srodek[0][0][0] * visionSystem.coefX)
print(DeltaX)
DeltaY = (visionSystem.robotBaseSquare[56].srodek[0][0][1] * visionSystem.coefY) - (visionSystem.robotBaseSquare[0].srodek[0][0][1] * visionSystem.coefY)
print(DeltaY)

server = Server()


def upuscpionek(i):
    movementFinished = 0

    velocity = 30
    movementFinished = server.Transmission([visionSystem.robotBaseSquare[i].srodek[0][0][0] * visionSystem.coefX,visionSystem.robotBaseSquare[i].srodek[0][0][1] * visionSystem.coefY, zOver,True, velocity])  # False = otwarty chwytak
    while True:
        if movementFinished == 1:
            movementFinished = 0
            break

    velocity = 10
    movementFinished = server.Transmission([visionSystem.robotBaseSquare[i].srodek[0][0][0] * visionSystem.coefX,visionSystem.robotBaseSquare[i].srodek[0][0][1] * visionSystem.coefY, z0,False, velocity])  # False = otwarty chwytak
    while True:
        if movementFinished == 1:
            movementFinished = 0
            break

    velocity = 10
    movementFinished = server.Transmission([visionSystem.robotBaseSquare[i].srodek[0][0][0] * visionSystem.coefX,visionSystem.robotBaseSquare[i].srodek[0][0][1] * visionSystem.coefY, zOver,False, velocity])  # False = otwarty chwytak
    while True:
        if movementFinished == 1:
            return 1

def podniespionek(i):
    piece = visionSystem.VisionPieceCenter(i)
    velocity = 30
    movementFinished = 0

    movementFinished = server.Transmission([piece[0][0][0] * visionSystem.coefX,piece[0][0][1] * visionSystem.coefY, zOver,False, velocity])  # False = otwarty chwytak
    while True:
        if movementFinished == 1:
            movementFinished = 0
            break

    velocity = 10
    movementFinished = server.Transmission([piece[0][0][0] * visionSystem.coefX, piece[0][0][1] * visionSystem.coefY, z0,True, velocity])  # False = otwarty chwytak
    while True:
        if movementFinished == 1:
            movementFinished = 0
            break

    velocity = 10
    movementFinished = server.Transmission([piece[0][0][0] * visionSystem.coefX, piece[0][0][1] * visionSystem.coefY, zOver,True, velocity])  # False = otwarty chwytak
    while True:
        if movementFinished == 1:
            return 1


def origin():

    velocity = 30
    movementFinished = server.Transmission(OriginPosition)
    while True:
        if movementFinished == 1:
            return 1


def dismiss():
    velocity = 30
    movementFinished = server.Transmission(DismissPosition)
    while True:
        if movementFinished == 1:
            return 1

def przesuniecie(p1,p2):
    finish = podniespionek(p1);
    while True:
        if finish == 1:
            break
    finish = upuscpionek(p2);
    while True:
        if finish == 1:
            break
    finish = origin();
    while True:
        if finish == 1:
            break

def bicie(p):
    finish = podniespionek(p);
    while True:
        if finish == 1:
            break
    finish = dismiss();
    while True:
        if finish == 1:
            break



colorbalance()
while True:
    fromPos, toPos, isCapture = gui.loop(game)

    visionSystem.chessBoard.startImg, visionSystem.chessBoard.smallImg = loadImage('Corrected.jpg')
    if isCapture:
        bicie(toPos)
    przesuniecie(fromPos, toPos)
    colorbalance()

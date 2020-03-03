class SettingsVisionSystem:
    p1 = 0
    p2 = 0
    # Hough w FindMarkerPosition do wyszukiwania środków markerów
    h1 = 0
    h2 = 0
    h3 = 0
    h4 = 0
    h5 = 0
    h6 = 0
    # parametry wyszukiwanych kolorów markerów w VisionInitializate
    mx1 = 0
    mx2 = 0
    mx3 = 0

    mc1 = 0
    mc2 = 0
    mc3 = 0

    my1 = 0
    my2 = 0
    my3 = 0

    DebugMode = False

    realXdistance = 0
    realYdistance = 0
    def __init__(self):
        #canny w FindMarkerPosition do wyszukiwania krawędzi markerów
        self.p1 = 160
        self.p2 = 200
        #Hough w FindMarkerPosition do wyszukiwania środków markerów
        self.h1 = 1
        self.h2 = 100
        self.h3 = 150
        self.h4 = 25
        self.h5 = 10
        self.h6 = 80
        #parametry wyszukiwanych kolorów markerów w VisionInitializate
        self.mx1 = 0
        self.mx2 = 50
        self.mx3 = 100

        self.mc1 = 50
        self.mc2 = 80
        self.mc3 = 150

        self.my1 = 120
        self.my2 = 80
        self.my3 = 150

        self.realXdistance = 1
        self.realYdistance = 1
        self.DebugMode = False

class SettingsChessSquare:
    # Hough w kolo do wyszukiwania środków bierek
    h1 = 0
    h2 = 0
    h3 = 0
    h4 = 0
    DebugMode = False

    def __init__(self):
        #Hough w kolo do wyszukiwania środków bierek
        self.h1 = 2.5
        self.h2 = 40
        self.h3 = 12
        self.h4 = 500
        self.DebugMode = False

class SettingsChessBoard:
    DebugMode = False

    def __init__(self):
        self.DebugMode = False
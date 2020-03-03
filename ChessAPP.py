import pygame
import chess


class ChessAI:
    def __init__(self):
        self.ileruchow = 0
        self.PlayerMove = True

        self.pawnEvalWhite = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0,
            0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0,
            0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5,
            0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pawnEvalBlack = self.pawnEvalWhite[::-1]

        self.knightEval = [
            -5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0,
            -4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0,
            -3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0,
            -3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0,
            -3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0,
            -3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0,
            -4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0,
            -5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]
        self.bishopEvalWhite = [
            -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0,
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
            -1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0,
            -1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0,
            -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0,
            -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
            -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0,
            -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0
        ]
        self.bishopEvalBlack = self.bishopEvalWhite[::-1]
        self.rookEvalWhite = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
            -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5,
            -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5,
            -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5,
            -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5,
            -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5,
            0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
        self.rookEvalBlack = self.rookEvalWhite[::-1]
        self.queenEval = [
            -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0,
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
            -1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0,
            -0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5,
            0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5,
            -1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0,
            -1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0,
            -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
        self.kingEvalWhite = [
            -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0,
            -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0,
            -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0,
            -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0,
            -2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0,
            -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0,
            2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0,
            2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0]
        self.kingEvalBlack = self.kingEvalWhite[::-1]

        self.EvalTable = [self.pawnEvalWhite, self.knightEval, self.bishopEvalWhite, self.rookEvalWhite, self.queenEval,
                          self.kingEvalWhite]

        self.board = chess.Board()

    def setPlayerMove(self, playermove):
        self.PlayerMove = playermove

    def AIMove(self, board):
        move = self.GetBestMove(board, 3, True)
        isCapture = board.is_capture(move)
        board.push(move)
        self.PlayerMove = True
        return move.from_square, move.to_square, isCapture

    def GetBestMove(self, board, depth, isMaxingPlayer):
        bestMove = -9999
        bestMoveFound = []
        for i in board.legal_moves:
            board.push(i)
            value = self.MiniMax(board, depth - 1, -10000, 10000, not isMaxingPlayer)
            board.pop()
            if value >= bestMove:
                bestMove = value
                bestMoveFound = i
        return bestMoveFound

    def MiniMax(self, board, depth, alfa, beta, isMaxingPlayer):
        if depth == 0:
            return self.BoardEval(board)
        if isMaxingPlayer:
            bestMove = -9999
            for i in board.legal_moves:
                board.push(i)
                bestMove = max(bestMove, self.MiniMax(board, depth - 1, alfa, beta, not isMaxingPlayer))
                board.pop()
                alfa = max(alfa, bestMove)
                if beta <= alfa:
                    return bestMove
            return bestMove
        else:
            bestMove = 9999
            for i in board.legal_moves:
                board.push(i)
                bestMove = min(bestMove, self.MiniMax(board, depth - 1, alfa, beta, not isMaxingPlayer))
                board.pop()
                beta = min(beta, bestMove)
                if beta <= alfa:
                    return bestMove
            return bestMove

    def BoardEval(self, board):
        PiecesValue = [10, 30, 30, 50, 90, 900]
        BoardValue = 0
        for i in range(1, 7):
            for j in board.pieces(i, False):
                BoardValue = BoardValue + PiecesValue[i - 1] + self.EvalTable[i - 1][j]
            for j in board.pieces(i, True):
                BoardValue = BoardValue - PiecesValue[i - 1] - (self.EvalTable[i - 1][::-1])[j]
        return BoardValue


class GUI:
    def __init__(self):
        self.GREY = [128, 128, 128]
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.width = 360
        self.height = 360
        self.gridwidth = self.width / 8
        self.gridheight = self.height / 8
        self.highlightGreen = []
        self.pressed = False

        self.selectedFigurePosition = ""
        self.bbishop = pygame.image.load("pieces/bbishop.png")
        self.bking = pygame.image.load("pieces/bking.png")
        self.bknight = pygame.image.load("pieces/bknight.png")
        self.bpawn = pygame.image.load("pieces/bpawn.png")
        self.bqueen = pygame.image.load("pieces/bqueen.png")
        self.brook = pygame.image.load("pieces/brook.png")
        self.wbishop = pygame.image.load("pieces/wbishop.png")
        self.wking = pygame.image.load("pieces/wking.png")
        self.wknight = pygame.image.load("pieces/wknight.png")
        self.wpawn = pygame.image.load("pieces/wpawn.png")
        self.wqueen = pygame.image.load("pieces/wqueen.png")
        self.wrook = pygame.image.load("pieces/wrook.png")

        pygame.init()
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("chess")
        self.clock = pygame.time.Clock()
        self.crashed = False

    def drawboard(self, board):
        for i in range(8):
            for j in range(8):
                if board.piece_at(i * 8 + j) == chess.Piece(1, True):
                    self.drawpiece(self.wpawn, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(2, True):
                    self.drawpiece(self.wknight, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(3, True):
                    self.drawpiece(self.wbishop, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(4, True):
                    self.drawpiece(self.wrook, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(5, True):
                    self.drawpiece(self.wqueen, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(6, True):
                    self.drawpiece(self.wking, j, i)

                if board.piece_at(i * 8 + j) == chess.Piece(1, False):
                    self.drawpiece(self.bpawn, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(2, False):
                    self.drawpiece(self.bknight, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(3, False):
                    self.drawpiece(self.bbishop, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(4, False):
                    self.drawpiece(self.brook, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(5, False):
                    self.drawpiece(self.bqueen, j, i)
                if board.piece_at(i * 8 + j) == chess.Piece(6, False):
                    self.drawpiece(self.bking, j, i)

    def makemove(self, game):
        board = game.board
        x, y = pygame.mouse.get_pos()

        for i in 'abcdefgh':
            j = ord(i) - 97
            if j * self.gridwidth <= x <= (j + 1) * self.gridwidth:
                figposx = i
        for i in range(8):
            if (7 - i) * self.gridheight <= y <= (8 - i) * self.gridheight:
                figposy = i + 1
        pos = figposx + str(figposy)
        move = selectedFigurePosition + pos
        move = chess.Move.from_uci(move)
        if move in board.legal_moves:
            isCapture = board.is_capture(move)
            board.push(move)
            game.setPlayerMove(False)
            return move.from_square, move.to_square, isCapture
        else:
            return 0, 0, False

    def drawlegalmoves(self, board):
        global selectedFigurePosition
        x, y = pygame.mouse.get_pos()
        for i in 'abcdefgh':
            j = ord(i) - 97
            if j * self.gridwidth <= x <= (j + 1) * self.gridwidth:
                figposx = i
        for i in range(8):
            if (7 - i) * self.gridheight <= y <= (8 - i) * self.gridheight:
                figposy = i + 1
        selectedFigurePosition = (figposx + str(figposy))
        for i in range(8):
            for j in range(8):
                x = chr(i + 97)
                y = str(j + 1)
                if chess.Move.from_uci(selectedFigurePosition + x + y) in board.legal_moves:
                    self.highlightGreen.append((i, 7 - j))

    def drawpiece(self, piece, polex, poley):
        x = polex * self.gridwidth
        y = self.height - (poley + 1) * self.gridheight
        piece = pygame.transform.scale(piece, (int(self.gridwidth), int(self.gridheight)))
        self.win.blit(piece, (x, y))

    def drawbackground(self, game):
        board = game.board
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    pygame.draw.rect(self.win, self.GREY,
                                     [i * self.gridwidth, j * self.gridheight, self.gridwidth + 1, self.gridheight + 1])

        for i in range(len(self.highlightGreen)):
            pygame.draw.rect(self.win, self.GREEN,
                             [self.highlightGreen[i][0] * self.gridwidth, self.highlightGreen[i][1] * self.gridheight,
                              self.gridwidth + 1,
                              self.gridheight + 1])
        if board.is_check() & game.PlayerMove is True:
            pygame.draw.rect(self.win, self.RED, [self.gridwidth * int(list(board.pieces(6, True))[0] % 8),
                                                  self.height - self.gridheight * (
                                                              (int(list(board.pieces(6, True))[0] / 8)) + 1),
                                                  self.gridwidth + 1, self.gridheight + 1])

        for i in range(8):
            pygame.draw.line(self.win, self.BLACK, [0, i * self.gridheight], [self.width, i * self.gridheight], 4)
            pygame.draw.line(self.win, self.BLACK, [i * self.gridwidth, 0], [i * self.gridwidth, self.height], 4)

    def loop(self, game):
        while True:
            if not self.crashed:
                fromSquare, toSquare = 0, 0
                self.win.fill(self.WHITE)
                self.drawbackground(game)
                self.drawboard(game.board)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.crashed = True
                    if game.PlayerMove == True:

                        if event.type == pygame.MOUSEBUTTONDOWN:
                            self.drawlegalmoves(game.board)
                        if event.type == pygame.MOUSEBUTTONUP:
                            fromSquare, toSquare, isCapture = self.makemove(game)
                            self.highlightGreen.clear()
                            if fromSquare != 0 and toSquare != 0:
                                return fromSquare, toSquare, isCapture

                if game.PlayerMove == False:
                    fromSquare, toSquare, isCapture = game.AIMove(game.board)
                pygame.display.update()
                pygame.display.flip()
                if fromSquare != 0 and toSquare != 0:
                    return fromSquare, toSquare, isCapture


AI = ChessAI()
gui = GUI()
while True:
    gui.loop(AI)
    print("aa")

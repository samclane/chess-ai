from joueur.base_ai import BaseAI
from math import inf
from timeit import default_timer as timer
from collections import namedtuple, defaultdict
from itertools import count
from random import getrandbits
from operator import xor
import re

'''
This board representation is based off the Sunfish Python Chess Engine 
Several changes have been made (most notably to value()) 
This method of board representation is more compact, and significantly faster than the old method
Most notably, it does not use any form of copying or deep-copying
'''

A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  # 0 -  9
    '         \n'  # 10 - 19
    ' rnbqkbnr\n'  # 20 - 29
    ' pppppppp\n'  # 30 - 39
    ' ........\n'  # 40 - 49
    ' ........\n'  # 50 - 59
    ' ........\n'  # 60 - 69
    ' ........\n'  # 70 - 79
    ' PPPPPPPP\n'  # 80 - 89
    ' RNBQKBNR\n'  # 90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

N, E, S, W = -10, 1, 10, -1

# valid moves for each piece
directions = {
    'P': (N, N + N, N + W, N + E),
    'N': (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    'B': (N + E, S + E, S + W, N + W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N + E, S + E, S + W, N + W),
    'K': (N, E, S, W, N + E, S + E, S + W, N + W)
}

piece_values = {
    'P': 1,
    'N': 3,
    'B': 3,
    'R': 5,
    'Q': 9,
    'K': 200
}

z_indicies = {
    'P': 1,
    'N': 2,
    'B': 3,
    'R': 4,
    'Q': 5,
    'K': 6,
    'p': 7,
    'n': 8,
    'b': 9,
    'r': 10,
    'q': 11,
    'k': 12
}

# initialize Zobrist hash table
z_table = [[None] * 12] * 64
for i in range(0, 64):
    for j in range(0, 12):
        z_table[i][j] = getrandbits(16)


class Position(namedtuple('Position', 'board score wc bc ep kp depth captured')):
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    depth - the node depth of the position
    captured - the piece that was captured as the result of the last move
    """

    def gen_moves(self):
        for i, p in enumerate(self.board):
            # i - initial position index
            # p - piece code

            # if the piece doesn't belong to us, skip it
            if not p.isupper(): continue
            for d in directions[p]:
                # d - potential action for a given piece
                for j in count(i + d, d):
                    # j - final position index
                    # q - occupying piece code
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (N, N + N) and q != '.': break
                    if p == 'P' and d == N + N and (i < A1 + N or self.board[i + N] != '.'): break
                    if p == 'P' and d in (N + W, N + E) and q == '.' and j not in (self.ep, self.kp): break
                    # Move it
                    yield (i, j)
                    # Stop non-sliders from sliding and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling by sliding rook next to king
                    if i == A1 and self.board[j + E] == 'K' and self.wc[0]: yield (j + E, j + W)
                    if i == H1 and self.board[j + W] == 'K' and self.wc[1]: yield (j + W, j + E)

    def rotate(self):
        # Rotates the board, preserving enpassant
        # Allows logic to be reused, as only one board configuration must be considered
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep else 0,
            119 - self.kp if self.kp else 0, self.depth, None)

    def nullmove(self):
        # Like rotate, but clears ep and kp
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 0, 0, self.depth + 1, None)

    def move(self, move):
        # i - original position index
        # j - final position index
        i, j = move
        # p - piece code of moving piece
        # q - piece code at final square
        p, q = self.board[i], self.board[j]
        # put replaces string character at i with character p
        put = lambda board, i, p: board[:i] + p + board[i + 1:]
        # copy variables and reset eq and kp and increment depth
        board = self.board
        wc, bc, ep, kp, depth = self.wc, self.bc, 0, 0, self.depth + 1
        # score = self.score + self.value(move)
        # perform the move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # update castling rights, if we move our rook or capture the opponent's rook
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling Logic
        if p == 'K':
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        # Pawn promotion, double move, and en passant capture
        if p == 'P':
            if A8 <= j <= H8:
                # Promote the pawn to Queen
                board = put(board, j, 'Q')
            if j - i == 2 * N:
                ep = i + N
            if j - i in (N + W, N + E) and q == '.':
                board = put(board, j + S, '.')
        # Rotate the returned position so it's ready for the next player
        return Position(board, 0, wc, bc, ep, kp, depth, q.upper()).rotate()

    def value(self):
        score = 0
        # evaluate material advantage
        for k, p in enumerate(self.board):
            # k - position index
            # p - piece code
            if p.isupper(): score += piece_values[p]
            if p.islower(): score -= piece_values[p.upper()]
        return score

    def is_check(self):
        # returns if the state represented by the current position is check
        op_board = self.nullmove()
        for move in op_board.gen_moves():
            i, j = move
            p, q = op_board.board[i], op_board.board[j]
            # opponent can take our king
            if q == 'k':
                return True
        return False

    def z_hash(self):
        # Zobrist Hash of board position
        # strip all whitespace from board
        stripboard = re.sub(r'[\s+]', '', self.board)
        h = 0
        for i in range(0, 64):
            j = z_indicies.get(stripboard[i], 0)
            h = xor(h, z_table[i][j - 1])
        return h


####################################
# square formatting helper functions
####################################

def square_index(file_index, rank_index):
    # Gets a square index by file and rank index
    file_index = ord(file_index.upper()) - 65
    rank_index = int(rank_index) - 1
    return A1 + file_index - (10 * rank_index)


def square_file(square_index):
    file_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    return file_names[(square_index % 10) - 1]


def square_rank(square_index):
    return 10 - (square_index // 10)


def square_san(square_index):
    # convert square index (21 - 98) to Standard Algebraic Notation
    square = namedtuple('square', 'file rank')
    return square(square_file(square_index), square_rank(square_index))


def fen_to_position(fen_string):
    # generate a Position object from a FEN string
    board, player, castling, enpassant, halfmove, move = fen_string.split()
    board = board.split('/')
    board_out = '         \n         \n'
    for row in board:
        board_out += ' '
        for piece in row:
            if piece.isdigit():
                for _ in range(int(piece)):
                    board_out += '.'
            else:
                board_out += piece
        board_out += '\n'
    board_out += '         \n         \n'

    wc = (False, False)
    bc = (False, False)
    if 'K' in castling: wc = (True, wc[1])
    if 'Q' in castling: wc = (wc[0], True)
    if 'k' in castling: bc = (True, bc[1])
    if 'q' in castling: bc = (bc[0], True)

    if enpassant != '-':
        enpassant = square_index(enpassant[0], enpassant[1])
    else:
        enpassant = 0

    # Position(board score wc bc ep kp depth)
    if player == 'w':
        return Position(board_out, 0, wc, bc, enpassant, 0, 0, None)
    else:
        return Position(board_out, 0, wc, bc, enpassant, 0, 0, None).rotate()


class AI(BaseAI):
    """ The basic AI functions that are the same between games. """

    def get_name(self):
        """ This is the name you send to the server so your AI will control the
        player named this string.

        Returns
            str: The name of your Player.
        """

        return "Sawyer McLane"

    def start(self):
        """ This is called once the game starts and your AI knows its playerID
        and game. You can initialize your AI here.
        """
        # store a sign controlling addition or subtraction so pieces move in the right direction
        self.board = fen_to_position(self.game.fen)

    def game_updated(self):
        """ This is called every time the game's state updates, so if you are
        tracking anything you can update it here.
        """

        # replace with your game updated logic
        self.update_board()

    def end(self, won, reason):
        """ This is called when the game ends, you can clean up your data and
        dump files here if need be.

        Args:
            won (bool): True means you won, False means you lost.
            reason (str): The human readable string explaining why you won or
                          lost.
        """
        pass
        # replace with your end logic

    def run_turn(self):
        """ This is called every time it is this AI.player's turn.

        Returns:
            bool: Represents if you want to end your turn. True means end your
                  turn, False means to keep your turn going and re-call this
                  function.
        """

        # Here is where you'll want to code your AI.

        # We've provided sample code that:
        #    1) prints the board to the console
        #    2) prints the opponent's last move to the console
        #    3) prints how much time remaining this AI has to calculate moves
        #    4) makes a random (and probably invalid) move.

        # 1) print the board to the console
        self.print_current_board()

        # 2) print the opponent's last move to the console
        if len(self.game.moves) > 0:
            print("Opponent's Last Move: '" + self.game.moves[-1].san + "'")

        # 3) print how much time remaining this AI has to calculate moves
        print("Time Remaining: " + str(self.player.time_remaining) + " ns")

        # 4) make a move
        (piece_index, move_index) = self.tlabiddl_minimax()

        # flip board indicies if playing from other side
        if self.player.color == "Black":
            piece_index = 119 - piece_index
            move_index = 119 - move_index

        # convert indices to SAN
        piece_pos = square_san(piece_index)
        move_pos = square_san(move_index)
        piece = self.get_game_piece(piece_pos.rank, piece_pos.file)
        piece.move(move_pos.file, move_pos.rank, promotionType="Queen")

        return True  # to signify we are done with our turn.

    def get_game_piece(self, rank, file):
        # used to go between rank and file notation and actual game object
        return next((piece for piece in self.game.pieces if piece.rank == rank and piece.file == file), None)

    def update_board(self):
        # update current board state by converting current FEN to Position object
        self.board = fen_to_position(self.game.fen)

    def tlabiddl_minimax(self):
        # Time Limited Alpha Beta Iterative-Deepening Depth-Limited MiniMax
        initial_board = self.board
        l_depth = 0
        depth_limit = 4
        # time limiting stuff
        time_limit = 10  # 10 seconds to find the best move
        start_time = timer()
        # history stuff
        history = defaultdict(dict)

        def quiescence(board, alpha=(-inf), beta=(inf)):
            stand_pat = board.value()
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat

            for move in board.gen_moves():
                if (timer() - start_time) >= time_limit:
                    # if time limit has been reached, give us the best move
                    return alpha
                next_board = board.move(move)
                score = -quiescence(next_board, -beta, -alpha)
                if score >= beta:
                    history[initial_board.z_hash()][board.z_hash()] = board.depth * board.depth
                    return beta
                if score > alpha:
                    alpha = score
            return alpha

        while l_depth <= depth_limit:
            frontier = [initial_board]
            visited = [initial_board]
            while len(frontier) != 0:
                # sort frontier by prune history
                frontier = sorted(frontier, key=lambda x: history[initial_board.z_hash()].get(x.z_hash(), 0))
                board = frontier.pop(0)
                best_score = -inf
                for move in board.gen_moves():
                    next_board = board.move(move)
                    if next_board.is_check(): continue
                    score = quiescence(next_board)
                    if score > best_score:
                        best_move = move
                        best_score = score
                    if not (next_board in visited) and not (next_board in frontier):
                        visited.append(next_board)
                    if (timer() - start_time) >= time_limit:
                        # if time limit has been reached, give us the best move
                        return best_move
            if len(frontier) == 0:
                l_depth += 1
        return best_move

    def print_current_board(self):
        """Prints the current board using pretty ASCII art
        Note: you can delete this function if you wish
        """

        # iterate through the range in reverse order
        for r in range(9, -2, -1):
            output = ""
            if r == 9 or r == 0:
                # then the top or bottom of the board
                output = "   +------------------------+"
            elif r == -1:
                # then show the ranks
                output = "     a  b  c  d  e  f  g  h"
            else:  # board
                output = " " + str(r) + " |"
                # fill in all the files with pieces at the current rank
                for file_offset in range(0, 8):
                    # start at a, with with file offset increasing the char
                    f = chr(ord("a") + file_offset)
                    current_piece = None
                    for piece in self.game.pieces:
                        if piece.file == f and piece.rank == r:
                            # then we found the piece at (file, rank)
                            current_piece = piece
                            break

                    code = "."  # default "no piece"
                    if current_piece:
                        # the code will be the first character of their type
                        # e.g. 'Q' for "Queen"
                        code = current_piece.type[0]

                        if current_piece.type == "Knight":
                            # 'K' is for "King", we use 'N' for "Knights"
                            code = "N"

                        if current_piece.owner.id == "1":
                            # the second player (black) is lower case.
                            # Otherwise it's uppercase already
                            code = code.lower()

                    output += " " + code + " "

                output += "|"
            print(output)

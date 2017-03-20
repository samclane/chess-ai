# This is where you build your AI for the Chess game.

from joueur.base_ai import BaseAI
import random
import re
import numpy as np
from queue import *
from math import inf
import copy

FORWARD = np.array((1, 0))
RIGHT = np.array((0, 1))
BACKWARD = np.array((-1, 0))
LEFT = np.array((0, -1))

VALID_MOVES = {
    'Pawn': [FORWARD, FORWARD + FORWARD, FORWARD + LEFT, FORWARD + RIGHT],
    'Knight': [
        FORWARD + FORWARD + RIGHT, RIGHT + FORWARD + RIGHT, RIGHT + BACKWARD + RIGHT, BACKWARD + BACKWARD + RIGHT,
        BACKWARD + BACKWARD + LEFT,
        LEFT + BACKWARD + LEFT, LEFT + FORWARD + LEFT, FORWARD + FORWARD + LEFT],
    'Bishop': [FORWARD + RIGHT, BACKWARD + RIGHT, BACKWARD + LEFT, FORWARD + LEFT],
    'Rook': [FORWARD, RIGHT, BACKWARD, LEFT],
    'Queen': [FORWARD, RIGHT, BACKWARD, LEFT, FORWARD + RIGHT, BACKWARD + RIGHT, BACKWARD + LEFT, FORWARD + LEFT],
    'King': [FORWARD, RIGHT, BACKWARD, LEFT, FORWARD + RIGHT, BACKWARD + RIGHT, BACKWARD + LEFT, FORWARD + LEFT]
}

PIECE_COST = {
    'Pawn': 1,
    'Knight': 3,
    'Bishop': 3,
    'Rook': 5,
    'Queen': 9,
    'King': 200
}

PIECE_CODES = {
    'p': 'Pawn',
    'n': 'Knight',
    'b': 'Bishop',
    'r': 'Rook',
    'q': 'Queen',
    'k': 'King'
}


# holds team specific information
class Player:
    def __init__(self, color):
        self.color = color
        if self.color is 'w':
            self.direction = 1
            self.case = str.upper
            self.case_check = str.isupper
            self.opponent_color = 'b'
        elif self.color is 'b':
            self.direction = -1
            self.case = str.lower
            self.case_check = str.islower
            self.opponent_color = 'w'
        else:
            raise TypeError
        self.pieces = []

    def __eq__(self, other):
        return self.color == other.color

    # returns instantiation of opponent
    def get_opponent(self):
        if self.color is 'w':
            return Player('b')
        else:
            return Player('w')


# holds info on pieces. Called actor to avoid naming conflict with built-in game class
class Actor:
    def __init__(self, parent_board, rank, file, type, color, captured=False, has_moved=False):
        self._parent_board = parent_board
        self.rank = rank
        self.file = file
        self.type = PIECE_CODES[type.lower()]  # type is entire word
        self.color = color
        self.captured = captured
        self.has_moved = has_moved

    def move(self, rank, file, promotion_type):
        self._parent_board.move_piece(self, rank, file, promotion_type)


# AI representation of board state. Can be created and modified independent of actual game state
class Board:
    def __init__(self, fen_state_data):
        self._board = {}
        fen = fen_state_data.split()
        board_data = fen[0]
        self.active_player = fen[1]
        self.castle = fen[2]
        self.enpasse = fen[3]
        self.half_move = fen[4]
        self.full_move = fen[5]
        self.is_gameover = False
        # build internal board representation
        rank = 8
        files = [c for c in "abcdefgh"]
        for row in board_data.split('/'):
            new_row = []
            for file in row:
                if file.isdigit():
                    for _ in range(int(file)):
                        new_row.append(None)
                else:
                    new_row.append(file)
            self._board[rank] = dict(zip(files, new_row))
            rank -= 1

    def move_piece(self, old_rank, old_file, new_rank, new_file):
        piece = self._board[old_rank][old_file]
        captured_piece = self._board[new_rank][new_file]
        if captured_piece is not None and captured_piece.lower() is 'k':
            if captured_piece.isupper() and self.active_player is 'b':
                self.is_gameover = True
            elif captured_piece.islower() and self.active_player is 'w':
                self.is_gameover = True
        self._board[old_rank][old_file] = None
        self._board[new_rank][new_file] = piece
        if captured_piece is None and piece.lower() is not 'p':
            self.half_move = chr(int(self.half_move) + 1)
        self.full_move = chr(int(self.full_move) + 1)
        return captured_piece

    def get_fen(self):
        fen_string = []
        for rank in range(8, 0, -1):
            row = self._board[rank]
            empty_count = 0
            for file in file_range('a', 'i'):
                piece = row[file]
                if piece is not None:
                    fen_string.append(row[file])
                else:
                    empty_count += 1
                    if chr(ord(file) + 1) not in row.keys() or row[chr(ord(file) + 1)] is not None:
                        fen_string.append(str(empty_count))
                        empty_count = 0
            if rank != 1:
                fen_string.append('/')
        fen_string.append(' ')
        fen_string.append(' '.join([self.active_player, self.castle, self.enpasse, self.half_move, self.full_move]))
        return ''.join(fen_string)

    def get_piece(self, rank, file):
        if not ((1 <= rank <= 8) and (ord('a') <= ord(file) <= ord('h'))):
            return None
        piece = self._board[rank][file]
        if piece is None:
            return None
        if piece.isupper():
            color = 'w'
        else:
            color = 'b'
        return Actor(self, rank, file, piece, color)

    def __copy__(self):
        return Board(self.get_fen())


# Holds metadata surrounding board state. Comparable to node in tree
class State:
    def __init__(self, board, parent=None, depth=0):
        self.board = board
        self.parent = parent
        self.children = []
        self.parent_move = None
        self.depth = depth
        self.heuristic_value = 0
        self.player = Player(self.board.active_player)
        self.opponent = self.player.get_opponent()
        # create and assign Actor objects for both sides
        for rank in range(1, 9):
            for file in file_range('a', 'i'):
                piece = self.board.get_piece(rank, file)
                if piece is not None:
                    if piece.color == self.player.color:
                        self.player.pieces.append(piece)
                    elif piece.color == self.opponent.color:
                        self.opponent.pieces.append(piece)
                    else:
                        raise Exception("Unclaimed piece")

    # is this a game winning state?
    def is_gameover(self):
        return self.board.is_gameover

    # generates all available moves from this state. Comparable to the ACTION(s) function
    def get_available_moves(self):
        action_list = []
        for piece in self.player.pieces:
            moveset = list(VALID_MOVES[piece.type])
            for move in moveset:
                curr_cell = np.array((piece.rank, ord(piece.file)))
                new_cell = curr_cell + (self.player.direction * move)
                new_cell = np.array((new_cell[0], new_cell[1]))
                # ensure move is in bounds
                if not ((1 <= new_cell[0] <= 8) and (ord('a') <= new_cell[1] <= ord('h'))):
                    continue
                occupying_piece = self.board.get_piece(int(new_cell[0]), chr(new_cell[1]))
                # ensure we are not taking our own piece
                if occupying_piece is not None:
                    if occupying_piece.color is self.player.color:
                        continue
                if piece.type == "Pawn":
                    # check if we're blocked
                    front_cell = curr_cell + (self.player.direction * FORWARD)
                    if (move == FORWARD).all() and self.board.get_piece(int(front_cell[0]),
                                                                        chr(front_cell[1])) is not None:
                        continue
                    # ensure double jump may occur
                    if (move == FORWARD + FORWARD).all():
                        if self.player.color == "w":
                            req_rank = 2
                        else:
                            req_rank = 7
                        if int(curr_cell[0]) != req_rank:
                            continue
                        # check if pawn is blocked
                        front_cell2 = front_cell + (self.player.direction * FORWARD)
                        if self.board.get_piece(int(front_cell[0]),
                                                chr(front_cell[1])) is not None or self.board.get_piece(
                            int(front_cell2[0]), chr(front_cell2[1])) is not None:
                            continue
                    # must take an opponent piece if moving diagonally
                    if (move == FORWARD + RIGHT).all() or (move == FORWARD + LEFT).all():
                        # if move is the enpassant, allow it
                        if self.board.enpasse is not '-' and move == self.board.enpasse:
                            pass
                        # otherwise must take a piece if moving diagonally
                        elif occupying_piece is None or occupying_piece.color is self.player.color:
                            continue
                if piece.type == "King":
                    # if castle is allowed in the FEN
                    if len(self.board.castle) > 0:
                        # check if castle is allowed and append it to the moveset
                        if self.check_castle(piece, 'qs'):
                            moveset.append(RIGHT + RIGHT)
                        if self.check_castle(piece, 'ks'):
                            moveset.append(LEFT + LEFT)
                if not self.find_if_check(piece, new_cell):
                    # move is valid, append to list
                    action_list.append((piece, new_cell))
                # if piece is not a slider or has taken a piece, end movement
                if piece.type in ("Pawn", "Knight", "King") or (
                                occupying_piece is not None and occupying_piece.color is not self.player.color):
                    continue
                # otherwise continue in direction
                else:
                    moveset.append(incr_move(move))
        return action_list

    # spaghetti code to see if a castle is available.
    def check_castle(self, piece, side):
        if side == "qs":
            sidemod = 1
        if side == "ks":
            sidemod = -1
        if self.board.get_piece(piece.rank, chr(ord(piece.file) + (1 * sidemod))) is not None:
            return False
        if self.board.get_piece(piece.rank, chr(ord(piece.file) + (2 * sidemod))) is not None:
            return False
        if self.board.get_piece(piece.rank, chr(ord(piece.file) + (3 * sidemod))) is None:
            return False
        if self.board.get_piece(piece.rank, chr(ord(piece.file) + (3 * sidemod))).type is not "Rook":
            return False
        if self.board.get_piece(piece.rank, chr(ord(piece.file) + (3 * sidemod))).has_moved:
            return False
        return True

    # Helper function to get_available_moves(), checking if the proposed action would result in a check
    def find_if_check(self, moved_piece, move_space):
        """
        Given the current state and an action to complete, gives the resultant state
        :return:
        """
        is_check = False
        # simulate move
        temp_board = copy.copy(self.board)
        temp_board.move_piece(moved_piece.rank, moved_piece.file, move_space[0], chr(move_space[1]))
        # generate opponent moves
        for piece in self.opponent.pieces:
            # if piece is captured (or will be captured)
            if piece.captured or (piece.rank, piece.file) == (move_space[0], chr(move_space[1])):
                continue
            moveset = list(VALID_MOVES[piece.type])
            for move in moveset:
                curr_cell = np.array((piece.rank, ord(piece.file)))
                new_cell = curr_cell + (self.opponent.direction * move)
                new_cell = np.array((new_cell[0], new_cell[1]))
                # ensure move is in bounds
                if not ((1 <= new_cell[0] <= 8) and (ord('a') <= new_cell[1] <= ord('h'))):
                    continue
                if piece.type == "Pawn":
                    # only add danger zones
                    if not (move == FORWARD + LEFT).all() and not (move == FORWARD + RIGHT).all():
                        continue
                occupying_piece = temp_board.get_piece(int(new_cell[0]), chr(new_cell[1]))
                # if move intersects piece
                if occupying_piece is not None:
                    # if it takes our king
                    if occupying_piece.type == "King" and occupying_piece.color is self.player.color:
                        is_check = True
                    # end of piece movement regardless
                    continue
                # see if piece is slider
                if piece.type in ("Pawn", "Knight", "King") or occupying_piece is not None:
                    continue
                else:
                    moveset.append(incr_move(move))
        return is_check

    def __eq__(self, other):
        if isinstance(other, State):
            return self.board == other.board
        return False


class Action:
    def __init__(self, state, piece, move_space):
        self.state_in = state
        self.piece = piece
        self.move_space = move_space
        self.board_out = copy.copy(state.board)

    def perform_move(self):
        self.board_out.move_piece(self.piece, self.move_space)
        return State(self.board_out, self.state_in, depth=self.state_in.depth + 1)


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
        self.board = Board(self.game.fen)
        self.state = State(self.board)
        # create FEN regex to extract info
        self.fen_regex = re.compile('[^/]+/[^/]+/[^/]+/[^/]+/[^/]+/[^/]+/[^/]+/\S+ . (\S+) (\S+) (\d+)')
        # create initial representation of the board
        self.update_board()

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

        # 4) make a random (and probably invalid) move.
        action_list = self.state.get_available_moves()
        (piece, move) = random.choice(action_list)
        piece = self.get_game_piece(piece.rank, piece.file)
        print(piece.id + ":" + ", ".join(
            [str(chr(pair[1][1]) + str(pair[1][0])) for pair in action_list if
             (pair[0].rank, pair[0].file) == (piece.rank, piece.file)]))
        piece.move(chr(move[1]), int(move[0]), promotionType="Queen")
        return True  # to signify we are done with our turn.

    # used to go between rank and file notation and actual game object
    def get_game_piece(self, rank, file):
        return next((piece for piece in self.game.pieces if piece.rank == rank and piece.file == file), None)

    def update_board(self):
        self.board = Board(self.game.fen)
        self.state = State(self.board)

    def check_castle(self, piece, side):
        if side == "qs":
            sidemod = 1
        if side == "ks":
            sidemod = -1
        if self.get_game_piece(piece.rank, chr(ord(piece.file) + (1 * sidemod))) is not None:
            return False
        if self.get_game_piece(piece.rank, chr(ord(piece.file) + (2 * sidemod))) is not None:
            return False
        if self.get_game_piece(piece.rank, chr(ord(piece.file) + (3 * sidemod))) is None:
            return False
        if self.get_game_piece(piece.rank, chr(ord(piece.file) + (3 * sidemod))).type is not "Rook":
            return False
        if self.get_game_piece(piece.rank, chr(ord(piece.file) + (3 * sidemod))).has_moved:
            return False
        return True

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


# returns iterator of characters between char1 and char2
def file_range(char1, char2, incr=1):
    for c in range(ord(char1), ord(char2), incr):
        yield chr(c)


def incr_move(move_tuple):
    ret_list = []
    for dist in move_tuple:
        if dist < 0:
            ret_list.append(dist - 1)
        elif dist > 0:
            ret_list.append(dist + 1)
        else:
            ret_list.append(0)
    return np.array(ret_list)


# check if array is in list
def is_arr_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem is myarr), False)

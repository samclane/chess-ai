# This is where you build your AI for the Chess game.

from joueur.base_ai import BaseAI
import random
import re
import numpy as np

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
    'King': [FORWARD, RIGHT, BACKWARD, LEFT]
}


class AI(BaseAI):
    """ The basic AI functions that are the same between games. """

    def get_name(self):
        """ This is the name you send to the server so your AI will control the
        player named this string.

        Returns
            str: The name of your Player.
        """

        return "Sawyer McLane"  # REPLACE THIS WITH YOUR TEAM NAME

    def start(self):
        """ This is called once the game starts and your AI knows its playerID
        and game. You can initialize your AI here.
        """
        # store a sign controlling addition or subtraction so pieces move in the right direction
        self.board = {}
        if self.player.color == "White":
            self.direction = 1
            self.case = str.isupper
        else:
            self.direction = -1
            self.case = str.islower
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
        action_list = self.actions()
        (piece, move) = random.choice(action_list)
        print(", ".join([str(chr(pair[1][1])+str(pair[1][0])) for pair in action_list if pair[0] == piece]))
        piece.move(chr(move[1]), int(move[0]))
        return True  # to signify we are done with our turn.

    def actions(self):
        """
        Generate a list of all possible actions that can be made
        :return:
        """
        action_list = []
        for piece in self.player.pieces:
            # ignore captured pieces
            if piece.captured:
                continue
            moveset = list(VALID_MOVES[piece.type])
            for move in moveset:
                curr_cell = np.array((piece.rank, ord(piece.file)))
                new_cell = curr_cell + (self.direction * move)
                new_cell = np.array((new_cell[0], new_cell[1]))
                occupying_piece = self.get_piece(int(new_cell[0]), chr(new_cell[1]))
                # ensure move is in bounds
                if not ((1 <= new_cell[0] <= 8) and (ord('a') <= new_cell[1] <= ord('h'))):
                    continue
                # ensure we are not taking our own piece
                if occupying_piece is not None:
                    if occupying_piece.owner is self.player:
                        continue
                if piece.type == "Pawn":
                    # check if we're blocked
                    front_cell = curr_cell + (self.direction * FORWARD)
                    if (move == FORWARD).all() and self.get_piece(int(front_cell[0]), chr(front_cell[1])) is not None:
                        continue
                    # ensure double jump may occur
                    if (move == FORWARD + FORWARD).all():
                        if self.player.color == "White":
                            req_rank = 2
                        else:
                            req_rank = 7
                        if int(curr_cell[0]) != req_rank:
                            continue
                        # check if pawn is blocked
                        front_cell2 = front_cell + (self.direction * FORWARD)
                        if self.get_piece(int(front_cell[0]), chr(front_cell[1])) is not None or self.get_piece(
                                int(front_cell2[0]), chr(front_cell2[1])) is not None:
                            continue
                    # must take an opponent piece if moving diagonally
                    if (move == FORWARD + RIGHT).all() or (move == FORWARD + LEFT).all():
                        if occupying_piece is None or occupying_piece.owner is self.player:
                            continue
                if not self.find_if_check(piece, new_cell):
                    # move is valid, append to list
                    action_list.append((piece, new_cell))
                # if piece is not a slider or has taken a piece, end movement
                if piece.type in ("Pawn", "Knight", "King") or (
                                occupying_piece is not None and occupying_piece.owner is not self.player):
                    continue
                # otherwise continue in direction
                else:
                    moveset.append(itermove(move))
        return action_list

    def find_if_check(self, moved_piece, move_space):
        """
        Given the current state and an action to complete, gives the resultant state
        :return:
        """
        self.update_board()
        is_check = False
        # simulate move
        temp_piece = self.get_piece(move_space[0], move_space[1])
        self.board[moved_piece.rank][moved_piece.file] = None
        self.board[move_space[0]][move_space[1]] = moved_piece
        # generate opponent moves
        for piece in self.player.opponent.pieces:
            # if piece is captured (or is to be captured)
            if piece.captured or (piece.rank, piece.file) == move_space:
                continue
            moveset = list(VALID_MOVES[piece.type])
            for move in moveset:
                curr_cell = np.array((piece.rank, ord(piece.file)))
                new_cell = curr_cell + (self.direction * move)
                new_cell = np.array((new_cell[0], new_cell[1]))
                # ensure move is in bounds
                if not ((1 <= new_cell[0] <= 8) and (ord('a') <= new_cell[1] <= ord('h'))):
                    continue
                occupying_piece = self.board[int(new_cell[0])][chr(new_cell[1])]
                if piece.type == "Pawn":
                    # only add danger zones
                    if not (move == FORWARD + LEFT).all() or not (move == FORWARD + RIGHT).all():
                        continue
                # if move intersects piece
                if occupying_piece is not None:
                    if occupying_piece == self.get_king():
                        is_check = True
                    # end of piece movement regardless
                    continue
                # see if piece is slider
                if piece.type in ("Pawn", "Knight", "King"):
                    continue
                else:
                    moveset.append(itermove(move))
        # reset the board
        self.board[move_space[0]][move_space[1]] = temp_piece
        self.board[moved_piece.rank][moved_piece.file] = moved_piece
        self.update_board()
        return is_check

    def get_piece(self, rank, file):
        return next((piece for piece in self.game.pieces if piece.rank == rank and piece.file == file), None)

    def get_king(self):
        return next((piece for piece in self.player.pieces if piece.type == "King"))

    def update_board(self):
        for rank in range(1, 9):
            self.board[rank] = {}
            for file in file_range('a', 'i'):
                self.board[rank][file] = self.get_piece(rank, file)
        match_group = self.fen_regex.findall(self.game.fen)[0]
        self.castle = list(filter(lambda x: self.case(x), match_group[0]))
        self.enpasse = match_group[1]
        self.half_turn = match_group[2]

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
def file_range(char1, char2):
    for c in range(ord(char1), ord(char2)):
        yield chr(c)


def itermove(move_tuple):
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

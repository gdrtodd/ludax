from dataclasses import dataclass
from collections import namedtuple

import jax.numpy as jnp
from lark import Lark
from lark.visitors import Visitor

from config import BoardShapes

@dataclass
class GameInfo:
    board_shape: str = None
    observation_shape: tuple = None
    board_dims: tuple = None
    board_size: int = None
    hex_diameter: int = None
    game_state_class: type = None
    game_state_attributes: list = None

    def __repr__(self):
        return f"GameInfo(board_shape={self.board_shape}, observation_shape={self.observation_shape}, board_size={self.board_size}, hex_diameter={self.hex_diameter})"

class GameInfoExtractor(Visitor):
    def __init__(self):
        self.game_info = GameInfo()

        # These attributes are shared across all games
        self.game_state_attributes = [
            "board",
            "current_player",
            "phase_idx",
            "phase_step_count",
            "previous_actions"
        ]

        self.defaults = []
    
    def __call__(self, tree):
        self.visit_topdown(tree)
        
        # We dynamically create a class for the game state that includes only the necessary attributes
        defaults = self.defaults if self.defaults else None
        game_state_class = namedtuple("GameState", self.game_state_attributes, defaults=defaults)
        self.game_info.game_state_class = game_state_class
        self.game_info.game_state_attributes = self.game_state_attributes

        return self.game_info

    def board(self, tree):

        shape_tree = tree.children[0]
        board_shape = str(shape_tree.data)
        self.game_info.board_shape = board_shape

        if board_shape == BoardShapes.SQUARE:
            size = int(shape_tree.children[0])
            self.game_info.observation_shape = (size, size, 2)
            self.game_info.board_dims = (size, size)
            self.game_info.board_size = size ** 2        

        elif board_shape == BoardShapes.RECTANGLE:
            width, height = map(int, shape_tree.children)
            self.game_info.observation_shape = (width, height, 2)
            self.game_info.board_dims = (width, height)
            self.game_info.board_size = width * height

        elif board_shape == BoardShapes.HEXAGON:
            diameter = int(shape_tree.children[0])
            assert diameter % 2 == 1, "Hexagon board diameter must be odd!"

            self.game_info.observation_shape = (diameter, 2*diameter-1, 2)
            self.game_info.board_dims = (diameter, 2*diameter-1)
            self.game_info.hex_diameter = diameter

            # Consider rings of tiles moving outward from the center tile.
            # The total number of tiles in a hex-hex board is: 1 + [(i* 6) for i in range(1, diameter-1)]
            self.game_info.board_size = 1 + sum([(i * 6) for i in range(1, diameter//2 + 1)])

        elif board_shape == BoardShapes.HEX_RECTANGLE:
            width, height = map(int, shape_tree.children)
            self.game_info.observation_shape = (width, height, 2)
            self.game_info.board_dims = (width, height)
            self.game_info.board_size = width * height    

        else:
            raise NotImplementedError(f"Board shape {board_shape} not implemented yet!")
    
    def force_pass(self, tree):
        if "passed" not in self.game_state_attributes:
            self.game_state_attributes.append("passed")
            self.defaults.append(jnp.zeros(2, dtype=jnp.bool_))

    def function_connected(self, tree):
        '''
        Checking if two regions are connected requires computing the connected components of the board
        '''
        if "connected_components" not in self.game_state_attributes:
            self.game_state_attributes.append("connected_components")
            self.defaults.append(jnp.zeros(self.game_info.board_size, dtype=jnp.int16))

    def play_effects(self, tree):
        '''
        Play effects might require referencing the score. Currently, effects are the only way
        to change the score so we don't need to separately look for things like the "score" function
        '''
        if "scores" not in self.game_state_attributes:
            self.game_state_attributes.append("scores")
            self.defaults.append(jnp.zeros(2, dtype=jnp.float32))

if __name__ == '__main__':
    grammar = open('grammar.lark').read()
    parser = Lark(grammar, start='game')

    game = open('games/tic_tac_toe.ldx').read()
    tree = parser.parse(game)

    info = GameInfoExtractor()(tree)

    print(info)
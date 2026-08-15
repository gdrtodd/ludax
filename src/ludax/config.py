from typing import Any
try:
    from enum import StrEnum  # Python ≥ 3.11
except ImportError: # Python ≤ 3.10
    from enum import Enum
    class StrEnum(str, Enum):
        """Minimal stand-in for the 3.11 enum.StrEnum."""
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

import jax.numpy as jnp

from .struct import dataclass

Array = Any
PRNGKey = Any

# Configurable dtypes for quantization profiling
BOARD_DTYPE = jnp.int8     # board cells, piece/player indices, integer masks
ACTION_DTYPE = jnp.int16   # action indices, step/phase counters
REWARD_DTYPE = jnp.float32 # rewards and value estimates

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

INVALID = BOARD_DTYPE(-2)  # used to represent invalid cells when projecting hex grid onto rectangular array
EMPTY = BOARD_DTYPE(-1)
P1 = BOARD_DTYPE(0)
P2 = BOARD_DTYPE(1)

MAX_STEP_COUNT = 2000

@dataclass
class State():
    '''
    The current state of the environment

    Attributes:
    - game_state: a custom class that contains all the information necessary to advance the game state (e.g. board, current player, ...)
    - current_player: the identity of the current player (copied from game_state for convenience)
    - legal_action_mask: a mask indicating which actions are legal
    - winner: the player who won the game
    - rewards: the rewards for each player
    - mover_reward: the reward for the player who made the last move
    - terminated: whether the game has ended
    - truncated: whether the game has been truncated for lasting too many steps
    - global_step_count: the number of steps taken in the game
    '''
    game_state: type
    current_player: Array
    legal_action_mask: Array
    winners: Array = EMPTY * jnp.ones(2, BOARD_DTYPE)
    rewards: Array = jnp.array([0.0, 0.0], dtype=REWARD_DTYPE)
    mover_reward: Array = REWARD_DTYPE(0.0)
    terminated: Array = FALSE
    truncated: Array = FALSE
    global_step_count: Array = ACTION_DTYPE(0)

class ActionTypes(StrEnum):
    TO = 'action_to'
    FROM_DIR = 'action_from_dir'
    FROM_TO = 'action_from_to'

class Shapes(StrEnum):
    SQUARE = 'square_shape'
    RECTANGLE = 'rectangle_shape'
    HEXAGON = 'hexagon_shape'
    HEX_RECTANGLE = 'hex_rectangle_shape'

class EdgeTypes(StrEnum):
    TOP = 'top'
    BOTTOM = 'bottom'
    LEFT = 'left'
    RIGHT = 'right'

    # Hexagonal edges
    TOP_LEFT = 'top_left'
    TOP_RIGHT = 'top_right'
    BOTTOM_LEFT = 'bottom_left'
    BOTTOM_RIGHT = 'bottom_right'

class Directions(StrEnum):
    ANY = 'any'
    BACK_DIAGONAL = 'back_diagonal'
    DOWN = 'down'
    DOWN_LEFT = 'down_left'
    DOWN_RIGHT = 'down_right'
    FORWARD_DIAGONAL = 'forward_diagonal'
    DIAGONAL = 'diagonal'
    HORIZONTAL = 'horizontal'
    LEFT = 'left'
    ORTHOGONAL = 'orthogonal'
    RIGHT = 'right'
    UP = 'up'
    UP_LEFT = 'up_left'
    UP_RIGHT = 'up_right'
    VERTICAL = 'vertical'

class RelativeDirections(StrEnum):
    FORWARD = 'forward'
    BACKWARD = 'backward'
    FORWARD_LEFT = 'forward_left'
    FORWARD_RIGHT = 'forward_right'
    BACKWARD_LEFT = 'backward_left'
    BACKWARD_RIGHT = 'backward_right'

class Orientations(StrEnum):
    ANY = 'any'
    BACK_DIAGONAL = 'back_diagonal'
    FORWARD_DIAGONAL = 'forward_diagonal'
    DIAGONAL = 'diagonal'
    HORIZONTAL = 'horizontal'
    ORTHOGONAL = 'orthogonal'
    VERTICAL = 'vertical'

class PieceShapes(StrEnum):
    CIRCLE = 'circle'
    SQUARE = 'square'
    TRIANGLE = 'triangle'
    STAR = 'star'
    DIAMOND = 'diamond'

class PlayPhases(StrEnum):
    SPECIFIC_PLAYER = 'phase_specific_player'
    ALTERNATE = 'phase_alternate'
    ALTERNATURE_UNTIL = 'phase_alternate_until'

class PlayTypes(StrEnum):
    MOVE = 'play_move'
    PLACE = 'play_place'
    REMOVE = 'play_remove'

class PlayerAndMoverRefs(StrEnum):
    MOVER = 'mover'
    OPPONENT = 'opponent'
    P1 = 'P1'
    P2 = 'P2'
    BOTH = 'both'

class GameResult(StrEnum):
    WIN = 'result_win'
    LOSS = 'result_lose'
    DRAW = 'result_draw'
    BY_SCORE = 'result_by_score'

class Functions(StrEnum):
    CONNECTED = 'function_connected'
    LINE = 'function_line'
    PATTERN = 'function_pattern'

class Masks(StrEnum):
    ADJACENT = 'mask_adjacent'
    CORNER_CUSTODIAL = 'mask_corner_custodial'
    CUSTODIAL = 'mask_custodial'
    EMPTY = 'mask_empty'
    LINE_OF = 'mask_line_of'
    LOOP = 'mask_loop'
    OCCUPIED = 'mask_occupied'

class MoveTypes(StrEnum):
    PLACE = 'move_place'
    HOP = 'move_hop'
    JUMP = 'move_jump'
    SLIDE = 'move_slide'
    STEP = 'move_step'

# A bit odd, but pieces are either "any" or are defined by name elsewhere
class PieceRefs(StrEnum):
    ANY = 'any'

class PlayEffects(StrEnum):
    CAPTURE = 'effect_capture'
    EXTRA_TURN = 'effect_extra_turn'
    FLIP = 'effect_flip'
    PROMOTE = 'effect_promote'

class Predicates(StrEnum):
    CAN_HOP = 'predicate_can_hop'
    EXISTS = 'predicate_exists'

class OptionalArgs(StrEnum):
    CAPTURE = 'capture_arg'
    DIRECTION = 'direction_arg'
    DISTANCE = 'distance_arg'
    EXACT = 'exact_arg'
    EXCLUDE = 'exclude_arg'
    HOP_OVER = 'hop_over_arg'
    INCREMENT_SCORE = 'increment_score_arg'
    INDICES = 'indices_arg'
    ORIENTATION = 'orientation_arg'
    MOVER = 'mover_arg'
    MULTI_MASK = 'multi_mask_arg'
    PATTERN = 'pattern_arg'
    PIECE = 'piece_arg'
    PLAYER = 'player_arg'
    PRIORITY = 'priority_arg'
    ROTATE = 'rotate_arg'
    SAME_PIECE = 'same_piece_arg'
    SHAPE = 'shape_arg'

DEFAULT_ARGUMENTS = {
    Functions.CONNECTED: {OptionalArgs.PIECE: 'any', OptionalArgs.MOVER: 'mover', OptionalArgs.DIRECTION: 'any'},
    Functions.LINE: {OptionalArgs.ORIENTATION: 'any', OptionalArgs.EXACT: False, OptionalArgs.PLAYER: 'mover', OptionalArgs.EXCLUDE: None},
    Functions.PATTERN: {OptionalArgs.ROTATE: False, OptionalArgs.PLAYER: 'mover', OptionalArgs.EXCLUDE: None},

    Masks.ADJACENT: {OptionalArgs.DIRECTION: 'any'},
    Masks.CORNER_CUSTODIAL: {OptionalArgs.MOVER: 'mover'},
    Masks.CUSTODIAL: {OptionalArgs.MOVER: 'mover', OptionalArgs.ORIENTATION: 'any'},
    Masks.LINE_OF: {OptionalArgs.DIRECTION: 'any', OptionalArgs.DISTANCE: None},
    Masks.LOOP: {OptionalArgs.MOVER: 'mover'},
    Masks.OCCUPIED: {OptionalArgs.MOVER: 'mover'},

    MoveTypes.HOP: {OptionalArgs.DIRECTION: 'any', OptionalArgs.PIECE: 'any', OptionalArgs.HOP_OVER: 'both', OptionalArgs.CAPTURE: False, OptionalArgs.PRIORITY: 0},
    MoveTypes.JUMP: {OptionalArgs.DISTANCE: None, OptionalArgs.PRIORITY: 0},
    MoveTypes.SLIDE: {OptionalArgs.DIRECTION: 'any', OptionalArgs.DISTANCE: None, OptionalArgs.PRIORITY: 0},
    MoveTypes.STEP: {OptionalArgs.DIRECTION: 'any', OptionalArgs.PRIORITY: 0},

    PlayEffects.CAPTURE: {OptionalArgs.MOVER: 'opponent', OptionalArgs.INCREMENT_SCORE: False},
    PlayEffects.EXTRA_TURN: {OptionalArgs.SAME_PIECE: False},
    PlayEffects.FLIP: {OptionalArgs.MOVER: 'opponent'},
    PlayEffects.PROMOTE: {OptionalArgs.MOVER: 'mover'},

    Predicates.CAN_HOP: {OptionalArgs.DIRECTION: 'any', OptionalArgs.PIECE: 'any', OptionalArgs.MOVER: 'both'},
}


RENDER_CONFIG = {
    "cell_size": 50,
    "piece_radius": 35,
    "legal_radius": 15,
    "hexagon_orientation": "pointy",

    # Colors
    "light_blue": "#d2e6ff",
    "light_grey": "#c3cdd8",
    "white": "#fafafa",
    "black": "#323232",
    "dark_grey": "#a6a6a6",
    "purple": "#8a7af0",

    # Region colors (cycled through for multiple regions)
    "region_colors": [
        "#ffe4b5",  # light orange
        "#e4ffb5",  # light lime
        "#ffb5e4",  # light pink
        "#b5ffb5",  # light green
        "#e4b5ff",  # light purple
    ],

    # SVG Style information
    "svg_style": """
        <style>
        .fade-in-out {
            animation: fadeOutIn 2s ease-in-out;
        }
        @keyframes fadeOutIn {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }
        </style>
    """
}
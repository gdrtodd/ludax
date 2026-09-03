from functools import partial
import logging
import typing

import jax
import jax.numpy as jnp
import numpy as np

from .config import EMPTY, INVALID, Shapes, Directions, RelativeDirections, EdgeTypes, Orientations, OptionalArgs, PieceRefs, PlayerAndMoverRefs, P1, P2, BOARD_DTYPE, ACTION_DTYPE
from .game_info import GameInfo

BOARD_SHAPE_TO_DIRECTIONS = {
    Shapes.SQUARE: [
            Directions.UP_LEFT, Directions.UP, Directions.UP_RIGHT, 
            Directions.LEFT, Directions.RIGHT, 
            Directions.DOWN_LEFT, Directions.DOWN, Directions.DOWN_RIGHT
    ],
    Shapes.RECTANGLE: [
            Directions.UP_LEFT, Directions.UP, Directions.UP_RIGHT, 
            Directions.LEFT, Directions.RIGHT, 
            Directions.DOWN_LEFT, Directions.DOWN, Directions.DOWN_RIGHT
    ],
    Shapes.HEXAGON: [
            Directions.UP_LEFT, Directions.UP_RIGHT, 
            Directions.LEFT, Directions.RIGHT, 
            Directions.DOWN_LEFT, Directions.DOWN_RIGHT
    ],
    Shapes.HEX_RECTANGLE: [
            Directions.UP_LEFT, Directions.UP_RIGHT, 
            Directions.LEFT, Directions.RIGHT, 
            Directions.DOWN_LEFT, Directions.DOWN_RIGHT
    ]
}

META_DIRECTION_MAPPING = {
    Directions.DIAGONAL: [Directions.UP_LEFT, Directions.UP_RIGHT, Directions.DOWN_LEFT, Directions.DOWN_RIGHT],
    Directions.ORTHOGONAL: [Directions.UP, Directions.DOWN, Directions.LEFT, Directions.RIGHT],
    Directions.VERTICAL: [Directions.UP, Directions.DOWN],
    Directions.HORIZONTAL: [Directions.LEFT, Directions.RIGHT],
    Directions.FORWARD_DIAGONAL: [Directions.UP_RIGHT, Directions.DOWN_LEFT],
    Directions.BACK_DIAGONAL: [Directions.UP_LEFT, Directions.DOWN_RIGHT],
    Directions.ANY: [
        Directions.UP_LEFT, Directions.UP, Directions.UP_RIGHT, 
        Directions.LEFT, Directions.RIGHT, 
        Directions.DOWN_LEFT, Directions.DOWN, Directions.DOWN_RIGHT
    ]
}

# Maps from a player's forward direction (first key) and a relative direction (second key) to an absolute direction
RELATIVE_DIRECTION_MAPPING = {
    Directions.UP: {RelativeDirections.FORWARD: Directions.UP, RelativeDirections.BACKWARD: Directions.DOWN,
                    RelativeDirections.FORWARD_LEFT: Directions.UP_LEFT, RelativeDirections.FORWARD_RIGHT: Directions.UP_RIGHT,
                    RelativeDirections.BACKWARD_LEFT: Directions.DOWN_LEFT, RelativeDirections.BACKWARD_RIGHT: Directions.DOWN_RIGHT},

    Directions.DOWN: {RelativeDirections.FORWARD: Directions.DOWN, RelativeDirections.BACKWARD: Directions.UP,
                        RelativeDirections.FORWARD_LEFT: Directions.DOWN_RIGHT, RelativeDirections.FORWARD_RIGHT: Directions.DOWN_LEFT,
                        RelativeDirections.BACKWARD_LEFT: Directions.UP_RIGHT, RelativeDirections.BACKWARD_RIGHT: Directions.UP_LEFT},

    Directions.LEFT: {RelativeDirections.FORWARD: Directions.LEFT, RelativeDirections.BACKWARD: Directions.RIGHT,
                      RelativeDirections.FORWARD_LEFT: Directions.DOWN_LEFT, RelativeDirections.FORWARD_RIGHT: Directions.UP_LEFT,
                      RelativeDirections.BACKWARD_LEFT: Directions.UP_RIGHT, RelativeDirections.BACKWARD_RIGHT: Directions.DOWN_RIGHT},

    Directions.RIGHT: {RelativeDirections.FORWARD: Directions.RIGHT, RelativeDirections.BACKWARD: Directions.LEFT,
                      RelativeDirections.FORWARD_LEFT: Directions.UP_RIGHT, RelativeDirections.FORWARD_RIGHT: Directions.DOWN_RIGHT,
                      RelativeDirections.BACKWARD_LEFT: Directions.DOWN_LEFT, RelativeDirections.BACKWARD_RIGHT: Directions.UP_LEFT}
}

# Maps from a player's forward direction (first key) and a relative edge type (second key) to an absolute edge type
# indicating the furthest edge in that direction
RELATIVE_EDGE_TYPE_MAPPING = {
    Directions.UP: {RelativeDirections.FORWARD: EdgeTypes.TOP, RelativeDirections.BACKWARD: EdgeTypes.BOTTOM},
    Directions.DOWN: {RelativeDirections.FORWARD: EdgeTypes.BOTTOM, RelativeDirections.BACKWARD: EdgeTypes.TOP},
    Directions.LEFT: {RelativeDirections.FORWARD: EdgeTypes.LEFT, RelativeDirections.BACKWARD: EdgeTypes.RIGHT},
    Directions.RIGHT: {RelativeDirections.FORWARD: EdgeTypes.RIGHT, RelativeDirections.BACKWARD: EdgeTypes.LEFT}
}

def _get_adjacency_kernel(game_info: GameInfo, optional_args: dict):
    '''
    Returns a two-dimensional kernel for the current game that can be used to precompute the adjacency
    lookup information
    '''
    if game_info.board_shape == Shapes.SQUARE or game_info.board_shape == Shapes.RECTANGLE:
        kernel = jnp.zeros((1, 1, 3, 3), dtype=BOARD_DTYPE)
        if optional_args[OptionalArgs.DIRECTION] in [Directions.UP_LEFT, Directions.DIAGONAL, Directions.BACK_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 0, 0].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.UP, Directions.VERTICAL, Directions.ORTHOGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 0, 1].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.UP_RIGHT, Directions.DIAGONAL, Directions.FORWARD_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 0, 2].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.LEFT, Directions.HORIZONTAL, Directions.ORTHOGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 1, 0].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.RIGHT, Directions.HORIZONTAL, Directions.ORTHOGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 1, 2].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.DOWN_LEFT, Directions.DIAGONAL, Directions.FORWARD_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 2, 0].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.DOWN, Directions.VERTICAL, Directions.ORTHOGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 2, 1].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.DOWN_RIGHT, Directions.DIAGONAL, Directions.BACK_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 2, 2].set(1)

    elif game_info.board_shape == Shapes.HEX_RECTANGLE:
        kernel = jnp.zeros((1, 1, 3, 3), dtype=BOARD_DTYPE)
        # The "back diagonal" in hex-rectangle board is, oddly, the same X position as the original
        if optional_args[OptionalArgs.DIRECTION] in [Directions.UP_LEFT, Directions.DIAGONAL, Directions.BACK_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 0, 1].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.UP_RIGHT, Directions.DIAGONAL, Directions.FORWARD_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 0, 2].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.LEFT, Directions.HORIZONTAL, Directions.ORTHOGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 1, 0].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.RIGHT, Directions.HORIZONTAL, Directions.ORTHOGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 1, 2].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.DOWN_LEFT, Directions.DIAGONAL, Directions.FORWARD_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 2, 0].set(1)

        # See above
        if optional_args[OptionalArgs.DIRECTION] in [Directions.DOWN_RIGHT, Directions.DIAGONAL, Directions.BACK_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 2, 1].set(1)
    
    elif game_info.board_shape == Shapes.HEXAGON:
        kernel = jnp.zeros((1, 1, 5, 5), dtype=BOARD_DTYPE)
        if optional_args[OptionalArgs.DIRECTION] in [Directions.UP_LEFT, Directions.DIAGONAL, Directions.BACK_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 1, 1].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.UP_RIGHT, Directions.DIAGONAL, Directions.FORWARD_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 1, 3].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.LEFT, Directions.HORIZONTAL, Directions.ORTHOGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 2, 0].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.RIGHT, Directions.HORIZONTAL, Directions.ORTHOGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 2, 4].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.DOWN_LEFT, Directions.DIAGONAL, Directions.FORWARD_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 3, 1].set(1)

        if optional_args[OptionalArgs.DIRECTION] in [Directions.DOWN_RIGHT, Directions.DIAGONAL, Directions.BACK_DIAGONAL, Directions.ANY]:
            kernel = kernel.at[:, :, 3, 3].set(1)

        # NOTE: hexagonal boards do not have a canonical 'up' or 'down' direction. We just ignore those inputs, since they're technically
        # grammatical (i.e. we can't determine that they're invalid when checking syntax, since we don't know the board shape yet)

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")
    
    return kernel

def _get_adjacency_lookup(game_info: GameInfo):
    '''
    Constructs an array that can be used to look up the adjacencies of any position on the board. The resulting
    array will be [C, M, M] where C is the number of channels (8 for rectangular boards, 6 for hexagonal boards)
    and M is the number of positions on the board. The value at [c, m1, m2] indicates whether m2 is adjacent to
    m1 in direction c.
    '''

    directions = BOARD_SHAPE_TO_DIRECTIONS[game_info.board_shape]
    mask_to_board, idx_to_pos, board_to_mask = _get_mask_board_conversion_fns(game_info)

    adjacency_array = jnp.zeros((len(directions), game_info.board_size, game_info.board_size), dtype=BOARD_DTYPE)
    for channel_idx, direction in enumerate(directions):
        kernel = _get_adjacency_kernel(game_info, {OptionalArgs.DIRECTION: direction})

        for i in range(game_info.board_size):
            mask = jnp.zeros((game_info.board_size,), dtype=BOARD_DTYPE).at[i].set(1)
            board_2d = mask_to_board(mask)
            board_2d = board_2d[jnp.newaxis, jnp.newaxis, :, :]
            board_2d = jnp.where(board_2d == INVALID, EMPTY, board_2d)

            board_2d = board_2d.astype(BOARD_DTYPE)
            kernel = kernel.astype(BOARD_DTYPE)

            conv_out = jax.lax.conv(board_2d, kernel, (1, 1), 'SAME')[0][0] > 0
            out_mask = board_to_mask(conv_out)

            adjacency_array = adjacency_array.at[channel_idx, i].set(out_mask)

    return adjacency_array

def _get_direction_indices(game_info: GameInfo, directions: typing.Union[Directions, list[Directions]]):
    '''
    Returns the indices corresponding to the channels in the adjacency lookup for the given (meta)direction.
    This function also supports "relative" directions like "forward" if they've been defined in the game's
    player assignments.

    Returns the direction indices for each player separately (which will be the same for non-relative directions)
    '''
    all_directions = BOARD_SHAPE_TO_DIRECTIONS[game_info.board_shape]

    p1_query_dirs = []
    p2_query_dirs = []

    if isinstance(directions, str):
        directions = [directions]

    for direction in directions:

        if direction in map(str, RelativeDirections):
            p1_direction = _get_relative_direction(game_info, RelativeDirections(direction), P1)
            p2_direction = _get_relative_direction(game_info, RelativeDirections(direction), P2)

        else:
            p1_direction = direction
            p2_direction = direction
        

        p1_query_dirs += META_DIRECTION_MAPPING.get(p1_direction, [p1_direction])
        p2_query_dirs += META_DIRECTION_MAPPING.get(p2_direction, [p2_direction])

    p1_dir_indices = [all_directions.index(_dir) for _dir in p1_query_dirs if _dir in all_directions]
    p2_dir_indices = [all_directions.index(_dir) for _dir in p2_query_dirs if _dir in all_directions]

    p1_dir_indices = jnp.array(list(sorted(p1_dir_indices)), dtype=BOARD_DTYPE)
    p2_dir_indices = jnp.array(list(sorted(p2_dir_indices)), dtype=BOARD_DTYPE)

    return p1_dir_indices, p2_dir_indices

def _get_relative_direction(game_info: GameInfo, direction: RelativeDirections, player: int):
    forward = game_info.forward_directions[player]
    relative_direction = RELATIVE_DIRECTION_MAPPING[forward][direction]
    return relative_direction

def _get_relative_edge_type(game_info: GameInfo, direction: RelativeDirections, player: int):
    forward = game_info.forward_directions[player]
    relative_edge_type = RELATIVE_EDGE_TYPE_MAPPING[forward][direction]
    return relative_edge_type

def _get_slide_lookup(game_info: GameInfo):
    '''
    Returns an array that can be used to get the board indices that are in a line with
    a given position on the board in a given direction. The resulting array will be of
    shape [C, M, N] where C is the number of channels (8 for rectangular boards, 6 for
    hexagonal boards), M is the number of positions on the board, and N is the maximum
    number of positions in a line in that direction.
    '''
    directions = BOARD_SHAPE_TO_DIRECTIONS[game_info.board_shape]
    num_board_positions = game_info.board_size

    num_line_positions = max(game_info.board_dims) if game_info.board_shape != Shapes.HEXAGON else game_info.hex_diameter

    mask_to_board, idx_to_pos, board_to_mask = _get_mask_board_conversion_fns(game_info)
    dummy_board = mask_to_board(jnp.zeros(num_board_positions, dtype=BOARD_DTYPE))
    n_rows, n_cols = dummy_board.shape

    slide_lookup = jnp.zeros((len(directions), num_board_positions, num_line_positions), dtype=ACTION_DTYPE)

    for channel_idx, direction in enumerate(directions):
        for i in range(num_board_positions):
            row, col = idx_to_pos(i)

            # For hex-rectangle boards, the "up-left" and "down-right" directions are just straight up and down, so we have to handle those differently
            # NOTE: this behavior is visually intuitive but somewhat mechanically unintuitive, since it means certain "diagonal" moves only displace
            # a piece in one axis instead of two
            if direction == Directions.UP_LEFT:
                if game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE]:
                    indices = [np.ravel_multi_index((r, c), dummy_board.shape) for r, c in zip(range(row, -1, -1), range(col, -1, -1))]
                
                elif game_info.board_shape == Shapes.HEX_RECTANGLE:
                    indices = [np.ravel_multi_index((r, col), dummy_board.shape) for r in range(row, -1, -1)]
                
                elif game_info.board_shape == Shapes.HEXAGON:
                    board_indices = np.array([[r, c] for r, c in zip(range(row, -1, -1), range(col, -1, -1))], dtype=BOARD_DTYPE)
                    occupied_board = dummy_board.at[board_indices[:, 0], board_indices[:, 1]].set(1)
                    indices = np.argwhere(board_to_mask(occupied_board) == 1).flatten()[::-1].tolist()

            elif direction == Directions.UP:
                indices = [np.ravel_multi_index((r, col), game_info.board_dims) for r in range(row, -1, -1)]

            elif direction == Directions.UP_RIGHT:
                if game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE, Shapes.HEX_RECTANGLE]:
                    indices = [np.ravel_multi_index((r, c), game_info.board_dims) for r, c in zip(range(row, -1, -1), range(col, n_cols))]
                
                elif game_info.board_shape == Shapes.HEXAGON:
                    board_indices = np.array([[r, c] for r, c in zip(range(row, -1, -1), range(col, n_cols))], dtype=BOARD_DTYPE)
                    occupied_board = dummy_board.at[board_indices[:, 0], board_indices[:, 1]].set(1)
                    indices = np.argwhere(board_to_mask(occupied_board) == 1).flatten()[::-1].tolist()

            elif direction == Directions.LEFT:
                if game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE, Shapes.HEX_RECTANGLE]:
                    indices = [np.ravel_multi_index((row, c), game_info.board_dims) for c in range(col, -1, -1)]
                
                elif game_info.board_shape == Shapes.HEXAGON:
                    start_col = abs((n_rows // 2) - row) - 1
                    board_indices = np.array([[row, c] for c in range(col, start_col, -2)], dtype=BOARD_DTYPE)
                    occupied_board = dummy_board.at[board_indices[:, 0], board_indices[:, 1]].set(1)
                    indices = np.argwhere(board_to_mask(occupied_board) == 1).flatten()[::-1].tolist()

            elif direction == Directions.RIGHT:
                if game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE, Shapes.HEX_RECTANGLE]:
                    indices = [np.ravel_multi_index((row, c), game_info.board_dims) for c in range(col, n_cols)]
                
                elif game_info.board_shape == Shapes.HEXAGON:
                    end_col = n_cols - abs((n_rows // 2) - row) + 1
                    board_indices = np.array([[row, c] for c in range(col, end_col, 2)], dtype=BOARD_DTYPE)
                    occupied_board = dummy_board.at[board_indices[:, 0], board_indices[:, 1]].set(1)
                    indices = np.argwhere(board_to_mask(occupied_board) == 1).flatten().tolist()


            elif direction == Directions.DOWN_LEFT:
                if game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE, Shapes.HEX_RECTANGLE]:
                    indices = [np.ravel_multi_index((r, c), game_info.board_dims) for r, c in zip(range(row, n_rows), range(col, -1, -1))]
                
                elif game_info.board_shape == Shapes.HEXAGON:
                    board_indices = np.array([[r, c] for r, c in zip(range(row, n_rows), range(col, -1, -1))], dtype=BOARD_DTYPE)
                    occupied_board = dummy_board.at[board_indices[:, 0], board_indices[:, 1]].set(1)
                    indices = np.argwhere(board_to_mask(occupied_board) == 1).flatten().tolist()

            elif direction == Directions.DOWN:
                indices = [np.ravel_multi_index((r, col), game_info.board_dims) for r in range(row, n_rows)]

            # See note above about hex-rectangle boards
            elif direction == Directions.DOWN_RIGHT:
                if game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE]:
                    indices = [np.ravel_multi_index((r, c), game_info.board_dims) for r, c in zip(range(row, n_rows), range(col, n_cols))]
                
                elif game_info.board_shape == Shapes.HEX_RECTANGLE:
                    indices = [np.ravel_multi_index((r, col), game_info.board_dims) for r in range(row, n_rows)]
                
                elif game_info.board_shape == Shapes.HEXAGON:
                    board_indices = np.array([[r, c] for r, c in zip(range(row, n_rows), range(col, n_cols))], dtype=BOARD_DTYPE)
                    occupied_board = dummy_board.at[board_indices[:, 0], board_indices[:, 1]].set(1)
                    indices = np.argwhere(board_to_mask(occupied_board) == 1).flatten().tolist()

            # Pad the indices with num_board_positions+1 to ensure that the resulting array has the correct shape
            indices = jnp.array(indices + [num_board_positions + 1] * (num_line_positions - len(indices)), dtype=ACTION_DTYPE)
            
            slide_lookup = slide_lookup.at[channel_idx, i].set(indices)
    
    return slide_lookup

def _get_mask_board_conversion_fns(game_info: GameInfo):
    '''
    Return functions for the current game that can be used to convert between a flattened
    board mask and a two-dimensional board representation
    '''
    if game_info.board_shape == Shapes.SQUARE or game_info.board_shape == Shapes.RECTANGLE:
        mask_to_board = lambda mask: mask.reshape(game_info.board_dims)
        mask_idx_to_board_pos = lambda idx: jnp.unravel_index(idx, game_info.board_dims)
        board_to_mask = lambda board: board.flatten()
        return mask_to_board, mask_idx_to_board_pos, board_to_mask

    elif game_info.board_shape == Shapes.HEX_RECTANGLE:
        mask_to_board = lambda mask: mask.reshape(game_info.board_dims)
        mask_idx_to_board_pos = lambda idx: jnp.unravel_index(idx, game_info.board_dims)
        board_to_mask = lambda board: board.flatten()
        return mask_to_board, mask_idx_to_board_pos, board_to_mask
    
    elif game_info.board_shape == Shapes.HEXAGON:
        diameter = game_info.hex_diameter
        offset = (diameter // 2) % 2

        # Apply checkerboard pattern
        base = np.zeros(game_info.board_dims, dtype=BOARD_DTYPE)
        offset_indices = np.argwhere(np.ones_like(base))[offset::2]
        base[tuple(offset_indices.T)] = 1

        # Mask out the corners
        stair_width = diameter // 2
        tri_indices = np.triu_indices(stair_width)

        ur_stair = np.ones((stair_width, stair_width), dtype=bool)
        ur_stair[tri_indices] = False
        ul_stair = np.fliplr(ur_stair)
        lr_stair = np.flipud(ur_stair)
        ll_stair = np.flipud(ul_stair)

        ul_slices = (slice(0, stair_width), slice(0, stair_width))
        ur_slices = (slice(0, stair_width), slice(-stair_width, None))
        ll_slices = (slice(-stair_width, None), slice(0, stair_width))
        lr_slices = (slice(-stair_width, None), slice(-stair_width, None))

        stairs = [ul_stair, ur_stair, ll_stair, lr_stair]
        corner_slices = [ul_slices, ur_slices, ll_slices, lr_slices]

        for stair_mask, corner_slice in zip(stairs, corner_slices):
            base[corner_slice] = base[corner_slice] & stair_mask

        valid_hex_indices = jnp.argwhere(base == 1).T
        base_board = jnp.ones_like(base, dtype=BOARD_DTYPE) * INVALID

        def mask_to_board(mask):
            return base_board.at[tuple(valid_hex_indices)].set(mask)
        
        def mask_idx_to_board_pos(idx):
            return valid_hex_indices.T[idx]
        
        def board_to_mask(board):
            return board[tuple(valid_hex_indices)].flatten()
        
        return mask_to_board, mask_idx_to_board_pos, board_to_mask

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")

def _get_column_indices(game_info: GameInfo, column_idx: int):
    '''
    Return the mask indices corresponding to the given column on the current game's board. Recall that
    in the canonical orientation of hexagonal boards, there are no columns
    '''
    if game_info.board_shape == Shapes.SQUARE or game_info.board_shape == Shapes.RECTANGLE:
        height, width = game_info.board_dims
        indices = jnp.arange(column_idx, height * width, width)

    elif game_info.board_shape == Shapes.HEXAGON:
        indices = jnp.array([])

    elif game_info.board_shape == Shapes.HEX_RECTANGLE:
        indices = jnp.array([])

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")

    return indices.astype(ACTION_DTYPE)

def _get_corner_indices(game_info: GameInfo):
    '''
    Return the mask indices corresponding to the corners of the current game's board
    '''
    if game_info.board_shape == Shapes.SQUARE or game_info.board_shape == Shapes.RECTANGLE:
        height, width = game_info.board_dims
        indices = jnp.array([
            0, width-1,
            (height-1)*width, height*width-1
        ])

    elif game_info.board_shape == Shapes.HEXAGON:
        half_diameter = game_info.hex_diameter // 2
        midpoint = game_info.board_size // 2
        indices = jnp.array([
            0, half_diameter,
            midpoint - half_diameter, midpoint + half_diameter,
            game_info.board_size - half_diameter - 1, game_info.board_size - 1
        ])

    elif game_info.board_shape == Shapes.HEX_RECTANGLE:
        height, width = game_info.board_dims
        indices = jnp.array([
            0, width-1,
            (height-1)*width, height*width-1
        ])

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")
    
    return indices.astype(ACTION_DTYPE)

def _get_edge_indices(game_info: GameInfo, edge_type: EdgeTypes):
    '''
    Return the mask indices corresponding to the edges of the current game's board and
    of the specific edge type
    '''
    if game_info.board_shape == Shapes.SQUARE or game_info.board_shape == Shapes.RECTANGLE:
        height, width = game_info.board_dims

        if edge_type == EdgeTypes.TOP:
            indices = jnp.arange(width)

        elif edge_type == EdgeTypes.BOTTOM:
            indices = jnp.arange((height-1)*width, height*width)

        elif edge_type == EdgeTypes.LEFT:
            indices = jnp.arange(0, height*width, width)

        elif edge_type == EdgeTypes.RIGHT:
            indices = jnp.arange(width-1, height*width, width)

        else:
            indices = jnp.array([])

    elif game_info.board_shape == Shapes.HEXAGON:
        diameter = game_info.hex_diameter
        stair_width = diameter // 2

        height, width = game_info.board_dims
        smallest_width = diameter // 2 + 1

        row_widths = [diameter + x for x in range(-stair_width, 0)] + [diameter + x for x in range(-stair_width, 1)][::-1]

        if edge_type == EdgeTypes.TOP:
            indices = jnp.arange(smallest_width)

        elif edge_type == EdgeTypes.BOTTOM:
            indices = jnp.arange(game_info.board_size - smallest_width, game_info.board_size)

        elif edge_type == EdgeTypes.TOP_LEFT:
            indices = [0]
            for width in row_widths[:diameter//2]:
                indices.append(indices[-1] + width)
            indices = jnp.array(indices)

        elif edge_type == EdgeTypes.TOP_RIGHT:
            indices = [row_widths[0] - 1]
            for width in row_widths[1:diameter//2+1]:
                indices.append(indices[-1] + width)
            indices = jnp.array(indices)

        elif edge_type == EdgeTypes.BOTTOM_LEFT:
            midpoint = game_info.board_size // 2
            indices = [midpoint - diameter // 2]
            for width in row_widths[diameter//2:-1]:
                indices.append(indices[-1] + width)
            indices = jnp.array(indices)

        elif edge_type == EdgeTypes.BOTTOM_RIGHT:
            midpoint = game_info.board_size // 2
            indices = [midpoint + diameter // 2]
            for width in row_widths[diameter//2+1:]:
                indices.append(indices[-1] + width)
            indices = jnp.array(indices)

        else:
            indices = jnp.array([])

    elif game_info.board_shape == Shapes.HEX_RECTANGLE:
        height, width = game_info.board_dims

        if edge_type == EdgeTypes.TOP:
            indices = jnp.arange(width)

        elif edge_type == EdgeTypes.BOTTOM:
            indices = jnp.arange((height-1)*width, height*width)

        elif edge_type == EdgeTypes.LEFT:
            indices = jnp.arange(0, height*width, width)

        elif edge_type == EdgeTypes.RIGHT:
            indices = jnp.arange(width-1, height*width, width)

        else:
            indices = jnp.array([])

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")

    return indices.astype(ACTION_DTYPE)

def _get_row_indices(game_info: GameInfo, row_idx: int):
    '''
    Return the mask indices corresponding to the given row on the current game's board
    '''
    if game_info.board_shape == Shapes.SQUARE or game_info.board_shape == Shapes.RECTANGLE:
        height, width = game_info.board_dims
        indices = jnp.arange(row_idx * width, (row_idx + 1) * width)

    elif game_info.board_shape == Shapes.HEXAGON:
        diameter = game_info.hex_diameter
        row_widths = [diameter - x for x in range(diameter // 2, -1, -1)] + [diameter - x for x in range(1, diameter // 2 + 1)]
        row_starts = [0] + np.cumsum(row_widths)[:-1].tolist()

        indices = jnp.arange(row_starts[row_idx], row_starts[row_idx] + row_widths[row_idx])

    elif game_info.board_shape == Shapes.HEX_RECTANGLE:
        height, width = game_info.board_dims
        indices = jnp.arange(row_idx * width, (row_idx + 1) * width)

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")
    
    return indices.astype(ACTION_DTYPE)

def _get_valid_edge_types(game_info: GameInfo):
    '''
    Return the types of edges that appear on the current game's board
    '''
    if game_info.board_shape == Shapes.SQUARE or game_info.board_shape == Shapes.RECTANGLE:
        edge_types = [
            EdgeTypes.TOP, EdgeTypes.BOTTOM,
            EdgeTypes.LEFT, EdgeTypes.RIGHT
        ]
    elif game_info.board_shape == Shapes.HEXAGON:
        edge_types = [
            EdgeTypes.TOP, EdgeTypes.BOTTOM,
            EdgeTypes.TOP_LEFT, EdgeTypes.TOP_RIGHT,
            EdgeTypes.BOTTOM_LEFT, EdgeTypes.BOTTOM_RIGHT
        ]
    elif game_info.board_shape == Shapes.HEX_RECTANGLE:
        edge_types = [
            EdgeTypes.TOP, EdgeTypes.BOTTOM,
            EdgeTypes.LEFT, EdgeTypes.RIGHT
        ]
    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")

    return edge_types

def _get_flood_fill_fn(adjacency_lookup: jnp.array):
    '''
    Returns a function that can be used to flood fill from a particular position
    on the board according to the adjacency kernel.
    '''

    def flood_fill(mask, idx):
        val_at_start = mask[idx]

        fill_out = jnp.zeros_like(mask, dtype=BOARD_DTYPE).at[idx].set(1)
        occupied = jnp.where(mask == val_at_start, 1, 0)

        def cond_fn(args):
            cur_mask, prev_mask = args
            return ~jnp.all(cur_mask == prev_mask)
        
        def body_fn(args):
            cur_mask, _ = args
            adjacent_mask = (cur_mask * adjacency_lookup).any(axis=(0, 2))
            new_mask = (cur_mask | (occupied & adjacent_mask))

            return new_mask, cur_mask
        fill_out, _ = jax.lax.while_loop(cond_fn, body_fn, (fill_out, jnp.zeros_like(fill_out)))

        return fill_out
    
    return flood_fill

def _get_connected_components_fn(game_info: GameInfo, adjacency_lookup: jnp.array):
    '''
    This implementation is based on the PGX code for Hex. It relies on the fact that
    we compute the connected components *after each action* is taken. This means 
    that we don't need to iterate over the entire board, only over each of the 
    adjacency directions
    '''

    num_directions = adjacency_lookup.shape[0]
    num_actions = game_info.board_size
    num_pieces = game_info.num_piece_types

    neighbor_indices = []
    for action_idx in range(num_actions):
        sub = []
        for direction_idx in range(num_directions):
            adjacency_mask = adjacency_lookup[direction_idx, action_idx]
            if adjacency_mask.any():
                sub.append(jnp.argmax(adjacency_mask))
            else:
                sub.append(-1)
        neighbor_indices.append(jnp.array(sub, dtype=ACTION_DTYPE))
    neighbor_indices = jnp.array(neighbor_indices, dtype=ACTION_DTYPE)

    def get_connected_components_piece(state, action, piece_idx):
        cur_components = state.connected_components[piece_idx]
        set_val = (action + 1).astype(ACTION_DTYPE)

        board_occupant = state.board[piece_idx, action]
        cur_components = cur_components.at[action].set(set_val)

        neighbor_positions = neighbor_indices[action]

        def merge(direction_index, components):
            adj_pos = neighbor_positions[direction_index]
            condition = (adj_pos >= 0) & (state.board[piece_idx, adj_pos] == board_occupant)

            components = jax.lax.cond(condition, lambda: jnp.where(components == components[adj_pos], set_val, components), lambda: components)

            return components
        
        cur_components = jax.lax.fori_loop(0, num_directions, merge, cur_components)

        return cur_components
    
    # TODO: special case (no VMAP) for when there's only one piece type?
    def get_connected_components(state, action):
        piece_indices = jnp.arange(num_pieces, dtype=ACTION_DTYPE)
        cur_components = jax.vmap(get_connected_components_piece, in_axes=(None, None, 0))(state, action, piece_indices)
        state = state._replace(connected_components=cur_components)
        return state
    
    return get_connected_components

def _get_line_indices(game_info: GameInfo, n: int, orientation: Orientations):
    '''
    The function precomputes the set of indices (into the flattened board mask) that correspond
    to every possible line of n in each of the specified directions. This means that checking for
    the presence of a line can be reduced to a single multi-dimensional query on the board mask.

    The code for rectangular boards is pretty straightforward, but it's much more involved for
    hexagonal boards because the width of each row changes. Recall that hexagonal boards are
    always arranged such that they have horizontal adjacencies and no vertical adjacencies.
    '''
    if game_info.board_shape == Shapes.SQUARE or game_info.board_shape == Shapes.RECTANGLE:
        height, width = game_info.board_dims
        indices = []

        if orientation in [Orientations.HORIZONTAL, Orientations.ORTHOGONAL, Orientations.ANY]:
            for row in range(height):
                for col in range(width - n + 1):
                    start = row * width + col
                    indices.append(jnp.arange(start, start + n))

        if orientation in [Orientations.VERTICAL, Orientations.ORTHOGONAL, Orientations.ANY]:
            for col in range(width):
                for row in range(height - n + 1):
                    start = row * width + col
                    indices.append(jnp.arange(start, start + n * width, width))

        if orientation in [Orientations.BACK_DIAGONAL, Orientations.DIAGONAL, Orientations.ANY]:
            for row in range(height - n + 1):
                for col in range(width - n + 1):
                    start = row * width + col
                    indices.append(jnp.arange(start, start + n * (width + 1), width + 1))

        if orientation in [Orientations.FORWARD_DIAGONAL, Orientations.DIAGONAL, Orientations.ANY]:
            for row in range(height - n + 1):
                for col in range(n - 1, width):
                    start = row * width + col
                    indices.append(jnp.arange(start, start + n * (width - 1), width - 1))

    elif game_info.board_shape == Shapes.HEXAGON:
        diameter = game_info.hex_diameter
        indices = []

        row_widths = [diameter - x for x in range(diameter // 2, -1, -1)] + [diameter - x for x in range(1, diameter // 2 + 1)]
        row_starts = [0] + np.cumsum(row_widths)[:-1].tolist()

        if orientation in [Orientations.HORIZONTAL, Orientations.ORTHOGONAL, Orientations.ANY]:
            for start, width in zip(row_starts, row_widths):
                for col in range(width - n + 1):
                    indices.append(jnp.arange(start + col, start + col + n))
        
        if orientation in [Orientations.BACK_DIAGONAL, Orientations.DIAGONAL, Orientations.ANY]:
            for row_idx, row_start in enumerate(row_starts):
                offset = min(row_idx, diameter // 2)
                diagonal_lengths = [diameter - row_idx - max(i-offset, 0) for i in range(row_widths[row_idx])]

                for col, length in enumerate(diagonal_lengths):
                    if length >= n:
                        start = row_start + col
                        local_offsets = [row_widths[idx] + 1 if idx < (diameter // 2) else row_widths[idx] for idx in range(row_idx, row_idx + n - 1)]
                        next_positions = np.cumsum(local_offsets)
                        line_indices = [start] + (next_positions + start).tolist()
                        indices.append(jnp.array(line_indices))

        if orientation in [Orientations.FORWARD_DIAGONAL, Orientations.DIAGONAL, Orientations.ANY]:
            for row_idx, row_start in enumerate(row_starts):
                offset = min(row_idx, diameter // 2)
                diagonal_lengths = [diameter - row_idx - max(i-offset, 0) for i in range(row_widths[row_idx])][::-1]

                for col, length in enumerate(diagonal_lengths):
                    if length >= n:
                        start = row_start + col
                        local_offsets = [row_widths[idx] - 1 if idx >= (diameter // 2) else row_widths[idx] for idx in range(row_idx, row_idx + n - 1)]
                        next_positions = np.cumsum(local_offsets)
                        line_indices = [start] + (next_positions + start).tolist()
                        indices.append(jnp.array(line_indices))
    
    elif game_info.board_shape == Shapes.HEX_RECTANGLE:
        height, width = game_info.board_dims
        indices = []

        if orientation in [Orientations.HORIZONTAL, Orientations.ORTHOGONAL, Orientations.ANY]:
            for row in range(height):
                for col in range(width - n + 1):
                    start = row * width + col
                    indices.append(jnp.arange(start, start + n))

        if orientation in [Orientations.BACK_DIAGONAL, Orientations.DIAGONAL, Orientations.ANY]:
            for row in range(height - n + 1):
                for col in range(width - n + 1):
                    start = row * width + col
                    indices.append(jnp.arange(start, start + n * width, width))

        if orientation in [Orientations.FORWARD_DIAGONAL, Orientations.DIAGONAL, Orientations.ANY]:
            for row in range(height - n + 1):
                for col in range(n - 1, width):
                    start = row * width + col
                    indices.append(jnp.arange(start, start + n * (width - 1), width - 1))

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")

    return jnp.array(indices, dtype=ACTION_DTYPE)

def _get_custodial_indices(game_info: GameInfo, inner_n: int, orientation: Orientations):
    '''
    Related to _get_line_indices above, this function returns the indices that correspond
    to 'custodial' captures -- i.e. a line of n pieces in a row surrounded by pieces that
    belong to the other player
    '''

    # We extract all the lines of n+2 to account for the two outer pieces
    line_indices = _get_line_indices(game_info, inner_n+2, orientation)

    # If there aren't any valid custodial arrangements, we can just return empty arrays
    if line_indices.shape[0] == 0:
        return jnp.array([], dtype=ACTION_DTYPE), jnp.array([], dtype=ACTION_DTYPE)

    inner_indices = line_indices[:, 1:-1]
    outer_indices = jnp.stack([line_indices[:, 0], line_indices[:, -1]], axis=1)

    return inner_indices, outer_indices

def _get_pattern_indices(game_info: GameInfo, arg_type: str, pattern: list, rotate: bool = False):
    '''
    Like _get_line_indices, this function returns the indices of the board corresponding to a particular
    pattern, defined either by manually listing local indices with respect to a "width" parameter or by
    specifying a particular shape (e.g. 2x2 square)
    '''

    # Manual pattern specification
    if arg_type == OptionalArgs.PATTERN:
        pattern_width, (_, indices) = pattern
        shape = None

        # Determine the *actual* dimensions of the pattern (which might be smaller than the provided width if the pattern is irregular)
        pattern_height = (max(indices) // pattern_width) + 1
        max_pattern_width  = max([idx % pattern_width for idx in indices]) + 1

        # Convert the pattern indices to local (row, col) coordinates
        local_pattern = [(idx // pattern_width, idx % pattern_width) for idx in indices]

        if game_info.board_shape == Shapes.HEXAGON:
            logging.warning("Manual pattern specification works with hexagonal boards, but the resulting patterns may not be visually intuitive due to the changing row widths.")

    # Shape-based pattern specification (e.g. square, rectangle, ...)
    elif arg_type == OptionalArgs.SHAPE:
        shape, shape_dims = pattern

        # Rectangular shapes are impossible on a hexagonal board and vice-versa
        if (shape in [Shapes.SQUARE, Shapes.RECTANGLE]) and (game_info.board_shape == Shapes.HEXAGON):
            return jnp.array([], dtype=ACTION_DTYPE)
        
        if (shape in [Shapes.HEXAGON, Shapes.HEX_RECTANGLE]) and (game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE]):
            return jnp.array([], dtype=ACTION_DTYPE)
        
        if shape == Shapes.SQUARE:
            pattern_width = shape_dims
            max_pattern_width = pattern_width
            pattern_height = shape_dims
            local_pattern = [(r, c) for r in range(pattern_height) for c in range(pattern_width)]

        elif shape == Shapes.RECTANGLE:
            pattern_width, pattern_height = shape_dims
            max_pattern_width = pattern_width
            local_pattern = [(r, c) for r in range(pattern_height) for c in range(pattern_width)]

        elif shape == Shapes.HEX_RECTANGLE:
            pattern_width, pattern_height = shape_dims
            max_pattern_width = pattern_width
            local_pattern = [(r, c) for r in range(pattern_height) for c in range(pattern_width)]

        elif shape == Shapes.HEXAGON:
            # By grammar construction, the diameter of the hexagon is odd
            diameter = shape_dims

            row_widths = [diameter + x for x in range(-diameter//2 + 1, 0)] + [diameter + x for x in range(-diameter//2 + 1, 1)][::-1]

            local_pattern = []
            for r, width in enumerate(row_widths):
                # If the row is above the diameter, the offset is towards the right
                if r < diameter // 2:
                    offset = diameter - row_widths[r]
                    for c in range(offset, width + offset):
                        local_pattern.append((r, c))
                else:
                    for c in range(width):
                        local_pattern.append((r, c))
            
            pattern_height = diameter
            max_pattern_width = diameter

        else:
            raise NotImplementedError(f"Shape {shape} not implemented yet for shape-based pattern specification!")

    else:
        raise ValueError(f"Invalid argument type for pattern indices: {arg_type}")
    
    # Compute the rotated versions of the pattern if necessary
    if rotate and shape not in [Shapes.HEXAGON]:
        cur_pattern = local_pattern[:]
        rotated_patterns = [cur_pattern]
        for _ in range(3):
            rotated_pattern = [(c, pattern_height - 1 - r) for r, c in cur_pattern]

            # Shift negative coordinates to ensure the pattern is in the top-left quadrant
            min_row = min(r for r, c in rotated_pattern)
            min_col = min(c for r, c in rotated_pattern)
            rotated_pattern = [(r - min_row, c - min_col) for r, c in rotated_pattern]

            rotated_patterns.append(rotated_pattern)
            cur_pattern = rotated_pattern

    if game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE, Shapes.HEX_RECTANGLE]:
        board_height, board_width = game_info.board_dims
        indices = []

        for row in range(board_height):
            for col in range(board_width):
                start = row * board_width + col

                # Check if the pattern can fit at this position
                can_fit = (row + pattern_height <= board_height) and (col + max_pattern_width <= board_width)
                if can_fit:    
                    transformed_indices = [start + (r * board_width) + c for r, c in local_pattern]
                    indices.append(jnp.array(transformed_indices, dtype=ACTION_DTYPE))

                if rotate and shape not in [Shapes.HEXAGON]:
                    # Add the 180 degree rotation of the pattern (rotated[2])
                    if can_fit:
                        transformed_indices_180 = [start + (r * board_width) + c for r, c in rotated_patterns[2]]
                        indices.append(jnp.array(transformed_indices_180, dtype=ACTION_DTYPE))

                    # Check if the rotated pattern can fit at this position
                    can_fit_rotated = (row + max_pattern_width <= board_height) and (col + pattern_height <= board_width)
                    if can_fit_rotated:
                        transformed_indices_90 = [start + (r * board_width) + c for r, c in rotated_patterns[1]]
                        transformed_indices_270 = [start + (r * board_width) + c for r, c in rotated_patterns[3]]
                        indices.append(jnp.array(transformed_indices_90, dtype=ACTION_DTYPE))
                        indices.append(jnp.array(transformed_indices_270, dtype=ACTION_DTYPE))

    elif game_info.board_shape == Shapes.HEXAGON:
        indices = []
        board_diameter = game_info.hex_diameter
        board_row_widths = [board_diameter + x for x in range(-board_diameter//2 + 1, 0)] + [board_diameter + x for x in range(-board_diameter//2 + 1, 1)][::-1]
        row_starts = [0] + np.cumsum(board_row_widths)[:-1].tolist()

        def check_fit(row, col, pattern):
            for r, c in pattern:
                board_r = row + r
                board_c = col + c
                if (board_r >= board_diameter) or (board_c < 0) or (board_c >= board_row_widths[board_r]):
                    return False
            return True
        
        def get_exceeded_width(start_row, local_row):
            return sum([board_row_widths[start_row + i] + (1 if start_row + i < board_diameter // 2 else 0) for i in range(local_row)])

        for row in range(board_diameter):
            for col in range(board_row_widths[row]):
                start = row_starts[row] + col

                if rotate and shape not in [Shapes.HEXAGON]:
                    for p in rotated_patterns:
                        can_fit = check_fit(row, col, p)
                        if can_fit:
                            transformed_indices = [start + get_exceeded_width(row, r) + c for r, c in p]
                            indices.append(jnp.array(transformed_indices, dtype=ACTION_DTYPE))

                else:
                    # Check if the pattern can fit at this position
                    can_fit = check_fit(row, col, local_pattern)
                    
                    if can_fit:
                        transformed_indices = [start + get_exceeded_width(row, r) + c for r, c in local_pattern]
                        indices.append(jnp.array(transformed_indices, dtype=ACTION_DTYPE))

    else:
        raise NotImplementedError(f"Pattern-based indices not implemented yet for board shape {game_info.board_shape}")
    
    return jnp.array(indices, dtype=ACTION_DTYPE)

def _get_jump_masks(game_info: GameInfo, distance: int):
    '''
    Return the masks corresponding to jumps up to a particular distance by iteratively
    applying the adjacency lookup to get the positions that are reachable in a certain 
    number of steps
    '''
    jump_masks = jnp.zeros((distance, game_info.board_size, game_info.board_size), dtype=ACTION_DTYPE)
    jump_masks = jump_masks.at[:, jnp.arange(game_info.board_size), jnp.arange(game_info.board_size)].set(1)

    # Collapse all the directions of the adjacency lookup
    adjacency_lookup = _get_adjacency_lookup(game_info)
    adjacency_lookup = adjacency_lookup.any(axis=0)

    # Iteratively apply the adjacency lookup to get the positions that are reachable in a certain number of steps
    for i in range(distance):
        for row_idx, row in enumerate(jump_masks[i]):
            active = jnp.argwhere(row == 1).flatten()
            for idx in active:
                adj_idxs = jnp.argwhere(adjacency_lookup[idx, :] == 1).flatten()
                jump_masks = jump_masks.at[i:, row_idx, adj_idxs].set(1)

    # Can't jump to the same position, so we set the diagonal to 0
    jump_masks = jump_masks.at[:, jnp.arange(game_info.board_size), jnp.arange(game_info.board_size)].set(0)

    return jump_masks

def _get_collect_values_fn(outer_children, vmap=False):
    '''
    This returns a function which will collect the values of each of the children
    using either jax.lax.map or jax.vmap. If there are exactly one or two children,
    then we just return the values directly.

    This function can be used interchangeably with masks, functions, and predicates
    and is most useful for rules that can have a variable number of children (e.g.
    super_mask_and, predicate_equals, ...)
    '''

    n = len(outer_children)

    if n == 1:
        def collect_values(children, *args):
            return jnp.array([children[0](*args)])
        
    elif n == 2:
        def collect_values(children, *args):
            return jnp.array([children[0](*args), children[1](*args)])
        
    else:
        indices = jnp.arange(n)
        if vmap:
            def collect_values(children, *args):
                values = jax.vmap(lambda i: jax.lax.switch(i, children, *args))(indices)
                return values
            
        else:
            def collect_values(children, *args):
                body_fn = lambda i: jax.lax.switch(i, children, *args)
                values = jax.lax.map(body_fn, indices)
                return values

        
    return partial(collect_values, outer_children)

def _get_occupied_mask_fn(piece, player_or_mover):
    '''
    Returns a function that can be used to get a mask of the positions occupied
    by the given player/mover and piece. Lots of other functions and masks first
    extract the occupied positions before doing further processing, so this
    function helps to reduce code duplication.
    '''

    if piece == PieceRefs.ANY:
        if player_or_mover == PlayerAndMoverRefs.MOVER:
            def get_mask(state):
                return (state.board == state.current_player).any(axis=0).astype(BOARD_DTYPE)
        
        elif player_or_mover == PlayerAndMoverRefs.OPPONENT:
            def get_mask(state):
                return (state.board == (state.current_player + 1) % 2).any(axis=0).astype(BOARD_DTYPE)
        
        elif player_or_mover == PlayerAndMoverRefs.P1:
            def get_mask(state):
                return (state.board == P1).any(axis=0).astype(BOARD_DTYPE)
        
        elif player_or_mover == PlayerAndMoverRefs.P2:
            def get_mask(state):
                return (state.board == P2).any(axis=0).astype(BOARD_DTYPE)
            
        elif player_or_mover == PlayerAndMoverRefs.BOTH:
            def get_mask(state):
                return (state.board != EMPTY).any(axis=0).astype(BOARD_DTYPE)
            
        else:
            raise ValueError(f"Invalid player or mover reference: {player_or_mover}")

    else:
        if player_or_mover == PlayerAndMoverRefs.MOVER:
            def get_mask(state):
                return (state.board[piece] == state.current_player).astype(BOARD_DTYPE)
        
        elif player_or_mover == PlayerAndMoverRefs.OPPONENT:
            def get_mask(state):
                return (state.board[piece] == (state.current_player + 1) % 2).astype(BOARD_DTYPE)
        
        elif player_or_mover == PlayerAndMoverRefs.P1:
            def get_mask(state):
                return (state.board[piece] == P1).astype(BOARD_DTYPE)
        
        elif player_or_mover == PlayerAndMoverRefs.P2:
            def get_mask(state):
                return (state.board[piece] == P2).astype(BOARD_DTYPE)
        
        elif player_or_mover == PlayerAndMoverRefs.BOTH:
            def get_mask(state):
                return (state.board[piece] != EMPTY).astype(BOARD_DTYPE)

        else:
            raise ValueError(f"Invalid player or mover reference: {player_or_mover}")
    
    return get_mask
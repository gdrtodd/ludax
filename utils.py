from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from config import EMPTY, INVALID, BoardShapes, Directions, EdgeTypes, Orientations, OptionalArgs
from game_info import GameInfo

BOARD_SHAPE_TO_DIRECTIONS = {
    BoardShapes.SQUARE: [
            Directions.UP_LEFT, Directions.UP, Directions.UP_RIGHT, 
            Directions.LEFT, Directions.RIGHT, 
            Directions.DOWN_LEFT, Directions.DOWN, Directions.DOWN_RIGHT
    ],
    BoardShapes.RECTANGLE: [
            Directions.UP_LEFT, Directions.UP, Directions.UP_RIGHT, 
            Directions.LEFT, Directions.RIGHT, 
            Directions.DOWN_LEFT, Directions.DOWN, Directions.DOWN_RIGHT
    ],
    BoardShapes.HEXAGON: [
            Directions.UP_LEFT, Directions.UP_RIGHT, 
            Directions.LEFT, Directions.RIGHT, 
            Directions.DOWN_LEFT, Directions.DOWN_RIGHT
    ],
    BoardShapes.HEX_RECTANGLE: [
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

def _get_adjacency_kernel(game_info: GameInfo, optional_args: dict):
    '''
    Returns a two-dimensional kernel for the current game that can be used to precompute the adjacency
    lookup information
    '''
    if game_info.board_shape == BoardShapes.SQUARE or game_info.board_shape == BoardShapes.RECTANGLE:
        kernel = jnp.zeros((1, 1, 3, 3), dtype=jnp.int16)
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

    elif game_info.board_shape == BoardShapes.HEX_RECTANGLE:
        kernel = jnp.zeros((1, 1, 3, 3), dtype=jnp.int16)
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
    
    elif game_info.board_shape == BoardShapes.HEXAGON:
        kernel = jnp.zeros((1, 1, 5, 5), dtype=jnp.int16)
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

    adjacency_array = jnp.zeros((len(directions), game_info.board_size, game_info.board_size), dtype=jnp.int8)
    for channel_idx, direction in enumerate(directions):
        kernel = _get_adjacency_kernel(game_info, {OptionalArgs.DIRECTION: direction})

        for i in range(game_info.board_size):
            mask = jnp.zeros((game_info.board_size,), dtype=jnp.int8).at[i].set(1)
            board_2d = mask_to_board(mask)
            board_2d = board_2d[jnp.newaxis, jnp.newaxis, :, :]
            board_2d = jnp.where(board_2d == INVALID, EMPTY, board_2d)

            board_2d = board_2d.astype(jnp.int8)
            kernel = kernel.astype(jnp.int8)

            conv_out = jax.lax.conv(board_2d, kernel, (1, 1), 'SAME')[0][0] > 0
            out_mask = board_to_mask(conv_out)

            adjacency_array = adjacency_array.at[channel_idx, i].set(out_mask)

    return adjacency_array

def _get_adjacency_direction_indices(game_info: GameInfo, optional_args: dict):
    '''
    Returns the indices corresponding to the channels in the adjacency lookup for the given (meta)direction
    '''
    all_directions = BOARD_SHAPE_TO_DIRECTIONS[game_info.board_shape]

    direction = optional_args[OptionalArgs.DIRECTION]
    query_directions = META_DIRECTION_MAPPING.get(direction, [direction])

    direction_indices = []
    for dir in query_directions:
        if dir in all_directions:
            direction_indices.append(all_directions.index(dir))

    direction_indices = jnp.array(list(sorted(direction_indices)), dtype=jnp.int16)

    return direction_indices
    
def _get_mask_board_conversion_fns(game_info: GameInfo):
    '''
    Return functions for the current game that can be used to convert between a flattened
    board mask and a two-dimensional board representation
    '''
    if game_info.board_shape == BoardShapes.SQUARE or game_info.board_shape == BoardShapes.RECTANGLE:
        mask_to_board = lambda mask: mask.reshape(game_info.board_dims)
        mask_idx_to_board_pos = lambda idx: jnp.unravel_index(idx, game_info.board_dims)
        board_to_mask = lambda board: board.flatten()
        return mask_to_board, mask_idx_to_board_pos, board_to_mask

    elif game_info.board_shape == BoardShapes.HEX_RECTANGLE:
        mask_to_board = lambda mask: mask.reshape(game_info.board_dims)
        mask_idx_to_board_pos = lambda idx: jnp.unravel_index(idx, game_info.board_dims)
        board_to_mask = lambda board: board.flatten()
        return mask_to_board, mask_idx_to_board_pos, board_to_mask
    
    elif game_info.board_shape == BoardShapes.HEXAGON:
        diameter = game_info.hex_diameter
        offset = (diameter // 2) % 2

        # Apply checkerboard pattern
        base = np.zeros(game_info.board_dims, dtype=np.int16)
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
        base_board = jnp.ones_like(base, dtype=jnp.int16) * INVALID

        def mask_to_board(mask):
            return base_board.at[*valid_hex_indices].set(mask)
        
        def mask_idx_to_board_pos(idx):
            return valid_hex_indices.T[idx]
        
        def board_to_mask(board):
            return board[*valid_hex_indices].flatten()
        
        return mask_to_board, mask_idx_to_board_pos, board_to_mask

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")

def _get_corner_indices(game_info: GameInfo):
    '''
    Return the mask indices corresponding to the corners of the current game's board
    '''
    if game_info.board_shape == BoardShapes.SQUARE or game_info.board_shape == BoardShapes.RECTANGLE:
        height, width = game_info.board_dims
        indices = jnp.array([
            0, width-1,
            (height-1)*width, height*width-1
        ])

    elif game_info.board_shape == BoardShapes.HEXAGON:
        half_diameter = game_info.hex_diameter // 2
        midpoint = game_info.board_size // 2
        indices = jnp.array([
            0, half_diameter,
            midpoint - half_diameter, midpoint + half_diameter,
            game_info.board_size - half_diameter - 1, game_info.board_size - 1
        ])

    elif game_info.board_shape == BoardShapes.HEX_RECTANGLE:
        height, width = game_info.board_dims
        indices = jnp.array([
            0, width-1,
            (height-1)*width, height*width-1
        ])

    else:
        raise NotImplementedError(f"Board shape {game_info.board_shape} not implemented yet!")
    
    return indices.astype(jnp.int16)

def _get_edge_indices(game_info: GameInfo, edge_type: EdgeTypes):
    '''
    Return the mask indices corresponding to the edges of the current game's board and
    of the specific edge type
    '''
    if game_info.board_shape == BoardShapes.SQUARE or game_info.board_shape == BoardShapes.RECTANGLE:
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

    elif game_info.board_shape == BoardShapes.HEXAGON:
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

    elif game_info.board_shape == BoardShapes.HEX_RECTANGLE:
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

    return indices.astype(jnp.int16)

def _get_valid_edge_types(game_info: GameInfo):
    '''
    Return the types of edges that appear on the current game's board
    '''
    if game_info.board_shape == BoardShapes.SQUARE or game_info.board_shape == BoardShapes.RECTANGLE:
        edge_types = [
            EdgeTypes.TOP, EdgeTypes.BOTTOM,
            EdgeTypes.LEFT, EdgeTypes.RIGHT
        ]
    elif game_info.board_shape == BoardShapes.HEXAGON:
        edge_types = [
            EdgeTypes.TOP, EdgeTypes.BOTTOM,
            EdgeTypes.TOP_LEFT, EdgeTypes.TOP_RIGHT,
            EdgeTypes.BOTTOM_LEFT, EdgeTypes.BOTTOM_RIGHT
        ]
    elif game_info.board_shape == BoardShapes.HEX_RECTANGLE:
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

        fill_out = jnp.zeros_like(mask, dtype=jnp.int16).at[idx].set(1)
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

    neighbor_indices = []
    for action_idx in range(num_actions):
        sub = []
        for direction_idx in range(num_directions):
            adjacency_mask = adjacency_lookup[direction_idx, action_idx]
            if adjacency_mask.any():
                sub.append(jnp.argmax(adjacency_mask))
            else:
                sub.append(-1)
        neighbor_indices.append(jnp.array(sub, dtype=jnp.int16))
    neighbor_indices = jnp.array(neighbor_indices, dtype=jnp.int16)

    def get_connected_components(state, action):
        cur_components = state.connected_components
        set_val = action + 1

        board_occupant = state.board[action]
        cur_components = cur_components.at[action].set(set_val)

        neighbor_positions = neighbor_indices[action]

        def merge(direction_index, components):
            adj_pos = neighbor_positions[direction_index]
            condition = (adj_pos >= 0) & (state.board[adj_pos] == board_occupant)

            components = jax.lax.cond(condition, lambda: jnp.where(components == components[adj_pos], set_val, components), lambda: components)

            return components
        
        cur_components = jax.lax.fori_loop(0, num_directions, merge, cur_components)

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
    if game_info.board_shape == BoardShapes.SQUARE or game_info.board_shape == BoardShapes.RECTANGLE:
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

    elif game_info.board_shape == BoardShapes.HEXAGON:
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
    
    elif game_info.board_shape == BoardShapes.HEX_RECTANGLE:
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

    return jnp.array(indices, dtype=jnp.int16)

def _get_custodial_indices(game_info: GameInfo, inner_n: int, orientation: Orientations):
    '''
    Related to _get_line_indices above, this function returns the indices that correspond
    to 'custodial' captures -- i.e. a line of n pieces in a row surrounded by pieces that
    belong to the other player
    '''

    # We extract all the lines of n+2 to account for the two outer pieces
    line_indices = _get_line_indices(game_info, inner_n+2, orientation)

    inner_indices = line_indices[:, 1:-1]
    outer_indices = jnp.stack([line_indices[:, 0], line_indices[:, -1]], axis=1)

    return inner_indices, outer_indices

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
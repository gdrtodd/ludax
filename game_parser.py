import inspect
import jax
import jax.numpy as jnp
from lark.visitors import Transformer

from config import EMPTY, P1, P2, TRUE, FALSE, DEFAULT_ARGUMENTS, PlayerAndMoverRefs, OptionalArgs
from game_info import GameInfo
import utils

class GameRuleParser(Transformer):
    def __init__(self, game_info: GameInfo):
        self.game_info = game_info
        self.end_rule_info = []
        self.adjacency_lookup = utils._get_adjacency_lookup(self.game_info)

    def __default__(self, data, children, meta):
        if len(children) == 1:
            to_return = children[0]
        else:
            to_return = children

        # In order to handle optional arguments, they need
        # to return their name along with their value
        if data.endswith("_arg"):
            return str(data), to_return
        
        return to_return
    
    def __default_token__(self, token):
        '''
        Attempt to parse tokens into the data types they represent,
        but fall back to a string if you can't
        '''
        if token == 'true':
            return True
        elif token == 'false':
            return False

        try:
            return int(token)
        except ValueError:
            return str(token)
    
    def game(self, children):
        _, _, _, *game_rule_dicts = children
        game_rule_dicts = game_rule_dicts[0]

        # Handle the case where there are no start rules
        if len(game_rule_dicts) == 2:
            play_rule_dict, end_rule_dict = game_rule_dicts
            info_dict = {'start_rules': lambda board: board, **play_rule_dict, **end_rule_dict}
        
        else:
            start_rule_dict, play_rule_dict, end_rule_dict = game_rule_dicts
            info_dict = {**start_rule_dict, **play_rule_dict, **end_rule_dict}

        # Handle any optional game state attributes
        addl_info_functions = []
        for attribute in self.game_info.game_state_attributes:
            if attribute == "connected_components":
                addl_info_functions.append(utils._get_connected_components_fn(self.game_info, self.adjacency_lookup))

        if len(addl_info_functions) > 0:
            n = len(addl_info_functions)
            def addl_info_fn(state, action):
                def body_fn(i, args):
                    state, action = args
                    state = jax.lax.switch(i, addl_info_functions, state, action)
                    return state, action
    
                result, _ = jax.lax.fori_loop(0, n, body_fn, (state, action))
                return result
            
            info_dict['addl_info_fn'] = addl_info_fn

        else:
            info_dict['addl_info_fn'] = lambda state, action: state

        return info_dict
    
    '''
    =========Start rules=========
    '''
    def start_rules(self, children):
        '''
        Start rules are optional, but we assume that if they are present then
        there's at least one. Start rules are also unique in that they operate
        over boards instead of states
        '''
        rules = children
        n_rules = len(rules)
        
        def combined_start_rules(board):
            body_fn = lambda i, board: jax.lax.switch(i, rules, board)
            result = jax.lax.fori_loop(0, n_rules, body_fn, board)
            return result

        return {'start_rules': combined_start_rules}

    def start_place(self, children):
        '''
        Begin the game by placing one or more pieces belong to a specific
        player at specified locations on the board
        '''
        player_ref, (_, pattern) = children

        if player_ref == PlayerAndMoverRefs.P1:
            player = P1
        elif player_ref == PlayerAndMoverRefs.P2:
            player = P2

        pattern = jnp.array(pattern, dtype=jnp.int16)

        def start_fn(board):
            return board.at[pattern].set(player)
        
        return start_fn

    '''
    =========Play rules=========
    '''
    def play_rules(self, children):
        '''
        The children of play_rules are play phases, so this function is responsible for collecting
        each phase and constructing the general play rule dictionary.
        '''

        (action_sizes, apply_action_fns, legal_action_mask_fns, 
         apply_effects_fns, next_phase_fns, next_player_fns) = zip(*children)

        def apply_action_fn(state, action):
            return jax.lax.switch(state.phase_idx, apply_action_fns, state, action)

        def legal_action_mask_fn(state):
            return jax.lax.switch(state.phase_idx, legal_action_mask_fns, state)
        
        def apply_effects_fn(state, original_player):
            return jax.lax.switch(state.phase_idx, apply_effects_fns, state, original_player)

        def next_phase_fn(state):
            return jax.lax.switch(state.phase_idx, next_phase_fns, state)

        def next_player_fn(state):
            return jax.lax.switch(state.phase_idx, next_player_fns, state)

        play_rule_dict = {
            'action_size': max(action_sizes),
            'apply_action_fn': apply_action_fn,
            'legal_action_mask_fn': legal_action_mask_fn,
            'apply_effects_fn': apply_effects_fn,
            'next_phase_fn': next_phase_fn,
            'next_player_fn': next_player_fn
        }

        return play_rule_dict

    '''
    =========Play phases=========
    '''
    
    def phase_once_through(self, children):
        '''
        Proceed once through the specified order of players and then advance the phase
        '''
        (next_player_fn, num_moves), play_mechanic = children
        action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn = play_mechanic

        # The phase advances after all the moves in the sequence have been made
        next_phase_fn = lambda state: (state.phase_idx + (state.phase_step_count == num_moves-1), (state.phase_step_count != num_moves-1) * (state.phase_step_count + 1))

        return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn, next_phase_fn, next_player_fn
    
    def phase_repeat(self, children):
        '''
        Repeat the specified order of players until the game is over
        '''
        (next_player_fn, _), play_mechanic = children
        action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn = play_mechanic

        # The phase never advances in this case
        next_phase_fn = lambda state: (state.phase_idx, state.phase_step_count + 1)

        return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn, next_phase_fn, next_player_fn
    
    def play_mover_order(self, children):
        '''
        Specifies a finite sequence of P1 and P2 that determines the player order
        '''
        sequence = jnp.array([P1 if child == PlayerAndMoverRefs.P1 else P2 for child in children])
        n = len(sequence)

        def next_player_fn(state):
            return sequence[state.phase_step_count % n]
        
        return next_player_fn, n

    '''
    =========Play mechanics=========
    '''

    def play_mechanic(self, children):
        '''
        A play mechanic is responsible for collecting the information about how to apply actions,
        determine legal actions, and apply effects before making any high-level modifications
        (e.g. applying a "force pass") rule
        '''
        (action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn), *force_pass = children
        force_pass = True if len(force_pass) > 0 else False
        
        # The "force_pass" rule adds an additional action which passes the turn to the next player
        # that can only be taken (and must be taken) when no other legal moves are available
        if force_pass:
            action_size += 1

            # This relies on the fgact that the "passed" property has been added to the state
            # by the GameInfo class
            def new_apply_action_fn(state, action):
                is_pass = (action == action_size - 1)

                pass_fn = lambda state, action: state
                state = jax.lax.cond(is_pass, pass_fn, apply_action_fn, state, action)
                passed = state.passed.at[state.current_player].set(is_pass)

                return state._replace(passed=passed)
            
            def new_legal_action_mask_fn(state):
                base_mask = legal_action_mask_fn(state)
                can_pass = ~base_mask.any()
                mask = jnp.concatenate((base_mask, jnp.array([can_pass], dtype=jnp.int16)))
                return mask
            
            return action_size, new_apply_action_fn, new_legal_action_mask_fn, apply_effects_fn
        
        else:
            return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn

    def play_place(self, children):
        '''
        The default behavior of games currently. Players put a piece onto one position
        on the board. This has one required argument which specifies the mask of legal
        destinations (typically empty squares) and then optional arguments which specify:
        - who the placed piece belongs to (default: the current player)
        - any constraints over the *result* of the action (e.g. in Reversi, where actions
          must be custodial)
        - any additional effects of the action (e.g. incrementing a score)
        '''

        # Case 1: the 'mover' argument is specified
        if isinstance(children[0], str):
            mover_ref, (destination_constraint_fn, _), *optional_args = children
            if mover_ref == PlayerAndMoverRefs.MOVER:
                offset = 0
            elif mover_ref == PlayerAndMoverRefs.OPPONENT:
                offset = 1

        # Case 2 (default): the mover is the current player 
        else:
            (destination_constraint_fn, _), *optional_args = children
            offset = 0

        # Case 1: no optional arguments -- legal actions determined by the destination constraint
        # and there are no effects
        if len(optional_args) == 0:
            legal_action_mask_fn = destination_constraint_fn
            apply_effects_fn = lambda state, original_player: state

        # Case 2: one optional argument is specified -- either a result constraint or an effect
        elif len(optional_args) == 1:
            arg = optional_args[0]
            if arg.__name__ == "result_constraint_fn" or arg.__name__ == "lookahead_mask_fn":
                legal_action_mask_fn = lambda state: destination_constraint_fn(state) & arg(state)
                apply_effects_fn = lambda state, original_player: state

            elif arg.__name__ == "apply_effects_fn":
                legal_action_mask_fn = destination_constraint_fn
                apply_effects_fn = arg

        # Case 3: two optional arguments are specified -- result constraints and effects, always
        # in that order
        else:
            result_constraint_fn, apply_effects_fn = optional_args
            legal_action_mask_fn = lambda state: destination_constraint_fn(state) & result_constraint_fn(state)

        action_size = self.game_info.board_size
        def apply_action_fn(state, action):
            board = state.board.at[action].set((state.current_player + offset) % 2)
            return state._replace(board=board)
        
        return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn
    
    '''
    Play constraints
    '''
    def place_result_constraint(self, children):
        '''
        Result constraints refer to things that must be true about the board
        immediately following an action. Each kind of action has its own
        'result_constraint' function that knows how to apply that action
        '''
        (predicate_fn, predicate_fn_info) = children[0]

        # The predicate function defines a condition that must be satisfied over the resulting board
        # in order for a move to be legal. However, certain constraints (specifically, custodial and
        # line constraints) at the current moment can be defined in terms of the *current* board state.
        # The predicate_fn_info dictionary will contain as 'lookahead_mask_fn' if and only if the game's
        # result constraints can be expressed in this way, in which case we use it to optimize the
        # constraint checking. Otherwise, we map the predicate function over all possible resulting
        # board states
        if predicate_fn_info.get('lookahead_mask_fn') is None:
            def apply_action(state, action):
                pred_val = predicate_fn(state._replace(
                    board=state.board.at[action].set(state.current_player),
                    previous_actions=state.previous_actions.at[state.current_player].set(action)
                ))
                return pred_val
            
            apply = jax.jit(apply_action)

            # TODO: maybe this logic should be moved to play_place and actually use
            # the 'apply_action_fn' defined there instead of having a separate one
            def result_constraint_fn(state):
                resulting_mask = jax.vmap(apply, in_axes=(None, 0))(state, jnp.arange(self.game_info.board_size, dtype=jnp.int16))
                return resulting_mask.astype(jnp.int16)

        else:
            result_constraint_fn = predicate_fn_info['lookahead_mask_fn']

        return result_constraint_fn

    '''
    Play effects
    '''
    def play_effects(self, children):
        '''
        Each effect is a function that takes a state and the player
        who began the turn, and then returns a new state
        '''

        n = len(children)
        def apply_effects_fn(state, original_player):
            def body_fn(i, args):
                cur_state, original_player = args
                cur_state = jax.lax.switch(i, children, cur_state, original_player)
                return cur_state, original_player
            
            result, _ = jax.lax.fori_loop(0, n, body_fn, (state, original_player))
            return result
        
        return apply_effects_fn
    
    def effect_capture(self, children):
        '''
        Remove pieces from the board according to a mask
        '''
        (child_mask_fn, _), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        increment_score = optional_args[OptionalArgs.INCREMENT_SCORE_ARG]

        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            child_mask = child_mask_fn(updated_state)
            occupied_mask = (updated_state.board == (original_player + offset) % 2)

            to_capture = (occupied_mask * child_mask).astype(jnp.int16)
            new_board = jnp.where(to_capture, EMPTY, updated_state.board)

            # If specified, increment the mover's score by the amount of pieces captured
            score_update = jax.lax.select(increment_score, to_capture.sum(), 0)
            scores = state.scores.at[original_player].set(state.scores[original_player] + score_update)

            return state._replace(board=new_board, scores=scores)
        
        return apply_effects_fn
    
    def effect_flip(self, children):
        '''
        Flip pieceson the board belonging to an opponent according to a mask
        '''
        (child_mask_fn, _), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            child_mask = child_mask_fn(updated_state)
            occupied_mask = (updated_state.board == (original_player + offset) % 2)

            to_flip = (occupied_mask * child_mask).astype(jnp.int16)
            new_board = jnp.where(to_flip, (updated_state.board + 1) % 2, updated_state.board)

            return state._replace(board=new_board)
        
        return apply_effects_fn

    def effect_increment_score(self, children):
        '''
        Increment the score for a specified player by the result of a function
        '''
        mover_ref, (amount_fn, _) = children
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        # We need to keep track of the player that started the turn
        # in order to correctly increment the score
        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            idx = (updated_state.current_player + offset) % 2
            amount = amount_fn(updated_state)

            scores = state.scores.at[idx].set(state.scores[idx] + amount)
            return state._replace(scores=scores)
        
        return apply_effects_fn
    
    def effect_set_score(self, children):
        '''
        Set the score for a specified player to the result of a function
        '''
        mover_ref, (amount_fn, _) = children
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        # We need to keep track of the player that started the turn
        # in order to correctly increment the score
        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            idx = (updated_state.current_player + offset) % 2
            amount = amount_fn(updated_state)

            scores = state.scores.at[idx].set(amount)
            return state._replace(scores=scores)
        
        return apply_effects_fn

    '''
    =========End rules=========
    '''
    def end_rules(self, children):
        '''
        Compile information about the end rules of the game
        '''
        rules = children
        n_rules = len(rules)
        
        # Multiple end rules should be applied in order -- if any of them flag
        # the game is terminated. Rewards are determined by the first rule that
        # flags the game as terminated
        def combined_end_rules(state):
            index = jnp.arange(n_rules)
            winners, ends = jax.vmap(lambda i: jax.lax.switch(i, rules, state))(index)

            # Take winner determined by the first active end rule
            winner = jax.lax.select(ends.any(), winners[jnp.argmax(ends)], EMPTY)
            end = ends.any()
            
            return winner, end

        return {'end_rules': combined_end_rules}
    
    def end_rule(self, children):
        '''
        Parse out a given ending rule, which takes the form of a
        binary-valued predicate and a game result
        '''
        (predicate_fn, predicate_info), (get_winner, winner_info) = children
        
        # Record the joint information about the predicate and the winner
        joint_info = {**predicate_info, **winner_info}
        self.end_rule_info.append(joint_info)

        def end_rule_fn(state):
            pred_val = predicate_fn(state)
            winner = jax.lax.select(pred_val, get_winner(state), EMPTY)
            termination = jax.lax.select(pred_val, TRUE, FALSE)

            return winner, termination

        return end_rule_fn
    
    def result_win(self, children):
        '''
        The specified player wins the game
        '''
        mover_ref = children[0]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def get_winner(state):
            return state.current_player + offset
        
        info = {}

        return get_winner, info
    
    def result_lose(self, children):
        '''
        The specified player loses the game
        '''
        mover_ref = children[0]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def get_winner(state):
            return (state.current_player + offset + 1)%2
        
        info = {}

        return get_winner, info
    
    def result_draw(self, children):
        '''
        The game ends in a draw
        '''
        def get_winner(state):
            return EMPTY
        
        info = {}

        return get_winner, info
    
    def result_by_score(self, children):
        '''
        The game is won by the player with the highest score (draw if scores are equal)
        '''
        def get_winner(state):
            is_draw = state.scores[0] == state.scores[1]
            p1_wins = state.scores[0] > state.scores[1]

            return jax.lax.select(is_draw, EMPTY, jax.lax.select(p1_wins, P1, P2))
        
        info = {}

        return get_winner, info

    '''
    =========Masks=========
    '''
    def _parse_optional_args(self, children):
        '''
        Returns the setting of the optional arguments for a given predicate
        by using the name of the calling function as a key
        '''
        predicate_name = inspect.stack()[1][3]
        default_args = DEFAULT_ARGUMENTS.get(predicate_name, {})
        optional_args = {rule: value for rule, value in children}

        return {**default_args, **optional_args}
    
    def super_mask_and(self, children):
        '''
        Apply a location-wise boolean AND operation to the masks of all children
        '''
        children_mask_fns, children_infos = zip(*children)

        collect_values = utils._get_collect_values_fn(children_mask_fns)
        def mask_fn(state):
            all_masks = collect_values(state)
            result = all_masks.all(axis=0).astype(jnp.int16)

            return result
        
        info = {}

        return mask_fn, info

    def super_mask_or(self, children):
        '''
        Apply a location-wise boolean OR operation to the masks of all children
        '''
        children_mask_fns, children_infos = zip(*children)

        collect_values = utils._get_collect_values_fn(children_mask_fns)
        def mask_fn(state):
            all_masks = collect_values(state)
            result = all_masks.any(axis=0).astype(jnp.int16)

            return result
        
        info = {"mask_or_children": children_infos}

        return mask_fn, info
    
    def super_mask_not(self, children):
        '''
        Apply a location-wise boolean NOT operation to the mask of the child
        '''
        child_mask_fn, child_info = children[0]

        def mask_fn(state):
            return (child_mask_fn(state) == 0).astype(jnp.int16)
        
        info = {}

        return mask_fn, info
    
    def mask_adjacent(self, children):
        '''
        Return a mask of all the positions on the board which are adjacent in the
        specified direction to any of the active positions in the child mask
        '''

        (child_mask_fn, child_info), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        direction_indices = utils._get_adjacency_direction_indices(self.game_info, optional_args)
        local_lookup = self.adjacency_lookup[direction_indices]

        def mask_fn(state):
            child_mask = child_mask_fn(state)
            adjacent_mask = (child_mask * local_lookup).any(axis=(0, 2)).astype(jnp.int16)

            return adjacent_mask

        return mask_fn, child_info
    
    def mask_center(self, children):
        '''
        Return the center of the board (defined as halfway through the board array). For
        boards with an even number of positions, this will be empty
        '''
        center_idx = self.game_info.board_size // 2 if self.game_info.board_size % 2 == 1 else self.game_info.board_size + 1
        mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[center_idx].set(1)

        def mask_fn(state):
            return mask

        return mask_fn, {}

    def mask_corners(self, children):
        '''
        Return the corners of the board (the number and location of which depends on the shape)
        '''
        corner_indices = utils._get_corner_indices(self.game_info)
        mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
        mask = mask.at[corner_indices].set(1)

        def mask_fn(state):
            return mask
        
        return mask_fn, {}
    
    def mask_custodial(self, children):
        '''
        Returns a mask of all of the board positions which are part of a 'custodial'
        arrangement -- i.e. a line of pieces of the same color with pieces of the 
        opposite color on each end. This is a mask (instead of just a function)
        because we need to be able to count / capture / flip the specific pieces
        that appear in the arrangement

        NOTE: mask_custodial is one of the few masks / functions that has a
        lookahead_mask_fn defined
        '''
        (_, n), *optional_args = children

        optional_args = self._parse_optional_args(optional_args)
        orientation = optional_args[OptionalArgs.ORIENTATION]

        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1
        
        # Concatenate the custodial checks for all valid lengths into one lookup
        if n == "any":
            max_dimension = max(self.game_info.board_dims) - 2

            line_indices = []
            inner_indices, outer_indices = [], []

            for i in range(1, max_dimension+1):
                local_line_indices = utils._get_line_indices(self.game_info, i+2, orientation)
                local_inner_indices, local_outer_indices = utils._get_custodial_indices(self.game_info, i, orientation)
                local_line_indices = jnp.pad(local_line_indices, ((0, 0), (0, max_dimension - i)), mode='maximum')
                local_inner_indices = jnp.pad(local_inner_indices, ((0, 0), (0, max_dimension - i)), mode='maximum')

                line_indices.append(local_line_indices)
                inner_indices.append(local_inner_indices)
                outer_indices.append(local_outer_indices)

            line_indices = jnp.concatenate(line_indices, axis=0)
            inner_indices = jnp.concatenate(inner_indices, axis=0)
            outer_indices = jnp.concatenate(outer_indices, axis=0)

        else:
            n = int(n)
            line_indices = utils._get_line_indices(self.game_info, n+2, orientation)
            inner_indices, outer_indices = utils._get_custodial_indices(self.game_info, n, orientation)

        full_match_width = line_indices.shape[1]

        def mask_fn(state):
            outer_player = (state.current_player + offset) % 2
            inner_player = (outer_player + 1) % 2
            outer_mask = (state.board == outer_player)
            inner_mask = (state.board == inner_player)

            # Only keep the custodial arrangements which include the last move
            # among the outer indices. TODO: make this an argument?
            last_move = state.previous_actions[outer_player]
            valid_outer = (outer_indices == last_move).any(axis=1)[:, jnp.newaxis]

            outer_match = (outer_mask[outer_indices] == 1).all(axis=1)[:, jnp.newaxis]
            inner_match = (inner_mask[inner_indices] == 1).all(axis=1)[:, jnp.newaxis]
            full_match = jnp.tile(outer_match & inner_match & valid_outer, full_match_width)

            # Ensure invalid indices are set to one larger than the board size so that they're not indexed
            matched_indices = jnp.where(full_match, line_indices, self.game_info.board_size+1).flatten()
            mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[matched_indices].set(1)

            return mask
        
        # The lookahead mask version looks for lines that have all of the inner indices set and
        # exactly one of the outer indices set -- "left" matches have the first outer index set 
        # and the second empty, and "right" matches are the reverse
        left_indices = outer_indices[:, 0]
        right_indices = outer_indices[:, 1]

        def lookahead_mask_fn(state):
            outer_player = (state.current_player + offset) % 2
            inner_player = (outer_player + 1) % 2

            outer_mask = (state.board == outer_player)
            inner_mask = (state.board == inner_player)
            
            inner_match = (inner_mask[inner_indices] == 1).all(axis=1)
            left_match = (outer_mask[left_indices] == 1) & (outer_mask[right_indices] == 0) & inner_match
            right_match = (outer_mask[right_indices]  == 1) & (outer_mask[left_indices] == 0) & inner_match

            left_match_indices = jnp.where(left_match, right_indices, self.game_info.board_size+1)
            right_match_indices = jnp.where(right_match, left_indices, self.game_info.board_size+1)

            mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
            mask = mask.at[left_match_indices].set(1)
            mask = mask.at[right_match_indices].set(1)

            return mask
        
        return mask_fn, {"lookahead_mask_fn": lookahead_mask_fn}
    
    def mask_edge(self, children):
        '''
        Masks the edges of the board (which depends on the shape of the board)
        '''
        edge_type = children[0]
        indices = utils._get_edge_indices(self.game_info, edge_type)

        def mask_fn(state):
            mask = jnp.zeros_like(state.board).astype(jnp.bool_)
            mask = mask.at[indices].set(True)
            return mask.astype(jnp.int16)
        
        return mask_fn, {}

    def mask_empty(self, children):
        '''
        Return all the positions on the board which are empty
        '''
        def mask_fn(state):
            return (state.board == EMPTY).astype(jnp.int16)
        
        return mask_fn, {}

    def mask_occupied(self, children):
        '''
        Return the positions on the board which are occupied
        by the specified player. If no player is specified,
        then the mask checks for either player
        '''
        if len(children) == 0:
            def mask_fn(state):
                return (state.board != EMPTY).astype(jnp.int16)
            
            return mask_fn, {}
        
        mover_ref = children[0]
        
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1

        def mask_fn(state):
            return (state.board == ((state.current_player + offset)%2)).astype(jnp.int16)
        
        return mask_fn, {}
    
    def mask_prev_move(self, children):
        '''
        Returns a mask that is only active at the position of the current player's
        last move. If it's a player's first move, then the mask is empty (for a game
        with a condition like 'your move must be adjacent to your previous move', you
        may need to express it as multiple phases, where the condition is only applied
        in the second phase after both players have made their first move)
        '''
        mover_ref = children[0]

        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1

        def mask_fn(state):
            mask = jnp.zeros_like(state.board).astype(jnp.bool_)
            prev_move = state.previous_actions[(state.current_player + offset) % 2]
            mask = jax.lax.select(prev_move != -1, mask.at[prev_move].set(True), mask)

            return mask.astype(jnp.int16)
        
        return mask_fn, {}

    '''
    ==========Multi-masks==========
    '''
    def _get_static_mask(self, indices):
        mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
        mask = mask.at[indices].set(1)
        return mask

    def multi_mask_corners(self, children):
        '''
        Returns a list of masks where each mask corresponds to one corner
        '''
        corner_indices = utils._get_corner_indices(self.game_info)
        
        # This mapping over a build function appears necessary to stop the
        # local variable 'idx' from being retroactively bound to the last
        #  value for every mask function
        def build(idx):
            mask = self._get_static_mask(idx)
            return lambda state: mask
        
        mask_fns = list(map(build, corner_indices))
        mask_infos = [{} for _ in corner_indices]

        return list(zip(mask_fns, mask_infos))
    
    def multi_mask_edges(self, children):
        '''
        Returns a list of masks where each mask corresponds to one edge
        '''
        edge_types = utils._get_valid_edge_types(self.game_info)

        def build(edge_type):
            mask = self._get_static_mask(utils._get_edge_indices(self.game_info, edge_type))
            return lambda state: mask
        
        mask_fns = list(map(build, edge_types))
        mask_infos = [{} for _ in edge_types]

        return list(zip(mask_fns, mask_infos))
    
    def multi_mask_edges_no_corners(self, children):
        '''
        Returns a list of masks where each mask corresponds to one edge but
        corners are removed
        '''
        edge_types = utils._get_valid_edge_types(self.game_info)
        corner_indices = utils._get_corner_indices(self.game_info)

        def build(edge_type):
            mask = self._get_static_mask(utils._get_edge_indices(self.game_info, edge_type))
            mask = mask.at[corner_indices].set(0)
            return lambda state: mask
        
        mask_fns = list(map(build, edge_types))
        mask_infos = [{} for _ in edge_types]
        
        return list(zip(mask_fns, mask_infos))

    '''
    ==========Predicates==========
    '''
    def super_predicate_and(self, children):
        '''
        Apply a boolean AND operation over all the child predicates
        '''
        children_pred_fns, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_pred_fns)

        def predicate_fn(state):
            all_values = collect_values(state)
            result = all_values.all()
            return result
        
        info = {}

        # If all of the children have a 'lookahead_mask_fn' then we can combine them
        if all([info.get('lookahead_mask_fn', False) for info in children_info]):
            lookahead_mask_fns = [info['lookahead_mask_fn'] for info in children_info]
            collect_lookahead_values = utils._get_collect_values_fn(lookahead_mask_fns)   

            def lookahead_mask_fn(state):
                all_masks = collect_lookahead_values(state)
                result = (all_masks > 0).all(axis=0)
                return result
            
            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def super_predicate_or(self, children):
        '''
        Apply a boolean OR operation over all the child predicates
        '''
        children_pred_fns, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_pred_fns)

        def predicate_fn(state):
            all_values = collect_values(state)
            result = all_values.any()
            return result
        
        info = {}

        # If all of the children have a 'lookahead_mask_fn' then we can combine them
        if all([info.get('lookahead_mask_fn', False) for info in children_info]):
            lookahead_mask_fns = [info['lookahead_mask_fn'] for info in children_info]
            collect_lookahead_values = utils._get_collect_values_fn(lookahead_mask_fns)

            def lookahead_mask_fn(state):
                all_masks = collect_lookahead_values(state)
                result = (all_masks > 0).any(axis=0)
                return result
            
            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def super_predicate_not(self, children):
        '''
        Apply a boolean NOT operation to the child predicate
        '''
        child_pred_fn, child_info = children[0]

        def predicate_fn(state):
            return ~child_pred_fn(state)
        
        info = {}

        if child_info.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn = child_info['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask = child_lookahead_mask_fn(state)
                return (mask == 0).astype(jnp.int16)

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info

    def predicate_equals(self, children):
        '''
        Return whether all of the child function values are equal
        '''
        children_functions, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_functions)

        def predicate_fn(state):
            all_values = collect_values(state)
            result = (all_values == all_values[0]).all(axis=0)
            return result
        
        info = {}

        # If all of the children have a 'lookahead_mask_fn' then we can combine them
        if all([info.get('lookahead_mask_fn', False) for info in children_info]):
            lookahead_mask_fns = [info['lookahead_mask_fn'] for info in children_info]
            collect_lookahead_values = utils._get_collect_values_fn(lookahead_mask_fns)

            def lookahead_mask_fn(state):
                all_masks = collect_lookahead_values(state)
                result = (all_masks == all_masks[0]).all(axis=0)
                return result

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def predicate_exists(self, children):
        '''
        Return whether the given mask is active anywhere on the board
        '''

        child_mask_fn, child_info = children[0]

        def predicate_fn(state):
            mask = child_mask_fn(state)
            return mask.any()
        
        info = {}
        if child_info.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn = child_info['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask = child_lookahead_mask_fn(state)
                return (mask >= 1).astype(jnp.int16)

            info['lookahead_mask_fn'] = lookahead_mask_fn
        
        return predicate_fn, info
    
    def predicate_full_board(self, children):
        '''
        Return whether the board is completely full of pieces
        '''
        def predicate_fn(state):
            return (state.board != EMPTY).all()
        
        return predicate_fn, {}
    
    def predicate_function(self, children):
        '''
        Special syntax: returns whether a function has a value greater than 0,
        which is equivalent to (>= function 1)
        '''
        child_fn, child_info = children[0]

        def predicate_fn(state):
            return child_fn(state) > 0
        
        info = {}
        if child_info.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn = child_info['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask = child_lookahead_mask_fn(state)
                return (mask >= 1).astype(jnp.int16)

            info['lookahead_mask_fn'] = lookahead_mask_fn
        
        return predicate_fn, child_info
    
    def predicate_greater_equals(self, children):
        '''
        Returns whether the first child function vlaue is greater than or equal to the second
        '''
        (child_fn_left, child_fn_right), (child_info_left, child_info_right) = zip(*children)

        def predicate_fn(state):
            return child_fn_left(state) >= child_fn_right(state)
        
        info = {}
        if child_info_left.get('lookahead_mask_fn', False) and child_info_right.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn_left = child_info_left['lookahead_mask_fn']
            child_lookahead_mask_fn_right = child_info_right['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask_left = child_lookahead_mask_fn_left(state)
                mask_right = child_lookahead_mask_fn_right(state)
                return (mask_left >= mask_right).astype(jnp.int16)

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def predicate_less_equals(self, children):
        '''
        Returns whether the first child function vlaue is less than or equal to the second
        '''
        (child_fn_left, child_fn_right), (child_info_left, child_info_right) = zip(*children)

        def predicate_fn(state):
            return child_fn_left(state) <= child_fn_right(state)
        
        info = {}
        if child_info_left.get('lookahead_mask_fn', False) and child_info_right.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn_left = child_info_left['lookahead_mask_fn']
            child_lookahead_mask_fn_right = child_info_right['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask_left = child_lookahead_mask_fn_left(state)
                mask_right = child_lookahead_mask_fn_right(state)
                return (mask_left <= mask_right).astype(jnp.int16)

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def predicate_mover_is(self, children):
        '''
        Returns whether the state's current player is the specified player
        '''
        player_ref = children[0]
        
        if player_ref == PlayerAndMoverRefs.P1:
            player = P1
        elif player_ref == PlayerAndMoverRefs.P2:
            player = P2

        def predicate_fn(state):
            return state.current_player == player
        
        info = {}

        return predicate_fn, info
    
    def predicate_passed(self, children):
        '''
        Returns whether one or both players passed their last turn
        '''
        player_ref = children[0]
        
        if player_ref == PlayerAndMoverRefs.BOTH:
            def predicate_fn(state):
                return state.passed.all()

        else:
            if player_ref == PlayerAndMoverRefs.P1:
                idx = 0
            elif player_ref == PlayerAndMoverRefs.P2:
                idx = 1

            def predicate_fn(state):
                return state.passed[idx]

        return predicate_fn, {}
    
    '''
    ==========Functions==========
    '''

    def function_add(self, children):
        '''
        Return the sum of each of the child function values
        '''
        children_fns, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_fns)

        def function_fn(state):
            all_values = collect_values(state)
            result = all_values.sum(axis=0)
            return result
            
        return function_fn, children_info

    def function_constant(self, children):
        '''
        Return a constant specified value

        NOTE: this is one of a few functions / masks which returns a "lookahead_mask_fn"
        '''
        value = children[0]

        def lookahead_mask_fn(state):
            return jnp.ones(self.game_info.board_size, dtype=jnp.int16) * value
        
        info = {"lookahead_mask_fn": lookahead_mask_fn}

        return lambda state: value, info
    
    def function_connected(self, children):
        '''
        Return the number of the specified masks which are connected to each other
        via the last move made by the current player.

        NOTE: to simplify syntax, it's possible to specify "multi-masks" (see above)
        instead of manually enumerating each of the target masks
        '''
        (_, target_masks_infos), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        # Parse out the target mask functions
        target_mask_fns, _ = zip(*target_masks_infos)
        num_targets = len(target_mask_fns)

        if optional_args[OptionalArgs.MOVER] == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def function_fn(state):
            mover = (state.current_player + offset) % 2
            target_masks = jax.vmap(lambda i: jax.lax.switch(i, target_mask_fns, state))(jnp.arange(num_targets))

            # The connected components are necessarily computed in the update_additional_info call
            set_val = state.previous_actions[mover] + 1
            fill_mask = (state.connected_components == set_val).astype(jnp.int16)
            
            target_intersections = jnp.sum(target_masks & fill_mask, axis=1)
            return jnp.sum(target_intersections > 0)

        return function_fn, {}

    def function_count(self, children):
        '''
        Return the number of active positions in the child mask
        '''
        child_mask_fn, child_info = children[0]

        def function_fn(state):
            mask = child_mask_fn(state)
            return mask.sum()
        
        return function_fn, child_info
    
    def function_line(self, children):
        '''
        Returns the number of distinct lines of a particular length present on the board

        NOTE: 'line' is one of a few functions / masks which returns a "lookahead_mask_fn"
        that can be used to optimize certain games with "result constraints"
        '''
        n, *optional_args = children

        n = int(n)
        optional_args = self._parse_optional_args(optional_args)
        orientation = optional_args[OptionalArgs.ORIENTATION]

        if optional_args[OptionalArgs.EXACT] and n < max(self.game_info.board_dims):
            line_indices = utils._get_line_indices(self.game_info, n, orientation)
            overshoot_line_indices = utils._get_line_indices(self.game_info, n+1, orientation)

            def function_fn(state):
                occupied_mask = (state.board == state.current_player)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)
                num_lines = line_matches.sum()
                num_overshoot_lines = jax.lax.cond(num_lines, lambda: 0, lambda: (occupied_mask[overshoot_line_indices] == 1).all(axis=1).sum())

                return num_lines - 2 * num_overshoot_lines
            
            info = {}

        else:
            line_indices = utils._get_line_indices(self.game_info, n, orientation)

            def function_fn(state):
                occupied_mask = (state.board == state.current_player)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)
                return line_matches.sum()
            
            num_lines = line_indices.shape[0]
            def lookahead_mask_fn(state):
                occupied_mask = (state.board == state.current_player)

                line_mask = (occupied_mask[line_indices] == 1)
                almost_line_mask = (line_mask.sum(axis=1) == n-1)

                arange = jnp.arange(num_lines).astype(jnp.int16)
                missing_line_indices = jnp.argmin(line_mask, axis=1).astype(jnp.int16)
                missing_positions = jnp.where(almost_line_mask, line_indices[arange, missing_line_indices], self.game_info.board_size+1)
                unique_positions, counts = jnp.unique_counts(missing_positions, size=num_lines, fill_value=self.game_info.board_size+1)

                unique_positions = unique_positions.astype(jnp.int16)
                counts = counts.astype(jnp.int16)

                mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
                mask = mask.at[unique_positions].set(counts)

                return mask

            info = {"lookahead_mask_fn": lookahead_mask_fn}
        
        return function_fn, info

    def function_multiply(self, children):
        '''
        Return the product of each of the child function values
        '''
        children_fns, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_fns)

        def function_fn(state):
            all_values = collect_values(state)
            result = all_values.prod(axis=0)
            return result
            
        return function_fn, {}
    
    def function_score(self, children):
        '''
        Return the score of the specified player
        '''
        mover_ref = children[0]
        
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1

        def function_fn(state):
            score = state.scores[(state.current_player + offset) % 2]
            return score
        
        return function_fn, {}

    def function_subtract(self, children):
        '''
        Return the difference between the first and second child function values
        '''
        (child_fn1, _), (child_fn2, _) = children

        def function_fn(state):
            return child_fn1(state) - child_fn2(state)
        
        return function_fn, {}
import jax
import jax.numpy as jnp
from lark import Lark

import pgx_core as core

from config import Array, PRNGKey, State, EMPTY
from game_info import GameInfoExtractor
from game_parser import GameRuleParser

class LudaxEnvironment(core.Env):
    def __init__(self,
                 game_path: str = None,
                 game_str: str = None,
                 grammar_path: str = 'grammar.lark'):
        super().__init__()
        
        assert game_path is not None or game_str is not None, "Must provide either a game path or a game string!"
        game_str = open(game_path).read() if game_str is None else game_str

        parser = Lark(open(grammar_path).read(), start='game')

        game_tree = parser.parse(game_str)
        self.game_info = GameInfoExtractor()(game_tree)
        game_rules = GameRuleParser(self.game_info).transform(game_tree)

        self.game_state_cls = self.game_info.game_state_class

        self.obs_shape = self.game_info.observation_shape
        self.board_size = self.game_info.board_size

        self.action_size = game_rules['action_size']
        self._initialize_board = game_rules['start_rules']
        self._apply_action = game_rules['apply_action_fn']
        self._update_info = game_rules['addl_info_fn']
        self._get_legal_action_mask = game_rules['legal_action_mask_fn']
        self._apply_effects = game_rules['apply_effects_fn']
        self._get_phase_idx = game_rules['next_phase_fn']
        self._get_next_player = game_rules['next_player_fn']

        self._get_winner = game_rules['end_rules']
        self._get_terminal = lambda board, player: (board != EMPTY).all()

    def _init(self, rng: PRNGKey) -> State:

        board = jnp.ones(self.board_size, dtype=jnp.int16) * EMPTY
        board = self._initialize_board(board)

        # Temporarily hard-coding the init of the game state
        temp_current_player = jnp.int16(0)
        game_state = self.game_state_cls(
            board=board,
            current_player=temp_current_player,
            phase_idx=jnp.int16(0),
            phase_step_count=jnp.int16(0),
            previous_actions=jnp.int16([-1, -1]),
        )

        # Compute the actual starting player, using the global player offset
        current_player = self._get_next_player(game_state)
        game_state = game_state._replace(current_player=current_player)

        state = State(
            game_state=game_state,
            observation=jnp.zeros(self.obs_shape, dtype=jnp.bool_),
            legal_action_mask=jnp.ones(self.action_size, dtype=jnp.bool_),
            first_player=current_player
        )

        legal_action_mask = self._get_legal_action_mask(game_state).astype(jnp.bool_)
        state = state.replace(legal_action_mask=legal_action_mask)

        return state
    
    def _step(self, state: State, action: Array, key) -> State:
        del key

        # The game state is separate from the "environment state" and contains
        # only the information necessary to progress the game
        game_state = state.game_state

        # Track the player who started the turn
        original_player = game_state.current_player

        # Compute the new board state
        game_state = self._apply_action(game_state, action)

        # Compute any 'additional information' required by later steps. For instance, connected
        # components
        game_state = self._update_info(game_state, action)

        # Record the action
        previous_actions = game_state.previous_actions.at[game_state.current_player].set(action)
        game_state = game_state._replace(previous_actions=previous_actions)

        # Compute the new phase index
        new_phase_idx, phase_step_count = self._get_phase_idx(game_state)
        game_state = game_state._replace(phase_idx=new_phase_idx, phase_step_count=phase_step_count)

        # Use the new phase and the global player offset to determine the next player but don't apply it yet
        next_player = self._get_next_player(game_state)
        
        # Apply any effects that occur at the end of the turn
        game_state = self._apply_effects(game_state, original_player)

        # Compute the legal action mask for the upcoming player (which is used in some scoring conditions)
        new_legal_action_mask = self._get_legal_action_mask(game_state._replace(current_player=next_player)).astype(jnp.bool_)
        state = state.replace(legal_action_mask=new_legal_action_mask)

        # Use the new board to compute the winner, terminal, and rewards
        winner, terminated = self._get_winner(game_state)
        rewards = jax.lax.select(
            winner != EMPTY,
            jnp.float32([-1, -1]).at[winner].set(1),
            jnp.zeros(2, jnp.float32)
        )
        mover_reward = rewards[game_state.current_player]
        state = state.replace(winner=winner, rewards=rewards, terminated=terminated, mover_reward=mover_reward)

        # If the game hasn't ended, then update the current player
        game_state = jax.lax.cond(
            terminated,
            lambda: game_state,
            lambda: game_state._replace(current_player=next_player)
        )

        # Update the game state in the state
        state = state.replace(game_state=game_state)

        # Increment the global step count
        state = state.replace(global_step_count=state.global_step_count + 1)

        return state

    def _observe(self, state: core.State, player_id: Array) -> Array:
        return jnp.zeros(self.obs_shape, dtype=jnp.bool_)

    @property
    def num_players(self) -> int:
        return 2
    
if __name__ == '__main__':
    import time
    from tqdm import tqdm 

    BATCH_SIZE = 100
    NUM_GAMES = 10000
    VERBOSE = False

    if VERBOSE:
        BATCH_SIZE = 1
        NUM_GAMES = 1    

    GAME_PATH = "games/tic_tac_toe.ldx"
    env = LudaxEnvironment(GAME_PATH)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))

    def _run_batch(state, key):
        def cond_fn(args):
            state, _ = args
            return ~(state.terminated | state.truncated).all()
        
        def body_fn(args):
            state, key = args
            key, subkey = jax.random.split(key)
            logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
            action = jax.random.categorical(key, logits=logits, axis=1)
            state = step(state, action)
            return state, key
        
        state, key = jax.lax.while_loop(cond_fn, body_fn, (state, key))

        return state, key

    run_batch = jax.jit(_run_batch)

    key = jax.random.PRNGKey(40)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, BATCH_SIZE)

    start = time.perf_counter()

    first_player_wins = 0
    second_player_wins = 0
    draws = 0
    for _ in tqdm(range(NUM_GAMES)):
        state = init(keys)
        state, key = run_batch(state, key)

        first_player_wins += (state.winner == state.first_player).sum()
        second_player_wins += (state.winner == ((state.first_player + 1) % 2)).sum()
        draws += (state.winner == EMPTY).sum()

        key, subkey = jax.random.split(key)

    end = time.perf_counter()
    print(f"Completed {NUM_GAMES} games with batch_size={BATCH_SIZE} in {(end-start):.2f} seconds!")
    print(f"Speed: {(NUM_GAMES * BATCH_SIZE) / (end - start):.2f} games / second")

    print(f"Wins for first player: {first_player_wins} ({100 * first_player_wins / (BATCH_SIZE * NUM_GAMES):.2f}%)")
    print(f"Wins for second player: {second_player_wins} ({100 * second_player_wins / (BATCH_SIZE * NUM_GAMES):.2f}%)")
    print(f"Draws: {draws} ({100 * draws / (BATCH_SIZE * NUM_GAMES):.2f}%)")
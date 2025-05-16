import math
import os
import time

import jax
from flask import render_template, request
from markupsafe import Markup

from app import app
from config import BoardShapes, RENDER_CONFIG
from environment import LudaxEnvironment
from app.render import InteractiveBoardHandler

ENV, HANDLER, STATE = None, None, None

def cube_round(q, r, s):
    q_round, r_round, s_round = round(q), round(r), round(s)
    q_diff, r_diff, s_diff = abs(q_round - q), abs(r_round - r), abs(s_round - s)

    if q_diff > r_diff and q_diff > s_diff:
        q_round = -r_round - s_round

    elif r_diff > s_diff:
        r_round = -q_round - s_round

    else:
        s_round = -q_round - r_round

    return q_round, r_round, s_round

@app.route('/') 
def index():
    game_names = [file_name.replace(".ldx", "") for file_name in os.listdir("./games") if file_name.endswith(".ldx")]
    return render_template('index.html', games=game_names)

@app.route('/game/<id>') 
def render_game(id):

    global ENV
    global HANDLER
    global STATE

    ENV = LudaxEnvironment(f"games/{id}.ldx")
    HANDLER = InteractiveBoardHandler(ENV.game_info)

    STATE = ENV.init(jax.random.PRNGKey(42))
    
    HANDLER.render(STATE)
    time.sleep(0.1)
    game_svg = open(RENDER_CONFIG['output_filename']).read()

    return render_template('game.html', game_svg=Markup(game_svg))

@app.route('/step', methods=['POST'])
def step():
    global ENV
    global HANDLER
    global STATE
    if ENV is None:
        return "No game loaded"
    
    # Get x and y from the request
    data = request.get_json()
    x = float(data['x'])
    y = float(data['y'])

    action_idx = HANDLER.pixel_to_action((x, y))

    # Temporary workaround: if there is only one legal action, then
    # we always take it
    legal_action_mask = STATE.legal_action_mask
    if legal_action_mask.sum() == 1:
        action_idx = int(legal_action_mask.argmax())
        print(f"Only one legal action available, taking action {action_idx}!")
    else:
        action = HANDLER.action_indices[action_idx]

    STATE = ENV.step(STATE, action_idx)

    if ENV.game_info.board_shape != BoardShapes.HEXAGON:
        shaped_board = STATE.game_state.board.reshape(ENV.obs_shape[:2])
        for row in shaped_board:
            pretty_row = ' '.join(str(cell) for cell in row + 1)
            print(pretty_row.replace('0', '.').replace('1', 'X').replace('2', 'O'))
        print()
        if hasattr(STATE.game_state, "connected_components"):
            shaped_components = STATE.game_state.connected_components.reshape(ENV.obs_shape[:2])
            print(shaped_components)
    else:
        print(f"Observation shape: {ENV.obs_shape}")
        print(f"Board: {STATE.game_state.board}")
        if hasattr(STATE.game_state, "connected_components"):
            print(f"Components: {STATE.game_state.connected_components}")

    HANDLER.render(STATE)
    time.sleep(0.1)

    svg_data = open(RENDER_CONFIG['output_filename']).read()
    terminated = bool(STATE.terminated)
    winner = int(STATE.winner)
    if hasattr(STATE.game_state, "scores"):
        scores = list(map(float, STATE.game_state.scores))
    else:
        scores = [0.0, 0.0]

    print(f"Current player: {STATE.game_state.current_player}")
    print(f"Scores: {scores}")

    return {"svg": svg_data, "terminated": terminated, "winner": winner, "current_player": int(STATE.game_state.current_player),
            "scores": scores}

@app.route('/reset', methods=['POST'])
def reset():
    global ENV
    global HANDLER
    global STATE
    if ENV is None:
        return "No game loaded"
    
    STATE = ENV.init(jax.random.PRNGKey(42))
    HANDLER.render(STATE)
    time.sleep(0.1)

    svg_data = open(f"renders/test.svg").read()

    return {"svg": svg_data}
import argparse
from functools import partial
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import pandas as pd
import pgx
from thefuzz import process
from tqdm import tqdm

from environment import LudaxEnvironment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='tic_tac_toe', help='Game name')
    parser.add_argument('--num_games', type=int, default=500, help='Number of games to simulate in each sample')
    parser.add_argument('--num_warmup_games', type=int, default=100, help='Number of games to simulate before measuring each sample')
    parser.add_argument('--batch_size_step', type=int, default=2, help='Multiplier for batch size between runs')
    parser.add_argument('--num_batch_sizes', type=int, default=11, help='Number of batch sizes to evaluate')
    parser.add_argument('--ludii_cache_prefix', type=str, default='yaldabaoth')
    parser.add_argument('--ludii_thread_nums', type=int, nargs='+', default=[1, 16, 32], help='Number of threads to use for Ludii')
    parser.add_argument('--save-graphs', type=bool, default=True, help='Save graphs instead of showing them')
    parser.add_argument('--cache', type=bool, default=False, help='Load cached results if available')
    return parser.parse_args()


@partial(jax.jit, static_argnames=['step'])
def run_batch(state, step, key):
    def cond_fn(args):
        state, _ = args
        return ~(state.terminated | state.truncated).all()

    def body_fn(args):
        state, key = args
        key, subkey = jax.random.split(key)
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        action = jax.random.categorical(key, logits=logits, axis=1).astype(jnp.int16)
        state = step(state, action)
        return state, key

    state, key = jax.lax.while_loop(cond_fn, body_fn, (state, key))
    return state, key


def retrieve_ludii(args):

    ludii_caches = [pd.read_csv(f"./data/{args.ludii_cache_prefix}_speeds_{num_threads}_threads.csv") for num_threads in args.ludii_thread_nums]
    game_names = ludii_caches[0]['Name'].unique()
    closest_game = process.extractOne(args.game, game_names, scorer=process.fuzz.ratio)[0]

    playouts_per_second = [cache[cache['Name'] == closest_game]['p/s'].values[0] for cache in ludii_caches]
    moves_per_second = [cache[cache['Name'] == closest_game]['m/s'].values[0] for cache in ludii_caches]
    
    formatted_playouts = " | ".join([f"{ps:.2f}" for ps in playouts_per_second])
    formatted_moves = " | ".join([f"{ms:.2f}" for ms in moves_per_second])
    formatted_threads = " | ".join([f"{num_threads}" for num_threads in args.ludii_thread_nums])

    print(f"Closest Ludii game: '{closest_game}' @ ({formatted_playouts}) games/sec and ({formatted_moves}) moves/sec ({formatted_threads} threads)")

    return playouts_per_second, moves_per_second


def evaluate(env, num_games, num_warmup_games, batch_sizes, ludii_playouts_per_second, ludii_moves_per_second):
    times = np.zeros((len(batch_sizes), num_games))
    total_steps = np.zeros((len(batch_sizes), num_games))
    warmup_times = np.zeros((len(batch_sizes), num_warmup_games))

    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))

    for i, batch_size in enumerate(batch_sizes):
        print(f"Batch {i}: size={batch_size}")
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)

        for j in tqdm(range(num_games + num_warmup_games)):

            state = init(keys)

            t0 = time.perf_counter()
            state, key = run_batch(state, step, key)
            key.block_until_ready()
            t1 = time.perf_counter()

            if j < num_warmup_games:
                warmup_times[i, j] = t1 - t0
            else:
                times[i, j - num_warmup_games] = t1 - t0

                if isinstance(env, LudaxEnvironment):
                    total_steps[i, j - num_warmup_games] = state.global_step_count.sum()
                else:
                    total_steps[i, j - num_warmup_games] = state._step_count.sum()

        average_batch_time = np.mean(times[i])
        playouts_per_second = batch_size / average_batch_time
        moves_per_second = np.mean(total_steps[i]) / average_batch_time

        formatted_playouts = " | ".join([f"{playouts_per_second / ludii_ps:.2f}×" for ludii_ps in ludii_playouts_per_second])
        formatted_moves = " | ".join([f"{moves_per_second / ludii_ms:.2f}×" for ludii_ms in ludii_moves_per_second])

        print(f"  → {playouts_per_second:.2f} games/sec ({formatted_playouts})")
        print(f"  → {moves_per_second:.2f} moves/sec ({formatted_moves})")

    return times, warmup_times, total_steps


def cached_eval(args):
    cache_file = (
        f"./data/benchmarks/cache-"
        f"{args.game}-{args.num_games}-{args.num_warmup_games}-"
        f"{args.batch_size_step}-{args.num_batch_sizes}.npz"
    )
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    batch_sizes = np.array([args.batch_size_step ** i for i in range(args.num_batch_sizes)])

    # ——— Load from cache if present ———
    if os.path.exists(cache_file) and args.cache:
        print(f"Loading cached results from {cache_file}")
        with np.load(cache_file) as cache:
            ludii_playouts_per_second  = float(cache['ludii_playouts_per_second'])
            ludii_moves_per_second     = float(cache['ludii_moves_per_second'])
            ldx_times                  = cache['ldx_times']
            ldx_warmup_times           = cache['ldx_warmup_times']
            ldx_total_steps            = cache['ldx_total_steps']
            pgx_times                  = cache['pgx_times']
            pgx_warmup_times           = cache['pgx_warmup_times']
            pgx_total_steps            = cache['pgx_total_steps']
    else:
        ludii_playouts_per_second, ludii_moves_per_second = retrieve_ludii(args)

        eval_args = args.num_games, args.num_warmup_games, batch_sizes, ludii_playouts_per_second, ludii_moves_per_second

        print(f"=========================EVALUATING LUDII-JAX on '{args.game}'=========================")
        try:
            ldx_env = LudaxEnvironment(f"games/{args.game}.ldx")
            ldx_times, ldx_warmup_times, ldx_total_steps = evaluate(ldx_env, *eval_args)
        except Exception as e:
            print(f"Ludii-JAX evaluation failed: {e}")
            ldx_times, ldx_warmup_times = None, None
            raise e

        print(f"============================EVALUATING PGX on '{args.game}'============================")
        # Special case -- Reversi is Othello in PGX
        if args.game == "reversi":
            game = "othello"
        else:
            game = args.game

        try:
            pgx_env = pgx.make(game)
            pgx_times, pgx_warmup_times, pgx_total_steps = evaluate(pgx_env, *eval_args)
        except Exception as e:
            print(f"PGX evaluation failed: {e}")
            pgx_times, pgx_warmup_times, pgx_total_steps = None, None, None

        # ——— Save to cache ———
        save_kwargs = dict(
            batch_sizes=batch_sizes,
            ludii_playouts_per_second=ludii_playouts_per_second,
            ludii_moves_per_second=ludii_moves_per_second,
            ldx_times=ldx_times,
            ldx_warmup_times=ldx_warmup_times,
            ldx_total_steps=ldx_total_steps,
            pgx_times=pgx_times,
            pgx_warmup_times=pgx_warmup_times,
            pgx_total_steps=pgx_total_steps
        )

        np.savez(cache_file, **save_kwargs)
        print(f"Saved cached results to {cache_file}")

    return save_kwargs

def plot_graphs(cmd_args, **kwargs):

    # ——— Unpack data ———
    batch_sizes = kwargs['batch_sizes']
    ludii_playouts_per_second = kwargs['ludii_playouts_per_second']
    ludii_moves_per_second = kwargs['ludii_moves_per_second']
    ldx_times = kwargs['ldx_times']
    ldx_warmup_times = kwargs['ldx_warmup_times']
    ldx_total_steps = kwargs['ldx_total_steps']
    pgx_times = kwargs['pgx_times'] if 'pgx_times' in kwargs else None
    pgx_warmup_times = kwargs['pgx_warmup_times'] if 'pgx_warmup_times' in kwargs else None
    pgx_total_steps = kwargs['pgx_total_steps'] if 'pgx_total_steps' in kwargs else None

    # ——— Compute summary speeds ———
    stacked_batch_sizes = np.repeat(batch_sizes[:, np.newaxis], cmd_args.num_games, axis=1)

    average_ldx_playouts_per_second = np.mean(stacked_batch_sizes / ldx_times, axis=1)
    std_ldx_playouts_per_second = np.std(stacked_batch_sizes / ldx_times, axis=1)

    average_ldx_moves_per_second = np.mean(ldx_total_steps / ldx_times, axis=1)
    std_ldx_moves_per_second = np.std(ldx_total_steps / ldx_times, axis=1)

    if pgx_times is not None:
        average_pgx_playouts_per_second = np.mean(stacked_batch_sizes / pgx_times, axis=1)
        std_pgx_playouts_per_second = np.std(stacked_batch_sizes / pgx_times, axis=1)

        average_pgx_moves_per_second = np.mean(pgx_total_steps / pgx_times, axis=1)
        std_pgx_moves_per_second = np.std(pgx_total_steps / pgx_times, axis=1)


    # ——— Styling helpers ———
    def style_log_axes(ax, left_factor=0.6, right_factor=1.5):
        max_bs = max(batch_sizes)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(left_factor, right_factor * max_bs)
        ax.xaxis.set_major_locator(LogLocator(base=2, subs=[1.0], numticks=len(batch_sizes)))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2))
        ax.xaxis.set_minor_locator(LogLocator(base=2, subs='auto', numticks=len(batch_sizes)*8))
        ax.xaxis.set_minor_formatter(LogFormatterMathtext(base=2))
        ax.grid(True, which='major', linestyle=':', linewidth=0.8, alpha=0.8)


    eb_kwargs = {
        'linestyle':       '-',
        'linewidth':        1.5,
        'capsize':          4,
        'elinewidth':       1,
        'markeredgewidth':  1
    }


    # ——— Create a new save directory ———
    save_dir = os.path.join("./data/benchmarks", cmd_args.game)
    os.makedirs(save_dir, exist_ok=True)

    # ——— Plot #1: Mean Throughput (Playouts) ———
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(batch_sizes, average_ldx_playouts_per_second, yerr=std_ldx_playouts_per_second, marker='s', label='Ludax', **eb_kwargs)

    if pgx_times is not None:
        ax.errorbar(batch_sizes, average_pgx_playouts_per_second, yerr=std_pgx_playouts_per_second, marker='o', label='PGX', **eb_kwargs)

    line_styles = [':', '--', '-.']
    for i, num_threads in enumerate(cmd_args.ludii_thread_nums):
        ludii_ps = ludii_playouts_per_second[i]
        ax.axhline(y=ludii_ps, linestyle=line_styles[i], label=f'Ludii ({num_threads} threads)', color='C0', alpha=0.75)

    style_log_axes(ax)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Mean Throughput (games/sec)')

    formatted_title = cmd_args.game.title().replace('_', ' ')
    ax.set_title(formatted_title, fontweight='bold')
    ax.legend()
    fig.tight_layout()

    if cmd_args.save_graphs:
        fig.savefig(os.path.join(save_dir, f'{cmd_args.game}_playouts_per_second.png'))
    else:
        plt.show()

    # ——— Plot #2: Mean Throughput (moves) ———
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(batch_sizes, average_ldx_moves_per_second, yerr=std_ldx_moves_per_second, marker='s', label='Ludax', **eb_kwargs)

    if pgx_times is not None:
        ax.errorbar(batch_sizes, average_pgx_moves_per_second, yerr=std_pgx_moves_per_second, marker='o', label='PGX', **eb_kwargs)

    line_styles = [':', '--', '-.']
    for i, num_threads in enumerate(cmd_args.ludii_thread_nums):
        ludii_ms = ludii_moves_per_second[i]
        ax.axhline(y=ludii_ms, linestyle=line_styles[i], label=f'Ludii ({num_threads} threads)', color='C0', alpha=0.75)

    style_log_axes(ax)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Mean Throughput (moves/sec)')

    formatted_title = cmd_args.game.title().replace('_', ' ')
    ax.set_title(formatted_title, fontweight='bold')
    ax.legend()
    fig.tight_layout()

    if cmd_args.save_graphs:
        fig.savefig(os.path.join(save_dir, f'{cmd_args.game}_moves_per_second.png'))
    else:
        plt.show()

    # ——— Plot #3: Warmup Bars ———
    # We'll visualize the *first batch* (index 0) warmup times
    batch_idx = 1

    # Ludii‑JAX warmup
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(cmd_args.num_warmup_games), ldx_warmup_times[batch_idx])
    ax.set_yscale('log')
    ax.set_xlabel('Warmup Call Index')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title(f'Ludii‑JAX Warmup Times (bs={batch_sizes[batch_idx]})')
    fig.tight_layout()
    if cmd_args.save_graphs:
        fig.savefig(f'./data/benchmarks/{cmd_args.game}-ldx_warmup_bs{batch_sizes[batch_idx]}.png')
    else:
        plt.show()

    # PGX warmup (if available)
    if pgx_times is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(cmd_args.num_warmup_games), pgx_warmup_times[batch_idx])
        ax.set_yscale('log')
        ax.set_xlabel('Warmup Call Index')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title(f'PGX Warmup Times (bs={batch_sizes[batch_idx]})')
        fig.tight_layout()
        if cmd_args.save_graphs:
            fig.savefig(os.path.join(save_dir, f'{cmd_args.game}_warmul_bs{batch_sizes[batch_idx]}.png'))
        else:
            plt.show()


if __name__ == "__main__":
    args = parse_args()
    data_kwargs = cached_eval(args)
    plot_graphs(args, **data_kwargs)

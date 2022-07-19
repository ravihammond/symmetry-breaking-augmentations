import os
import sys
import argparse
import pprint
pprint = pprint.pprint
import json
from collections import defaultdict

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
from calculate_convention_stats import extract_convention_stats

def extract_dubious_conventions(args): 
    weights = load_weight_file(args.weights)
    conventions = extract_all_conventions(weights, args)
    pprint(conventions)
    # dubious_sequences = extract_dubious_sequences(weights, args.d_thresh)
    # dubious_conventions = filter_dubious_conventions(
            # conventions, dubious_sequences)

ACTION_TYPES = "DPCR"
COLOURS = "RYGWB"
RANKS = "12345"
CARDS = "01234"
ACTION_MAP = {
    "C": COLOURS,
    "R": RANKS,
    "P": CARDS,
    "D": CARDS
}


def load_weight_file(weights_path):
    if weights_path == None:
        return []

    weights_file = open(weights_path)
    weights = json.load(weights_file)
    return weights


def run_evaluation(args, weight_files):
    _, _, _, _, actors = evaluate_saved_model(
        weight_files,
        args.num_game,
        args.seed,
        args.bomb,
        num_run=args.num_run,
        device=args.device
    )

    return actors


def extract_all_conventions(weights, args):
    conventions = {}

    for weight in weights:
        conventions[weight] = extract_conventions(weight, args)

    return conventions


def extract_conventions(weight, args):
    weight_files = [weight, weight]
    actors = run_evaluation(args, weight_files)
    two_step_sequences = extract_convention_stats(actors, args)

    pprint(two_step_sequences)
    conventions = {}

    return conventions


def extract_dubious_sequences(weights, dubious_threshold):
    # for i, actor in enumerate(actors):
        # actor_stats = defaultdict(int, actor.get_stats())
        # for s_type in ACTION_TYPES:
            # for s_action in ACTION_MAP[s_type]:
                # for r_type in ACTION_TYPES:
                    # for r_action in ACTION_MAP[r_type]:
                        # two_step = f"{s_type}{s_action}_{r_type}{r_action}"
                        # player_idx = i % 2
                        # callback(actor_stats, two_step, 
                                # player_idx, **kwargs)
    return {}


def filter_dubious_conventions(conventions, dubious_sequences):
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--bomb", default=0, type=int)
    parser.add_argument("--num_game", default=1000, type=int)
    parser.add_argument("--num_run", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--c_thresh", default=0.5, type=float)
    parser.add_argument("--split", default=0, type=int)
    args = parser.parse_args()

    extract_dubious_conventions(args)

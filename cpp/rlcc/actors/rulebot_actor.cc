#include <stdlib.h>
#include <iostream>
#include <array>
#include <algorithm>
#include <random>
#include <chrono>

#include "rulebot_actor.h"

using namespace std;
namespace hle = hanabi_learning_env;

void RulebotActor::act(HanabiEnv& env, const int curPlayer) {
    if (curPlayer != playerIdx_) {
        return;
    }
    //printf("rulebot act\n");

    // If last action was anything else, discard oldest card
    hle::HanabiMove move = hle::HanabiMove(
        hle::HanabiMove::kDiscard,
        0, // Card index.
        -1, // Hint target offset (which player).
        -1, // Hint card colour.
        -1 // Hint card rank.
    );

    int last_action = env.getLastAction();
    const auto& state = env.getHleState();
    if (last_action == -1 || env.getInfo() == 8) {
        std::array<int,5> vals {0,1,2,3,4};
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle (vals.begin(), vals.end(), std::default_random_engine(seed));

        if (rand() % 2 == 0) {
            for (int colour: vals) {
                move = hle::HanabiMove(
                    hle::HanabiMove::kRevealColor,
                    -1, // Card index.
                    1, // Hint target offset (which player).
                    colour, // Hint card colour.
                    -1 // Hint card rank.
                );
                if (state.MoveIsLegal(move)) {
                    break;
                }
            } 
        } else {
            for (int cardrank: vals) {
                move = hle::HanabiMove(
                    hle::HanabiMove::kRevealRank,
                    -1, // Card index.
                    1, // Hint target offset (which player).
                    -1, // Hint card colour.
                    cardrank // Hint card rank.
                );
                if (state.MoveIsLegal(move)) {
                    break;
                }
            } 
        }

    }

    auto last_move = env.getMove(env.getLastAction());
    if (last_move.MoveType() == hle::HanabiMove::kRevealColor &&
        last_move.Color() == 0) {
        // If last action was a colour hint, play oldest card
        move = hle::HanabiMove(
            hle::HanabiMove::kPlay,
            0, // Card index.
            -1, // Hint target offset (which player).
            -1, // Hint card colour.
            -1 // Hint card rank.
        );
    }

    move = overrideMove(env, move);
    incrementPlayedCardKnowledgeCount(env, move);
    incrementStats(env, move);

    //cout << "Playing move: " << move.ToString() << endl;
    env.step(move);
}

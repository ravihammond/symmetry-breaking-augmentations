#include <stdlib.h>
#include <iostream>

#include "hanabi-learning-environment/hanabi_lib/hanabi_observation.h"

#include "rulebot_2_actor.h"

using namespace std;
namespace hle = hanabi_learning_env;

void Rulebot2Actor::act(HanabiEnv& env, const int curPlayer) {
    if (curPlayer != playerIdx_) {
        return;
    }
    //printf("rulebot 2 act\n");

    // Default action is to discard oldest card
    hle::HanabiMove move = hle::HanabiMove(
        hle::HanabiMove::kDiscard,
        0, // Card index.
        -1, // Hint target offset (which player).
        -1, // Hint card colour.
        -1 // Hint card rank.
    );

    int last_action = env.getLastAction();
    const auto& state = env.getHleState();
    if (last_action == -1 || env.getInfo() >= 4) {
        int card_rank = 0;
        do {
            move = hle::HanabiMove(
                hle::HanabiMove::kRevealRank,
                -1, // Card index.
                1, // Hint target offset (which player).
                -1, // Hint card colour.
                card_rank // Hint card rank.
            );
            card_rank++;
        } while (not state.MoveIsLegal(move));
    }

    hle::HanabiObservation obs = env.getObsShowCards();
    auto& all_hands = obs.Hands();
    auto partner_hand = all_hands[(playerIdx_ + 1) % 2];
    auto oldest_card = partner_hand.Cards()[0];

    if (state.CardPlayableOnFireworks(oldest_card)) {
        // If last action was a colour hint, play oldest card
        auto reveal_red_move = hle::HanabiMove(
            hle::HanabiMove::kRevealColor,
            -1, // Card index.
            1, // Hint target offset (which player).
            0, // Hint card colour.
            -1 // Hint card rank.
        );
        if (state.MoveIsLegal(reveal_red_move)) {
            move = reveal_red_move;
        }
    }

    move = overrideMove(env, move);
    incrementPlayedCardKnowledgeCount(env, move);
    incrementStats(env, move);

    //cout << "Playing move: " << move.ToString() << endl;
    env.step(move);
}

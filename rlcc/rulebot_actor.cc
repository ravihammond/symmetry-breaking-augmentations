#include <stdlib.h>
#include <iostream>

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
        int colour = 0;

        do {
            move = hle::HanabiMove(
                hle::HanabiMove::kRevealColor,
                -1, // Card index.
                1, // Hint target offset (which player).
                colour, // Hint card colour.
                -1 // Hint card rank.
            );
            colour++;
        } while (not state.MoveIsLegal(move));

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

    incrementPlayedCardKnowledgeCount(env, move);

    //cout << "Playing move: " << move.ToString() << endl;
    env.step(move);
}

void RulebotActor::incrementPlayedCardKnowledgeCount(const HanabiEnv& env, hle::HanabiMove move) {
    const auto& state = env.getHleState();
    const auto& game = env.getHleGame();
    auto obs = hle::HanabiObservation(state, state.CurPlayer(), true);
    auto encoder = hle::CanonicalObservationEncoder(&game);
    auto [privV0, cardCount] =
        encoder.EncodePrivateV0Belief(obs, std::vector<int>(), false, std::vector<int>());
    perCardPrivV0_ =
        extractPerCardBelief(privV0, env.getHleGame(), obs.Hands()[0].Cards().size());

    if (move.MoveType() == hle::HanabiMove::kPlay) {
        auto cardBelief = perCardPrivV0_[move.CardIndex()];
        auto [colorKnown, rankKnown] = analyzeCardBelief(cardBelief);

        if (colorKnown && rankKnown) {
            ++bothKnown_;
        } else if (colorKnown) {
            ++colorKnown_;
        } else if (rankKnown) {
            ++rankKnown_;
        } else {
            ++noneKnown_;
        }
    }
}

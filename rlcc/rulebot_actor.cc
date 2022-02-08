#include <stdlib.h>
#include <iostream>

#include "hanabi-learning-environment/hanabi_lib/hanabi_observation.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_hand.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_move.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"

#include "rulebot_actor.h"

using namespace std;
namespace hle = hanabi_learning_env;

void RulebotActor::act(HanabiEnv& env, const int curPlayer) {
    // Display cards and knowedge of all players.
    hle::HanabiObservation obs = env.getObsShowCards();
    const vector<hle::HanabiHand>& all_hands = obs.Hands();
    for (auto hand: all_hands) {
        cout << hand.ToString() << endl;
    }

    printf("a\n");
    // Check what the last action was
    int last_action = env.getLastAction();
    hle::HanabiMove last_move = env.getMove(env.getLastAction());
    printf("b\n");

        // If last action was anything else, discard oldest card
    hle::HanabiMove move = hle::HanabiMove(
        hle::HanabiMove::kDiscard,
        0, // Card index.
        -1, // Hint target offset (which player).
        -1, // Hint card colour.
        -1 // Hint card rank.
    );
    printf("c\n");
    cout << "Last action: " << last_action << endl;

    if (last_action == -1) {
        int colour = 0;
        do {
            move = hle::HanabiMove(
                hle::HanabiMove::kRevealColor,
                -1, // Card index.
                -1, // Hint target offset (which player).
                colour, // Hint card colour.
                -1 // Hint card rank.
            );
            colour++;
        } while (not move.IsValid());
    } else if (last_move.MoveType() == hle::HanabiMove::kRevealColor) {
        // If last action was a colour hint, play oldest card
        printf("d\n");
        move = hle::HanabiMove(
            hle::HanabiMove::kPlay,
            0, // Card index.
            -1, // Hint target offset (which player).
            -1, // Hint card colour.
            -1 // Hint card rank.
        );
        printf("e\n");
    }
    printf("f\n");

    cout << "Playing move: " << move.ToString() << endl;

    env.step(move);
}

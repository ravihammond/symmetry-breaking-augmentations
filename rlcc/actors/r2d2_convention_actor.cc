#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "rlcc/actors/r2d2_convention_actor.h"

#define PR true

using namespace std;

hle::HanabiMove R2D2ConventionActor::getFicticiousTeammateMove(
        const HanabiEnv& env, hle::HanabiState& fictState) {
    (void)env;
    auto originalMove = R2D2Actor::getFicticiousTeammateMove(env, fictState);
    auto signalMove = strToMove(convention_[conventionIdx_][0]);
    auto responseMove = strToMove(convention_[conventionIdx_][1]);

    auto moveHistory = fictState.MoveHistory();
    auto lastMove = moveHistory[moveHistory.size() - 1].move;
    if (lastMove.MoveType() == hle::HanabiMove::kDeal) {
        lastMove = moveHistory[moveHistory.size() - 2].move;
        if (lastMove.MoveType() == hle::HanabiMove::kDeal) {
            return originalMove;
        }
    }

    if (lastMove == signalMove && fictState.MoveIsLegal(signalMove)) {
        if(PR)printf("previous move: %s\n", lastMove.ToString().c_str());
        if(PR)printf("convention move: %s\n", responseMove.ToString().c_str());
        return responseMove;
    }
    if(PR)printf("original move: %s\n", originalMove.ToString().c_str());

    return originalMove;
}

hle::HanabiMove R2D2ConventionActor::strToMove(string key) {
    auto move = hle::HanabiMove(hle::HanabiMove::kInvalid, -1, -1, -1, -1);

    assert(key.length() == 2);
    char move_type = key[0];
    int index = key[1] - '0';

    switch (move_type) {
        case 'P':
            move.SetMoveType(hle::HanabiMove::kPlay);
            move.SetCardIndex(index);
            break;
        case 'D':
            move.SetMoveType(hle::HanabiMove::kDiscard);
            move.SetCardIndex(index);
            break;
        case 'C':
            move.SetMoveType(hle::HanabiMove::kRevealColor);
            move.SetTargetOffset(1);
            move.SetColor(index);
            break;
        case 'R':
            move.SetMoveType(hle::HanabiMove::kRevealRank);
            move.SetTargetOffset(1);
            move.SetRank(index);
            break;
        default:
            move.SetMoveType(hle::HanabiMove::kInvalid);
            break;
    }
    assert(move.MoveType() != hle::HanabiMove::kInvalid);

    return move;
}

void R2D2ConventionActor::updateStats(
        const HanabiEnv& env, hle::HanabiMove move) {
    //auto state = env.getHleState();
    //if (signalConventionApplies(env, state)) {
        //auto element = stats_.emplace("convention_total", 0);
        //stats_["convention_total"] = element.second + 1;
    //}
}


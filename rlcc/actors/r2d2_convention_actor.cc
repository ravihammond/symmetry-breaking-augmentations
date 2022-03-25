#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "rlcc/actors/r2d2_convention_actor.h"

#define PR false
using namespace std;

hle::HanabiMove R2D2ConventionActor::getFicticiousTeammateMove(
        const HanabiEnv& env, hle::HanabiState& fictState) {
    auto hands = fictState_->Hands();

    auto originalMove = R2D2Actor::getFicticiousTeammateMove(env, fictState);
    auto senderMove = strToMove(convention_[conventionIdx_][0]);
    auto responseMove = strToMove(convention_[conventionIdx_][1]);

    auto moveHistory = fictState.MoveHistory();
    auto lastMove = moveHistory[moveHistory.size() - 1].move;
    if (lastMove.MoveType() == hle::HanabiMove::kDeal) {
        lastMove = moveHistory[moveHistory.size() - 2].move;
        if (lastMove.MoveType() == hle::HanabiMove::kDeal) {
            return originalMove;
        }
    }

    if (lastMove == senderMove && fictState.MoveIsLegal(responseMove)) {
        return responseMove;
    }

    return originalMove;
}



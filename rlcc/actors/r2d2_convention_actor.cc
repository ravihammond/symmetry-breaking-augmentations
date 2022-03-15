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



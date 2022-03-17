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

    for(auto hand: hands)
        if(PR)printf("%s\n", hand.ToString().c_str());

    if(PR)printf("previous move: %s\n", lastMove.ToString().c_str());
    if(PR)printf("senderMove move: %s\n", senderMove.ToString().c_str());
    if(PR)printf("legal: %d\n", fictState.MoveIsLegal(senderMove));

    if (lastMove == senderMove && fictState.MoveIsLegal(responseMove)) {
        if(PR)printf("convention move: %s\n", responseMove.ToString().c_str());
        return responseMove;
    }
    if(PR)printf("original move: %s\n", originalMove.ToString().c_str());

    return originalMove;
}



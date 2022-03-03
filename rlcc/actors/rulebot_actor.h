#pragma once

#include "actor.h"
#include <stdio.h>

class RulebotActor: public Actor {
public:
    RulebotActor(int playerIdx): Actor(playerIdx) {}
    void act(HanabiEnv& env, const int curPlayer);
};


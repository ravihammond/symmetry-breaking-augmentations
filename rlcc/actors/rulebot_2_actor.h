#pragma once

#include "actor.h"
#include <stdio.h>

class Rulebot2Actor: public Actor {
public:
    Rulebot2Actor(int playerIdx): Actor(playerIdx) {}
    void act(HanabiEnv& env, const int curPlayer);
};


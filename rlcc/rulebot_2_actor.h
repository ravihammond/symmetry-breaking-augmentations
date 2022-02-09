#pragma once

#include "r2d2_actor.h"
#include <stdio.h>

class Rulebot2Actor: public R2D2Actor {
public:
    Rulebot2Actor(int playerIdx): R2D2Actor(playerIdx) {}

    void act(HanabiEnv& env, const int curPlayer);

    void observeBeforeAct(const HanabiEnv& env) { (void)env; }
    void fictAct(const HanabiEnv& env) { (void)env; }
    void observeAfterAct(const HanabiEnv& env) { (void)env; }
    void reset(const HanabiEnv& env) { (void)env; }
};


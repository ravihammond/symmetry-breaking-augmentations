#pragma once

#include "actor.h"
#include <stdio.h>

class Rulebot2Actor: public Actor {
    using Actor::Actor;
    void act(HanabiEnv& env, const int curPlayer) override;
};


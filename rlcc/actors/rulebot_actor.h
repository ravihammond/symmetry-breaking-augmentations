#pragma once

#include "actor.h"
#include <stdio.h>

class RulebotActor: public Actor {
    using Actor::Actor;
    void act(HanabiEnv& env, const int curPlayer) override;
};


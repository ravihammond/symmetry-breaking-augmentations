#pragma once

#include <stdio.h>
#include <iostream>

#include "rlcc/actors/r2d2_actor.h"

class R2D2ConventionActor: public R2D2Actor {
    using R2D2Actor::R2D2Actor;

private:
    hle::HanabiMove getFicticiousTeammateMove(
        const HanabiEnv& env, hle::HanabiState& fictState) override;
};

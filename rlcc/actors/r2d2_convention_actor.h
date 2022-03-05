#pragma once

#include "rlcc/actors/r2d2_actor.h"

class R2D2ConventionActor: public R2D2Actor {
    using R2D2Actor::R2D2Actor;

private:
    void changeStateForBeliefSampler(hle::HanabiState& state);
    bool conventionApplies(hle::HanabiState& state);
    vector<hle::HanabCard> getElligibleConventionCards(hle::HanabiState& state);
};

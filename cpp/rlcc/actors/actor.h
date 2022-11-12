#pragma once

#include "rlcc/hanabi_env.h"

class Actor {
public:
    virtual void reset(const HanabiEnv& env) = 0; 
    virtual void observeBeforeAct(HanabiEnv& env) = 0;
    virtual void act(HanabiEnv& env, const int curPlayer) = 0;
    virtual void fictAct(const HanabiEnv& env) = 0;
    virtual void observeAfterAct(const HanabiEnv& env) = 0;
    virtual void pushToReplayBuffer() = 0;
}; 

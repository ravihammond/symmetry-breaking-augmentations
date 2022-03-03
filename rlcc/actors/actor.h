#pragma once

#include "rlcc/hanabi_env.h"

class Actor{
public:
    Actor(int playerIdx) : playerIdx_(playerIdx) {}

    virtual void reset(const HanabiEnv& env) { (void)env; }
    virtual void observeBeforeAct(const HanabiEnv& env) { (void)env; }
    virtual void act(HanabiEnv& env, const int curPlayer) { 
        (void)env; (void)curPlayer;}
    virtual void fictAct(const HanabiEnv& env) { (void)env; }
    virtual void observeAfterAct(const HanabiEnv& env) { (void)env; }

    std::tuple<int, int, int, int> getPlayedCardInfo() const {
        return {noneKnown_, colorKnown_, rankKnown_, bothKnown_};
    }

protected:
    std::tuple<bool, bool> analyzeCardBelief(const std::vector<float>& b);
    void incrementPlayedCardKnowledgeCount(
            const HanabiEnv& env, hle::HanabiMove move);

    const int playerIdx_;
    std::vector<std::vector<float>> perCardPrivV0_;
    int noneKnown_ = 0;
    int colorKnown_ = 0;
    int rankKnown_ = 0;
    int bothKnown_ = 0;
};

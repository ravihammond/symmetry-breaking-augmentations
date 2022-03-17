#pragma once

#include "rlcc/hanabi_env.h"

class Actor{
public:
    Actor(
            int playerIdx,
            std::vector<std::vector<std::string>> convention,
            bool conventionSender,
            bool conventionOverride)
        : playerIdx_(playerIdx) 
        , convention_(convention) 
        , conventionSender_(conventionSender) 
        , conventionIdx_(0) 
        , conventionOverride_(conventionOverride) {}

    virtual void reset(const HanabiEnv& env) { (void)env; }
    virtual void observeBeforeAct(HanabiEnv& env) { (void)env; }
    virtual void act(HanabiEnv& env, const int curPlayer) { 
        (void)env; (void)curPlayer;}
    virtual void fictAct(const HanabiEnv& env) { (void)env; }
    virtual void observeAfterAct(const HanabiEnv& env) { (void)env; }

    std::tuple<int, int, int, int> getPlayedCardInfo() const {
        return {noneKnown_, colorKnown_, rankKnown_, bothKnown_};
    }

    std::unordered_map<std::string, float> getStats() const { 
        return stats_;
    }

    std::tuple<int> getTestVariable() const {
        return {testVariable_};
    }

protected:
    std::tuple<bool, bool> analyzeCardBelief(const std::vector<float>& b);
    void incrementPlayedCardKnowledgeCount(
            const HanabiEnv& env, hle::HanabiMove move);

    void incrementStat(std::string key);
    virtual void incrementStats(const HanabiEnv& env, hle::HanabiMove move);
    void incrementStatsConvention(const HanabiEnv& env, hle::HanabiMove move);
    hle::HanabiMove overrideMove(const HanabiEnv& env, hle::HanabiMove move);
    hle::HanabiMove randomMove(const HanabiEnv& env, 
            std::vector<hle::HanabiMove> exclude, hle::HanabiMove originalMove);
    bool partnerCardPlayableOnFireworks(const HanabiEnv& env);
    hle::HanabiMove strToMove(std::string key);

    const int playerIdx_;
    std::vector<std::vector<float>> perCardPrivV0_;
    std::unordered_map<std::string,float> stats_;
    int noneKnown_ = 0;
    int colorKnown_ = 0;
    int rankKnown_ = 0;
    int bothKnown_ = 0;
    int testVariable_ = 0;
    std::vector<std::vector<std::string>> convention_;
    bool conventionSender_;
    int conventionIdx_;
    bool conventionOverride_;
};

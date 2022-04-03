#pragma once

#include "rlcc/hanabi_env.h"

class Actor{
public:
    Actor(
            int playerIdx,
            std::vector<std::vector<std::vector<std::string>>> convention,
            bool conventionSender,
            bool conventionOverride,
            bool recordStats)
        : playerIdx_(playerIdx) 
        , convention_(convention) 
        , conventionSender_(conventionSender) 
        , conventionIdx_(0) 
        , conventionOverride_(conventionOverride) 
        , recordStats_(recordStats) {}

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

    void incrementStatsTwoStep(const HanabiEnv& env, hle::HanabiMove move);

    hle::HanabiMove overrideMove(const HanabiEnv& env, hle::HanabiMove move);

    bool moveInVector(std::vector<hle::HanabiMove> moves, hle::HanabiMove move);

    std::tuple<hle::HanabiMove, bool> availableSenderMove( const HanabiEnv& env, 
            std::vector<std::vector<std::string>> convention);

    hle::HanabiMove matchingResponseMove(
            std::vector<std::vector<std::string>> convention,
        hle::HanabiMove senderMove);

    hle::HanabiMove randomMove(const HanabiEnv& env, 
            std::vector<hle::HanabiMove> exclude, hle::HanabiMove originalMove);

    bool movePlayableOnFireworks(const HanabiEnv& env, hle::HanabiMove move);

    bool discardMovePlayable(const HanabiEnv& env, hle::HanabiMove move);

    hle::HanabiMove strToMove(std::string key);

    const int playerIdx_;
    std::vector<std::vector<float>> perCardPrivV0_;
    std::unordered_map<std::string,float> stats_;
    int noneKnown_ = 0;
    int colorKnown_ = 0;
    int rankKnown_ = 0;
    int bothKnown_ = 0;
    int testVariable_ = 0;
    std::vector<std::vector<std::vector<std::string>>> convention_;
    bool conventionSender_;
    int conventionIdx_;
    bool conventionOverride_;
    bool recordStats_;
    std::unordered_map<char, int> colourMoveToIndex_{
        {'R', 0}, {'Y', 1}, {'G', 2}, {'W', 3}, {'B', 4} };
    std::unordered_map<char, int> rankMoveToIndex_{
        {'1', 0}, {'2', 1}, {'3', 2}, {'4', 3}, {'5', 4} };
};

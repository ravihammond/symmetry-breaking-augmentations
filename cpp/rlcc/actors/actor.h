#pragma once

#include "rlcc/hanabi_env.h"

class Actor{
public:
    Actor(
            int seed,
            int playerIdx,
            std::vector<std::vector<std::vector<std::string>>> convention,
            int conventionIdx,
            int conventionOverride,
            bool recordStats);

    virtual void reset(const HanabiEnv& env);
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

    int getConventionIndex() {
        return {conventionIdx_};
    }

protected:
    std::tuple<bool, bool> analyzeCardBelief(const std::vector<float>& b);

    void incrementPlayedCardKnowledgeCount(
            const HanabiEnv& env, hle::HanabiMove move);

    void incrementStat(std::string key);

    virtual void incrementStatsBeforeMove(
            const HanabiEnv& env, hle::HanabiMove move);

    void incrementStatsConvention(const HanabiEnv& env, hle::HanabiMove move);

    void incrementStatsConventionRole(bool shouldHavePlayed, std::string role,
        hle::HanabiMove movePlayed, hle::HanabiMove moveRole);

    void incrementStatsTwoStep(const HanabiEnv& env, hle::HanabiMove move);

    std::string conventionString();

    virtual void incrementStatsAfterMove(const HanabiEnv& env);

    hle::HanabiMove overrideMove(const HanabiEnv& env, hle::HanabiMove move,
            std::vector<float> actionQ, bool exploreAction,
            std::vector<float> legalMoves);

    bool moveInVector(std::vector<hle::HanabiMove> moves, hle::HanabiMove move);

    std::tuple<hle::HanabiMove, bool> availableSignalMove( const HanabiEnv& env, 
            std::vector<std::vector<std::string>> convention);

    hle::HanabiMove matchingResponseMove(
            std::vector<std::vector<std::string>> convention,
        hle::HanabiMove signalMove);

    hle::HanabiMove different_action(const HanabiEnv& env, 
            std::vector<hle::HanabiMove> exclude,
            std::vector<float> actionQ, bool exploreAction,
            std::vector<float> legalMoves);

    bool movePlayableOnFireworks(const HanabiEnv& env, hle::HanabiMove move, int player);

    bool discardMovePlayable(const HanabiEnv& env, hle::HanabiMove move);

    hle::HanabiMove strToMove(std::string key);

    virtual void incrementBeliefStatsConvention(const HanabiEnv& env,
        std::vector<hle::HanabiCardValue> sampledCards);

    std::mt19937 rng_;
    const int playerIdx_;

    std::vector<std::vector<float>> perCardPrivV0_;
    std::unordered_map<std::string,float> stats_;
    int noneKnown_ = 0;
    int colorKnown_ = 0;
    int rankKnown_ = 0;
    int bothKnown_ = 0;
    int testVariable_ = 0;

    std::vector<std::vector<std::vector<std::string>>> convention_;
    int conventionIdx_;
    int conventionOverride_;
    bool sentSignal_;
    bool sentSignalStats_;
    bool recordStats_;
    std::unordered_map<char, int> colourMoveToIndex_{
        {'R', 0}, {'Y', 1}, {'G', 2}, {'W', 3}, {'B', 4} };
    std::unordered_map<char, int> rankMoveToIndex_{
        {'1', 0}, {'2', 1}, {'3', 2}, {'4', 3}, {'5', 4} };
    int livesBeforeMove_;
    std::string currentTwoStep_;
};

#pragma once

#include <stdio.h>
#include <iostream>

#include "rlcc/actors/r2d2_actor.h"

class R2D2ConventionActor: public R2D2Actor {
public:
    R2D2ConventionActor(
            std::shared_ptr<rela::BatchRunner> runner,
            int seed,
            int numPlayer,
            int playerIdx,
            const std::vector<float>& epsList,
            const std::vector<float>& tempList,
            bool vdn,
            bool sad,
            bool shuffleColor,
            bool hideAction,
            bool trinary,
            std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer,
            int multiStep,
            int seqLen,
            float gamma,
            std::vector<std::vector<std::string>> convention)
        : R2D2Actor(
            runner,
            seed,
            numPlayer,
            playerIdx,
            epsList,
            tempList,
            vdn,
            sad,
            shuffleColor,
            hideAction,
            trinary,
            replayBuffer,
            multiStep,
            seqLen,
            gamma)
        , convention_(convention)
        , conventionIdx_(0) {} 

    // simpler constructor for eval mode
    R2D2ConventionActor(
            std::shared_ptr<rela::BatchRunner> runner,
            int numPlayer,
            int playerIdx,
            bool vdn,
            bool sad,
            bool hideAction,
            std::vector<std::vector<std::string>> convention)
        : R2D2Actor(
            runner,
            numPlayer,
            playerIdx,
            vdn,
            sad,
            hideAction)
        , convention_(convention)
        , conventionIdx_(0) {} 

private:
    hle::HanabiMove getFicticiousTeammateMove(
        const HanabiEnv& env, hle::HanabiState& fictState) override;
    hle::HanabiMove strToMove(std::string key);

    void updateStats(const HanabiEnv& env, hle::HanabiMove move) override;

    std::vector<std::vector<std::string>> convention_;
    int conventionIdx_;
};

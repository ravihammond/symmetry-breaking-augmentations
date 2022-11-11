#pragma once

#include <iostream>
#include <stdio.h>

#include "rlcc/actors/actor.h"
#include "rlcc/hanabi_env.h"
#include "rela/prioritized_replay.h"

class SADActor: public Actor {
public:
    SADActor(
            std::shared_ptr<rela::BatchRunner> runner,
            int numEnvs,
            float eta,
            int numPlayer,
            int playerIdx,
            bool shuffleColor,
            std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer)
        : runner_(std::move(runner))
          , numEnvs_(numEnvs)
          , numPlayer_(numPlayer)
          , playerIdx_(playerIdx)
          , sad_(true)
          , shuffleColor_(shuffleColor)
          , replayBuffer_(replayBuffer)
          , eta_(eta)
          , hidden_(getH0(numEnvs, numPlayer))
          , numAct_(0) 
          , colorPermutes_(1)
          , invColorPermutes_(1) {
    }

    SADActor(
            std::shared_ptr<rela::BatchRunner> runner, 
            int numPlayer,
            int playerIdx)
        : runner_(std::move(runner))
          , numEnvs_(1)
          , numPlayer_(numPlayer)
          , playerIdx_(playerIdx)
          , sad_(true)
          , shuffleColor_(false)
          , replayBuffer_(nullptr)
          , eta_(0)
          , hidden_(getH0(1, numPlayer))
          , numAct_(0) 
          , colorPermutes_(1)
          , invColorPermutes_(1) {
    }

    void reset(const HanabiEnv& env) override { (void)env; }
    void observeBeforeAct(HanabiEnv& env) override;
    void act(HanabiEnv& env, const int curPlayer) override;
    void fictAct(const HanabiEnv& env) override { (void)env; }
    void observeAfterAct(const HanabiEnv& env) override { (void)env; }
    void pushToReplayBuffer() override {}

    int numAct() const {
        return numAct_;
    }

private:
    rela::TensorDict getH0(int numEnvs, int numPlayer) {
        std::vector<torch::jit::IValue> input{numEnvs * numPlayer};
        auto model = runner_->jitModel();
        model.get_method("get_h0")(input);
        auto output = model.get_method("get_h0")(input);
        auto h0 = rela::tensor_dict::fromIValue(output, torch::kCPU, true);

        return h0;
    }

    std::shared_ptr<rela::BatchRunner> runner_;
    const int numEnvs_;
    const int numPlayer_;
    const int playerIdx_;
    const bool sad_;
    const bool shuffleColor_;

    std::deque<rela::TensorDict> historyHidden_;
    std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer_;

    const float eta_;

    rela::TensorDict hidden_;
    std::atomic<int> numAct_;

    std::vector<std::vector<int>> colorPermutes_;
    std::vector<std::vector<int>> invColorPermutes_;

    rela::FutureReply futReply_;
};

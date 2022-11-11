#include <stdio.h>

#include "rlcc/actors/sad_actor.h"
#include "rlcc/utils.h"

#define PR false

using namespace std;

void SADActor::observeBeforeAct(HanabiEnv& env) {
    torch::NoGradGuard ng;
    const auto& state = env.getHleState();

    auto input = observe(
        state,
        playerIdx_,
        shuffleColor_,
        colorPermutes_[0],
        invColorPermutes_[0],
        false,
        false,
        sad_,
        false,
        true);

    vector<float> eps(1, 0); 
    input["eps"] = torch::tensor(eps);

    assert(!hidden_.empty());
    if (replayBuffer_ != nullptr) {
        historyHidden_.push_back(hidden_);
    }

    for (auto& kv : hidden_) {
        // convert to batch_first
        auto ret = input.emplace(kv.first, kv.second.transpose(0, 1));
        assert(ret.second);
    }

    futReply_ = runner_->call("act", input);
}

void SADActor::act(HanabiEnv& env, const int curPlayer) {
    torch::NoGradGuard ng;

    auto reply = futReply_.get();

    for (auto& kv : hidden_) {
        auto newHidIt = reply.find(kv.first);
        assert(newHidIt != reply.end());
        assert(newHidIt->second.sizes() == kv.second.transpose(0, 1).sizes());
        hidden_[kv.first] = newHidIt->second.transpose(0, 1);
        reply.erase(newHidIt);
    }

    // perform action for only current player
    int action = reply.at("a").item<int>();

    if (curPlayer != playerIdx_) {
        assert(action == env.noOpUid());
        return;
    }

    const auto& state = env.getHleState();
    hle::HanabiMove move = state.ParentGame()->GetMove(action);

    if(PR)printf("Playing move: %s\n", move.ToString().c_str());
    env.step(move);
}


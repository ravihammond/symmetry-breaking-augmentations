#include <stdio.h>

#include "rlcc/actors/sad_actor.h"
#include "rlcc/utils.h" 

#define PR false

using namespace std;

void SADActor::reset(const HanabiEnv& env) {
    (void)env;
    //// if ith state is terminal, reset hidden states
    //// h0: [num_layers * num_directions, batch, hidden_size]
    //TensorDict h0 = getH0(1, numPlayer_);
    //auto terminal = t.accessor<bool, 1>();
    //// std::cout << "terminal size: " << t.sizes() << std::endl;
    //// std::cout << "hid size: " << hidden_["h0"].sizes() << std::endl;
    //for (int i = 0; i < terminal.size(0); i++) {
        //if (!terminal[i]) {
            //continue;
        //}
        //for (auto& kv : hidden_) {
            //// [numLayer, numEnvs, hidDim]
            //// [numLayer, numEnvs, numPlayer (>1), hidDim]
            //kv.second.narrow(1, i * numPlayer_, numPlayer_) = h0.at(kv.first);
        //}
    //}
}

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

void SADActor::observeAfterAct(const HanabiEnv& env) {
    (void)env;
    //torch::NoGradGuard ng;
    //if (replayBuffer_ == nullptr) {
        //return;
    //}

    ////multiStepBuffer_->pushRewardAndTerminal(r, t);
    //float reward = env.stepReward();
    //bool terminated = env.terminated();
    //r2d2Buffer_->pushReward(reward);
    //r2d2Buffer_->pushTerminal(float(terminated));

    //assert(multiStepBuffer_->size() == historyHidden_.size());

    //if (!multiStepBuffer_->canPop()) {
        //assert(!r2d2Buffer_->canPop());
        //return;
    //}

    //if (terminated) {
        //lastEpisode_ = r2d2Buffer_->popTransition();
        //TensorDict hid = historyHidden_.front();
        //TensorDict nextHid = historyHidden_.back();
        //historyHidden_.pop_front();

        //auto input = lastEpisode_.toDict();
        //for (auto& kv : hid) {
            //auto ret = input.emplace(kv.first, kv.second.transpose(0, 1));
            //assert(ret.second);
        //}
        //for (auto& kv : nextHid) {
            //auto ret = input.emplace("next_" + kv.first, kv.second.transpose(0, 1));
            //assert(ret.second);
        //}

        //futPriority_ = runner_->call("compute_priority", input);
        //auto priority = futPriority_->get()["priority"].item<float>();
        //replayBuffer_->add(std::move(lastEpisode_), priority);
    //}

    //{
        //FFTransition transition = multiStepBuffer_->popTransition();
        //TensorDict hid = historyHidden_.front();
        //TensorDict nextHid = historyHidden_.back();
        //historyHidden_.pop_front();

        //auto input = transition.toDict();
        //for (auto& kv : hid) {
            //auto ret = input.emplace(kv.first, kv.second.transpose(0, 1));
            //assert(ret.second);
        //}
        //for (auto& kv : nextHid) {
            //auto ret = input.emplace("next_" + kv.first, kv.second.transpose(0, 1));
            //assert(ret.second);
        //}

        //int slot = -1;
        //auto futureReply = runner_->call("compute_priority", input, &slot);
        //auto priority = futureReply->get(slot)["priority"];

        //r2d2Buffer_->push(transition, priority, hid);
    //}

    //if (!r2d2Buffer_->canPop()) {
        //return;
    //}

    //std::vector<RNNTransition> batch;
    //torch::Tensor seqBatchPriority;
    //torch::Tensor batchLen;

    //std::tie(batch, seqBatchPriority, batchLen) = r2d2Buffer_->popTransition();
    //auto priority = aggregatePriority(seqBatchPriority, batchLen, eta_);
    //replayBuffer_->add(batch, priority);
}

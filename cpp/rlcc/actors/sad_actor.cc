#include <stdio.h>

#include "rlcc/actors/sad_actor.h"

#define PR false

using namespace std;

void SADActor::observeBeforeAct(HanabiEnv& env) {
    torch::NoGradGuard ng;
    std::vector<torch::Tensor> privS; 
    std::vector<torch::Tensor> legalMove; 
    std::vector<torch::Tensor> legalMatrix; 
    std::vector<torch::Tensor> ownHand; 

    rela::TensorDict input; 
    const auto& state = env.getHleState();

    const auto& game = env.getHleGame();
    auto obs = hle::HanabiObservation(state, playerIdx_, false);
    auto encoder = hle::CanonicalObservationEncoder(&game);

    std::vector<float> vS = encoder.Encode(
            obs,
            false,
            vector<int>(),
            shuffleColor_,
            colorPermutes_[0],
            invColorPermutes_[0],
            false);

    if (sad_) {
        auto extraObs = hle::HanabiObservation(state, playerIdx_, false); 
        std::vector<float> vGreedyAction = encoder.EncodeLastAction(
                extraObs, vector<int>(), shuffleColor_, colorPermutes_[0]);
        vS.insert(vS.end(), vGreedyAction.begin(), vGreedyAction.end());
    }

    privS.push_back(torch::tensor(vS)); 

    auto cheatObs = hle::HanabiObservation(state, playerIdx_, true);
    auto vOwnHand = encoder.EncodeOwnHandTrinary(cheatObs);
    ownHand.push_back(torch::tensor(vOwnHand));

    // legal moves
    auto legalMoves = state.LegalMoves(playerIdx_);
    std::vector<float> moveUids(env.numAction(), 0);

    for (auto move : legalMoves) {
        if (shuffleColor_ &&
                move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
            int permColor = colorPermutes_[0][move.Color()];
            move.SetColor(permColor);
        }
        auto uid = game.GetMoveUid(move);
        if (uid >= env.noOpUid()) {
            std::cout << "Error: legal move id should be < " << env.numAction() - 1 << std::endl;
            assert(false);
        }
        moveUids[uid] = 1;
    }
    if (legalMoves.size() == 0) {
        moveUids[env.noOpUid()] = 1;
    }

    legalMove.push_back(torch::tensor(moveUids)); 
    vector<float> eps(1, 0); 

    input["priv_s"] = torch::stack(privS, 0);
    input["legal_move"] = torch::stack(legalMove, 0);
    input["eps"] = torch::tensor(eps);
    input["own_hand"] = torch::stack(ownHand, 0);

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

rela::TensorDict SADActor::computeFeatureAndLegalMove(
        HanabiEnv& env) {
    std::vector<torch::Tensor> privS;
    std::vector<torch::Tensor> legalMove;
    std::vector<torch::Tensor> legalMatrix;
    std::vector<torch::Tensor> ownHand;

    const auto& state = env.getHleState();

    const auto& game = env.getHleGame();
    auto obs = hle::HanabiObservation(state, playerIdx_, false);
    auto encoder = hle::CanonicalObservationEncoder(&game);
    std::vector<int> shuffleOrder;

    std::vector<float> vS = encoder.Encode(
            obs,
            false,
            shuffleOrder,
            shuffleColor_,
            colorPermutes_[0],
            invColorPermutes_[0],
            false);

    if (sad_) {
        auto extraObs = hle::HanabiObservation(state, playerIdx_, false);
        std::vector<float> vGreedyAction = encoder.EncodeLastAction(
                extraObs, shuffleOrder, shuffleColor_, colorPermutes_[0]);
        vS.insert(vS.end(), vGreedyAction.begin(), vGreedyAction.end());
    }

    privS.push_back(torch::tensor(vS));

    auto cheatObs = hle::HanabiObservation(state, playerIdx_, true);
    auto vOwnHand = encoder.EncodeOwnHandTrinary(cheatObs);
    ownHand.push_back(torch::tensor(vOwnHand));

    // legal moves
    auto legalMoves = state.LegalMoves(playerIdx_);
    std::vector<float> moveUids(env.numAction(), 0);
    // auto moveUids = torch::zeros({numAction()});
    // auto moveAccessor = moveUids.accessor<float, 1>();
    for (auto move : legalMoves) {
        if (shuffleColor_ &&
                // fixColorPlayer_ == i &&
                move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
            int permColor = colorPermutes_[0][move.Color()];
            move.SetColor(permColor);
        }
        auto uid = game.GetMoveUid(move);
        if (uid >= env.noOpUid()) {
            std::cout << "Error: legal move id should be < " << env.numAction() - 1 << std::endl;
            assert(false);
        }
        moveUids[uid] = 1;
    }
    if (legalMoves.size() == 0) {
        moveUids[env.noOpUid()] = 1;
    }

    legalMove.push_back(torch::tensor(moveUids));

    vector<float> eps(1, 0);

    rela::TensorDict dict = {
        {"priv_s", torch::stack(privS, 0)},
        {"legal_move", torch::stack(legalMove, 0)},
        {"eps", torch::tensor(eps)},
        {"own_hand", torch::stack(ownHand, 0)},
    };

    return dict;
}

#include <stdio.h>
#include <algorithm>
#include <random>

#include "actor.h"

using namespace std;

#define CV false

Actor::Actor(
        int playerIdx,
        std::vector<std::vector<std::vector<std::string>>> convention,
        int conventionIdx,
        int conventionOverride,
        bool recordStats)
    : playerIdx_(playerIdx) 
    , convention_(convention) 
    , conventionIdx_(conventionIdx) 
    , conventionOverride_(conventionOverride) 
    , sentSignal_(false)
    , sentSignalStats_(false)
    , recordStats_(recordStats)
    , livesBeforeMove_(-1) 
    , currentTwoStep_("X") {
}

void Actor::reset(const HanabiEnv& env) {
    (void)env;
    sentSignal_ = false;
    sentSignalStats_ = false;
}

tuple<bool, bool> Actor::analyzeCardBelief(const vector<float>& b) {
    assert(b.size() == 25);
    set<int> colors;
    set<int> ranks;
    for (int c = 0; c < 5; ++c) {
        for (int r = 0; r < 5; ++r) {
            if (b[c * 5 + r] > 0) {
                colors.insert(c);
                ranks.insert(r);
            }
        }
    }
    return {colors.size() == 1, ranks.size() == 1};
}

void Actor::incrementPlayedCardKnowledgeCount(
        const HanabiEnv& env, hle::HanabiMove move) {
    const auto& state = env.getHleState();
    const auto& game = env.getHleGame();
    auto obs = hle::HanabiObservation(state, state.CurPlayer(), true);
    auto encoder = hle::CanonicalObservationEncoder(&game);
    auto [privV0, cardCount] =
        encoder.EncodePrivateV0Belief(obs, std::vector<int>(), 
                false, std::vector<int>());
    perCardPrivV0_ =
        extractPerCardBelief(privV0, env.getHleGame(), 
                obs.Hands()[0].Cards().size());

    if (move.MoveType() == hle::HanabiMove::kPlay) {
        auto cardBelief = perCardPrivV0_[move.CardIndex()];
        auto [colorKnown, rankKnown] = analyzeCardBelief(cardBelief);

        if (colorKnown && rankKnown) {
            ++bothKnown_;
        } else if (colorKnown) {
            ++colorKnown_;
        } else if (rankKnown) {
            ++rankKnown_;
        } else {
            ++noneKnown_;
        }
    }
}

hle::HanabiMove Actor::overrideMove(const HanabiEnv& env, hle::HanabiMove move, 
        vector<float> action_q) {
    if (conventionOverride_ == 0|| convention_.size() == 0 || 
            convention_[conventionIdx_].size() == 0) {
        return move;
    }

    auto lastMove = env.getMove(env.getLastAction());
    auto signalMove = strToMove(convention_[conventionIdx_][0][0]);
    auto responseMove = strToMove(convention_[conventionIdx_][0][1]);
    auto& state = env.getHleState();

    if ((conventionOverride_ == 1 || conventionOverride_ == 3)
            && (lastMove.MoveType() == hle::HanabiMove::kPlay 
            || lastMove.MoveType() == hle::HanabiMove::kDiscard) 
            && sentSignal_
            && lastMove.CardIndex() <= responseMove.CardIndex()) {
        sentSignal_ = false;
        if(CV)printf("seen signal\n");
    }

    if (conventionOverride_ == 2 || conventionOverride_ == 3 ) {
        if (lastMove == signalMove) {
            if(CV)printf("play response\n");
            return responseMove;
        } else if (move == responseMove) {
            vector<hle::HanabiMove> exclude = {responseMove};
            if (conventionOverride_ == 3) {
                exclude.push_back(signalMove);
                if (!sentSignal_ && movePlayableOnFireworks(env, responseMove) &&
                        state.MoveIsLegal(signalMove)) {
                    sentSignal_ = true;
                    if(CV)printf("play signal (move was response)\n");
                    return signalMove;
                }
            }
            if(CV)printf("playing next best move (move was response)\n");
            return action_argmax(env, exclude, action_q);
        }
    }

    if (conventionOverride_ == 1 || conventionOverride_ == 3) {
        if (!sentSignal_ && movePlayableOnFireworks(env, responseMove) &&
                state.MoveIsLegal(signalMove)) {
            sentSignal_ = true;
            if(CV)printf("play signal\n");
            return signalMove;
        } else if (move == signalMove) {
            vector<hle::HanabiMove> exclude = {signalMove};
            if (conventionOverride_ == 3) exclude.push_back(responseMove);
            if(CV)printf("playing next best move (move was signal)\n");
            return action_argmax(env, exclude, action_q);
        }
    }

    return move;
}

bool Actor::movePlayableOnFireworks(const HanabiEnv& env, hle::HanabiMove move) {
    auto& state = env.getHleState();
    hle::HanabiObservation obs = env.getObsShowCards();
    auto& allHands = obs.Hands();
    auto partnerCards = allHands[(playerIdx_ + 1) % 2].Cards();
    auto focusCard = partnerCards[move.CardIndex()];

    if (state.CardPlayableOnFireworks(focusCard))
        return true;

    return false;
}

hle::HanabiMove Actor::action_argmax(const HanabiEnv& env, 
        vector<hle::HanabiMove> exclude, vector<float> action_q) {
    assert(action_q.size() == 21);
    auto game = env.getHleGame();

    for (auto exclude_move: exclude) {
        action_q[game.GetMoveUid(exclude_move)] = 0;
    }

    auto next_best_move = distance(action_q.begin(), max_element(
                action_q.begin(), action_q.end()));

    return game.GetMove(next_best_move);
}

hle::HanabiMove Actor::strToMove(string key) {
    auto move = hle::HanabiMove(hle::HanabiMove::kInvalid, -1, -1, -1, -1);

    assert(key.length() == 2);
    char move_type = key[0];
    char move_target = key[1];

    switch (move_type) {
        case 'P':
            move.SetMoveType(hle::HanabiMove::kPlay);
            move.SetCardIndex(move_target - '0');
            break;
        case 'D':
            move.SetMoveType(hle::HanabiMove::kDiscard);
            move.SetCardIndex(move_target - '0');
            break;
        case 'C':
            move.SetMoveType(hle::HanabiMove::kRevealColor);
            move.SetColor(colourMoveToIndex_[move_target]);
            move.SetTargetOffset(1);
            break;
        case 'R':
            move.SetMoveType(hle::HanabiMove::kRevealRank);
            move.SetRank(rankMoveToIndex_[move_target]);
            move.SetTargetOffset(1);
            break;
        default:
            move.SetMoveType(hle::HanabiMove::kInvalid);
            break;
    }
    assert(move.MoveType() != hle::HanabiMove::kInvalid);

    return move;
}

void Actor::incrementStat(std::string key) {
    if (stats_.find(key) == stats_.end()) stats_[key] = 0;
    stats_[key]++;
}

void Actor::incrementStatsBeforeMove(
        const HanabiEnv& env, hle::HanabiMove move) {
    if (!recordStats_) {
        return;
    }

    string colours[5] = {"red", "yellow", "green", "white", "blue"};
    string ranks[5] = {"1", "2", "3", "4", "5"};

    switch(move.MoveType()) {
        case hle::HanabiMove::kPlay:
            incrementStat("play");
            incrementStat("play_" + to_string(move.CardIndex()));
            break;
        case hle::HanabiMove::kDiscard:
            incrementStat("discard");
            incrementStat("discard_" + to_string(move.CardIndex()));
            break;
        case hle::HanabiMove::kRevealColor:
            incrementStat("hint_colour");
            incrementStat("hint_" + colours[move.Color()]);
            break;
        case hle::HanabiMove::kRevealRank:
            incrementStat("hint_rank");
            incrementStat("hint_" + ranks[move.Rank()]);
            break;
        default:
            break;
    }   

    incrementStatsConvention(env, move);
    incrementStatsTwoStep(env, move);

    livesBeforeMove_ = env.getLife();
}

void Actor::incrementStatsConvention(
        const HanabiEnv& env, hle::HanabiMove move) {
    if (convention_.size() == 0 || 
            convention_[conventionIdx_].size() == 0) {
        return;
    }

    auto lastMove = env.getMove(env.getLastAction());
    auto signalMove = strToMove(convention_[conventionIdx_][0][0]);
    auto responseMove = strToMove(convention_[conventionIdx_][0][1]);
    auto& state = env.getHleState();
    bool shouldHavePlayedSignal = false;
    bool shouldHavePlayedResponse = false;

    if ((lastMove.MoveType() == hle::HanabiMove::kPlay
            || lastMove.MoveType() == hle::HanabiMove::kDiscard)
            && sentSignalStats_
            && lastMove.CardIndex() <= responseMove.CardIndex()) {
        sentSignalStats_ = false;
        if(CV)printf("stats seen signal\n");
    }

    if (lastMove == signalMove && state.MoveIsLegal(responseMove)) {
        shouldHavePlayedResponse = true;
        if(CV)printf("stats should play response\n");
    }
    
    if (!shouldHavePlayedResponse
            && !sentSignalStats_ 
            && movePlayableOnFireworks(env, responseMove)
            && state.MoveIsLegal(signalMove)) {
        sentSignalStats_ = true;
        shouldHavePlayedSignal = true;
        if(CV)printf("stats should play signal\n");
    } 

    incrementStatsConventionRole(shouldHavePlayedResponse, "response", 
            move, responseMove);
    incrementStatsConventionRole(shouldHavePlayedSignal, "signal", move, signalMove);
}

void Actor::incrementStatsConventionRole(bool shouldHavePlayed, string role,
        hle::HanabiMove movePlayed, hle::HanabiMove moveRole) {
    string roleStr = role + "_" + conventionString();

    if (shouldHavePlayed) {
        incrementStat(roleStr + "_available");
        if(CV)printf("stats %s available\n", role.c_str());
    }

    if (movePlayed == moveRole) {
        incrementStat(roleStr + "_played");
        if (shouldHavePlayed) {
            incrementStat(roleStr + "_played_correct");
            if(CV)printf("stats %s played correct\n", role.c_str());
        } else {
            incrementStat(roleStr + "_played_incorrect");
            if(CV)printf("stats %s played incorrect\n", role.c_str());
        }
    }

}

string Actor::conventionString() {
    string conventionStr = "";

    auto conventionSet = convention_[conventionIdx_];
    for (size_t i = 0; i < conventionSet.size(); i++) {
        if (i > 0) {
            conventionStr += "-";
        }
        auto convention = conventionSet[i];
        conventionStr += convention[0] + convention[1];
    }

    return conventionStr;
}

void Actor::incrementStatsTwoStep(
        const HanabiEnv& env, hle::HanabiMove move) {
    auto lastMove = env.getMove(env.getLastAction());
    string colours[5] = {"R", "Y", "G", "W", "B"};
    string ranks[5] = {"1", "2", "3", "4", "5"};

    string stat = "";

    switch(lastMove.MoveType()) {
        case hle::HanabiMove::kRevealColor:
            stat = "C" + colours[lastMove.Color()] ;
            break;
        case hle::HanabiMove::kRevealRank:
            stat = "R" + ranks[lastMove.Rank()];
            break;
        case hle::HanabiMove::kPlay:
            stat = "P" + to_string(lastMove.CardIndex());
            break;
        case hle::HanabiMove::kDiscard:
            stat = "D" + to_string(lastMove.CardIndex());
            break;
        default:
            currentTwoStep_ = "X";
            return;
    }

    incrementStat(stat);

    switch(move.MoveType()) {
        case hle::HanabiMove::kPlay:
            stat += "_P" + to_string(move.CardIndex());
            break;
        case hle::HanabiMove::kDiscard:
            stat += "_D" + to_string(move.CardIndex());
            break;
        case hle::HanabiMove::kRevealColor:
            stat += "_C" + colours[move.Color()];
            break;
        case hle::HanabiMove::kRevealRank:
            stat += "_R" + ranks[move.Rank()];
            break;
        default:
            currentTwoStep_ = "X";
            return;
    }
 
    currentTwoStep_ = stat;
    incrementStat(stat);
}


void Actor::incrementStatsAfterMove(
        const HanabiEnv& env) {
    if (!recordStats_) {
        return;
    }

    if (env.getCurrentPlayer() != playerIdx_ &&
        env.getLife() != livesBeforeMove_ &&
        currentTwoStep_ != "X") {
        incrementStat("dubious_" + currentTwoStep_);
    }
}


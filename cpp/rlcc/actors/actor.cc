#include <stdio.h>
#include <algorithm>
#include <random>

#include "actor.h"

using namespace std;

#define CV false

Actor::Actor(
        int seed,
        int playerIdx,
        std::vector<std::vector<std::vector<std::string>>> convention,
        int conventionIdx,
        int conventionOverride,
        bool recordStats)
    : rng_(seed)
    , playerIdx_(playerIdx) 
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
        vector<float> actionQ, bool exploreAction, vector<float> legalMoves) {
    if (conventionOverride_ == 0|| convention_.size() == 0 || 
            convention_[conventionIdx_].size() == 0) {
        return move;
    }

    auto lastMove = env.getMove(env.getLastAction());
    auto signalMove = strToMove(convention_[conventionIdx_][0][0]);
    auto responseMove = strToMove(convention_[conventionIdx_][0][1]);
    auto& state = env.getHleState();
    int nextPlayer = (playerIdx_ + 1) % 2;

    if ((conventionOverride_ == 1 || conventionOverride_ == 3)
            && (lastMove.MoveType() == hle::HanabiMove::kPlay 
            || lastMove.MoveType() == hle::HanabiMove::kDiscard) 
            && sentSignal_
            && lastMove.CardIndex() <= responseMove.CardIndex()) {
        sentSignal_ = false;
        if(CV)printf("signal reset\n");
    }

    if (conventionOverride_ == 2 || conventionOverride_ == 3 ) {
        if (lastMove == signalMove) {
            if(CV)printf("play response =================================\n");
            return responseMove;
        } else if (move == responseMove) {
            vector<hle::HanabiMove> exclude = {responseMove};
            if (conventionOverride_ == 3) {
                exclude.push_back(signalMove);
                if (!sentSignal_ 
                        && movePlayableOnFireworks(env, responseMove, nextPlayer) 
                        && state.MoveIsLegal(signalMove)) {
                    sentSignal_ = true;
                    if(CV)printf("play signal (move was response) **********************************\n");
                    return signalMove;
                }
            }
            if(CV)printf("playing new move (move was response) =================================\n");
            return different_action(env, exclude, actionQ, exploreAction, legalMoves);
        }
    }

    if (conventionOverride_ == 1 || conventionOverride_ == 3) {
        if (!sentSignal_ && movePlayableOnFireworks(env, responseMove, nextPlayer) 
                && state.MoveIsLegal(signalMove)) {
            sentSignal_ = true;
            if(CV)printf("play signal ========================================\n");
            return signalMove;
        } else if (move == signalMove) {
            vector<hle::HanabiMove> exclude = {signalMove};
            if (conventionOverride_ == 3) exclude.push_back(responseMove);
            if(CV)printf("playing new move (move was signal) *********************************\n");
            return different_action(env, exclude, actionQ, exploreAction, legalMoves);
        }
    }

    return move;
}

bool Actor::movePlayableOnFireworks(const HanabiEnv& env, hle::HanabiMove move, 
        int player) {
    auto& state = env.getHleState();
    hle::HanabiObservation obs = env.getObsShowCards();
    auto& allHands = obs.Hands();
    auto partnerCards = allHands[player].Cards();
    auto focusCard = partnerCards[move.CardIndex()];

    if (state.CardPlayableOnFireworks(focusCard))
        return true;

    return false;
}

hle::HanabiMove Actor::different_action(const HanabiEnv& env, 
        vector<hle::HanabiMove> exclude, vector<float> actionQ, 
        bool exploreAction, vector<float> legalMoves) {
    assert(actionQ.size() == 21);
    assert(legalMoves.size() == 21);

    auto game = env.getHleGame();

    for (auto exclude_move: exclude) {
        actionQ[game.GetMoveUid(exclude_move)] = 0;
        legalMoves[game.GetMoveUid(exclude_move)] = 0;
    }

    int nextBestMove = -1;

    if (exploreAction) {
        vector<int> legalIndices;
        vector<int> output;
        for(size_t i = 0; i < legalMoves.size(); i++) 
            if(legalMoves[i]) 
                legalIndices.push_back((int)i);
        sample(legalIndices.begin(), legalIndices.end(),
                back_inserter(output), 1, rng_);

        nextBestMove = output[0];
    } else {
        nextBestMove = distance(actionQ.begin(), max_element(
                    actionQ.begin(), actionQ.end()));
    }
    assert(nextBestMove != -1);

    return game.GetMove(nextBestMove);
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
    int nextPlayer = (playerIdx_ + 1) % 2;

    // Have seen teammate play response move, or a card before response card
    if ((lastMove.MoveType() == hle::HanabiMove::kPlay
            || lastMove.MoveType() == hle::HanabiMove::kDiscard)
            && sentSignalStats_
            && lastMove.CardIndex() <= responseMove.CardIndex()) {
        sentSignalStats_ = false;
        if(CV)printf("stats seen signal\n");
    }

    // Should play the response move
    if (lastMove == signalMove && state.MoveIsLegal(responseMove)) {
        shouldHavePlayedResponse = true;
        if(CV)printf("stats should play response\n");
    }
    
    // Should play the signal move
    if (!shouldHavePlayedResponse
            && !sentSignalStats_ 
            && movePlayableOnFireworks(env, responseMove, nextPlayer)
            && state.MoveIsLegal(signalMove)) {
        shouldHavePlayedSignal = true;
        if(CV)printf("stats should play signal\n");
    } 
    
    // Signal move has been played
    if (move == signalMove) {
        if(CV)printf("stats signal move played\n");
        sentSignalStats_ = true;
    }

    // Current turn caused a life to be lost
    if (shouldHavePlayedResponse 
            && move == responseMove 
            && !movePlayableOnFireworks(env, move, playerIdx_)) {
        if(CV)printf("response played losing a life\n");
        incrementStat("response_played_life_lost");
    }

    incrementStatsConventionRole(shouldHavePlayedResponse, "response", move, responseMove);
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

void Actor::incrementBeliefStatsConvention(const HanabiEnv& env,
        std::vector<hle::HanabiCardValue> sampledCards) {
    (void)env;
    (void)sampledCards;
    //if (convention_.size() == 0 || convention_[conventionIdx_].size() == 0) {
        //return;
    //}

    //printf("Sampled cards\n");
    //string colours[5] = {"R","Y","G","W","B"};
    //int ranks[5] = {1,2,3,4,5};
    //for (auto cardValue: sampledCards) {
        //printf("%s\n", cardValue.ToString().c_str());
    //}

    //auto partnerLastMove = env.getMove(env.getLastAction());
    //auto myLastMove = env.getMove(env.getSecondLastAction());
    //auto signalMove = strToMove(convention_[conventionIdx_][0][0]);
    //auto responseMove = strToMove(convention_[conventionIdx_][0][1]);

    //if ((myLastMove.MoveType() == hle::HanabiMove::kPlay 
            //|| myLastMove.MoveType() == hle::HanabiMove::kDiscard) 
            //&& signalReceived_
            //&& myLastMove.CardIndex() <= responseMove.CardIndex()) {
        //signalReceived_ = false;
        //if(CV)printf("BELIEF STATS --- signal received reset\n");
    //}

    //if (partnerLastMove == signalMove) {
        //signalReceived_ = true;
    //}

    //if (signalReceived_) {
        //incrementStat("response_should_be_playable");
    //} else {
        //incrementStat("response_should_not_be_playable");
    //}

    //if (movePlayableOnFireworks(env, responseMove, playerIdx_)) {

    //}

    //auto& state = env.getHleState();
    //auto obs = env.getObsShowCards();
    //auto& allHands = obs.Hands();
    //auto partnerCards = allHands[playerIdx_].Cards();
    //auto focusCard = partnerCards[responseMove.CardIndex()];

    //printf("focusCard: %s, id: %d\n", focusCard.ToString().c_str(), focusCard.Id());


    //if (state.CardPlayableOnFireworks(focusCard))
        //return true;

    //return false;

    //auto& state = env.getHleState();
    //auto cardValue = sampledCards[responseMove.CardIndex()];
    //auto focusCard = hle::HanabiCard(cardValue, );

    //if (state.CardPlayableOnFireworks(focusCard))

}


    //if (conventionOverride_ == 2 || conventionOverride_ == 3 ) {
        //if (lastMove == signalMove) {
            //if(CV)printf("play response =================================\n");
            //return responseMove;
        //} else if (move == responseMove) {
            //vector<hle::HanabiMove> exclude = {responseMove};
            //if (conventionOverride_ == 3) {
                //exclude.push_back(signalMove);
                //if (!sentSignal_ 
                        //&& movePlayableOnFireworks(env, responseMove, nextPlayer) 
                        //&& state.MoveIsLegal(signalMove)) {
                    //sentSignal_ = true;
                    //if(CV)printf("play signal (move was response) **********************************\n");
                    //return signalMove;
                //}
            //}
            //if(CV)printf("playing new move (move was response) =================================\n");
            //return different_action(env, exclude, actionQ, exploreAction, legalMoves);
        //}
    //}

    //if (conventionOverride_ == 1 || conventionOverride_ == 3) {
        //if (!sentSignal_ && movePlayableOnFireworks(env, responseMove, nextPlayer) 
                //&& state.MoveIsLegal(signalMove)) {
            //sentSignal_ = true;
            //if(CV)printf("play signal ========================================\n");
            //return signalMove;
        //} else if (move == signalMove) {
            //vector<hle::HanabiMove> exclude = {signalMove};
            //if (conventionOverride_ == 3) exclude.push_back(responseMove);
            //if(CV)printf("playing new move (move was signal) *********************************\n");
            //return different_action(env, exclude, actionQ, exploreAction, legalMoves);
        //}
    //}


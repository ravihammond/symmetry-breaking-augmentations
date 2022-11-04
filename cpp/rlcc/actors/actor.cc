#include <stdio.h>
#include <algorithm>
#include <random>

#include "actor.h"

using namespace std;

#define CV false
#define BF false

Actor::Actor(
            int seed,
            int numPlayer,
            int playerIdx,
            std::vector<std::vector<std::vector<std::string>>> convention,
            int conventionIdx,
            int conventionOverride,
            bool recordStats)
        : rng_(seed)
        , numPlayer_(numPlayer) 
        , playerIdx_(playerIdx) 
        , convention_(convention) 
        , conventionIdx_(conventionIdx) 
        , conventionOverride_(conventionOverride) 
        , sentSignal_(false)
        , sentSignalStats_(false)
        , recordStats_(recordStats)
        , livesBeforeMove_(-1) 
        , currentTwoStep_("X") 
        , beliefStatsSignalReceived_(false) {
    auto responseMove = strToMove(convention_[conventionIdx_][0][1]);
    beliefStatsResponsePosition_ = responseMove.CardIndex();
}

void Actor::reset(const HanabiEnv& env) {
    (void)env;
    sentSignal_ = false;
    sentSignalStats_ = false;
    beliefStatsSignalReceived_ = false;
    auto responseMove = strToMove(convention_[conventionIdx_][0][1]);
    beliefStatsResponsePosition_ = responseMove.CardIndex();
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
    if(CV)printf("Move before override: %s\n", move.ToString().c_str());

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
    }

    if (conventionOverride_ == 2 || conventionOverride_ == 3 ) {
        if (lastMove == signalMove) {
            return responseMove;
        } else if (move == responseMove) {
            vector<hle::HanabiMove> exclude = {responseMove};
            if (conventionOverride_ == 3) {
                exclude.push_back(signalMove);
                if (!sentSignal_ 
                        && movePlayableOnFireworks(env, responseMove, nextPlayer) 
                        && state.MoveIsLegal(signalMove)) {
                    sentSignal_ = true;
                    return signalMove;
                }
            }
            return different_action(env, exclude, actionQ, exploreAction, legalMoves);
        }
    }

    if (conventionOverride_ == 1 || conventionOverride_ == 3) {
        if (!sentSignal_ && movePlayableOnFireworks(env, responseMove, nextPlayer) 
                && state.MoveIsLegal(signalMove)) {
            sentSignal_ = true;
            return signalMove;
        } else if (move == signalMove) {
            vector<hle::HanabiMove> exclude = {signalMove};
            if (conventionOverride_ == 3) exclude.push_back(responseMove);
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
    }

    // Should play the response move
    if (lastMove == signalMove && state.MoveIsLegal(responseMove)) {
        shouldHavePlayedResponse = true;
    }
    
    // Should play the signal move
    if (!shouldHavePlayedResponse
            && !sentSignalStats_ 
            && movePlayableOnFireworks(env, responseMove, nextPlayer)
            && state.MoveIsLegal(signalMove)) {
        shouldHavePlayedSignal = true;
    } 
    
    // Signal move has been played
    if (move == signalMove) {
        sentSignalStats_ = true;
    }

    // Current turn caused a life to be lost
    if (shouldHavePlayedResponse 
            && move == responseMove 
            && !movePlayableOnFireworks(env, move, playerIdx_)) {
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
    }

    if (movePlayed == moveRole) {
        incrementStat(roleStr + "_played");
        if (shouldHavePlayed) {
            incrementStat(roleStr + "_played_correct");
        } else {
            incrementStat(roleStr + "_played_incorrect");
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

tuple<int, int, vector<int>> Actor::beliefConventionPlayable(const HanabiEnv& env) {
    int curPlayer = env.getCurrentPlayer();
    auto partner = partners_[(playerIdx_ + 1) % 2];
    vector<int> playableCards(25, 0);
    if (true) {
        return make_tuple(0, beliefStatsResponsePosition_, playableCards);
    }
    if (curPlayer != playerIdx_ 
            || convention_.size() == 0 
            || convention_[conventionIdx_].size() == 0
            || partner->previousHand_ == nullptr) {
        return make_tuple(0, beliefStatsResponsePosition_, playableCards);
    }

    auto& state = env.getHleState();
    auto partnerLastMove = env.getMove(env.getLastAction());
    auto myLastMove = env.getMove(env.getSecondLastAction());
    auto signalMove = strToMove(convention_[conventionIdx_][0][0]);
    auto responseMove = strToMove(convention_[conventionIdx_][0][1]);

    // Reset if partners play may be the signal card 
    if (beliefStatsSignalReceived_
        && partnerLastMove.MoveType() == hle::HanabiMove::kPlay
        && playedCardPossiblySignalledCard(
            partnerLastMove, partner->previousHand_)) {
        beliefStatsSignalReceived_ = false;
        beliefStatsResponsePosition_ = responseMove.CardIndex();
    }

    // Reset or shift position if my last action was discard
    if (beliefStatsSignalReceived_
        && myLastMove.MoveType() == hle::HanabiMove::kDiscard) {
        if (myLastMove.CardIndex() < beliefStatsResponsePosition_) {
            beliefStatsResponsePosition_--;
        } else if (myLastMove.CardIndex() == beliefStatsResponsePosition_) {
            beliefStatsSignalReceived_ = false;
            beliefStatsResponsePosition_ = responseMove.CardIndex();
        }
    }

    // Reset or shift position if my last action was play
    if (beliefStatsSignalReceived_
        && myLastMove.MoveType() == hle::HanabiMove::kPlay) {
        if (myLastMove.CardIndex() == beliefStatsResponsePosition_) {
            beliefStatsSignalReceived_ = false;
            beliefStatsResponsePosition_ = responseMove.CardIndex();
        } else if (previousHand_ != nullptr
                && playedCardPossiblySignalledCard(myLastMove, previousHand_)) {
            beliefStatsSignalReceived_ = false;
            beliefStatsResponsePosition_ = responseMove.CardIndex();
        } else if (myLastMove.CardIndex() < beliefStatsResponsePosition_) {
            beliefStatsResponsePosition_--;
        }
    }

    if (partnerLastMove == signalMove) {
        beliefStatsSignalReceived_ = true;
    }

    auto obs = env.getObsShowCards();
    auto& all_hands = obs.Hands();
    auto myHand = all_hands[playerIdx_];
    auto conventionCard = myHand.Cards()[beliefStatsResponsePosition_];

    possibleResponseCards(playableCards);

    if (beliefStatsSignalReceived_) {
        incrementStat("response_should_be_playable");
        if (state.CardPlayableOnFireworks(conventionCard)) {
            incrementStat("response_is_playable");
        }
        return make_tuple(1, beliefStatsResponsePosition_, playableCards);
    }

    return make_tuple(0, beliefStatsResponsePosition_, playableCards);
}

bool Actor::playedCardPossiblySignalledCard(hle::HanabiMove playedMove,
        shared_ptr<hle::HanabiHand> playedHand) {
    auto myHandKnowledge = previousHand_->Knowledge();
    auto signalledCardKnowledge = myHandKnowledge[beliefStatsResponsePosition_];
    string colourKnowledgeStr = signalledCardKnowledge.ColorKnowledgeRangeString();
    string rankKnowledgeStr = signalledCardKnowledge.RankKnowledgeRangeString();

    auto playedCard = playedHand->Cards()[playedMove.CardIndex()];
    char colours[5] = {'R','Y','G','W','B'};
    char ranks[5] = {'1','2','3','4','5'};
    char playedColour = colours[playedCard.Color()];
    char playedRank = ranks[playedCard.Rank()];

    bool colourMatch = false;
    bool rankMatch = false;

    for (char& ch: colourKnowledgeStr) {
        if (ch == playedColour) {
            colourMatch = true;
            break;
        }
    }

    for (char& ch: rankKnowledgeStr) {
        if (ch == playedRank) {
            rankMatch = true;
            break;
        }
    }

    return colourMatch && rankMatch;
}

void Actor::possibleResponseCards(vector<int> playableCards) {
    (void)playableCards;
    //for (int i: playableCards) {
        //printf("%d", i);
    //}
    //printf("\n");
}

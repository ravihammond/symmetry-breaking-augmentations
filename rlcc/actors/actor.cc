#include <stdio.h>

#include "actor.h"

using namespace std;

#define PR false

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

hle::HanabiMove Actor::overrideMove(
        const HanabiEnv& env, hle::HanabiMove move) {
    if (not conventionOverride_ || convention_.size() == 0) {
        return move;
    }
    if(PR)printf("move before override: %s\n", move.ToString().c_str());

    auto last_move = env.getMove(env.getLastAction());
    auto senderMove = strToMove(convention_[conventionIdx_][0]);
    auto responseMove = strToMove(convention_[conventionIdx_][1]);
    auto& state = env.getHleState();

    if (conventionSender_) {
        if(PR)printf("sender\n");
        if (partnerCardPlayableOnFireworks(env) &&
                state.MoveIsLegal(senderMove)) {
            if(PR)printf("should play convention\n");
            return senderMove;
        } else {
            return randomMove(env, senderMove, move);
        }
    } else {
        if(PR)printf("responder\n");
        if (last_move == senderMove){
            if(PR)printf("should play convention\n");
            return responseMove;
        }
    }

    return move;
}

bool Actor::partnerCardPlayableOnFireworks(const HanabiEnv& env) {
    auto responseMove = strToMove(convention_[conventionIdx_][1]);

    auto& state = env.getHleState();
    hle::HanabiObservation obs = env.getObsShowCards();
    auto& allHands = obs.Hands();
    auto partnerCards = allHands[(playerIdx_ + 1) % 2].Cards();
    auto focusCard = partnerCards[responseMove.CardIndex()];

    if (state.CardPlayableOnFireworks(focusCard))
        return true;

    return false;
}

hle::HanabiMove Actor::strToMove(string key) {
    auto move = hle::HanabiMove(hle::HanabiMove::kInvalid, -1, -1, -1, -1);

    assert(key.length() == 2);
    char move_type = key[0];
    int index = key[1] - '0';

    switch (move_type) {
        case 'P':
            move.SetMoveType(hle::HanabiMove::kPlay);
            move.SetCardIndex(index);
            break;
        case 'D':
            move.SetMoveType(hle::HanabiMove::kDiscard);
            move.SetCardIndex(index);
            break;
        case 'C':
            move.SetMoveType(hle::HanabiMove::kRevealColor);
            move.SetTargetOffset(1);
            move.SetColor(index);
            break;
        case 'R':
            move.SetMoveType(hle::HanabiMove::kRevealRank);
            move.SetTargetOffset(1);
            move.SetRank(index);
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

void Actor::incrementStats(
        const HanabiEnv& env, hle::HanabiMove move) {
    string colours[5] = {"red", "yellow", "green", "white", "blue"};
    string ranks[5] = {"1", "2", "3", "4", "5"};

    switch(move.MoveType()) {
        case hle::HanabiMove::kPlay:
            incrementStat("play");
            break;
        case hle::HanabiMove::kDiscard:
            incrementStat("discard");
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
}

void Actor::incrementStatsConvention(
        const HanabiEnv& env, hle::HanabiMove move) {
    if (convention_.size() == 0)
        return;

    auto& state = env.getHleState();

    // Extract convention moves
    auto senderMove = strToMove(convention_[conventionIdx_][0]);
    auto responseMove = strToMove(convention_[conventionIdx_][1]);
    auto conventionMove = senderMove;

    bool shouldHavePlayedConvention = false;

    if (conventionSender_) {
        if (partnerCardPlayableOnFireworks(env) &&
                state.MoveIsLegal(senderMove)) {
            shouldHavePlayedConvention = true;
        }

        hle::HanabiObservation obs = env.getObsShowCards();
        auto& allHands = obs.Hands();
        auto partnerCards = allHands[(playerIdx_ + 1) % 2].Cards();
        for (unsigned long i = 0; i < partnerCards.size(); i++)  {
            auto card = partnerCards[i];
            if (conventionMove == move && 
                    state.CardPlayableOnFireworks(card) &&
                    state.MoveIsLegal(senderMove)) {
                incrementStat("convention_played_" + to_string(i) + "_playable");
            }
        }

    } else {
        conventionMove = responseMove;
        auto last_move = env.getMove(env.getLastAction());
        if (last_move == senderMove && state.MoveIsLegal(responseMove))
            shouldHavePlayedConvention = true;
    }

    if (shouldHavePlayedConvention) {
        incrementStat("convention_available");
        if(PR)printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
    }


    if (conventionMove == move) {
        incrementStat("convention_played");
        if (shouldHavePlayedConvention) {
            incrementStat("convention_played_correct");
        } else {
            incrementStat("convention_played_incorrect");
        }
    }
}


#include <stdio.h>
#include <algorithm>
#include <random>

#include "actor.h"

using namespace std;

Actor::Actor(
        int playerIdx,
        std::vector<std::vector<std::vector<std::string>>> convention,
        int conventionIdx,
        bool conventionSender,
        bool conventionOverride,
        bool recordStats)
    : playerIdx_(playerIdx) 
    , convention_(convention) 
    , conventionSender_(conventionSender) 
    , conventionIdx_(conventionIdx) 
    , conventionOverride_(conventionOverride) 
    , recordStats_(recordStats) {
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

hle::HanabiMove Actor::overrideMove(
        const HanabiEnv& env, hle::HanabiMove move) {
    if (not conventionOverride_ || convention_.size() == 0 || 
            convention_[conventionIdx_].size() == 0) {
        return move;
    }

    auto lastMove = env.getMove(env.getLastAction());

    vector<hle::HanabiMove> senderMoves;
    vector<hle::HanabiMove> responseMoves;
    auto currentConvention = convention_[conventionIdx_];
    for (auto convention: currentConvention) {
        senderMoves.push_back(strToMove(convention[0]));
        responseMoves.push_back(strToMove(convention[1]));
    }

    if (conventionSender_) {
        auto [senderMove, moveAvailable] = 
            availableSenderMove(env, currentConvention);
        if (moveAvailable) {
            return senderMove;
        }
        if (moveInVector(senderMoves, move)) {
            return randomMove(env, senderMoves, move);
        }
    } else {
        if (moveInVector(senderMoves, lastMove)) {
            return matchingResponseMove(currentConvention, lastMove);
        }
        if (moveInVector(responseMoves, move)) {
            return randomMove(env, responseMoves, move);
        }
    }

    return move;
}


bool Actor::moveInVector(vector<hle::HanabiMove> moves, hle::HanabiMove move) {
    if (find(moves.begin(), moves.end(), move) != moves.end()) {
        return true;
    }
    return false;
}


hle::HanabiMove Actor::matchingResponseMove(vector<vector<string>> convention,
        hle::HanabiMove senderMove) {
    for (auto twoStepConvention: convention) {
        if (strToMove(twoStepConvention[0]) == senderMove) {
            return strToMove(twoStepConvention[1]);
        }
    }
    return hle::HanabiMove(hle::HanabiMove::kInvalid, -1, -1, -1, -1);
}


tuple<hle::HanabiMove, bool> Actor::availableSenderMove(const HanabiEnv& env,
        vector<vector<string>> convention) {
    auto move = hle::HanabiMove(hle::HanabiMove::kInvalid, -1, -1, -1, -1);
    bool available = false;
    auto& state = env.getHleState();
    vector<hle::HanabiMove> possibleMoves;

    for (size_t i = 0; i < convention.size(); i++) {
        auto senderMove = strToMove(convention[i][0]);
        auto responseMove = strToMove(convention[i][1]);

        if (responseMove.MoveType() == hle::HanabiMove::kPlay) {
            if (movePlayableOnFireworks(env, responseMove)
                    && state.MoveIsLegal(senderMove)) {
                possibleMoves.push_back(senderMove);
                available = true;
            }
        } else if (responseMove.MoveType() == hle::HanabiMove::kDiscard) {
            if (discardMovePlayable(env, responseMove)
                    && state.MoveIsLegal(senderMove)) {
                possibleMoves.push_back(senderMove);
                available = true;
            }
        }
    }

    if (possibleMoves.size() > 0) {
        move = possibleMoves[rand() % possibleMoves.size()];
    }

    return {move, available};
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

bool Actor::discardMovePlayable(const HanabiEnv& env, hle::HanabiMove move) {
    // TODO: Implement discard response Convention Moves.
    (void)env;
    (void)move;
    return false;
}

hle::HanabiMove Actor::randomMove(const HanabiEnv& env, 
        vector<hle::HanabiMove> exclude, hle::HanabiMove originalMove) {
    auto game = env.getHleGame();

    // Get possible discard and hint moves, and shuffle them.
    vector<int> discard_moves = {0, 1, 2, 3, 4};
    vector<int> hint_moves = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    auto rd = random_device {};
    auto rng = default_random_engine { rd() };
    shuffle(begin(discard_moves), end(discard_moves), rng);
    shuffle(begin(hint_moves), end(hint_moves), rng);

    // Concatenate possible moves into single list, random order.
    auto moveList = discard_moves;
    auto appendList = hint_moves;
    if (rand() % 2 == 0) {
        moveList = hint_moves;
        appendList = discard_moves;
    }
    moveList.insert(moveList.end(), appendList.begin(), appendList.end());

    // Loop through all possible moves.
    auto& state = env.getHleState();
    for (auto moveUid: moveList) {
        auto move = game.GetMove(moveUid);
        // If current move should be excluded, skip it.
        if (find(exclude.begin(), exclude.end(), move) != exclude.end())
            continue;
        // If random move is legal, choose it.
        if (state.MoveIsLegal(move))
            return move;
    }

    return originalMove;
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

void Actor::incrementStats(
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
}

void Actor::incrementStatsConvention(
        const HanabiEnv& env, hle::HanabiMove move) {
    if (convention_.size() == 0) {
        return;
    }

    auto& state = env.getHleState();
    auto conventionMove = hle::HanabiMove(hle::HanabiMove::kInvalid, -1, -1, -1, -1);
    bool shouldHavePlayedConvention = false;

    // Extract convention moves
    for (auto convention: convention_[conventionIdx_]) {
        auto senderMove = strToMove(convention[0]);
        auto responseMove = strToMove(convention[1]);

        if (conventionSender_) {
            conventionMove = senderMove;
            if (movePlayableOnFireworks(env, responseMove) &&
                    state.MoveIsLegal(senderMove)) {
                shouldHavePlayedConvention = true;
            }

        } else {
            conventionMove = responseMove;
            auto lastMove = env.getMove(env.getLastAction());
            if (lastMove == senderMove && state.MoveIsLegal(responseMove)) {
                shouldHavePlayedConvention = true;
            }
        }

        if (shouldHavePlayedConvention && conventionMove == move) {
            break;
        }
    }

    string conventionStr = "convention_" + conventionString();

    if (shouldHavePlayedConvention) {
        incrementStat(conventionStr + "_available");
    }

    if (conventionMove == move) {
        incrementStat(conventionStr + "_played");
        if (shouldHavePlayedConvention) {
            incrementStat(conventionStr + "_played_correct");
        } else {
            incrementStat(conventionStr + "_played_incorrect");
        }
    }
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
            return;
    }
 
    incrementStat(stat);
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


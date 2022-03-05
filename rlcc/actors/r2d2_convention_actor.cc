#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "rlcc/actors/r2d2_convention_actor.h"

using namespace std;

void R2D2ConventionActor::changeStateForBeliefSampler(hle::HanabiState& state) {
    // Check if convention applies, and belief needs to be changed.
    if (!conventionApplies(state)) {
        return
    }

    // Get list of elligble cards
    auto elligibleCards = getElligibleConventionCards(state);

    // Select a card to put in belief
    vector<hle::HanabiCard> chosenCard;
    sample(
        elligibleCards.begin(),
        elligibleCards.end(),
        back_inserter(chosenCard),
        1,
        mt19937{random_device{}()}
    );

    // Replace Card in Hand

    // Update Card new card knowledge
    const auto& hand = state.Hands()[playerIdx_];
    cout << hand.ToString() << endl;
    auto& deck = state.Deck();
    deck.PutCardsBack(hand.Cards());
}

bool R2D2ConventionActor::conventionApplies(hle::HanabiState& state) {
    return false;
}

vector<hle::HanabCard> R2D2ConventionActor::getElligibleConventionCards(
        hle::HanabiState& state) {
    vector<hle::HanabCard> elligibleCards;
    return elligibleCards;
}

// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <stdio.h>
#include <iostream>

#include "rela/thread_loop.h"
#include "rlcc/actors/actor.h"

#define PR false
#define ST false

class HanabiThreadLoop : public rela::ThreadLoop {
    public:
        HanabiThreadLoop(
                std::vector<std::shared_ptr<HanabiEnv>> envs,
                std::vector<std::vector<std::shared_ptr<Actor>>> actors,
                bool eval)
            : envs_(std::move(envs))
              , actors_(std::move(actors))
              , done_(envs_.size(), -1)
              , eval_(eval) {
                  assert(envs_.size() == actors_.size());
              }

        virtual void mainLoop() override {
            while (!terminated()) {
                if(PR)printf("\n=======================================\n");

                // go over each envs in sequential order
                // call in seperate for-loops to maximize parallization
                for (size_t i = 0; i < envs_.size(); ++i) {
                    if (done_[i] == 1) {
                        continue;
                    }

                    auto& actors = actors_[i];

                    if (envs_[i]->terminated()) {
                        // we only run 1 game for evaluation
                        if (eval_) {
                            ++done_[i];
                            if (done_[i] == 1) {
                                numDone_ += 1;
                                if (numDone_ == (int)envs_.size()) {
                                    return;
                                }
                            }
                        }

                        envs_[i]->reset();
                        for (size_t j = 0; j < actors.size(); ++j) {
                            if(PR)printf("\n[player %ld resetting]\n", j);
                            actors[j]->reset(*envs_[i]);
                        }
                    }
                }

                if(ST)printf("\nScore: %d\n", envs_[0]->getScore());
                if(ST)printf("Lives: %d\n", envs_[0]->getLife());
                if(ST)printf("Information: %d\n", envs_[0]->getInfo());
                auto deck = envs_[0]->getHleState().Deck();
                if(ST)printf("Deck: %d\n", deck.Size());
                std::string colours = "RYGWB";
                auto fireworks = envs_[0]->getFireworks();
                if(ST)printf("Fireworks: ");
                for (unsigned long i = 0; i < colours.size(); i++)
                    if(ST)printf("%c%d ", colours[i], fireworks[i]);
                if(ST)printf("\n");
                auto hands = envs_[0]->getHleState().Hands();
                int cp = envs_[0]->getCurrentPlayer();
                for(unsigned long i = 0; i < hands.size(); i++) {
                    if(ST)printf("Actor %ld hand:%s\n", i,
                        cp == (int)i ? " <-- current player" : ""); 
                    auto hand = hands[i].ToString();
                    hand.pop_back();
                    if(ST)printf("%s\n", hand.c_str());
                }

                if(PR)printf("\n----\n");

                // go over each envs in sequential order
                // call in seperate for-loops to maximize parallization
                for (size_t i = 0; i < envs_.size(); ++i) {
                    if (done_[i] == 1) {
                        continue;
                    }

                    auto& actors = actors_[i];
                    int curPlayer = envs_[i]->getCurrentPlayer();
                    for (size_t j = 0; j < actors.size(); ++j) {
                        if(PR)
                            printf("\n[player %ld observe before acting]%s\n", j,
                            curPlayer == (int)j ? " <-- current player" : "");
                        actors[j]->observeBeforeAct(*envs_[i]);
                    }
                }
                if(PR)printf("\n----\n");

                for (size_t i = 0; i < envs_.size(); ++i) {
                    if (done_[i] == 1) {
                        continue;
                    }

                    auto& actors = actors_[i];
                    int curPlayer = envs_[i]->getCurrentPlayer();
                    for (size_t j = 0; j < actors.size(); ++j) {
                        if(PR)printf("\n[player %ld acting]%s\n", j, 
                                curPlayer == (int)j ? " <-- current player" : "");
                        actors[j]->act(*envs_[i], curPlayer);
                    }
                }
                if(PR)printf("\n----\n");

                //if(ST)printf("\nScore: %d\n", envs_[0]->getScore());
                //if(ST)printf("Lives: %d\n", envs_[0]->getLife());
                //if(ST)printf("Information: %d\n", envs_[0]->getInfo());
                //deck = envs_[0]->getHleState().Deck();
                //if(ST)printf("Deck: %d\n", deck.Size());
                //colours = "RYGWB";
                //fireworks = envs_[0]->getFireworks();
                //if(ST)printf("Fireworks: ");
                //for (unsigned long i = 0; i < colours.size(); i++)
                    //if(ST)printf("%c%d ", colours[i], fireworks[i]);
                //if(ST)printf("\n");
                //hands = envs_[0]->getHleState().Hands();
                //cp = envs_[0]->getCurrentPlayer();
                //for(unsigned long i = 0; i < hands.size(); i++) {
                    //if(ST)printf("Actor %ld hand:%s\n", i,
                        //cp == (int)i ? " <-- current player" : ""); 
                    //auto hand = hands[i].ToString();
                    //hand.pop_back();
                    //if(ST)printf("%s\n", hand.c_str());
                //}
                //if(PR)printf("\n----\n");

                for (size_t i = 0; i < envs_.size(); ++i) {
                    if (done_[i] == 1) {
                        continue;
                    }

                    auto& actors = actors_[i];
                    int curPlayer = envs_[i]->getCurrentPlayer();
                    for (size_t j = 0; j < actors.size(); ++j) {
                        if(PR)printf("\n[player %ld fictious acting]%s\n", j,
                                curPlayer == (int)j ? " <-- current player" : "");
                        actors[j]->fictAct(*envs_[i]);
                    }
                }
                if(PR)printf("\n----\n");

                for (size_t i = 0; i < envs_.size(); ++i) {
                    if (done_[i] == 1) {
                        continue;
                    }

                    auto& actors = actors_[i];
                    int curPlayer = envs_[i]->getCurrentPlayer();
                    for (size_t j = 0; j < actors.size(); ++j) {
                        if(PR)printf("\n[player %ld observe after acting]%s\n", j,
                                curPlayer == (int)j ? " <-- current player" : "");
                        actors[j]->observeAfterAct(*envs_[i]);
                    }
                }
            }
        }

    private:
        std::vector<std::shared_ptr<HanabiEnv>> envs_;
        std::vector<std::vector<std::shared_ptr<Actor>>> actors_;
        std::vector<int8_t> done_;
        const bool eval_;
        int numDone_ = 0;
};

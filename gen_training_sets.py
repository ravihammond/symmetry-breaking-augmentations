import argparse
import random
import json
import pprint
pprint = pprint.pprint
import copy


def generate_train_test(args):
    all_models = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    all_splits = []

    train_sets = set()

    for i in range(args.num_splits):
        train_test = {}

        while True:
            random.shuffle(all_models)
            train_set = all_models[:6]
            test_set = all_models[6:]
            train_set.sort()
            test_set.sort()
            train_set_key = '-'.join(str(x) for x in train_set)
            if train_set_key not in train_sets:
                break
                
        train_test["train"] = train_set
        train_test["test"] = test_set
        all_splits.append(train_test)

        train_sets.add(train_set_key)

    with open(args.output, "w") as outfile:
        json.dump(all_splits, outfile, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_splits", type=int, default=100)

    args = parser.parse_args()
    generate_train_test(args)

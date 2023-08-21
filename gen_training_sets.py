import argparse
import random
import json
import pprint
pprint = pprint.pprint
import copy


def generate_train_test(args):
    all_models = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    # all_models = [0,1,2,3,4,5,6,7,8,9,10,11]
    all_splits = []

    train_sets = set()
    test_occurances = {}

    for i in range(args.num_splits):
        train_test = {}

        while True:
            random.shuffle(all_models)
            train_set = all_models[:5]
            test_set = all_models[5:]
            train_set.sort()
            test_set.sort()
            train_set_key = '-'.join(str(x) for x in train_set)

            if train_set_key not in train_sets:
                if args.max_test_occurances != -1 and not \
                    test_count_okay(test_set, test_occurances, args.max_test_occurances):
                    continue
                break
                
        train_test["train"] = train_set
        train_test["test"] = test_set
        all_splits.append(train_test)

        train_sets.add(train_set_key)

    pprint(all_splits)

    with open(args.output, "w") as outfile:
        json.dump(all_splits, outfile, indent=4)


def test_count_okay(test_set, test_occurances, max_occurances):
    all_test_okay = True

    for test in test_set:
        if test not in test_occurances.keys():
            continue

        count = test_occurances[test]
        if count < max_occurances:
            continue

        all_test_okay = False

    if all_test_okay:
        for test in test_set:
            if test not in test_occurances.keys():
                test_occurances[test] = 1
                continue
            test_occurances[test] += 1

    return all_test_okay


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_splits", type=int, default=100)
    parser.add_argument("--max_test_occurances", type=int, default=-1)

    args = parser.parse_args()
    generate_train_test(args)

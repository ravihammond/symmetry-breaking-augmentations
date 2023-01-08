import argparse

if __name__ "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_test_splits", type=str, default="None")

    parser.add_argument("--act_policy", type=str, required=True)
    parser.add_argument("--comp_policies", type=str, default="None")
    parser.add_argument("--act_sad_legacy", type=int, default=0)
    parser.add_argument("--comp_sad_legacy", type=str, default="0")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--comp_names", type=str, default="None")
    parser.add_argument("--outdir", type=str, default="similarity_data")
    args = parser.parse_args()

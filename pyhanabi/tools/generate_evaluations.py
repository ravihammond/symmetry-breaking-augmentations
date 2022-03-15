import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--teammate", type=str, default="None")
    args = parser.parse_args()

    collect_data(args)

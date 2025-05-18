import argparse
from Inference.video_infer import run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["inference"])
    args = parser.parse_args()

    if args.mode == "inference":
        run_inference()

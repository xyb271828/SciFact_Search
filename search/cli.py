"""Command line entry points for repeatable experiments."""
import argparse

from search.experiment import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    bench = subs.add_parser("benchmark")
    bench.add_argument("--dataset", choices=("scifact", "lessons"), required=True)
    bench.add_argument("--data-dir", default=".cache")
    bench.add_argument("--output", required=True)
    bench.add_argument("--threads", type=int, default=2)
    bench.add_argument("--lexical-only", action="store_true")
    mslr = subs.add_parser("mslr")
    mslr.add_argument("--fold", required=True)
    mslr.add_argument("--output", required=True)
    mslr.add_argument("--threads", type=int, default=2)
    mslr.add_argument("--max-train-queries", type=int)
    args = parser.parse_args()
    if args.command == "benchmark":
        benchmark(args.dataset, args.data_dir, args.output, args.threads, args.lexical_only)
    else:
        from search.mslr import run_mslr
        run_mslr(args.fold, args.output, args.threads, args.max_train_queries)


if __name__ == "__main__":
    main()

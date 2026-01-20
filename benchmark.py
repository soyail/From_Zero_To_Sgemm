#!/usr/bin/env python3
import argparse
import csv
from typing import Dict, List, Tuple


def parse_k_list(k_list: str) -> List[int]:
    parts = [p.strip() for p in k_list.split(",") if p.strip()]
    return [int(p) for p in parts]


def plot_results(
    ks: List[int],
    series: List[Tuple[str, List[float]]],
    out_path: str,
    title: str,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for name, gflops in series:
        plt.plot(ks, gflops, marker="o", label=name)
    plt.xlabel("k")
    plt.ylabel("GFLOPS")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GEMM implementations.")
    parser.add_argument("--impl", required=True, help="GEMM implementation name")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument(
        "--k-list",
        default="256,512,1024,2048,4096",
        help="Comma-separated k sizes",
    )
    parser.add_argument("--out", default="benchmark.png", help="Output plot path")
    parser.add_argument("--csv", required=True, help="Path to CSV from bench_gemm")
    args = parser.parse_args()

    ks = parse_k_list(args.k_list)
    impl_name = args.impl

    impl_gflops = {k: None for k in ks}
    cublas_gflops = {k: None for k in ks}

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["m"]) != args.m or int(row["n"]) != args.n:
                continue
            k = int(row["k"])
            if k not in impl_gflops:
                continue
            if row["impl"] == impl_name:
                impl_gflops[k] = float(row["gflops"])
            if row["impl"] == "gpu_cublas":
                cublas_gflops[k] = float(row["gflops"])

    missing_impl = [k for k, v in impl_gflops.items() if v is None]
    missing_cublas = [k for k, v in cublas_gflops.items() if v is None]
    if missing_impl:
        raise RuntimeError("Missing {} data for k: {}".format(impl_name, missing_impl))
    if missing_cublas:
        raise RuntimeError("Missing gpu_cublas data for k: {}".format(missing_cublas))

    impl_series = [impl_gflops[k] for k in ks]
    cublas_series = [cublas_gflops[k] for k in ks]

    series = [(impl_name, impl_series), ("gpu_cublas", cublas_series)]
    title = "GEMM Benchmark (m=n={}, k varies)".format(args.m)
    plot_results(ks, series, args.out, title)
    print("Saved plot to {}".format(args.out))


if __name__ == "__main__":
    main()

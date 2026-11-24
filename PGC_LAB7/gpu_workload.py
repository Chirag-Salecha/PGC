
import argparse
import math
import time
import torch

# --------------------------------------------------
# Create GPU weights
# --------------------------------------------------

def make_weights(n):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(n, n, generator=g) / math.sqrt(n)).cuda()


# --------------------------------------------------
# Create input batches
# --------------------------------------------------

def make_host_batches(rows, cols, pinned, unique=4):
    g = torch.Generator().manual_seed(0)
    batches = []

    for _ in range(unique):
        t = torch.randn(rows, cols, generator=g)
        batches.append(t.pin_memory() if pinned else t)

    return batches


# --------------------------------------------------
# GPU computation
# --------------------------------------------------

def compute(x, w, reps):
    y = x

    for _ in range(reps):
        y = y @ w
        y = torch.relu(y)
        y = y.mul_(0.5).add_(1.0)

    return y.sum(dim=1)


# --------------------------------------------------
# BASELINE: Sequential execution
# --------------------------------------------------

def run_baseline(host, w, reps, nbatches):
    results = []

    for i in range(nbatches):
        x = host[i % len(host)].cuda()
        r = compute(x, w, reps)
        results.append(r.cpu())

    torch.cuda.synchronize()
    return torch.stack(results)


# --------------------------------------------------
# OPTIMIZED: Pinned memory + two CUDA streams
# --------------------------------------------------

def run_optimized(host, w, reps, nbatches):

    rows, cols = host[0].shape

    copy_s = torch.cuda.Stream()
    comp_s = torch.cuda.Stream()

    buf = [
        torch.empty(rows, cols, device="cuda")
        for _ in range(2)
    ]

    copied = [torch.cuda.Event() for _ in range(2)]
    freed = [torch.cuda.Event() for _ in range(2)]

    out_host = torch.empty(
        nbatches, rows
    ).pin_memory()

    torch.cuda.synchronize()

    for event in freed:
        event.record(comp_s)

    def issue_copy(i):
        k = i % 2

        copy_s.wait_event(freed[k])

        with torch.cuda.stream(copy_s):
            buf[k].copy_(
                host[i % len(host)],
                non_blocking=True
            )
            copied[k].record(copy_s)

    issue_copy(0)

    for i in range(nbatches):
        k = i % 2

        if i + 1 < nbatches:
            issue_copy(i + 1)

        comp_s.wait_event(copied[k])

        with torch.cuda.stream(comp_s):
            r = compute(buf[k], w, reps)

            freed[k].record(comp_s)

            out_host[i].copy_(
                r,
                non_blocking=True
            )

    torch.cuda.synchronize()

    return out_host


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        required=True
    )

    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=3)

    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-batches",
        type=int,
        default=10
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not available. Check GPU runtime."
        )

    pinned = args.mode == "optimized"

    fn = run_optimized if pinned else run_baseline

    host = make_host_batches(
        args.rows, args.cols, pinned
    )

    w = make_weights(args.cols)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Mode:", args.mode)

    print(
        "Batch size:",
        round(args.rows * args.cols * 4 / 1e6, 2),
        "MB"
    )

    # Warm-up
    fn(host, w, args.reps, 3)
    torch.cuda.synchronize()

    # Measure execution time
    t0 = time.perf_counter()

    res = fn(
        host, w, args.reps, args.batches
    )

    torch.cuda.synchronize()

    dt = time.perf_counter() - t0

    print("Batches:", args.batches)
    print("Total time:", round(dt, 3), "s")
    print(
        "Per batch:",
        round(dt / args.batches * 1000, 2),
        "ms"
    )
    print(
        "Throughput:",
        round(args.batches / dt, 2),
        "batches/s"
    )
    print(
        "Checksum:",
        f"{res.double().sum().item():.6e}"
    )

    # GPU profiling
    if args.profile:

        from torch.profiler import (
            profile,
            ProfilerActivity
        )

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ]
        ) as prof:

            fn(
                host,
                w,
                args.reps,
                args.profile_batches
            )

            torch.cuda.synchronize()

        filename = f"trace_{args.mode}.json"

        prof.export_chrome_trace(filename)

        print("Trace saved:", filename)

        for key in (
            "cuda_time_total",
            "device_time_total"
        ):
            try:
                print(
                    prof.key_averages().table(
                        sort_by=key,
                        row_limit=10
                    )
                )
                break
            except Exception:
                continue


if __name__ == "__main__":
    main()

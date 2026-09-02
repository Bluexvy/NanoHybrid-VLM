from nanovllm.engine.hybrid_cuda_graph import (
    HybridDecodeGraphPolicy,
)


def main() -> None:
    policy = HybridDecodeGraphPolicy(
        batch_sizes=(1, 2, 4),
        max_num_seqs=8,
    )

    # Prefill 一律使用 Eager。
    route = policy.route(
        batch_size=1,
        is_prefill=True,
        enforce_eager=False,
    )

    assert not route.use_graph
    assert route.bucket_size is None
    assert route.reason == "prefill_uses_eager"

    # 显式开启 Eager 时，即使 bucket 存在，
    # 也不能使用 Graph。
    route = policy.route(
        batch_size=1,
        is_prefill=False,
        enforce_eager=True,
    )

    assert not route.use_graph
    assert route.bucket_size is None
    assert route.reason == "enforce_eager"

    # 精确命中 B=1。
    route = policy.route(
        batch_size=1,
        is_prefill=False,
        enforce_eager=False,
    )

    assert route.use_graph
    assert route.batch_size == 1
    assert route.bucket_size == 1
    assert route.reason == "exact_graph_bucket"

    # 精确命中 B=2。
    route = policy.route(
        batch_size=2,
        is_prefill=False,
        enforce_eager=False,
    )

    assert route.use_graph
    assert route.bucket_size == 2

    # B=3 没有对应 Graph，安全回退 Eager。
    route = policy.route(
        batch_size=3,
        is_prefill=False,
        enforce_eager=False,
    )

    assert not route.use_graph
    assert route.bucket_size is None
    assert route.reason == "unsupported_batch_size"

    # 精确命中 B=4。
    route = policy.route(
        batch_size=4,
        is_prefill=False,
        enforce_eager=False,
    )

    assert route.use_graph
    assert route.bucket_size == 4

    # 非法 batch size。
    try:
        policy.route(
            batch_size=0,
            is_prefill=False,
            enforce_eager=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "batch_size=0 should fail"
        )

    # 非法 bucket：重复且不是严格递增。
    try:
        HybridDecodeGraphPolicy(
            batch_sizes=(1, 2, 2, 4),
            max_num_seqs=8,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Duplicated buckets should fail"
        )

    # bucket 超过调度器最大并发数。
    try:
        HybridDecodeGraphPolicy(
            batch_sizes=(1, 2, 16),
            max_num_seqs=8,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Bucket larger than max_num_seqs "
            "should fail"
        )

    print(
        "Graph buckets:",
        policy.batch_sizes,
    )

    print(
        "Part 3 passed: Hybrid Decode uses "
        "exact CUDA Graph buckets and safely "
        "falls back to Eager."
    )


if __name__ == "__main__":
    main()
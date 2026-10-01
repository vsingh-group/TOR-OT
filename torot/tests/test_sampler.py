import numpy as np

from torot.data.dataset import SizeBucketBatchSampler


def test_token_budget_and_batch_size_caps():
    rng = np.random.default_rng(0)
    sizes = rng.integers(10, 200, size=5000)             # n + m per instance
    sampler = SizeBucketBatchSampler(sizes, batch_size=64, max_tokens=8192, shuffle=True, seed=1)
    seen = []
    for batch in sampler:
        assert 1 <= len(batch) <= 64
        est = 6.5 * sizes[batch] + 2
        assert len(batch) * est.max() <= 8192
        seen += batch
    assert sorted(seen) == list(range(len(sizes)))       # every instance exactly once


def test_ddp_ranks_partition_batches():
    sizes = np.full(1000, 50)
    r0 = list(SizeBucketBatchSampler(sizes, 32, shuffle=False, rank=0, world_size=2))
    r1 = list(SizeBucketBatchSampler(sizes, 32, shuffle=False, rank=1, world_size=2))
    assert len(r0) == len(r1)
    assert not set(map(tuple, r0)) & set(map(tuple, r1))

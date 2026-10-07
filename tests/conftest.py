import pytest

from eeg_rx.features import extract
from eeg_rx.preprocess import preprocess
from eeg_rx.synthetic import SynthSpec, generate


def small(effect, seed=3):
    return SynthSpec(n_responders=8, n_nonresponders=8, seconds=60.0, effect=effect, seed=seed)


@pytest.fixture(scope="session")
def recs():
    return generate(small(0.35))


@pytest.fixture(scope="session")
def fs(recs):
    return extract(preprocess(recs), source="synthetic")


@pytest.fixture(scope="session")
def fs_null():
    return extract(preprocess(generate(small(0.0, seed=5))), source="synthetic")

"""Tests for KNeighborsTimeSeriesClassifier."""

import numpy as np
import pandas as pd
import pytest

from sktime.alignment.dtw_python import AlignerDTW
from sktime.classification.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesClassifier,
)
from sktime.datasets import load_unit_test
from sktime.dists_kernels import AggrDist, ScipyDist
from sktime.tests.test_switch import run_test_for_class

distance_functions = [
    "euclidean",
    "dtw",
    "wdtw",
    "msm",
    "erp",
    "lcss",
    "edr",
]

# expected correct on test set using default parameters.
expected_correct = {
    "euclidean": 19,
    "dtw": 21,
    "wdtw": 21,
    "msm": 20,
    "erp": 19,
    "lcss": 12,
    "edr": 20,
}

# expected correct on test set using window params.
expected_correct_window = {
    "euclidean": 19,
    "dtw": 21,
    "wdtw": 21,
    "msm": 10,
    "erp": 19,
    "edr": 20,
    "lcss": 12,
}


@pytest.mark.skipif(
    not run_test_for_class(KNeighborsTimeSeriesClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_on_unit_test(distance_key):
    """Test function for elastic knn, to be reinstated soon."""
    # load arrowhead data for unit tests
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    knn = KNeighborsTimeSeriesClassifier(
        distance=distance_key,
    )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    correct = 0
    for j in range(0, len(pred)):
        if pred[j] == y_test[j]:
            correct = correct + 1
    assert correct == expected_correct[distance_key]


@pytest.mark.skipif(
    not run_test_for_class(KNeighborsTimeSeriesClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_bounding_matrix(distance_key):
    """Test knn with custom bounding parameters."""
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    knn = KNeighborsTimeSeriesClassifier(
        distance=distance_key, distance_params={"window": 0.5}
    )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    correct = 0
    for j in range(0, len(pred)):
        if pred[j] == y_test[j]:
            correct = correct + 1
    assert correct == expected_correct_window[distance_key]


@pytest.mark.skipif(
    not run_test_for_class([KNeighborsTimeSeriesClassifier, AlignerDTW]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_knn_with_aligner():
    """Tests KNN classifier with alignment distance on unequal length data."""
    from sktime.dists_kernels.compose_from_align import DistFromAligner
    from sktime.utils._testing.hierarchical import _make_hierarchical

    X = _make_hierarchical((3,), min_timepoints=5, max_timepoints=10, random_state=0)
    y = np.array([0, 1, 1])

    dtw_dist = DistFromAligner(AlignerDTW())
    clf = KNeighborsTimeSeriesClassifier(distance=dtw_dist)

    clf.fit(X, y)


@pytest.mark.skipif(
    not run_test_for_class([KNeighborsTimeSeriesClassifier, AggrDist, ScipyDist]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_knn_with_aggrdistance():
    """Tests KNN classifier with alignment distance on unequal length data."""
    from sktime.utils._testing.hierarchical import _make_hierarchical

    X = _make_hierarchical((3,), min_timepoints=5, max_timepoints=10, random_state=0)
    y = np.array([0, 1, 1])

    eucl_dist = ScipyDist()
    aggr_dist = AggrDist(eucl_dist)
    clf = KNeighborsTimeSeriesClassifier(distance=aggr_dist)

    clf.fit(X, y)


@pytest.mark.skipif(
    not run_test_for_class(KNeighborsTimeSeriesClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_knn_kneighbors():
    """Tests kneighbors method and absence of bug #3798."""
    from sktime.utils._testing.hierarchical import _make_hierarchical

    Xtrain = _make_hierarchical(hierarchy_levels=(3,), n_columns=3)
    Xtest = _make_hierarchical(hierarchy_levels=(5,), n_columns=3)

    ytrain = pd.Series(["label_1", "label_2", "label_3"])

    kntsc = KNeighborsTimeSeriesClassifier(n_neighbors=1)
    kntsc.fit(Xtrain, ytrain)

    ret = kntsc.kneighbors(Xtest)
    assert isinstance(ret, tuple)
    assert len(ret) == 2

    dist, ind = ret
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (5, 1)
    assert isinstance(ind, np.ndarray)
    assert ind.shape == (5, 1)
    def test_knn_params_and_single_sample_consistency():
    """
    Test KNN consistency: parameter setting and single sample prediction.
    This ensures the model handles individual samples without crashing.
    """
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    from sktime.datasets import load_unit_test
    import numpy as np

    # 1. Parameter Consistency Check
    # Testing if the model correctly retains different k values
    for k in [1, 5]:
        clf = KNeighborsTimeSeriesClassifier(n_neighbors=k)
        assert clf.n_neighbors == k, f"Failed to set n_neighbors to {k}"

    # 2. Single Sample Prediction Check
    # Loading small dataset for speed
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, _ = load_unit_test(split="test", return_X_y=True)
    
    # Init and Fit
    clf_test = KNeighborsTimeSeriesClassifier(n_neighbors=3)
    clf_test.fit(X_train, y_train)
    
    # Predicting on a single sample (common edge case)
    # Using iloc[[0]] to keep the DataFrame structure for sktime compatibility
    single_sample = X_test.iloc[[0]]
    y_pred = clf_test.predict(single_sample)
    
    # 3. Assertions (The Proof)
    assert isinstance(y_pred, np.ndarray), "Prediction should be a numpy array"
    assert len(y_pred) == 1, "Should predict exactly 1 sample"

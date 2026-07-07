"""
Anomaly detection.

Implements Stage 2 of the ML pipeline described in docs/ml-pipeline.md:
unsupervised anomaly detection against a learned baseline from an initial
calibration period, run on the joint acoustic+environmental feature vectors
produced by simulation/pipeline/feature_extraction.py. Isolation Forest is
the concrete algorithm used here, one of the two candidates named in
DECISIONS.md (the other being autoencoder reconstruction error).
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class BaselineAnomalyDetector:
    """
    Unsupervised anomaly detector wrapping sklearn's IsolationForest.

    Why unsupervised fits the cold-start problem (docs/ml-pipeline.md Stage 2
    and Stage 3 gating):

    At first deployment there is no labeled anomaly data at all -- no field
    recordings exist yet to say "this window was a real anomaly, that one
    wasn't" (see DECISIONS.md: no hardware purchased, no implementation
    code, planning phase). A supervised classifier has nothing to train on
    at this point. Isolation Forest instead only needs unlabeled feature
    vectors from a period assumed to represent normal site conditions (the
    "calibration period" in docs/ml-pipeline.md) -- it isolates points that
    are structurally easy to separate from the rest of the calibration data
    via random recursive partitioning, and scores them as more anomalous the
    fewer partitions it takes to isolate them. This requires no notion of
    "anomaly class" at all, so it works identically whether an eventual
    anomaly turns out to be a vessel passage, a storm runoff event, a sensor
    fault, or something never seen during design -- exactly the property
    needed before Stage 3's labeled/reviewed dataset exists.

    This class is a thin wrapper providing two operations matching the
    pipeline's two distinct phases: `fit()` during the (offline) calibration
    period, `score()` on each new window during (near-real-time, on-device)
    normal operation.
    """

    def __init__(
        self,
        contamination: Union[str, float] = "auto",
        random_state: Optional[int] = None,
        threshold_sigma: float = 3.0,
        **isolation_forest_kwargs,
    ):
        """
        Args:
            contamination: expected proportion of anomalies in the fitted
                calibration data, passed through to IsolationForest. Default
                "auto" is appropriate here since the calibration period is
                assumed to be normal conditions -- there's no reason to
                expect a specific non-zero anomaly proportion in it.
            random_state: seed for IsolationForest's internal randomness,
                for reproducible fitting.
            threshold_sigma: is_anomaly flags a window when its anomaly_score
                exceeds (calibration mean + threshold_sigma * calibration
                std), both computed by scoring the calibration set itself
                once fitting completes -- not IsolationForest's own
                contamination-based predict() cutoff. That built-in cutoff
                sits close to the calibration set's own score mean (it
                expects contamination's fraction of the *calibration* data
                to already be outliers), which is a poor fit here since the
                calibration period is assumed 100% normal by construction:
                empirically (simulation/scripts/evaluate.py runs) it flagged
                over a quarter of genuinely normal evaluation windows,
                crushing per-type precision. 5 sigma keeps recall on actual
                events (their scores sit far outside the calibration std)
                while requiring a much larger deviation than ordinary
                calibration-period noise before flagging.
            **isolation_forest_kwargs: any other sklearn IsolationForest
                constructor arguments (e.g. n_estimators), passed through.
        """
        self._model = IsolationForest(
            contamination=contamination, random_state=random_state, **isolation_forest_kwargs
        )
        self._threshold_sigma = threshold_sigma
        self._threshold = None
        self._feature_names = None
        self._fitted = False

    def fit(self, feature_vectors: Union[pd.DataFrame, Sequence[pd.Series]]) -> "BaselineAnomalyDetector":
        """
        Establish the calibration baseline from a set of assumed-normal
        joint feature vectors.

        Args:
            feature_vectors: a DataFrame (one row per window) or a sequence
                of per-window pd.Series (e.g. from
                feature_extraction.build_joint_feature_vector()), all drawn
                from the initial calibration period. Column/index names are
                remembered so score() can validate and align later vectors
                against the same feature order.

        Returns:
            self, so fit() can be chained with construction.
        """
        df = (
            feature_vectors
            if isinstance(feature_vectors, pd.DataFrame)
            else pd.DataFrame(list(feature_vectors))
        )
        self._feature_names = list(df.columns)
        self._model.fit(df.values)
        self._fitted = True

        # is_anomaly's cutoff is derived from how the *fitted* model scores
        # the calibration data it was just fit on (see threshold_sigma
        # above), not from IsolationForest's own predict().
        calibration_scores = -self._model.decision_function(df.values)
        self._threshold = calibration_scores.mean() + self._threshold_sigma * calibration_scores.std()

        return self

    def score(self, feature_vector: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Score one window's joint feature vector against the fitted baseline.

        Args:
            feature_vector: a single window's joint feature vector, as a
                pd.Series (aligned by name against the fitted feature
                order) or a plain array already in that same feature order.

        Returns:
            dict with:
                "anomaly_score": float, higher means more anomalous.
                    (IsolationForest's own decision_function returns higher
                    values for more *normal* points; this is inverted here
                    so the sign convention matches docs/data-pipeline.md's
                    `anomaly_flags.anomaly_score` column, where higher
                    should read as "more anomalous".)
                "is_anomaly": bool, True if anomaly_score exceeds the
                    calibration-derived threshold (see threshold_sigma in
                    __init__ / fit()) -- not IsolationForest's own predict().
        """
        if not self._fitted:
            raise RuntimeError(
                "BaselineAnomalyDetector.score() called before fit() -- "
                "no calibration baseline established yet"
            )

        if isinstance(feature_vector, pd.Series):
            vector = feature_vector.reindex(self._feature_names).values
        else:
            vector = np.asarray(feature_vector)

        x = vector.reshape(1, -1)
        raw_normality_score = self._model.decision_function(x)[0]
        anomaly_score = -float(raw_normality_score)
        is_anomaly = anomaly_score > self._threshold

        return {"anomaly_score": anomaly_score, "is_anomaly": is_anomaly}

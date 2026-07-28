"""
Regression tests for statistical correctness in moseq2_viz.model.stat.

These cover bugs where the reported p-values were wrong rather than merely
imprecise, so they are worth pinning explicitly.
"""
import numpy as np
import pandas as pd
from scipy import stats
from unittest import TestCase
from statsmodels.stats.multitest import multipletests

from moseq2_viz.model.stat import (
    get_session_mean_df,
    mann_whitney,
    COUNT_STATISTICS,
)


def _make_stats_df(groups, n_sessions=4, max_syllable=4, statistic="usage"):
    """Build a minimal per-session syllable statistics dataframe."""
    rows = []
    for group, base in groups.items():
        for s in range(n_sessions):
            for syllable in range(max_syllable):
                rows.append(
                    {
                        "group": group,
                        "uuid": f"{group}{s}",
                        "syllable": syllable,
                        statistic: base + 0.01 * syllable,
                    }
                )
    return pd.DataFrame(rows)


def _pvalues(result):
    """mann_whitney returns either a df or a (df, ...) tuple depending on kwargs."""
    df = result[0] if isinstance(result, tuple) else result
    return df["pvalue"].values


class TestMannWhitneyGroupSelection(TestCase):
    def test_ignores_groups_not_being_compared(self):
        # A third group present in the dataframe must not leak into a pairwise
        # comparison. The old implementation pivoted every group and split the
        # rows positionally, so 'other' contaminated the ctrl-vs-mut test.
        df = _make_stats_df({"ctrl": 0.10, "mut": 0.20, "other": 0.50}, max_syllable=4)

        observed = _pvalues(mann_whitney(df, "ctrl", "mut", max_syllable=4))

        pivot = get_session_mean_df(df, "usage", 4)
        a, b = pivot.loc["ctrl"].values, pivot.loc["mut"].values
        expected = [
            stats.mannwhitneyu(a[:, i], b[:, i], alternative="two-sided").pvalue
            for i in range(4)
        ]

        np.testing.assert_allclose(observed, expected)

    def test_group_order_does_not_change_pvalues(self):
        # Two-sided test: swapping the group arguments must not change the result.
        # The old positional split used group sizes in argument order against a
        # pivot sorted alphabetically, so reversing the arguments changed answers.
        df = _make_stats_df({"ctrl": 0.10, "mut": 0.20}, max_syllable=4)

        forward = _pvalues(mann_whitney(df, "ctrl", "mut", max_syllable=4))
        reverse = _pvalues(mann_whitney(df, "mut", "ctrl", max_syllable=4))

        np.testing.assert_allclose(forward, reverse)


class TestMissingObservationHandling(TestCase):
    def test_count_statistics_fill_zero_but_scalars_stay_nan(self):
        # A session that never performs a syllable used it zero times (usage=0 is
        # a real observation), but has NO height measurement for it -- filling
        # that with 0 mm would invent a physically impossible data point.
        df = _make_stats_df({"ctrl": 0.10, "mut": 0.20}, max_syllable=4)
        df["height_ave_mm_mean"] = 50.0
        # ctrl never expresses syllable 3
        df = df[~((df.group == "ctrl") & (df.syllable == 3))]

        usage = get_session_mean_df(df, "usage", 4)
        height = get_session_mean_df(df, "height_ave_mm_mean", 4)

        assert "usage" in COUNT_STATISTICS
        assert (usage.loc["ctrl"][3] == 0).all()
        assert height.loc["ctrl"][3].isna().all()

    def test_missing_observations_do_not_drag_the_test_toward_zero(self):
        # Both groups have the same true height; they differ only in how often
        # they express the syllable. Treating "never expressed" as 0 mm creates a
        # large fake difference, so the test must not report significance.
        rng = np.random.RandomState(0)
        rows = []
        for group, n_expressed in (("ctrl", 4), ("mut", 10)):
            for s in range(10):
                # every session expresses syllable 1, so every session appears in
                # the pivot; only syllable 0 differs in how often it is performed
                rows.append(
                    {
                        "group": group,
                        "uuid": f"{group}{s}",
                        "syllable": 1,
                        "height_ave_mm_mean": rng.normal(50, 3),
                    }
                )
                if s < n_expressed:
                    rows.append(
                        {
                            "group": group,
                            "uuid": f"{group}{s}",
                            "syllable": 0,
                            "height_ave_mm_mean": rng.normal(50, 3),
                        }
                    )
        df = pd.DataFrame(rows)

        pivot = get_session_mean_df(df, "height_ave_mm_mean", 2)
        a = pivot.loc["ctrl"].values[:, 0]
        b = pivot.loc["mut"].values[:, 0]
        # sessions that never performed syllable 0 are present but unmeasured
        assert np.isnan(a).sum() == 6

        # complete-case comparison finds no difference (there is none)
        p_complete = stats.ttest_ind(
            a, b, equal_var=False, nan_policy="omit"
        ).pvalue
        # zero-filling the absent sessions manufactures one
        p_zero_filled = stats.ttest_ind(
            np.nan_to_num(a), np.nan_to_num(b), equal_var=False
        ).pvalue

        assert p_complete > 0.05
        assert p_zero_filled < 0.05


class TestMultipleComparisonAxis(TestCase):
    def test_correction_is_applied_across_syllables(self):
        # Dunn's p-values are indexed syllable x group-pair. Correcting along
        # axis=1 corrects across group pairs, which for a two-group experiment is
        # a single value per syllable and therefore no correction at all.
        pvals = pd.DataFrame({("ctrl", "mut"): np.linspace(0.001, 0.049, 20)})
        correct = lambda x: multipletests(x, alpha=0.05, method="fdr_bh")[1]

        across_pairs = pvals.apply(correct, axis=1, result_type="broadcast")
        across_syllables = pvals.apply(correct, axis=0, result_type="broadcast")

        # correcting across pairs leaves a 2-group experiment untouched
        np.testing.assert_allclose(across_pairs.values, pvals.values)
        # correcting across syllables actually raises the p-values (Benjamini-
        # Hochberg leaves the single largest p-value unchanged, hence >=)
        assert (across_syllables.values >= pvals.values).all()
        assert (across_syllables.values > pvals.values).sum() == len(pvals) - 1

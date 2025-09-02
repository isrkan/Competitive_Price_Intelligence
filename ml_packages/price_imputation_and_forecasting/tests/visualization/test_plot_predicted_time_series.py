import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from price_imputation_and_forecasting.visualization.plot_predicted_time_series import visualize_imputation_forecasting_results


class TestPlotPredictedTimeSeries(unittest.TestCase):
    """
    Extensive tests for visualize_imputation_forecasting_results.

    Coverage:
      - Returns a dict keyed by each row's index.
      - For each figure:
          * Is a Plotly Figure with exactly 3 traces.
          * Trace order: Forecasted (0), Imputed (1), Original (2).
          * Each trace's x/y exactly matches the corresponding series.
          * Styling applied: names, line styles/colors, marker config, layout.
      - Handles NaNs in original series (gaps in plot).
      - Works with MultiIndex.
      - Empty input returns empty dict.
      - Mismatched indices across inputs raises KeyError.
    """
    def setUp(self):
        # -----------------------------
        # Build deterministic toy data
        # -----------------------------
        self.hist_dates = pd.date_range("2024-01-01", periods=4, freq="D")
        self.future_dates = pd.date_range(self.hist_dates[-1] + pd.Timedelta(days=1), periods=3, freq="D")
        self.all_dates = self.hist_dates.append(self.future_dates)

        # Use a MultiIndex to ensure indexing works for tuples as well
        self.idx = pd.MultiIndex.from_tuples(
            [("StoreA", "Product1"), ("StoreB", "Product2")],
            names=["StoreID", "ProductDescription"]
        )

        # Original with NaNs (shape: 2 x 4)
        self.df_original = pd.DataFrame(
            [
                [1.0, np.nan, 3.0, 4.0],     # StoreA/Product1
                [2.0, 2.5, np.nan, 3.5],     # StoreB/Product2
            ],
            index=self.idx,
            columns=self.hist_dates
        )

        # Imputed (same shape as original, but filled)
        self.df_imputed = pd.DataFrame(
            [
                [1.0, 2.0, 3.0, 4.0],        # StoreA/Product1
                [2.0, 2.5, 3.0, 3.5],        # StoreB/Product2
            ],
            index=self.idx,
            columns=self.hist_dates
        )

        # Forecasted (hist + future, shape: 2 x 7)
        self.df_forecasted = pd.DataFrame(
            [
                [1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5],  # StoreA/Product1
                [2.0, 2.5, 3.0, 3.5, 3.7, 3.9, 4.1],  # StoreB/Product2
            ],
            index=self.idx,
            columns=self.all_dates
        )

    # -----------------------------
    # Utility helpers
    # -----------------------------
    @staticmethod
    def _to_timestamp_list(x_like):
        """Convert a Plotly x sequence (which can be datetime, strings, etc.) to pandas.Timestamp list."""
        return list(pd.to_datetime(list(x_like)))

    @staticmethod
    def _to_float_array(y_like):
        """Convert a Plotly y sequence to a NumPy float array (preserving NaNs)."""
        return np.array(list(y_like), dtype=float)

    # -----------------------------
    # Core functionality tests
    # -----------------------------
    def test_returns_figure_per_row_with_expected_keys(self):
        figs = visualize_imputation_forecasting_results(
            self.df_original, self.df_imputed, self.df_forecasted
        )

        # One figure per row in original
        self.assertEqual(len(figs), len(self.df_original.index))

        # Keys match the original index exactly (as a set to avoid order flakiness)
        self.assertEqual(set(figs.keys()), set(self.df_original.index))

        # Each value is a Plotly Figure with 3 traces
        for k, fig in figs.items():
            self.assertIsInstance(fig, go.Figure)
            self.assertEqual(len(fig.data), 3, "Figure must contain exactly 3 traces.")

    def test_trace_order_and_basic_styling(self):
        figs = visualize_imputation_forecasting_results(
            self.df_original, self.df_imputed, self.df_forecasted
        )

        for idx in self.df_original.index:
            fig = figs[idx]

            # Trace 0: Forecasted
            t0 = fig.data[0]
            self.assertIsInstance(t0, go.Scatter)
            self.assertEqual(t0.name, "Forecasted")
            # Styling
            self.assertEqual(getattr(t0.line, "dash", None), "dot")
            self.assertEqual(getattr(t0.line, "color", None), "firebrick")

            # Trace 1: Imputed
            t1 = fig.data[1]
            self.assertIsInstance(t1, go.Scatter)
            self.assertEqual(t1.name, "Imputed")
            self.assertEqual(getattr(t1.line, "dash", None), "solid")
            self.assertEqual(getattr(t1.line, "color", None), "blue")

            # Trace 2: Original
            t2 = fig.data[2]
            self.assertIsInstance(t2, go.Scatter)
            self.assertEqual(t2.name, "Original")
            self.assertEqual(getattr(t2.line, "color", None), "black")
            # mode should be markers+lines
            self.assertEqual(getattr(t2, "mode", None), "markers+lines")
            # marker styling
            self.assertEqual(getattr(t2.marker, "size", None), 6)
            self.assertEqual(getattr(t2.marker, "symbol", None), "circle")

    def test_x_y_alignment_with_series(self):
        figs = visualize_imputation_forecasting_results(
            self.df_original, self.df_imputed, self.df_forecasted
        )

        for idx in self.df_original.index:
            # Build expected series
            exp_forecast = self.df_forecasted.loc[idx]
            exp_imputed = self.df_imputed.loc[idx]
            exp_original = self.df_original.loc[idx]

            fig = figs[idx]
            t_forecast, t_imputed, t_original = fig.data

            # X alignment: convert both to Timestamps and compare element-wise
            self.assertEqual(
                self._to_timestamp_list(t_forecast.x),
                list(exp_forecast.index),
                "Forecasted x does not match expected index."
            )
            self.assertEqual(
                self._to_timestamp_list(t_imputed.x),
                list(exp_imputed.index),
                "Imputed x does not match expected index."
            )
            self.assertEqual(
                self._to_timestamp_list(t_original.x),
                list(exp_original.index),
                "Original x does not match expected index."
            )

            # Y alignment: numeric equality (with NaNs allowed in original)
            np.testing.assert_allclose(
                self._to_float_array(t_forecast.y),
                exp_forecast.values.astype(float),
                rtol=0, atol=0,
                err_msg="Forecasted y does not match expected values."
            )
            np.testing.assert_allclose(
                self._to_float_array(t_imputed.y),
                exp_imputed.values.astype(float),
                rtol=0, atol=0,
                err_msg="Imputed y does not match expected values."
            )

            # For original, specifically check NaN positions are preserved
            got_orig = self._to_float_array(t_original.y)
            exp_orig = exp_original.values.astype(float)
            self.assertEqual(len(got_orig), len(exp_orig))
            for g, e in zip(got_orig, exp_orig):
                if np.isnan(e):
                    self.assertTrue(np.isnan(g), "NaN in original series should remain NaN in plot.")
                else:
                    self.assertAlmostEqual(g, e, places=12)

    # -----------------------------
    # Edge / negative tests
    # -----------------------------
    def test_empty_input_returns_empty_dict(self):
        empty_idx = pd.MultiIndex.from_tuples([], names=["StoreID", "ProductDescription"])
        df_empty_original = pd.DataFrame(index=empty_idx, columns=self.hist_dates)
        df_empty_imputed = pd.DataFrame(index=empty_idx, columns=self.hist_dates)
        df_empty_forecasted = pd.DataFrame(index=empty_idx, columns=self.all_dates)

        figs = visualize_imputation_forecasting_results(df_empty_original, df_empty_imputed, df_empty_forecasted)
        self.assertIsInstance(figs, dict)
        self.assertEqual(len(figs), 0)

    def test_mismatched_indices_raises_key_error(self):
        """
        If imputed/forecasted are missing a row that exists in original, .loc[idx] inside the function should raise KeyError.
        """
        # Drop one row from imputed/forecasted to force a mismatched index
        partial_idx = self.idx[:1]  # Keep only first row
        df_imputed_partial = self.df_imputed.loc[partial_idx]
        df_forecasted_partial = self.df_forecasted.loc[partial_idx]

        with self.assertRaises(KeyError):
            _ = visualize_imputation_forecasting_results(self.df_original, df_imputed_partial, df_forecasted_partial)


if __name__ == "__main__":
    unittest.main()
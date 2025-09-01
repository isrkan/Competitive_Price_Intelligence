import unittest
import pandas as pd
import numpy as np
from price_imputation_and_forecasting.data.clean_data import filter_missing_values_from_price_data


class TestFilterMissingValuesFromPriceData(unittest.TestCase):
    """
    Unit tests for filter_missing_values_from_price_data.

    The function filters rows based on the minimum required count of non-NaN values per row. Internally it computes: thresh = int(threshold_ratio * num_cols) and calls DataFrame.dropna(thresh=thresh, axis=0).

    NOTE on rounding behavior: Because `thresh` is computed with `int(...)` (floor), very small ratios can produce `thresh == 0`, which means *no* rows are dropped (including rows with all NaNs). We test and document that explicitly.
    """

    def setUp(self):
        """
        Create a canonical DataFrame used across tests.

        4 columns → convenient to reason about counts:
        - Row 'r_all'   : 4/4 non-NaN (100%)
        - Row 'r_half'  : 2/4 non-NaN (50%)
        - Row 'r_quarter':1/4 non-NaN (25%)
        - Row 'r_none'  : 0/4 non-NaN (0%)
        """
        self.cols = ["c1", "c2", "c3", "c4"]
        self.df = pd.DataFrame(
            {
                "c1": [1.0, np.nan, np.nan, np.nan],
                "c2": [2.0, 2.0, np.nan, np.nan],
                "c3": [3.0, np.nan, 3.0, np.nan],
                "c4": [4.0, 4.0, np.nan, np.nan],
            },
            index=["r_all", "r_half", "r_quarter", "r_none"],
        )
        # Sanity checks of the test fixture:
        # r_all      -> 4 valid
        # r_half     -> 2 valid
        # r_quarter  -> 1 valid
        # r_none     -> 0 valid
        self.assertEqual(self.df.notna().sum(axis=1).to_dict(),
                         {"r_all": 4, "r_half": 2, "r_quarter": 1, "r_none": 0})

    # ---------------------------
    # Happy-path / correctness
    # ---------------------------
    def test_correctness_various_thresholds(self):
        """
        Verify rows kept for several threshold ratios.

        thresh = int(ratio * 4):
          - ratio=1.00 → thresh=4 → keep only rows with 4 valid → ['r_all']
          - ratio=0.50 → thresh=2 → keep rows with >=2 valid → ['r_all','r_half']
          - ratio=0.26 → thresh=1 → keep rows with >=1 valid → ['r_all','r_half','r_quarter']
          - ratio=0.24 → thresh=0 → keep ALL (including all-NaN) → all rows
        """
        test_cases = [
            (1.00, ["r_all"]),
            (0.50, ["r_all", "r_half"]),
            (0.26, ["r_all", "r_half", "r_quarter"]),
            (0.24, ["r_all", "r_half", "r_quarter", "r_none"]),  # thresh=0 → keep all
        ]

        for ratio, expected_index in test_cases:
            with self.subTest(ratio=ratio):
                out = filter_missing_values_from_price_data(self.df, ratio)
                self.assertListEqual(list(out.index), expected_index)
                self.assertListEqual(list(out.columns), self.cols)  # columns preserved

    def test_all_rows_dropped_when_ratio_is_strict(self):
        """
        If every row has at least one NaN and ratio==1.0 (i.e., require all 4), rows with any NaN should be removed. In a DataFrame where each row has at least one NaN, result should be empty.
        """
        # Make a version where every row has at least one NaN
        df_all_have_nan = pd.DataFrame(
            {
                "c1": [1.0, np.nan],
                "c2": [np.nan, 2.0],
                "c3": [3.0, np.nan],
                "c4": [np.nan, 4.0],
            },
            index=["a", "b"],
        )
        # Each row has 2 valid values (out of 4). With ratio 1.0 → thresh=4 → drop all.
        out = filter_missing_values_from_price_data(df_all_have_nan, 1.0)
        self.assertTrue(out.empty)
        self.assertListEqual(list(out.columns), ["c1", "c2", "c3", "c4"])

    def test_multiindex_is_preserved(self):
        """
        Ensure that a MultiIndex on rows is preserved and filtering works the same.
        """
        arrays = [
            ["CategoryA", "CategoryA", "CategoryA", "CategoryA"],
            ["Prod1", "Prod1", "Prod1", "Prod1"],
            [1, 2, 3, 4],
        ]
        idx = pd.MultiIndex.from_arrays(arrays, names=["category", "ProductDescription", "StoreID"])
        df_multi = pd.DataFrame(self.df.values, index=idx, columns=self.cols)

        # ratio=0.50 → thresh=2 → keep rows with >=2 valid → 'r_all','r_half'
        # Our df_multi has the same values as self.df
        out = filter_missing_values_from_price_data(df_multi, 0.50)

        # Expect 2 rows retained out of 4 with same index names and columns
        self.assertEqual(len(out), 2)
        self.assertListEqual(list(out.columns), self.cols)
        self.assertEqual(out.index.names, ["category", "ProductDescription", "StoreID"])

    # ---------------------------------
    # Input validation & error handling
    # ---------------------------------

    def test_empty_dataframe_raises_value_error(self):
        """
        Passing an empty DataFrame should raise a ValueError with a clear message.
        """
        empty_df = pd.DataFrame(columns=self.cols)
        with self.assertRaises(ValueError) as ctx:
            filter_missing_values_from_price_data(empty_df, 0.5)
        self.assertIn("Input DataFrame 'category_df' is empty", str(ctx.exception))

    def test_non_dataframe_input_raises_type_error(self):
        """
        Non-DataFrame input should raise TypeError.

        IMPORTANT: The function checks `category_df.empty` BEFORE checking type.
        To avoid tripping the "empty" check, we pass a **non-empty pandas Series**, which has an `.empty` attribute but is not a DataFrame → ensures the TypeError path.
        """
        non_df_input = pd.Series([1, 2, 3])  # not empty, has .empty attribute, not a DataFrame
        with self.assertRaises(TypeError) as ctx:
            filter_missing_values_from_price_data(non_df_input, 0.5)
        self.assertIn("Expected category_df to be a pandas DataFrame", str(ctx.exception))

    def test_threshold_ratio_zero_raises_value_error(self):
        """
        threshold_ratio must be in (0, 1]. Zero should fail.
        """
        with self.assertRaises(ValueError) as ctx:
            filter_missing_values_from_price_data(self.df, 0.0)
        self.assertIn("threshold_ratio must be in (0,1]", str(ctx.exception))

    def test_threshold_ratio_negative_raises_value_error(self):
        """
        threshold_ratio must be in (0, 1]. Negative ratios should fail.
        """
        with self.assertRaises(ValueError):
            filter_missing_values_from_price_data(self.df, -0.1)

    def test_threshold_ratio_greater_than_one_raises_value_error(self):
        """
        threshold_ratio must be in (0, 1]. >1 should fail.
        """
        with self.assertRaises(ValueError):
            filter_missing_values_from_price_data(self.df, 1.000001)

    def test_runtime_error_wrapped_when_dropna_fails(self):
        """
        Simulate an internal failure in DataFrame.dropna and assert it is wrapped as RuntimeError with a helpful message.
        """
        class ExplodingDataFrame(pd.DataFrame):
            # Ensure the subclass propagates properly in pandas ops (good practice, though not strictly required here)
            @property
            def _constructor(self):
                return ExplodingDataFrame

            def dropna(self, *args, **kwargs):
                raise Exception("Simulated dropna failure")

        bad_df = ExplodingDataFrame(self.df.copy())
        # Must not be empty to avoid early empty-check error
        self.assertFalse(bad_df.empty)

        with self.assertRaises(RuntimeError) as ctx:
            filter_missing_values_from_price_data(bad_df, 0.5)
        self.assertIn("Error filtering DataFrame", str(ctx.exception))
        self.assertIn("Simulated dropna failure", str(ctx.exception))

    # ---------------------------------
    # Behavioral edge cases
    # ---------------------------------

    def test_small_ratio_rounds_down_to_zero_keeps_all_rows(self):
        """
        When ratio * num_cols < 1, int(...) == 0 → pandas dropna(thresh=0) keeps ALL rows (even all-NaN). We assert that behavior explicitly.
        """
        tiny_ratio = 1e-6  # 0.000001 * 4 → 0.000004 → int(...) == 0
        out = filter_missing_values_from_price_data(self.df, tiny_ratio)
        self.assertListEqual(list(out.index), list(self.df.index))  # no filtering
        self.assertTrue(pd.isna(out.loc["r_none"]).all())  # even all-NaN row retained

    def test_columns_and_dtypes_preserved(self):
        """
        Ensure the function does not alter column order or dtypes of remaining rows.
        """
        # Add a non-numeric column w/ NaNs to ensure robustness
        df = self.df.copy()
        df["meta"] = ["A", "B", np.nan, "D"]

        out = filter_missing_values_from_price_data(df, 0.50)  # thresh=2 → keep r_all, r_half
        self.assertListEqual(list(out.columns), ["c1", "c2", "c3", "c4", "meta"])
        # dtypes should be preserved for retained rows; allow pandas' normal dtype upcasting
        # Just assert 'meta' is object dtype as expected
        self.assertEqual(out["meta"].dtype, object)
        self.assertListEqual(list(out.index), ["r_all", "r_half"])


if __name__ == "__main__":
    unittest.main()
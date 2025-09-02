import plotly.graph_objects as go

def visualize_imputation_forecasting_results(df_price_input,df_price_with_imputed_data,df_price_with_forecasted_data):
    """
    Visualize original, imputed, and forecasted price data for selected stores.

    Parameters:
    - df_price_input (pd.DataFrame): Original price data with NaNs.
    - df_price_with_imputed_data (pd.DataFrame): Prices with imputed values.
    - df_price_with_forecasted_data (pd.DataFrame): Prices with imputation + forecast.

    Returns:
    - figs (dict): Dictionary mapping store_id -> Plotly Figure object.
    """
    figs = {}  # Initialize dictionary to store figures for each row

    # Loop over each row in the original price data
    for idx in df_price_input.index:
        # Row-wise selection
        original_series = df_price_input.loc[idx]  # Original price series with NaNs
        imputed_series = df_price_with_imputed_data.loc[idx]  # Imputed series (filled missing values)
        forecasted_series = df_price_with_forecasted_data.loc[idx]  # Imputed + forecasted series

        fig = go.Figure()

        # Add forecasted series to show future predictions
        fig.add_trace(go.Scatter(
            x=forecasted_series.index,
            y=forecasted_series.values,
            name="Forecasted",
            line=dict(dash="dot", color="firebrick")
        ))

        # Add imputed series to show historical prices with missing values filled
        fig.add_trace(go.Scatter(
            x=imputed_series.index,
            y=imputed_series.values,
            name="Imputed",
            line=dict(dash="solid", color="blue")
        ))

        # Add original series to show raw historical data with NaNs
        fig.add_trace(go.Scatter(
            x=original_series.index,
            y=original_series.values,
            name="Original",
            mode="markers+lines",
            line=dict(color="black"),
            marker = dict(size=6, symbol="circle")
        ))

        # Configure figure layout
        fig.update_layout(
            title=f"Price Predictions for {idx}",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        figs[idx] = fig  # Save figure in dictionary keyed by row index

    return figs
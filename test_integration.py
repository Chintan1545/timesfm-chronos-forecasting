import os, pandas as pd, numpy as np
from timeseries_forecasting import (
    load_dataset, run_experiment, aggregate_results, build_excel_report
)

data           = load_dataset(r'complex_multi_product_order_forecasting_dataset.xlsx')
daily_results  = run_experiment(data, 'daily',  horizon=14)
daily_lb       = aggregate_results(daily_results, 'daily')
hourly_results = run_experiment(data, 'hourly', horizon=24)
hourly_lb      = aggregate_results(hourly_results, 'hourly')

# ── Check leaderboard structure ────────────────────────────────
assert set(daily_lb.columns) >= {"model","WMAPE_%","MAPE_%","MAE","RMSE"}
assert daily_lb["model"].nunique() == 3,  "Should have 3 models"
assert len(daily_lb) == 24,              "8 series × 3 models = 24 rows"

# ── WMAPE must always be between 0–200% ──────────────────────
assert daily_lb["WMAPE_%"].between(0, 200).all(), "WMAPE out of range"
assert daily_lb["MAPE_%"].between(0, 500).all(),  "MAPE out of range"

# ── Best model should have lowest WMAPE ──────────────────────
best = daily_lb.groupby("model")["WMAPE_%"].mean().idxmin()
print(f"✅ Best daily model by WMAPE: {best}")

# ── Build Excel and verify file ───────────────────────────────
build_excel_report(daily_lb, hourly_lb, daily_results, hourly_results,
                   'test_output.xlsx')
assert os.path.exists('test_output.xlsx'),    "Excel file not created"
assert os.path.getsize('test_output.xlsx') > 10_000, "Excel file too small"
print("✅ Excel report created and valid")

# ── Validate sheets ───────────────────────────────────────────
xl = pd.read_excel('test_output.xlsx', sheet_name=None)
required_sheets = ['README','Daily_Leaderboard','Hourly_Leaderboard',
                   'Daily_Detail','Hourly_Detail',
                   'Forecast_Table','Metrics_Walkthrough','Pipeline_Guide']
for sheet in required_sheets:
    assert sheet in xl, f"Missing sheet: {sheet}"
print(f"✅ All {len(required_sheets)} sheets present in Excel")

print("\n All integration tests passed!")
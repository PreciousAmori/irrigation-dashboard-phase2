import pandas as pd
from pathlib import Path

# =====================================================
# File locations
# =====================================================

base_dir = Path(
    r"G:\My Drive\PhD_Project_Phase2\data\processed\ml_ready_loso\vri"
)

keys_file = base_dir / "keys_test_external.parquet"
x_file = base_dir / "X_test_external.parquet"

output_file = base_dir / "ImplementationSET_phase2.csv"

# =====================================================
# Read files
# =====================================================

keys_df = pd.read_parquet(keys_file)
X_df = pd.read_parquet(x_file)

# =====================================================
# Sanity checks
# =====================================================

print(f"keys rows: {len(keys_df):,}")
print(f"X rows:    {len(X_df):,}")

if len(keys_df) != len(X_df):
    raise ValueError(
        "Row count mismatch between keys_test_external and X_test_external"
    )

# =====================================================
# Preserve row alignment
# =====================================================

keys_df = keys_df.reset_index(drop=True)
X_df = X_df.reset_index(drop=True)

# =====================================================
# Merge side-by-side
# =====================================================

merged_df = pd.concat([keys_df, X_df], axis=1)

# =====================================================
# Create dashboard-required columns
# =====================================================

# Convert Unix milliseconds to datetime
if "date" in merged_df.columns:

    merged_df["Date"] = pd.to_datetime(
        merged_df["date"],
        unit="ms",
        errors="coerce"
    )

# Create dashboard plot identifier
if "plot_id" in merged_df.columns:

    merged_df["Management_Plot_ID"] = merged_df["plot_id"]

# =====================================================
# Save NEW file
# =====================================================

merged_df.to_csv(output_file, index=False)

print("\nSaved:")
print(output_file)

print("\nShape:")
print(merged_df.shape)

print("\nFirst columns:")
print(merged_df.columns[:15].tolist())
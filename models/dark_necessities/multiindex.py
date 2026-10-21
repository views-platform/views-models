from pathlib import Path
from uuid import uuid4
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

candidate_paths = [
    #Path("/home/dylpin/views-platform/views-models/models/dark_necessities/data/raw/calibration_viewser_df.parquet"),
    Path("/home/dylpin/views-platform/views-models/models/dark_necessities/data/raw/validation_viewser_df.parquet"),
    Path("/home/dylpin/views-platform/views-models/models/dark_necessities/data/raw/forecasting_viewser_df.parquet"),
]

source_path = next((path for path in candidate_paths if path.exists()), None)
if source_path is None:
    raise FileNotFoundError(
        "Could not find Parquet file."
    )

# Overwrite the source file safely by streaming indexed batches into a temp file,
# then atomically replacing the original path.
target_path = source_path
staging_path = target_path.with_name(f"{target_path.stem}.tmp.{uuid4().hex}{target_path.suffix}")
parquet_file = pq.ParquetFile(source_path)
writer = None

try:
    for batch in parquet_file.iter_batches(batch_size=1_000_000):
        batch_df = batch.to_pandas()
        batch_df = batch_df.set_index(["priogrid_id", "month_id"]).sort_index()
        table = pa.Table.from_pandas(batch_df, preserve_index=True)
        if writer is None:
            writer = pq.ParquetWriter(staging_path, table.schema, compression="snappy")
        writer.write_table(table)
finally:
    if writer is not None:
        writer.close()

os.replace(staging_path, target_path)

# Verify the overwritten file round-trips with the expected MultiIndex.
verification = pd.read_parquet(target_path).head()
print(f"Overwrote {target_path}")
print(verification.index)
print(verification)

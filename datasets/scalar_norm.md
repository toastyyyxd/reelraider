# Scalar Normalization Processing & Output Schema

This document describes the process of min-max normalization applied to scalar columns in the dataset using the `ScalarNormalizer` class defined in `scalar_norm.py`.

## Overview

The scalar normalization workflow consists of:

1. **Compute Statistics**: Calculate the minimum and maximum values for each target column.
2. **Normalize Values**: Transform each value using the formula:
   ```text
   normalized = (value - min) / (max - min)
   ```
3. **Add Prefixed Columns**: Append the new normalized columns to the DataFrame with prefix `sn_` (e.g., `sn_year`).
4. **Collect Results**: Convert the LazyFrame back into a regular DataFrame for output.

## Default & Custom Columns

- By default, the following columns are normalized:
  - `year`
  - `runtime`
  - `votes`
- You can specify a custom comma-separated list via the `--columns` argument.

## CLI Usage

```bash
python datasets/scalar_norm.py \
  --input  /path/to/input.parquet \
  --columns year,runtime,votes \
  --output /path/to/output.parquet
```

- `--input`: Path to the input Parquet file.
- `--columns`: Comma-separated list of columns to normalize (default: `year,runtime,rating,votes,metascore`).
- `--output`: Path for the output Parquet file containing the normalized DataFrame.

## Relevant Output Columns & Schema

| Column       | Type   | Description                                                                                       |
|--------------|--------|---------------------------------------------------------------------------------------------------|
| **Original** |        | All original columns are preserved unchanged.                                                     |
| sn_year      | float  | Normalized `year`: (year - min(year)) / (max(year) - min(year)).                                  |
| sn_runtime   | float  | Normalized `runtime`: (runtime - min(runtime)) / (max(runtime) - min(runtime)).                   |
| sn_rating    | float  | Normalized `rating`: (rating - min(rating)) / (max(rating) - min(rating)).                        |
| sn_votes     | float  | Normalized `votes`: (votes - min(votes)) / (max(votes) - min(votes)).                             |
| sn_metascore | float  | Normalized `metascore`: (metascore - min(metascore)) / (max(metascore) - min(metascore)).        |

## Example Row

| year | runtime | rating | votes | metascore | sn_year | sn_runtime | sn_rating | sn_votes | sn_metascore |
|------|---------|--------|-------|-----------|---------|------------|-----------|----------|--------------|
| 2005 | 120     | 7.5    | 15000 | 85        | 0.50    | 0.60       | 0.75      | 0.75     | 0.85         |

## Notes

- Missing or null values in source columns will propagate to the corresponding `sn_` columns.
- If `max == min` for any column, normalization will output the default value 0.5; ensure your data has variation for each target column.
- Only columns listed in `--columns` are normalized; all others remain unchanged.
- The script uses Polars LazyFrame for efficient computation and requires Polars installed in your environment.

---
*Generated based on the implementation in `datasets/scalar_norm.py`*

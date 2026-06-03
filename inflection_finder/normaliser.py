import os
import re
import numpy as np
import pandas as pd


def combine_well_matrix_csvs(csv_paths, output_path, cap_size=1):
    """
    Combine multiple CN x NN CSV files into one CN x NN CSV.

    If a CN exists but a new NN column is empty, fill the base CN row.
    If the exact CN x NN cell is already occupied, create/fill CN-1, CN-2, etc.
    """

    def to_int_label(x):
        s = str(x).strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        return x

    def normalize_labels(labels):
        return pd.Index([to_int_label(x) for x in labels])

    def sort_labels(labels):
        def sort_key(label):
            if isinstance(label, (int, np.integer)):
                return (int(label), 0, "")

            s = str(label).strip()

            m = re.match(r"^(\d+)-(\d+)$", s)
            if m:
                return (int(m.group(1)), int(m.group(2)), "")

            if re.fullmatch(r"-?\d+", s):
                return (int(s), 0, "")

            return (10**9, 0, s)

        return sorted(list(labels), key=sort_key)

    def add_empty_row(df, row_label):
        new_row = pd.Series(np.nan, index=df.columns, dtype=float)
        return pd.concat(
            [df, pd.DataFrame([new_row], index=[row_label])],
            axis=0,
            sort=False,
        )

    combined = None

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, index_col=0)

        df.index = normalize_labels(df.index)
        df.columns = normalize_labels(df.columns)
        df = df.loc[:, ~df.columns.duplicated()]

        if combined is None:
            combined = df.copy()
            continue

        combined.index = normalize_labels(combined.index)
        combined.columns = normalize_labels(combined.columns)
        combined = combined.loc[:, ~combined.columns.duplicated()]

        for nn in df.columns:
            if nn not in combined.columns:
                combined[nn] = np.nan

        for cn in df.index:
            row_series = df.loc[cn]

            if row_series.dropna().empty:
                continue

            if cn not in combined.index:
                combined = add_empty_row(combined, cn)

            for nn, value in row_series.items():
                if pd.isna(value):
                    continue

                if nn not in combined.columns:
                    combined[nn] = np.nan

                # Fill base CN row if that exact CN x NN cell is empty
                if pd.isna(combined.at[cn, nn]):
                    combined.at[cn, nn] = value
                    continue

                # Otherwise this exact CN x NN pair is a replicate
                k = 1
                while True:
                    duplicate_cn = f"{cn}-{k}"

                    if duplicate_cn not in combined.index:
                        combined = add_empty_row(combined, duplicate_cn)

                    if pd.isna(combined.at[duplicate_cn, nn]):
                        combined.at[duplicate_cn, nn] = value
                        break

                    k += 1

    if combined is None:
        raise ValueError("No CSV files were provided.")
        
    combined = combined.reindex(columns=sort_labels(combined.columns))
    combined = combined.reindex(index=sort_labels(combined.index))

    # Make all values > 1 equal to 1
    combined = combined.clip(upper=cap_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index_label="CN")

    print(f"Saved combined CSV to {output_path}")
    return output_path
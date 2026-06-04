import os
import re
import numpy as np
import pandas as pd


def _check_file(exp_name, csv_name):
    """
    Helper function for normalise_plates().
    Checks that the requested experiment folder exists and contains the desired
    CSV file inside its output folder.

    Expected structure:
        experiments/
            {exp_name}/
                output/
                    {csv_name}

    Args:
        exp_name (str): Name of the experiment directory inside "experiments".
        csv_name (str): Name of the CSV file inside the experiment's output
            folder. Usually "inflection_points.csv" or "intensity_points.csv".

    Returns:
        str or None: Path to the CSV file if it exists. Returns None and prints
        an error message if the experiment folder or CSV file is missing.
    """
    exp_folder = os.path.join("..", "experiments", exp_name)

    # Checks if experiment is found in "experiments" folder.
    if not os.path.isdir(exp_folder):
        print(f"'{exp_name}' not found in 'experiments' folder.")
        return

    exp_path = os.path.join(exp_folder, "output", csv_name)

    # Checks if csv file is found in output folder.
    if not os.path.isfile(exp_path):
        print(f"'{csv_name}' not found in '{exp_name}/output.'")
        return
    
    return exp_path


def normalise_plates(uniform_name, exp_names, int_or_infl, cap_size=1):
    """
    Normalise experimental plate CSVs against a uniform reference plate.

    Plates should be stored inside the "experiments" folder. Each plate should
    contain an "output" folder with exported CSV files, such as
    "inflection_points.csv" and "intensity_points.csv".

    The function uses the uniform plate as a position-based correction map. For
    each experimental plate, values are corrected by dividing by the relative
    uniform intensity at the same CN/NN position.

    Expected structure:
        experiments/
            {uniform_name}/
                output/
                    intensity_points.csv or inflection_points.csv
            {plate_name}/
                output/
                    intensity_points.csv or inflection_points.csv

    Args:
        uniform_name (str): Name of the uniform/reference plate experiment.
        plate_names (list[str]): Names of experimental plate folders.
        int_or_infl (str): Which type of CSV to normalise. Use "intensity" or
            "inflection".
        cap_size (float): Maximum allowed normalised value. Values above this
            are capped. Default is 1.

    Produces:
        Corrected CSV files named "{plate_name}-corrected.csv" inside:
            output/{int_or_infl}/
    """
    output_dir = os.path.join("output", int_or_infl)
    csv = int_or_infl + "_points.csv"

    uniform_path = _check_file(uniform_name, csv)

    # Check if output csv exists at uniform_name
    if uniform_path:
        uniform_raw = pd.read_csv(uniform_path, index_col=0)    
    else: return

    for exp_name in exp_names:

        # Stop output if csv does not exist at experiment
        exp_path = _check_file(exp_name, csv)
        if not exp_path: break

        exp = pd.read_csv(exp_path, index_col=0)
        uniform = uniform_raw.copy()

        if uniform.shape != exp.shape:
            raise ValueError(
                f"Shape mismatch for {exp_name}: experiment {exp.shape}, uniform {uniform.shape}"
            )

        # Align by physical plate position, then copy experiment labels
        uniform.index = exp.index
        uniform.columns = exp.columns

        max_uniform = uniform.max().max()
        correction_factor = uniform / max_uniform

        corrected = exp / correction_factor.replace(0, np.nan)

        # Make all values > cap_size equal to cap_size
        corrected = corrected.clip(upper=cap_size)

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{exp_name}-corrected.csv")
        corrected.to_csv(output_path, index_label="CN")

        print(f"Saved {output_path}")



def combine_well_matrix_csvs(csv_paths, output_path):
    """
    Combine multiple CN x NN CSV files into one larger CN x NN CSV.

    Files are combined using CN row labels and NN column labels. If a CN/NN
    value already exists, the new value is treated as a replicate and saved in
    a duplicate CN row such as "1-1", "1-2", etc.

    Args:
        - csv_paths:   list of CSV paths to combine
        - output_path: path to save the combined CSV

    Produces:
        - a combined CSV with only CN row headings and NN column headings

    Returns:
        - the output_path after saving
    """
    def to_int_label(x):
        s = str(x).strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        return x

    def normalise_labels(labels):
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

        df.index = normalise_labels(df.index)
        df.columns = normalise_labels(df.columns)
        df = df.loc[:, ~df.columns.duplicated()]

        if combined is None:
            combined = df.copy()
            continue

        combined.index = normalise_labels(combined.index)
        combined.columns = normalise_labels(combined.columns)
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index_label="CN")

    print(f"Saved combined CSV to {output_path}")
    return output_path


def calculate_inflection_rates(
    intensity_csv,
    inflection_csv,
    start_intensity=1.0,
    positive_rate=True,
    max_inflection=None,
    output_csv="output/inflection_rate.csv",
):
    """
    Calculate inflection rates from intensity-at-inflection and inflection-time CSVs.

    Assumes each well starts at `start_intensity` at time 0 and changes roughly
    linearly until the inflection point.

    Formula:
        mathematical slope = (intensity_at_inflection - start_intensity) / inflection_time

        positive darkening rate = - mathematical slope

    If inflection time equals max_inflection, the rate is set to 0.

    Args:
        intensity_csv: path to CSV containing normalized intensity values at inflection.
        inflection_csv: path to CSV containing inflection times.
        start_intensity: starting normalized intensity, usually 1.0.
        positive_rate: if True, returns positive darkening rates.
                       if False, returns mathematical slope.
        max_inflection: end-of-experiment time. If inflection == max_inflection,
                        rate is set to 0.
        output_csv: optional output path. If None, does not save.

    Returns:
        rate_df: pandas DataFrame of rates.
    """

    intensity = pd.read_csv(intensity_csv, index_col=0)
    inflection = pd.read_csv(inflection_csv, index_col=0)

    # Align CN rows and NN columns
    intensity = intensity.reindex(index=inflection.index, columns=inflection.columns)

    # Convert all cells to numeric
    intensity = intensity.apply(pd.to_numeric, errors="coerce")
    inflection = inflection.apply(pd.to_numeric, errors="coerce")

    # Avoid divide-by-zero
    inflection_safe = inflection.replace(0, 0.01)

    if positive_rate: rate = (start_intensity - intensity) / inflection_safe
    else: rate = (intensity - start_intensity) / inflection_safe

    # If inflection happens at the end of the experiment, treat as no inflection
    if max_inflection is not None:
        rate = rate.mask(inflection == max_inflection, 0)

    if output_csv is not None:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        rate.to_csv(output_csv, index_label="CN")
        print(f"Saved inflection rates to {output_csv}")

    return rate
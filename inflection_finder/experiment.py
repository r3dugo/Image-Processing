import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy
import yaml
from sklearn.metrics import r2_score
# LOESS import is performed lazily inside process_data to avoid hard dependency at module import time
import re


class UserParameters(object):
    '''
    A class for handling user parameters from the user config file.
    '''

    #Given a file path, path, ending in .txt, read the contents of the file and 
    #return the contents as a string.
    def read_text_file(self, path):
        '''
        Given a file path to a text file, read the contents of the file and
        output the contents as a string.
        '''
        with open(path, "rt") as f:
            return f.read()

    def read_parameters_from_yaml(self, yaml_file_path):
        '''
        Given a path to a yaml file, read the file into python as a dictionary.
        '''
        return yaml.safe_load(self.read_text_file(yaml_file_path))

    def __init__(self, config_file_path):
        '''
        Given a user config file path, load the user parameter dictionary from the
        specified user config yaml file.

        Errors:
        1) If the config_file_path is not a yaml file, raise an exception.
        '''
        # Check that the user config file is a yaml file.
        if config_file_path.split(".")[-1] != "yaml":
            raise Exception("User config file must be a yaml file.")

        self.config_file_path = config_file_path

        self.user_parameters = self.read_parameters_from_yaml(config_file_path)

    def get(self, variable_name):
        '''
        Given a variable name string, attempt to fetch the associated value
        from the parameters loaded from the user config file.

        Errors:
        1) If the requested variable name is not in the user config file, raise
            a KeyError.
        '''
        try:
            variable_value = self.user_parameters[variable_name]
            return variable_value
        
        except KeyError:
            raise KeyError("Variable '" + variable_name + \
                            "' not found in config file '" + \
                            self.config_file_path + "'.")



class Experiment(object):

    def __init__(self, config_file_path):

        # Check that the user config file is a yaml file.
        if config_file_path.split(".")[-1] != "yaml":
            raise Exception("User config file must be a yaml file.")
        
        self.config_file_path = config_file_path
        self.user_parameters = UserParameters(config_file_path)

        self.inflection_points = []
        self.inflection_point_values = []  # Store intensity values at inflection points
        self.plateau_regions = []  # Store plateau regions for each well
        self.filtered_data = {}  # Store filtered time and intensity arrays
        self._process_debugs = []  # Per-call debug summaries from process_data
        # Read LOESS bandwidth from config, default to 0.3 if not specified
        self.loess_bandwidth = self.user_parameters.user_parameters.get('loess_bandwidth', 0.3)
    

    def filter_first_half(self, time_array, array):
        '''
        Filter data to only include the first half of the time range.
        
        Inputs:
            time_array: numpy array of time values
            array: numpy array of intensity values
            
        Returns:
            filtered_time: time array filtered to first half
            filtered_array: intensity array filtered to first half
            cutoff_index: index where the cutoff occurs
        '''
        time_array = np.asarray(time_array, dtype=np.float64)
        array = np.asarray(array, dtype=np.float64)
        max_time = np.max(time_array)
        cutoff_time = max_time / 2.0
        
        # Find the index where time exceeds cutoff
        cutoff_indices = np.where(time_array <= cutoff_time)[0]
        if len(cutoff_indices) == 0:
            cutoff_index = len(time_array)
        else:
            cutoff_index = cutoff_indices[-1] + 1
        
        filtered_time = time_array[:cutoff_index]
        filtered_array = array[:cutoff_index]
        
        return filtered_time, filtered_array, cutoff_index
    
    def detect_plateau(self, time_array, array, window_size=10, threshold=0.01):
        '''
        Detect plateaus in the data by finding regions where the local slope is close to zero.

        Selection behavior (optional user parameter 'plateau_selection'):
          - 'latest'  (default): choose the plateau whose END is latest in time
          - 'longest': choose the plateau with the most points
          - 'first':   choose the first plateau detected (old behavior)
        '''
        if len(array) < window_size:
            return None, None, None, None

        # Force float64 and drop NaNs so lengths match
        time_array = np.asarray(pd.to_numeric(time_array, errors='coerce'), dtype=np.float64)
        array = np.asarray(pd.to_numeric(array, errors='coerce'), dtype=np.float64)
        mask = np.isfinite(time_array) & np.isfinite(array)
        time_array = time_array[mask]
        array = array[mask]
        if len(array) < window_size:
            return None, None, None, None

        # Optional: adaptive threshold if user passes 'auto' or None
        if threshold is None or (isinstance(threshold, str) and threshold.lower() == 'auto'):
            # Estimate a typical slope scale from the median absolute derivative
            dt = np.diff(time_array)
            dy = np.diff(array)
            valid = np.isfinite(dt) & np.isfinite(dy) & (dt != 0)
            if np.any(valid):
                deriv = np.abs(dy[valid] / dt[valid])
                med = np.median(deriv)
                threshold = 0.2 * med  # conservative: plateau is much flatter than typical change
            else:
                threshold = 0.01

        # Rolling slope over windows via np.polyfit (slope only)
        slopes = np.empty(len(array) - window_size + 1, dtype=np.float64)
        slopes[:] = np.inf
        for i in range(len(slopes)):
            window_time = time_array[i:i + window_size]
            window_data = array[i:i + window_size]
            if len(np.unique(window_time)) > 1 and np.all(np.isfinite(window_time)) and np.all(np.isfinite(window_data)):
                slopes[i] = abs(np.polyfit(window_time, window_data, 1)[0])
            else:
                slopes[i] = np.inf

        # Build contiguous plateau regions where slope < threshold
        plateau_regions = []
        in_plateau = False
        plateau_start = None
        for i, slope in enumerate(slopes):
            if slope < threshold and not in_plateau:
                in_plateau = True
                plateau_start = i
            elif slope >= threshold and in_plateau:
                plateau_end = i + window_size - 1
                if plateau_end - plateau_start >= window_size:
                    plateau_regions.append((plateau_start, plateau_end))
                in_plateau = False
                plateau_start = None

        # Plateau continues to end
        if in_plateau and plateau_start is not None:
            plateau_end = len(array) - 1
            if plateau_end - plateau_start >= window_size:
                plateau_regions.append((plateau_start, plateau_end))

        if not plateau_regions:
            return None, None, None, None

        selection = self.user_parameters.user_parameters.get('plateau_selection', 'latest')
        selection = str(selection).lower()

        def score(region):
            s, e = region
            length = e - s + 1
            if selection == 'longest':
                return (length, e)  # primary: length, secondary: latest
            if selection == 'first':
                return (-s, -length)  # smallest start wins; invert for max()
            # default/latest
            return (e, length)  # primary: latest end, secondary: length

        best = max(plateau_regions, key=score)
        start_idx, end_idx = best
        plateau_time = time_array[start_idx]
        plateau_value = float(np.mean(array[start_idx:end_idx + 1]))
        return start_idx, end_idx, plateau_time, plateau_value

    def process_data(self, time_array, array, bandwidth=None):
        '''
        Locate inflection points using LOESS regression and zone-based detection.
        
        Args:
            time_array: Time values
            array: Intensity values
            bandwidth: LOESS bandwidth (fraction of data to use, 0.1-1.0). 
                      Defaults to self.loess_bandwidth (0.3)
        
        Algorithm:
        1. Apply LOESS regression to smooth noisy data
        2. Interpolate to equally-spaced points
        3. Compute first derivative (variation/slope)
        4. Identify zones: high-variation (red) vs low-variation/plateau (blue)
        5. Inflection points are boundaries between zones
        6. Filter out zones below a minimum size threshold
        '''
        if bandwidth is None:
            bandwidth = self.loess_bandwidth
        
        # Force numeric arrays and drop NaNs
        time_array = np.asarray(pd.to_numeric(time_array, errors='coerce'), dtype=np.float64)
        array = np.asarray(pd.to_numeric(array, errors='coerce'), dtype=np.float64)
        mask = np.isfinite(time_array) & np.isfinite(array)
        time_array = time_array[mask]
        array = array[mask]
        if len(array) < 5:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.nan, None

        # Apply LOESS regression for noise suppression.
        # Import lazily so this function can fall back when statsmodels isn't available.
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
            loess_result = _lowess(array, time_array, frac=bandwidth, it=3)
            smoothed_time = loess_result[:, 0]
            smoothed_array = loess_result[:, 1]
        except Exception:
            # Fallback to original data if LOESS (or statsmodels) fails
            smoothed_time = time_array
            smoothed_array = array

        # Compute first derivative (finite differences) to get variation/slope
        derivatives = np.gradient(smoothed_array, smoothed_time)
        abs_derivatives = np.abs(derivatives)
        
        # Find median derivative to use as threshold for zone determination
        median_derivative = np.nanmedian(abs_derivatives)
        threshold = median_derivative * 0.5  # Tolerance for plateau detection
        
        # Identify zones: where variation is significant vs. close to zero
        # High-variation zones: abs(derivative) > threshold
        # Low-variation zones (plateaus): abs(derivative) <= threshold
        in_active_zone = abs_derivatives > threshold
        
        # Find boundaries where zones change (inflection points)
        zone_changes = np.diff(in_active_zone.astype(int))
        inflection_indices = np.where(np.abs(zone_changes) > 0)[0]
        
        # If no clear inflection points, use the point with maximum curvature
        if len(inflection_indices) == 0:
            index = int(np.nanargmax(abs_derivatives))
        else:
            # Choose the first major inflection point within the curve
            # Filter out very small zones by taking the second inflection if it exists
            if len(inflection_indices) >= 2:
                # Prefer the first significant inflection
                index = int(inflection_indices[0])
            else:
                index = int(inflection_indices[0])
        
            # Check if the value at inflection point is reasonable (dips below configured threshold)
            # Read `inflection_threshold` from user config, default to 0.8 if not specified
            inflection_threshold = self.user_parameters.user_parameters.get('inflection_threshold', 0.8)

            # Determine the earliest time (index) where the signal falls below threshold.
            # Prefer the smoothed curve normally, but detect LOESS oversmoothing and
            # prefer the raw signal interpolated onto the smoothed time grid when that occurs.
            first_below = None
            use_raw_interp = False
            use_raw_interp = False
            try:
                # Interpolate raw values onto smoothed_time for comparison
                interp_raw = np.interp(smoothed_time, time_array, array)
                below_raw = np.where(interp_raw < inflection_threshold)[0]
                below_smoothed = np.where(smoothed_array < inflection_threshold)[0]

                # Compare minima to detect oversmoothing: if LOESS-min is notably higher than raw-interp-min
                min_smoothed = np.nanmin(smoothed_array) if smoothed_array.size else np.nan
                min_interp = np.nanmin(interp_raw) if interp_raw.size else np.nan
                oversmooth_tol = self.user_parameters.user_parameters.get('loess_oversmooth_tolerance', 0.03)
                oversmoothed = (min_smoothed - min_interp) > oversmooth_tol

                if oversmoothed and below_raw.size > 0:
                    # LOESS oversmoothed the minimum — prefer first crossing from interpolated raw signal
                    first_below = int(below_raw[0])
                    use_raw_interp = True
                else:
                    if below_smoothed.size > 0:
                        first_below = int(below_smoothed[0])
                    elif below_raw.size > 0:
                        # no smoothed crossing but raw crosses
                        first_below = int(below_raw[0])
            except Exception:
                first_below = None
                use_raw_interp = False

            n = len(smoothed_array)

            if smoothed_array[index] >= inflection_threshold:
                # If there is a crossing (smoothed or raw), enforce the inflection to be at that crossing.
                # If no crossing exists, strictly place the inflection at the end of the trace.
                if first_below is not None:
                    # Respect user-configured selection strictness: 0.0 (relaxed — search far after crossing)
                    # to 1.0 (strict — choose first crossing). Default 0.6.
                    selection_strictness = float(self.user_parameters.user_parameters.get('selection_strictness', 0.6))
                    if selection_strictness >= 0.999:
                        index = int(first_below)
                    else:
                        try:
                            # compute allowed search window end based on strictness
                            max_follow = n - first_below - 1
                            allowed_span = max(1, int((1.0 - selection_strictness) * max(5, max_follow)))
                            allowed_end = min(n - 1, first_below + allowed_span)

                            # pick the values to search (interpolated raw or smoothed)
                            if use_raw_interp:
                                tail_vals = interp_raw[first_below:allowed_end + 1]
                            else:
                                tail_vals = smoothed_array[first_below:allowed_end + 1]

                            if tail_vals.size == 0:
                                index = n - 1
                            else:
                                # find local minima in the constrained tail (neg->pos in derivative)
                                d_tail = np.diff(tail_vals)
                                minima_rel = np.where((d_tail[:-1] < 0) & (d_tail[1:] >= 0))[0] + 1 if d_tail.size >= 2 else np.array([], dtype=int)
                                if minima_rel.size > 0:
                                    minima_idx = first_below + minima_rel
                                    if use_raw_interp:
                                        minima_vals = interp_raw[minima_idx]
                                    else:
                                        minima_vals = smoothed_array[minima_idx]
                                    chosen = int(minima_idx[np.argmin(minima_vals)])
                                    index = chosen
                                else:
                                    # no clear local minima; pick the global minimum in the constrained tail
                                    min_rel = int(np.nanargmin(tail_vals))
                                    index = first_below + min_rel
                        except Exception:
                            index = n - 1
                else:
                    index = n - 1

            # Enforce the strict rule: if we detected any crossing, the inflection point must be at or after it
            if first_below is not None and index < first_below:
                index = int(first_below)

        time_at_inflection = smoothed_time[index]

        # Splitting the array around the index for plotting
        array_1_x = smoothed_time[:index + 1]
        array_1_y = smoothed_array[:index + 1]
        array_2_x = smoothed_time[index:]
        array_2_y = smoothed_array[index:]

        # Record a compact debug summary for this call (do not store large arrays)
        try:
            first_below_smoothed = int(np.where(smoothed_array < inflection_threshold)[0][0]) if np.any(smoothed_array < inflection_threshold) else None
        except Exception:
            first_below_smoothed = None
        try:
            interp_raw = np.interp(smoothed_time, time_array, array)
            first_below_interp = int(np.where(interp_raw < inflection_threshold)[0][0]) if np.any(interp_raw < inflection_threshold) else None
        except Exception:
            first_below_interp = None

        debug = {
            'loess_bandwidth': bandwidth,
            'min_filtered': float(np.nanmin(array)) if array.size else None,
            'min_smoothed': float(np.nanmin(smoothed_array)) if smoothed_array.size else None,
            'first_below_smoothed': first_below_smoothed,
            'first_below_interp': first_below_interp,
            'use_raw_interp': bool(use_raw_interp),
            'selection_strictness': float(self.user_parameters.user_parameters.get('selection_strictness', 0.6)),
            'allowed_search_end': (locals().get('allowed_end') if 'allowed_end' in locals() else None),
            'chosen_index': int(index) if index is not None else None,
            'chosen_value_smoothed': float(smoothed_array[index]) if (index is not None and 0 <= index < len(smoothed_array)) else None,
        }
        try:
            self._process_debugs.append(debug)
        except Exception:
            pass

        return array_1_x, array_1_y, array_2_x, array_2_y, time_at_inflection, index

    def _get_data_csv_path(self):
        """Resolve path to data.csv.

        Preference order:
        1) If a data.csv exists next to the YAML config file (typical when running from an experiment folder), use it.
        2) Otherwise, use Image-Analysis output convention: <parent_dir>/<date_and_reactor>/output/<camera>/<channel>/data.csv
        """
        config_dir = os.path.dirname(os.path.abspath(self.config_file_path))
        local_data_path = os.path.join(config_dir, "data.csv")
        if os.path.isfile(local_data_path):
            return local_data_path

        parent_dir = self.user_parameters.get("parent_dir")
        date_and_reactor = self.user_parameters.get("date_and_reactor")
        camera = self.user_parameters.get("camera")
        channel = self.user_parameters.get("channel")
        experiment_root = os.path.join(parent_dir, date_and_reactor)
        return os.path.join(experiment_root, "output", camera, channel, "data.csv")

    def get_inflection_points(self, bandwidth=None):
        """
        Load data from data.csv and compute per-well inflection points (and plateaus).

        Args:
            bandwidth: LOESS bandwidth (fraction of data to use, 0.1-1.0). 
                      Defaults to self.loess_bandwidth (0.3)

        Expected CSV structure (from image_data pipeline):
          - Row 0: independent_variable row — time values start at column 7 (0-indexed)
          - Next 2*num_refs rows: reference well traces (ignored here)
          - Remaining rows: sample wells, one row per well; intensity data from column 7
        """
        if bandwidth is not None:
            self.loess_bandwidth = bandwidth
        data_file_path_str = self._get_data_csv_path()
        df = pd.read_csv(data_file_path_str)

        # Row 1 = independent_variable row (time in columns 7+)
        time_row = df.iloc[0, 7:]
        time_array = np.asarray(time_row.values, dtype=np.float64)

        rows = self.user_parameters.get("rows")
        cols = self.user_parameters.get("cols")
        num_refs = self.user_parameters.get("num_refs")

        # Sample wells start after row 0 (time) and 2*num_refs reference rows
        sample_start_row = 1 + 2 * num_refs
        intensity_block = df.iloc[sample_start_row:, 7:]
        intensity_arrays = np.asarray(intensity_block.values, dtype=np.float64)
        # One row per well; restrict to rows*cols wells
        num_wells = rows * cols
        intensity_arrays = intensity_arrays[:num_wells]

        _, axs = plt.subplots(rows, cols, figsize=(2*cols, 1.7*rows))

        # Reset storage arrays
        self.inflection_points = []
        self.inflection_point_values = []
        self.plateau_regions = []
        self.filtered_data = {}
        # Reset debug summaries
        self._process_debugs = []

        for i, array in enumerate(intensity_arrays):
            # Filter to first half of time
            filtered_time, filtered_array, cutoff_idx = self.filter_first_half(time_array, array)
            
            # Store filtered data
            self.filtered_data[i] = {
                'time': filtered_time,
                'intensity': filtered_array,
                'full_time': time_array,
                'full_intensity': array
            }
            
            # Detect plateau in filtered data
            plateau_start, plateau_end, plateau_time, plateau_value = self.detect_plateau(
                filtered_time, filtered_array
            )
            
            if plateau_start is not None:
                self.plateau_regions.append({
                    'well_idx': i,
                    'start_idx': plateau_start,
                    'end_idx': plateau_end,
                    'start_time': filtered_time[plateau_start],
                    'end_time': filtered_time[plateau_end],
                    'time': plateau_time,
                    'value': plateau_value
                })
            
            # Process data for inflection point (using filtered data)
            array_1_x, array_1_y, array_2_x, array_2_y, time_at_inflection, greater_than_index = self.process_data(filtered_time, filtered_array, bandwidth=self.loess_bandwidth)
            self.inflection_points.append(time_at_inflection)
            self.inflection_point_values.append(filtered_array[greater_than_index])

            # Locate the subplot row and column
            ax = axs[i // cols, i % cols]
            
            # Plot filtered data before and after the inflection point
            ax.plot(array_1_x, array_1_y, marker='x', lw=0.5, ms=2.0, color="#1f77b4", label='Before inflection')
            ax.plot(array_2_x, array_2_y, marker='x', lw=0.5, ms=2.0, color="#ff7f0e", label='After inflection')

            # Avoid 96 legends; show one legend on the first subplot only
            if i == 0:
                ax.legend(fontsize=6)
            
            # Mark the inflection point
            ax.plot(time_at_inflection, filtered_array[greater_than_index],
                    marker='*', lw=0.5, ms=15.0, color="#d62728", label='Inflection point')
            
            # Mark plateau region if found
            if plateau_start is not None:
                plateau_time_region = filtered_time[plateau_start:plateau_end+1]
                plateau_intensity_region = filtered_array[plateau_start:plateau_end+1]
                ax.plot(plateau_time_region, plateau_intensity_region,
                       marker='o', lw=2.0, ms=4.0, color="#2ca02c", alpha=0.6, label='Plateau')
                ax.axvline(plateau_time, color="#2ca02c", linestyle='--', alpha=0.7)
            
            # Mark cutoff point (first half boundary)
            ax.axvline(filtered_time[-1], color='gray', linestyle=':', alpha=0.5, label='Halfway point')
            
            ax.set_title(f"Well ({i//cols+1}, {i%cols+1})")
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Intensity')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    def update_bandwidth_and_reprocess(self, bandwidth):
        """
        Re-run inflection point detection with a new LOESS bandwidth value.
        Useful for interactive exploration with ipywidgets slider.
        
        Args:
            bandwidth: LOESS bandwidth (fraction of data to use, 0.1-1.0)
                      - Lower values: less smoothing, more detail
                      - Higher values: more smoothing, cleaner curves
        
        Returns:
            List of new inflection point times (same as self.inflection_points)
        """
        print(f"Reprocessing with LOESS bandwidth: {bandwidth:.2f}")
        self.loess_bandwidth = bandwidth
        
        # Re-run analysis with new bandwidth
        data_file_path_str = self._get_data_csv_path()
        df = pd.read_csv(data_file_path_str)
        
        time_row = df.iloc[0, 7:]
        time_array = np.asarray(time_row.values, dtype=np.float64)
        
        rows = self.user_parameters.get("rows")
        cols = self.user_parameters.get("cols")
        num_refs = self.user_parameters.get("num_refs")
        
        sample_start_row = 1 + 2 * num_refs
        intensity_block = df.iloc[sample_start_row:, 7:]
        intensity_arrays = np.asarray(intensity_block.values, dtype=np.float64)
        num_wells = rows * cols
        intensity_arrays = intensity_arrays[:num_wells]
        
        # Clear previous results
        self.inflection_points = []
        self.inflection_point_values = []
        self.plateau_regions = []
        
        # Re-process all wells with new bandwidth
        for i, array in enumerate(intensity_arrays):
            filtered_time, filtered_array, cutoff_idx = self.filter_first_half(time_array, array)
            
            plateau_start, plateau_end, plateau_time, plateau_value = self.detect_plateau(
                filtered_time, filtered_array
            )
            
            if plateau_start is not None:
                self.plateau_regions.append({
                    'well_idx': i,
                    'start_idx': plateau_start,
                    'end_idx': plateau_end,
                    'start_time': filtered_time[plateau_start],
                    'end_time': filtered_time[plateau_end],
                    'time': plateau_time,
                    'value': plateau_value
                })
            
            # Process with new bandwidth
            array_1_x, array_1_y, array_2_x, array_2_y, time_at_inflection, greater_than_index = self.process_data(
                filtered_time, filtered_array, bandwidth=bandwidth
            )
            self.inflection_points.append(time_at_inflection)
            self.inflection_point_values.append(filtered_array[greater_than_index])
        
        print(f"✓ Reprocessing complete. {len(self.inflection_points)} wells analyzed.")
        return self.inflection_points

    def _export_well_matrix_csv(self, values, filename, output_dir=None, save_label="CSV"):
        """Build CN x NN table from per-well values and write/append CSV.

        Keeps column headers as-is. If an existing file exists it is read and
        the new rows are appended (columns are aligned by header names).
        """
        rows = self.user_parameters.get("rows")
        cols = self.user_parameters.get("cols")
        CNs = self.user_parameters.get("CNs") or list(range(1, rows + 1))
        NNs = self.user_parameters.get("NNs") or list(range(1, cols + 1))

        # Build matrix and DataFrame
        inf_matrix = np.full((rows, cols), np.nan)
        for i, val in enumerate(values[: rows * cols]):
            r = i // cols
            c = i % cols
            inf_matrix[r, c] = val

        df = pd.DataFrame(inf_matrix, index=CNs[:rows], columns=NNs[:cols])

        if output_dir is None:
            config_dir = os.path.dirname(os.path.abspath(self.config_file_path))
            output_dir = os.path.abspath(os.path.join(config_dir, ".."))

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            try:
                existing = pd.read_csv(output_path, index_col=0)

                # Normalise numeric-like labels to integers where possible to avoid
                # pandas treating '1' (str) and 1 (int) as different labels which
                # leads to duplicate-looking CSV headers.

                def _to_int_label(x):
                    s = str(x).strip()
                    # Handles normal labels like "1", "11", "21"
                    if re.fullmatch(r"-?\d+", s):
                        return int(s)
                    return x

                def _to_int_labels(labels):
                    return pd.Index([_to_int_label(x) for x in labels])

                def add_empty_row(combined, row_label):
                    new_row = pd.Series(np.nan, index=combined.columns, dtype=float)
                    return pd.concat(
                        [combined, pd.DataFrame([new_row], index=[row_label])],
                        axis=0,
                        sort=False
                    )

                def next_duplicate_cn(combined, cn):
                    k = 1
                    while f"{cn}-{k}" in combined.index: k += 1
                    return f"{cn}-{k}"

                existing.index = _to_int_labels(existing.index)
                existing.columns = _to_int_labels(existing.columns)
                df.index = _to_int_labels(df.index)
                df.columns = _to_int_labels(df.columns)

                combined = existing.copy()

                # Add missing NN columns.
                for nn in df.columns:
                    if nn not in combined.columns: combined[nn] = np.nan

                # Important: make sure columns are unique.
                combined = combined.loc[:, ~combined.columns.duplicated()]

                for cn in df.index:
                    row_series = df.loc[cn]

                    if row_series.dropna().empty: continue

                    # If this CN is new, use the base row.
                    if cn not in combined.index:
                        target_cn = cn
                        combined = add_empty_row(combined, target_cn)

                    # If this CN already exists, this whole run becomes CN-1, CN-2, etc.
                    else:
                        target_cn = next_duplicate_cn(combined, cn)
                        combined = add_empty_row(combined, target_cn)

                    # Put all NN values from this run into the same target CN row.
                    for nn, v in row_series.items():
                        if pd.isna(v): continue

                        combined.at[target_cn, nn] = v
            except Exception: combined = df
        else: combined = df


        # Sort NN columns and CN index when possible (numeric sort if labels numeric)
        def _sort_labels(labels):
            lab_list = list(labels)

            def sort_key(lab):
                # Try integer label
                if isinstance(lab, (int, np.integer)): return (int(lab), 0, "")
                s = str(lab)

                # Match primary-secondary like '9-1'
                m = re.match(r"^(\d+)-(\d+)$", s)
                if m: return (int(m.group(1)), int(m.group(2)), "")
                
                # Match plain integer in string
                if re.fullmatch(r"\s*-?\d+\s*", s):
                    try:                return (int(s), 0, "")
                    except Exception:   pass
                    
                # Fallback: place non-numeric labels after numeric, sort by string
                return (10**9, 0, s)

            sorted_list = sorted(lab_list, key=sort_key)
            return sorted_list

        try:
            sorted_cols = _sort_labels(combined.columns)
            combined = combined.reindex(columns=sorted_cols)
        except Exception:
            pass

        try:
            sorted_index = _sort_labels(combined.index)
            combined = combined.reindex(index=sorted_index)
        except Exception:
            pass

        combined.to_csv(output_path, index_label="CN")
        print(f"Saved {save_label} to {output_path}")
        return output_path

    def export_inflection_points_csv(self, filename="inflection_points.csv", output_dir=None):
        """Export per-well inflection times as a CN x NN CSV."""
        if len(self.inflection_points) == 0:
            self.get_inflection_points()
        return self._export_well_matrix_csv(
            self.inflection_points, filename, output_dir, save_label="inflection points CSV"
        )

    def export_intensity_points_csv(self, filename="intensity_points.csv", output_dir=None):
        """Export per-well intensity values at inflection points as a CN x NN CSV."""
        if len(self.inflection_point_values) == 0:
            self.get_inflection_points()
        return self._export_well_matrix_csv(
            self.inflection_point_values, filename, output_dir, save_label="intensity points CSV"
        )

    def plot_inflection_points(self):
        '''
        Create a summary plot showing inflection points and plateaus across all wells.
        This creates a grid visualisation where each well's inflection point and plateau
        are displayed in a heatmap format.
        '''
        rows = self.user_parameters.get("rows")
        cols = self.user_parameters.get("cols")
        
        if len(self.inflection_points) == 0:
            print("No inflection points found. Run get_inflection_points() first.")
            return
        
        # Prepare data matrices
        inflection_matrix = np.full((rows, cols), np.nan)
        inflection_value_matrix = np.full((rows, cols), np.nan)

        for i, inflection_time in enumerate(self.inflection_points):
            row_idx = i // cols
            col_idx = i % cols
            inflection_matrix[row_idx, col_idx] = inflection_time

        for i, inflection_value in enumerate(self.inflection_point_values):
            row_idx = i // cols
            col_idx = i % cols
            inflection_value_matrix[row_idx, col_idx] = inflection_value

        try:
            from .visualiser import plot_inflection_matrices
            plot_inflection_matrices(inflection_matrix, inflection_value_matrix)
        except Exception:
            # Fallback: simple plotting if import fails
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            im1 = ax1.imshow(inflection_matrix, cmap='viridis', aspect='auto')
            ax1.set_title('Inflection Points (Time in minutes)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Row')
            plt.colorbar(im1, ax=ax1)
            im2 = ax2.imshow(inflection_value_matrix, cmap='plasma', aspect='auto')
            ax2.set_title('Intensity at Inflection Point', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Row')
            plt.colorbar(im2, ax=ax2)
            plt.tight_layout()
            plt.show()

        # Print summary statistics (unchanged)
        print("\n=== Summary Statistics ===")
        print(f"Total wells analyzed: {len(self.inflection_points)}")
        print(f"Wells with detected plateaus: {len(self.plateau_regions)}")
        if len(self.inflection_points) > 0:
            print(f"\nInflection Points:")
            print(f"  Mean: {np.nanmean(inflection_matrix):.2f} min")
            print(f"  Median: {np.nanmedian(inflection_matrix):.2f} min")
            print(f"  Std: {np.nanstd(inflection_matrix):.2f} min")
            print(f"  Min: {np.nanmin(inflection_matrix):.2f} min")
            print(f"  Max: {np.nanmax(inflection_matrix):.2f} min")

        if len(self.inflection_point_values) > 0:
            print(f"\nInflection Point Values:")
            print(f"  Mean: {np.nanmean(inflection_value_matrix):.4f}")
            print(f"  Median: {np.nanmedian(inflection_value_matrix):.4f}")
            print(f"  Std: {np.nanstd(inflection_value_matrix):.4f}")
            print(f"  Min: {np.nanmin(inflection_value_matrix):.4f}")
            print(f"  Max: {np.nanmax(inflection_value_matrix):.4f}")

        if len(self.plateau_regions) > 0:
            plateau_times = [p['time'] for p in self.plateau_regions]
            print(f"\nPlateau Start Times:")
            print(f"  Mean: {np.mean(plateau_times):.2f} min")
            print(f"  Median: {np.median(plateau_times):.2f} min")
            print(f"  Std: {np.std(plateau_times):.2f} min")
            print(f"  Min: {np.min(plateau_times):.2f} min")
            print(f"  Max: {np.max(plateau_times):.2f} min")

        return



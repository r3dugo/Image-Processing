import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _to_numeric_labels(labels):
    new = []
    for x in labels:
        s = str(x)
        if re.fullmatch(r"\s*-?\d+\s*", s):
            try:
                new.append(int(s))
            except Exception:
                new.append(x)
        else:
            new.append(x)
    return pd.Index(new)


def plot_inflection_matrices(inflection_matrix, inflection_value_matrix=None, row_labels=None, col_labels=None, show_annotations=True):
    """Render inflection time (and optionally intensity) matrices as heatmaps.

    Args:
        inflection_matrix: 2D numpy array (rows x cols) with inflection times (floats or NaN)
        inflection_value_matrix: optional 2D numpy array same shape with intensity values
        row_labels: optional list-like labels for rows
        col_labels: optional list-like labels for columns
    """
    rows, cols = inflection_matrix.shape

    if inflection_value_matrix is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(inflection_matrix, cmap='viridis', aspect='auto')
        ax.set_title('Inflection Points (Time in minutes)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Time (min)', rotation=270, labelpad=20)

        # Set tick positions and labels if provided
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        if col_labels is not None and len(col_labels) == cols:
            ax.set_xticklabels([str(x) for x in col_labels])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        if row_labels is not None and len(row_labels) == rows:
            ax.set_yticklabels([str(x) for x in row_labels])

        if show_annotations:
            for i in range(rows):
                for j in range(cols):
                    if not np.isnan(inflection_matrix[i, j]):
                        ax.text(j, i, f'{inflection_matrix[i, j]:.1f}', ha='center', va='center', color='white', fontsize=8)

        plt.tight_layout()
        plt.show()
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(inflection_matrix, cmap='viridis', aspect='auto')
    ax1.set_title('Inflection Points (Time in minutes)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Time (min)', rotation=270, labelpad=20)
    # Set ticks and labels
    ax1.set_xticks(np.arange(cols))
    ax1.set_yticks(np.arange(rows))
    if col_labels is not None and len(col_labels) == cols:
        ax1.set_xticklabels([str(x) for x in col_labels])
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    if row_labels is not None and len(row_labels) == rows:
        ax1.set_yticklabels([str(x) for x in row_labels])
    if show_annotations:
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(inflection_matrix[i, j]):
                    ax1.text(j, i, f'{inflection_matrix[i, j]:.1f}', ha='center', va='center', color='white', fontsize=8)

    im2 = ax2.imshow(inflection_value_matrix, cmap='plasma', aspect='auto')
    ax2.set_title('Intensity at Inflection Point', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Intensity', rotation=270, labelpad=20)
    # Set ticks and labels
    ax2.set_xticks(np.arange(cols))
    ax2.set_yticks(np.arange(rows))
    if col_labels is not None and len(col_labels) == cols:
        ax2.set_xticklabels([str(x) for x in col_labels])
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    if row_labels is not None and len(row_labels) == rows:
        ax2.set_yticklabels([str(x) for x in row_labels])
    if show_annotations:
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(inflection_value_matrix[i, j]):
                    ax2.text(j, i, f'{inflection_value_matrix[i, j]:.3f}', ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_inflection_csv(csv_path, show_annotations=True):
    """Read a CSV produced by `export_inflection_points_csv` and visualise inflection times.

    The CSV is expected to have a first column as CN index and NN columns. Duplicate CN
    rows (e.g., '9-1') are treated as separate rows and displayed in order.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # Try to normalize numeric-like labels
    df.index = _to_numeric_labels(df.index)
    df.columns = _to_numeric_labels(df.columns)

    # Convert to float matrix
    mat = df.values.astype(float)

    plot_inflection_matrices(mat, inflection_value_matrix=None, row_labels=df.index, col_labels=df.columns, show_annotations=show_annotations)

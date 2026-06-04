# Image Processing / Inflection Analysis Pipeline

This project processes plate images from high-throughput experiments, extracts per-well intensity traces, detects inflection points, normalises intensity values, and combines the processed outputs into ML-ready CSV targets.

## Folder Structure

### `experiments/`

Contains the raw and processed experiment folders. Each plate folder typically contains:

* `image-data-vars.yaml`
* `image-data.ipynb`
* `processing-vars.yaml`
* `processing.ipynb`
* `output/inflection_points.csv`
* `output/intensity_points.csv`

Current active plate folders include `plate-1` through `plate-6` and `uniform-plate`. The `uniform-plate` experiment is used as the reference plate for intensity correction. Older archived analyses are stored under `experiments/old-analyses/`.

### `image_data/`

The original image-processing package used to load images, masks, and plate data. This code handles lower-level image analysis, including image datasets, masks, normalised intensity data, and per-well/per-plate data structures.

The original image-analysis software was authored as described in the provided `image_data/README.txt`: the photoreactor image-processing code was developed in the Bernhard Lab, with the image-processing software written by Andrew Kubaney and inspired by earlier Mathematica notebooks by Eric Lopato and Savannah Talledo, with assistance from Tomek Kowalewski.

Main files include:

* `experiment.py`
* `image.py`
* `image_dataset.py`
* `intensity_data.py`
* `per_well_data.py`
* `per_plate_data.py`

### `inflection_finder/`

Contains the main inflection analysis code. This module reads processed image intensity data, detects inflection points, extracts intensity values at those points, and visualises CN × NN plate matrices.

Main files include:

* `experiment.py`: inflection detection and CSV export logic
* `normaliser.py`: helper functions for normalising and combining CSV outputs
* `visualiser.py`: heatmap visualisation for inflection/intensity matrices

### `inflection_processor/`

Contains the final data-normalisation and consolidation workflow. This folder combines plate-level CSVs, corrects intensities using the uniform plate, and derives the final inflection-rate target for ML modelling.

Main files and outputs include:

* `normalise.ipynb`
* `output/intensity/combined_intensity.csv`
* `output/intensity/combined_corrected_intensity.csv`
* `output/inflection/combined_inflection.csv`
* `output/rate.csv`
* `output/corrected_rate.csv`

## Workflow

1. Run image processing for each plate in `experiments/`.
2. Use `inflection_finder` to detect inflection times and intensities at inflection.
3. Use the uniform plate to correct intensity values across plate positions.
4. Consolidate all plate CSVs into combined CN × NN matrices.
5. Combine inflection time and corrected intensity into a single inflection-rate target.
6. Use the final rate CSV as the ML prediction target.

## Final ML Target

The final target is the inflection rate, calculated from:

```text
inflection rate = (start intensity - intensity at inflection) / inflection time
```

If the inflection time equals the maximum experiment time, the rate is set to `0`, representing no detected inflection within the experiment window.

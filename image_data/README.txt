********************************************************************************
IMAGE ANALYSIS SOFTWARE
********************************************************************************

********************************************************************************
TABLE OF CONTENTS
********************************************************************************

1. OVERVIEW

2. REQUIRED MODULES AND DOCUMENTATION

3. INPUT DIRECTORY STRUCTURE

4. CONFIG FILE OVERVIEW

5. CALIBRATION: CONFIG FILE OPTIONS

6. ANALYSIS: CONFIG FILE OPTIONS

7. OUTPUT DIRECTORY STRUCTURE

8. CSV OUTPUT STRUCTURE

9. BRIEF OVERVIEW OF UNDERLYING CODE STRUCTURE

********************************************************************************
1. OVERVIEW:
********************************************************************************

This software handles image processing for the parallel hydrogen evolution and
other photoreactors developed in Stefan Bernhard's laboratory.

The photoreactors were envisioned, designed, and built by Stefan Bernhard, 
Kuba Kowalewski, Jaqueline Lewis, and Eric Lopato.

The code for the photoreactors was developed by Stefan Bernhard, 
Kuba Kowalewski, Jaqueline Lewis, Eric Lopato, and Andrew Kubaney.

The image processing software was writen by Andrew Kubaney, inspired by
Wolfram Mathematica notebooks written by Eric Lopato and Savannah Talledo
with assistance from Tomek Kowalewski.

********************************************************************************
2. REQUIRED MODULES AND DOCUMENTATION
********************************************************************************

  Below is a list of the modules required to run this image analysis software.
This list excludes modules included in the default version of Python.

  Required Modules:
    numpy
    cv2
    matplotlib
    scipy
    pyyaml

  Asociated Documentation for Modules:
    numpy: https://numpy.org/doc/1.24/
    cv2: https://docs.opencv.org/4.x/
    matplotlib: https://matplotlib.org/stable/index.html
    scipy: https://docs.scipy.org/doc/scipy/
    pyyaml: https://pyyaml.org/wiki/PyYAMLDocumentation

********************************************************************************
3. INPUT DIRECTORY STRUCTURE
********************************************************************************
      
  The input folder structure for calibration and analysis is similar. First,
you will need to make an overall folder for all projects. Within that folder,
you should make a folder for your project (a project folder should contain 
sub-folders of many experiments). Within the project folder, you should place
folders for each experiment (this folder will be referred to as the experiment
folder in the subsequent discussion).

  Within the experiment folder, there should be two folders. There should be a
folder labeled "input". Inside the "input" folder, you should create two 
folders: "images" and "masks". Inside "images", you should place folders
containing the images to be analyzed (for example, a folder named "top" with the
images from the top camera). Inside "masks", you should place folders containing
the corresponding masks. These folders should have the same names as the image
folders. Note, image datasets will only have their intensities processed and 
post-processed if there is a corresponding folder in "masks" with the mask(s)
need for image processing. This allows the user to only process certain image
datasets (by not including masks for the datasets that the user does not want
to process). Masks are not needed for movie generation, which is independent.

  There is one difference between analysis and calibration: the number of
masks inside each folder within the "masks" folder. For analysis, each folder
(for example, "top"), should only have one mask (to be used for all images). For
calibration, the folder inside "masks" should have one mask per image, and the
name of each mask image file must match the name of the corresponding image file
in the sub-folder of "images".

  Note, the picture times (as representing by the file names) must have EVENLY
spaced picture times (down to the second). This is due to how the rate 
calculation is implemented.

********************************************************************************
4. CONFIG FILE OVERVIEW
********************************************************************************

  After placing your images and masks in the proper folders, you should edit the
appropriate config file to contain the details of your experiment. Config files
should be placed in the "Bernhard_Lab_Image_Processing" directory. Two example 
config files have been included: "hydrogen_analysis_config.yaml" (for analysis) 
and "hydrogen_calibration_config.yaml" (for calibration). Descriptions of all
input variables can be found in the config files. Once the config file has been
edited, open the "run_image_processing.ipynb", and change the "config_file_name"
variable to match the file name of your config file. Then, run all of the cells
in the notebook.

********************************************************************************
5. CALIBRATION: CONFIG FILE OPTIONS
********************************************************************************

  An example of a config file for calibration 
("hydrogen_calibration_config.yaml") has been included below. To perform
calibration, you will have to change the variables to reflect the parameters of
your experiment. Once calibration is complete, you should add the calibration
constants to the analysis config file.

hydrogen_calibration_config.yaml
--------------------------------------------------------------------------------
# This variable specifies which type of image processing should be used.
# Currently, the options are "hydrogen" for hydrogen film-related image
# processing, and "intensity", for any image processing that only focuses on
# the intensities.
image_processing_type: "hydrogen"

# This variable specifies which type of image processing should be used.
# Currently, the options are "hydrogen" for hydrogen film-related image
# processing, and "intensity", for any image processing that only focuses on
# the intensities.
experiment_type: "hydrogen"

# This variable should be set to True if the image processing is for 
# calibration, and False otherwise.
is_calibration: True

# If this variable is set to True, figures will be displayed in
# image_processing.ipynb. Otherwise, no figures will be display.
#
# NOTE: aside from the figures related to the masks, the figures generated
# in this notebook are never saved.
display_figures: True

# This variable specifies the absolute parent directory that contains the
# data for all experiments.
parent_data_directory: "D:\\Research with Bernhard Lab\\Image Analysis Test Data"

# This variable specifes the name of the project. A sub-folder of the same
# name should exist in the parent_data_directory.
project_name: "Photoreactor Calibration"

# This variable specifes the name of the experiment. A sub-folder of the same
# name should exist with the project_name directory. This directory should
# contain an "input" sub-directory (the structure of this directory is
# specified in the README.txt). The "input" sub-directory will contain all
# image/mask inputs for the image processing.
experiment_name: "calibration_test"

# This variable specifies the number of columns in each row of wells, including 
# the reference wells. The rows are ordered from top to bottom in the mask
# image. The total number of wells must match the number of wells in the
# input mask image.
columns_per_row: [12, 12, 12, 12, 12, 12, 12, 12, 12, 1]

# This variable is used to specify the wells that are reference or empty
# wells in the masks.
#
# Reference wells are specified by mapping a string of the form "(a,b)", where
# a is the 0-indexed row and b is the 0-indexed column, to "ref" (for reference
# wells) or "empty" (for empty wells).
#
# NOTE: if you have multiple masks, the reference and empty wells should be in 
# the same positions.
#
# NOTE: "empty" wells may cause the intensity/post-processed data figures in the
# notebook to display in an unusual way (this is because the ordering does not
# account for columns, so labeling a well as "empty" will cause the sample 
# labels in the same row to shift as if the well was not there).
ref_or_empty: {"(9,0)": "ref"}

# This variable specifies the image channels to be used for analysis. Multiple
# channels can be passed in, and each channel will be analyzed separately.
# There are four options: "blue", "green", "red", and "average". "blue", 
# "green", and "red" analyze the blue, green, and red channels of the input
# images respectively. "average" causes the blue, green, and red channels
# to be averaged together and then analyzed.
analysis_channels: ["average"]

# If this variable is True, the reference and sample wells will be normalized
# by the average intensity of the reference wells in each image. If this
# variable is False, normalization by the average intensity of the reference
# wells will not occur.
normalize_by_reference: True

# This variable dictates the type of fit to use for calibration. Currently, the
# options are "quadratic" and "linear".
calibration_type: "quadratic"

# This variable specifies the wells to be used for the calibration. Each entry
# in this list should be a length-two list, with the first entry specifying the
# 0-indexed well row of the calibration well, and the second entry specifying
# the 0-indexed well column of the calibration well. Every well used for
# calibration should have an entry in this list.
# 
# NOTE: These coordinates refer to the sample well ordering (the ordering of the
# sample wells when the reference wells are not considered).
calibration_wells: [[0,2], [0,7], [0,10], [1,5], [1,11], [2,0], [2,3], [2,9], 
                    [3,6], [3,8], [4,1], [4,3], [5,2], [5,4], [5,7], [6,0], 
                    [6,5], [6,10], [7,1], [7,8], [8,6], [8,11]]

# This variable specifies the requested outputs for the image processing
# for each image dataset. NOTE, each mask should have an entry specified
# in this dictionary. Each sub-dictionary maps the desired output
# property. "True" means the property will be saved in the output csv.
# "False" means the property will not be saved.
requested_outputs: 
  {"top": {"Normalized Raw Reference Intensity" : True, 
           "Normalized Reference Intensity": True, 
           "Normalized Sample Intensity": True,
           "Means of Intensities": True,
           "Sample Standard Deviations of Intensities": True,
           "a": True,
           "b": True,
           "c": True,
           "R Squared": True}}

# This variable specifies the volume of the vials. The units are specified
# below with the "volume_units" variable.
vial_volume: 1.1

# This variable specifies the amount of hydrogen injected for each image. The
# units are specified below with the "volume_units" variable.
injected_hydrogen_volumes: [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.4, 1.9, 2.9]

# This variable specifes the volume units used for the calculations. The valid
# options are "L", "mL", and "uL" (micro liter).
volume_units: "mL"

# This list specifies the image datasets that should be used to generate movies.
# Each entry in this list should correspond to the name of the image directory
# for the desired image dataset. The list can be empty, in which case no
# movies will be generated.
movies_to_be_generated: ["top"]

# This variable specifies the frames per second to use in the generated movies.
frames_per_second: 15
--------------------------------------------------------------------------------

********************************************************************************
6. ANALYSIS: CONFIG FILE OPTIONS
********************************************************************************

  Similarly, a config file for analysis ("hydrogen_analysis_config.yaml") has 
been included below. To perform analysis, you will have to change the variables 
to reflect the parameters of your experiment.

hydrogen_analysis_config.yaml
--------------------------------------------------------------------------------
# This variable specifies which type of image processing should be used.
# Currently, the options are "hydrogen" for hydrogen film-related image
# processing, and "intensity", for any image processing that only focuses on
# the intensities.
image_processing_type: "hydrogen"

# This variable should be set to True if the image processing is for 
# calibration, and False otherwise.
is_calibration: False

# If this variable is set to True, figures will be displayed in
# image_processing.ipynb. Otherwise, no figures will be display.
#
# NOTE: aside from the figures related to the masks, the figures generated
# in this notebook are never saved.
display_figures: True

# This variable specifies the absolute parent directory that contains the
# data for all experiments.
parent_data_directory: "D:\\Research with Bernhard Lab\\Image Analysis Test Data"

# This variable specifes the name of the project. A sub-folder of the same
# name should exist in the parent_data_directory.
project_name: "Photoreactor Calibration"

# This variable specifes the name of the experiment. A sub-folder of the same
# name should exist with the project_name directory. This directory should
# contain an "input" sub-directory (the structure of this directory is
# specified in the README.txt). The "input" sub-directory will contain all
# image/mask inputs for the image processing.
experiment_name: "012023uniformG_test"

# This variable specifies the number of columns in each row of wells, including 
# the reference wells. The rows are ordered from top to bottom in the mask
# image. The total number of wells must match the number of wells in the
# input mask image.
columns_per_row: [12, 12, 12, 12, 12, 12, 12, 12, 12, 1]

# This variable is used to specify the wells that are reference or empty
# wells in the masks.
#
# Reference wells are specified by mapping a string of the form "(a,b)", where
# a is the 0-indexed row and b is the 0-indexed column, to "ref" (for reference
# wells) or "empty" (for empty wells).
#
# NOTE: if you have multiple masks, the reference and empty wells should be in 
# the same positions.
#
# NOTE: "empty" wells may cause the intensity/post-processed data figures in the
# notebook to display in an unusual way (this is because the ordering does not
# account for columns, so labeling a well as "empty" will cause the sample 
# labels in the same row to shift as if the well was not there).
ref_or_empty: {"(9,0)": "ref"}

# This variable specifies the image channels to be used for analysis. Multiple
# channels can be passed in, and each channel will be analyzed separately.
# There are four options: "blue", "green", "red", and "average". "blue", 
# "green", and "red" analyze the blue, green, and red channels of the input
# images respectively. "average" causes the blue, green, and red channels
# to be averaged together and then analyzed.
analysis_channels: ["average"]

# If this variable is True, the reference and sample wells will be normalized
# by the average intensity of the reference wells in each image. If this
# variable is False, normalization by the average intensity of the reference
# wells will not occur.
normalize_by_reference: True

# This variable specifies the requested outputs for the image processing
# for each image dataset. NOTE, each mask should have an entry specified
# in this dictionary. Each sub-dictionary maps the desired output
# property. "True" means the property will be saved in the output csv.
# "False" means the property will not be saved.
requested_outputs: 
  {"top": {"Normalized Raw Reference Intensity" : True, 
           "Normalized Reference Intensity": True, 
           "Normalized Sample Intensity": True,
           "Hydrogen Mole Fraction": True,
           "Corrected Hydrogen Volume": True,
           "Hydrogen Amount": True,
           "Max Hydrogen Amount": True,
           "Hydrogen Rate": True,
           "Max Hydrogen Rate": True,
           "Hydrogen Incubation Time": True,
           "Hydrogen Plateau Time": True}}

# This dictionary stores the calibration coefficients for each reactor.
# The entries should be labeled with the reactor name. The sub-dictionaries
# contain the coefficients for the quadratic or linear fit. For a quadratic fit, 
# there should be three entries in the sub-dictionary: "a", "b", and "c" (where 
# the fit is ax^2 + bx + c). For a linear fit, there should be two entries in
# the sub-dictionary: "a" and "b" (where the fit is ax + b).
# Once a calibration is complete, the constants should be added here.
reactor_calibration_coefficients:
  {"gastly": {"a": 0.943543, "b": -1.22109, "c": 0.973533},
   "rattata": {"a": 0.968379, "b": -1.26827, "c": 0.982771},
   "snorlax": {"a": 1.0029, "b": -1.26733, "c": 0.980978},
   "torchic": {"a": 1.3072, "b": -1.3966, "c": 1.0055}}

# The name of the reactor used for this experiment. This must match one of the
# entries in reactor_calibration_coefficients.
reactor_name: "gastly"

# This variable specifies the volume of the vials. The units are specified
# below with the "volume_units" variable.
vial_volume: 1.1

# This variable specifies the per-well solution volume. It should have the
# same shape as the sample well array. The units are specified below with the 
# "volume_units" variable.
solution_volumes: [[0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42], 
                   [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
                   [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
                   [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
                   [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
                   [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
                   [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
                   [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
                   [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42]]

# This variable specifies the time units used for the calculations. The valid
# options are "second", "minute", and "hour".
time_units: "minute"

# This variable specifes the volume units used for the calculations. The valid
# options are "L", "mL", and "uL" (micro liter).
volume_units: "mL"

# This variable specifes the amount units used for the calculations. The valid
# options are "mol", "mmol", and "umol" (micro mol).
amount_units: "umol"

# Variables used for the Savitsky-Golay Filter used to compute a smoothed
# derivative for the hydrogen rate.

# This variable determines the window size for the Savitzky-Golay filter, which
# is used to compute a smoothed derivative of the hydrogen amount (in order
# to compute the rate). This value should be less than or equal to the
# number of images.
savitzky_golay_window: 22

# This variable determines the polynomial degree for the Savitzky-Golay filter,
# which is used to compute a smoothed derivative of the hydrogen amount (in 
# order to compute the rate). This value should be less than the
# savitzky_golay_window value.
savitzky_golay_degree: 2

# This variable specifies the fraction of the per-well max hydrogen that
# must be reached for the incubation period to be considered complete.
# It should be greater than or equal to 0, less than or equal to 1, and
# less than the plateau_threshold_fraction.
incubation_threshold_fraction: 0.05

# This variable specifies the fraction of the per-well max hydrogen that
# must be reached for the plateau period to begin. It should be greater
# than 0, less than or equal to 1, and greater than the 
# incubation_threshold_fraction.
plateau_threshold_fraction: 0.90

# This list specifies the image datasets that should be used to generate movies.
# Each entry in this list should correspond to the name of the image directory
# for the desired image dataset. The list can be empty, in which case no
# movies will be generated.
movies_to_be_generated: ["top", "bottom"]

# This variable specifies the frames per second to use in the generated movies.
frames_per_second: 15
--------------------------------------------------------------------------------

********************************************************************************
7. OUTPUT DIRECTORY STRUCTURE
********************************************************************************

  The output folder will have three folders: "data", "masks", and "movies".
  
  For each mask folder in the "masks" sub-folder of "input", a sub-folder of the
same name will be created in "data". In this sub-folder, there will be
sub-folders for each analysis channel specified in the config file ("average", 
"blue", "green", or "red"). Each of these folders will contain a CSV file, with
the outputs requested by the user. 

  For each mask folder in the "masks" sub-folder of "input", a sub-folder of the
same name will be created in "masks". For each input mask, 4 validation figures
will be generated (one is an image of the mask, one displays the overall 
ordering of the wells, one displays the ordering of the wells in the reference 
grid, and the last displays the ordering of the wells in the sample grid).

  For each input image folder specified by the user, a sub-folder of the same
name will be created in "movies". Within this folder, a movie of the same name
will be created.
  
  Also note, the config file used for the image processing will be duplicated
and saved in the experiment folder (not the output folder).

********************************************************************************
8. CSV OUTPUT STRUCTURE
********************************************************************************

  The CSVs for analysis and calibration are structured the same. The column
headers are as follows: "Type", "Computed from Sample or Reference Wells", 
"Against Independent Variable", "Property Name", "Property Units", 
"Well Row (0-indexed)", "Well Column (0-indexed)", "Data".

  "Type": one of "independent_variable" (for the data on the independent 
    variable), "per_well" (for data that is per-well), or "per_plate" (for data 
    that is one point per-image or one data point).
  
  "Computed from Sample or Reference Wells": "NONE" for the independent_variable
    data. Otherwise, "sample" indicates that the data was calculated from the
    sample well intensities, and "ref" indicates that the data was calculated
    from the reference well intensities.
  
  "Against Independent Variable": "TRUE" indicates that there will be one data
    point per independent variable point. "FALSE" indicates that there will
    only be one data point.

  "Property Name": the name of the property.

  "Property Units": the units of the property. "NONE" for no units.

  "Well Row (0-indexed)": for per-well properties, the 0-indexed row of the
    well (either the sample grid or reference grid row depending on the value of
    the "Computed from Sample or Reference Wells" column). "NONE" for
    non-per-well data.

  "Well Column (0-indexed)": for per-well properties, the 0-indexed column of 
    the well (either the sample grid or reference grid column depending on the 
    value of the "Computed from Sample or Reference Wells" column). "NONE" for
    non-per-well data.
  
  "Data": If "Against Independent Variable" is "TRUE", there will be one data
    point. If "Against Independent Variable" if "FALSE", there will be one data 
    point per independent variable point, spread out over multiple columns.

********************************************************************************
9. BRIEF OVERVIEW OF UNDERLYING CODE STRUCTURE
********************************************************************************

  The file "run_image_processing.ipynb" contains the highest level code, which
constructs the Experiment object from the config file and input data and 
performs all image processing. The rest of the code is located in the
"image_processing" folder. The "experiment.py" code contains the definition of
the Experiment class. This class handles the high level functionality of
loading data and exporting data, as well as manipulating Mask/MaskDataset, 
ImageDataset, NormalizedIntensityData, and the post-processed data 
(PerPlateData and PerWellData). "image.py" defines the functionality of Image
and Mask (a sub-class of Image). "image_dataset.py" defines the functionality
of "ImageDataset" and "MaskDataset" (a sub-class of "ImageDataset"). Mask or
MaskDataset and ImageDataset are combined to process the images, producing
a NormalizedIntensityData (defined in "intensity_data.py"). The transformations
of the intensity data are defined in "per_plate_data.py" (PerPlateData) and
"per_well_data.py" (PerWellData). This transformations are used to compute
results from the intensities (hydrogen amounts, rates, calibration constants, 
etc.).
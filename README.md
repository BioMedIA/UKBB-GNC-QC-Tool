# SegQC: Ready-to-use Quality Control Tool for Automated Segmentations in Medical Imaging

SegQC is initially designed to do quality control of abdominal organ segmentations in population/epidemiological studies, specifically UK Biobank (UKBB) and German National Cohort (GNC). However, it can also be used as a general-purpose QC tool for checking automated segmentations in medical imaging.

### Disclaimer:
This software has been developed for research purposes only, and hence should not be used as a diagnostic tool. In no event shall the authors or distributors be liable to any direct, indirect, special, incidental, or consequential damages arising of the use of this software, its documentation, or any derivatives thereof, even if the authors have been advised of the possibility of such damage.


## Getting Started

After obtaining the datasets (e.g. UKBB and/or GNC whole-body data) and their corresponding segmentations, SegQC can be set up in 3 easy steps.

## Setting up SegQC

### Step 1: 
Ensure that all scripts (segQC.py, st_rerun.py, SessionState.py) and the config file (segQC_config.json) are in the same directory.


### Step 2: Edit segQC_config.json
Edit the config file based on the paths to your data and segmentations. Here are some helpful tips:

- The file structure for images and segmentations should follow below:
```
images        => /root/img_folder/subject_id/img_basename.nii.gz
segmentations => /root/seg_folder/subject_id/seg_basename.nii.gz
```
- "img_folder" and "seg_folder" could be set as the same path.
- Chrome is recommended as browser for SegQC.
- Please ensure that only 1 SegQC application is running at a time as concurrent runs (e.g. in multiple tabs in Chrome) may lead to unexpected behaviour.

```
"img_folder"     = Directory path for images
"seg_folder"     = Directory path for segmentations/predictions
"out_folder"     = Directory path for QC results
"img_basenames"  = List of file basenames for images to be visualized in "img_folder" (without .nii.gz extensions)
"seg_basename"   = File basename for segmentations (without .nii.gz extension)
"class_names"    = Names of the classes 
"qc_options"     = List of QC result options (e.g. ["Pass", "Fail"])
"qc_csv_basename"= File basename for QC results (e.g. "quality_check.csv")
```

### Step 3: Run SegQC

```
streamlit run segQC.py
```

## Acknowledgements

Original scripts of st_rerun.py and SessionState.py are available at
```
st_rerun.py     =  https://gist.github.com/tvst/ef477845ac86962fa4c92ec6a72bb5bd
SessionState.py =  https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
```

## Contact
If you have any questions, please reach out to me over email or Twitter.

- Email: t.kart@imperial.ac.uk
- Twitter: [@turkaykart](https://twitter.com/turkaykart)

# Analytical SEC Gaussian Fitting and Molecular Weight Estimation

## Overview

This package provides a workflow for fitting Gaussian peaks to
analytical size-exclusion chromatography (SEC) traces and estimating
molecular weights based on calibration standards.

The software was designed for analytical SEC experiments performed on:

-   Superdex 200 Increase 10/300 GL columns
-   ÄKTA Go
-   ÄKTA Purifier

The program fits Gaussian peaks to chromatographic data and calculates
molecular weights using:
1. The fitted Gaussian midpoints (elution
volumes)
1. A column calibration curve derived from known protein
standards


------------------------------------------------------------------------

## Features

-   Automatic detection of injection volumes from ÄKTA CSV files
-   Interactive selection of injection peaks for analysis
-   Gaussian fitting of chromatographic peaks
-   Molecular weight estimation from calibration standards
-   Designed specifically for Superdex 200 Increase datasets
-   Works with CSV exports from ÄKTA systems

------------------------------------------------------------------------

## Installation

Clone the repository:

    git clone github.com/james-r-barrett/SEC_fit
    cd SEC_fit

------------------------------------------------------------------------

## Usage

Run the analysis script from the command line:

    python main.py trace.csv

Where:

-   trace.csv is the chromatogram exported from the ÄKTA system

### Workflow

1.  The program reads the chromatogram CSV file.
2.  Injection volumes are automatically detected.
3.  The user is prompted to choose which injection to analyse.
4.  Peaks are selected interactively.
5.  Gaussian models are fitted to selected peaks.
6.  Elution midpoints are used to calculate molecular weights using the
    calibration curve.

------------------------------------------------------------------------

## Calibration Information

The repository includes a calibration.csv file generated from a
calibration run using:

Sigma Aldrich Gel Filtration Calibration Kit
https://www.sigmaaldrich.com/GB/en/product/sial/69385

### Column Used

Superdex 200 Increase 10/300 GL

### Calibration Standards

-   Thyroglobulin (670 kDa)
-   Gamma-globulins (150 kDa)
-   Ovalbumin (44.3 kDa)
-   Ribonuclease A Type I-A (13.7 kDa)
-   pABA (0.137 kDa)

Void volume was determined using Blue Dextran 2000.

------------------------------------------------------------------------

## Supported Input Data

The program is designed for CSV exports from:

-   ÄKTA Go
-   ÄKTA Purifier

Expected contents include:

-   Elution volume (mL)
-   Absorbance (mAU)
-   Injection markers

------------------------------------------------------------------------

## Assumptions and Limitations

-   Calibration is specific to the included column and standards
-   Molecular weights are estimates based on SEC behaviour
-   Non-globular or disordered proteins may deviate from calibration
-   Accurate baseline subtraction improves fitting results
-   Gaussian models assume approximately symmetric peaks
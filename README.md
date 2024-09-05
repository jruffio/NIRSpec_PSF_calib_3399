# NIRSpec_PSF_calib_3399

Scripts for the reduction of the JWST/NIRSpec data for the cycle 2 PSF calibration program #3399.

-----------------------
Install the JWST pipeline
-----------------------
To install the JWST pipeline, we recommend you use the STScI python environment: 

https://github.com/spacetelescope/stenv/releases

These scripts were tested with JWST pipeline version 1.15.1: 

stenv-Linux-X64-py3.11-2024.08.14.yaml
/!\ py3.11 was necessary to "pip install mkl-service" later

then install the following packages within this new environment:

pip install mkl-service
pip install py
pip install PyQt5
pip install webbpsf

-----------------------
Install BREADS
-----------------------
There is no installation guide yet, just download the repo and make sure it is visible by your Python environment.
https://github.com/jruffio/breads

-----------------------
Download the data
-----------------------
To download the uncalibrated data, download the following:
https://github.com/spacetelescope/jwst_mast_query

Use the jwst_query.cfg file in this repository. 
First, modify the output directory toward the top of the file: 

outrootdir: "/output/path/to/define/PSF_calib_3399/20240830_mast/"

Then run:

jwst_download.py  --config /path/to/config/file/jwst_query.cfg

-----------------------
Reduce the uncal data
-----------------------

Run the script:

20240902_make_calibrated_point_cloud.py 
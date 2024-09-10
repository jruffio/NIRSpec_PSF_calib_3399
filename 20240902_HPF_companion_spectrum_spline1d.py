import os
import time
import jwst
import multiprocessing as mp
import numpy as np
import datetime
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator
from copy import copy
from glob import glob
import astropy.units as u
from astropy import constants as const
from astropy.table import Table
from  scipy.interpolate import interp1d

# Print out what pipeline version we're using
print('JWST pipeline version',jwst.__version__)
from breads.jwst_tools.reduction_utils import find_files_to_process
from breads.jwst_tools.reduction_utils import compute_normalized_stellar_spectrum
from breads.jwst_tools.reduction_utils import compute_starlight_subtraction
from breads.jwst_tools.reduction_utils import get_combined_regwvs
from breads.jwst_tools.reduction_utils import get_2D_point_cloud_interpolator
from breads.instruments.jwstnirspec_cal import build_cube

from breads.utils import rotate_coordinates

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar

import mkl
mkl.set_num_threads(1)

if __name__ == "__main__":

    numthreads = 16

    # Main directory for the data and reduced products
    targetdir = "/stow/jruffio/data/JWST/nirspec/PSF_calib_3399/"
    # Output PSF models
    output_dir = "/stow/jruffio/data/JWST/nirspec/PSF_calib_3399/outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # exit()

    crds_dir = "/stow/jruffio/data/JWST/crds_cache/"
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data-1.3.0"

    # Sampling of the wavelength directions:
    # ra_vec = np.arange(-0.5,0.5,0.05)
    # dec_vec = np.arange(-0.5,0.5,0.05)
    ra_vec = np.arange(-1.5,1.5,0.05)
    dec_vec = np.arange(-1.5,1.5,0.05)


    # Target J1757132;  Ks = 11.15
    # Target HD1634665;  Ks = 6.33
    # for targetname in ["HD1634665","J1757132"]:
    # for targetname in ["J1757132"]:
    for targetname in ["HD1634665"]:

        if targetname == "J1757132":
            obsnum = "obsnum01"
            mask_charge_transfer_radius = None
        elif targetname == "HD1634665":
            obsnum = "obsnum02"
            mask_charge_transfer_radius = 0.16

        for detector in ["nrs1","nrs2"]:
        # for detector in ["nrs2"]:
            # for filter in ["G140H","G235H","G395H"]:
            for filter in ["G140H"]:

                if filter == "G140H":
                    filename_filter = "jw0339900"+obsnum[-1::]+"001_03102_000*_"+detector+"_cal.fits"
                elif filter == "G235H":
                    filename_filter = "jw0339900"+obsnum[-1::]+"001_03104_000*_"+detector+"_cal.fits"
                elif filter == "G395H":
                    filename_filter = "jw0339900"+obsnum[-1::]+"001_03106_000*_"+detector+"_cal.fits"

                stage2_clean_outdir = os.path.join(targetdir,targetname, filter+"_stage2_cleaned")
                utils_dir = os.path.join(targetdir,targetname, filter+"_utils")
                utils_before_cleaning_dir = os.path.join(targetdir,targetname, filter+"_utils_before_cleaning")
                cleaned_cal_files = find_files_to_process(stage2_clean_outdir, filetype=filename_filter)

                splitbasename = os.path.basename(cleaned_cal_files[0]).split("_")
                filename_suffix = "_webbpsf"
                poly2d_centroid_filename = os.path.join(utils_before_cleaning_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_poly2d_centroid" + filename_suffix + ".txt")
                # fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_fitpsf" + filename_suffix + ".fits")
                cube_filename = os.path.join(output_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+"_spectral_cube_ish.fits")

                mypool = mp.Pool(processes=numthreads)

                ##############
                ## Retrieve the coordinates calibration, ie the polynomial coefficient of the centroid trend
                output = np.loadtxt(poly2d_centroid_filename, delimiter=' ')
                poly_p_RA, poly_p_dec = output[0],output[1]

                ##############
                ## Compute normalized spectrum of the star in the FOV
                combined_star_func = compute_normalized_stellar_spectrum(cleaned_cal_files, utils_dir, crds_dir,
                                                    coords_offset=(poly_p_RA,poly_p_dec),
                                                    wv_nodes=None,
                                                    mask_charge_transfer_radius = mask_charge_transfer_radius,
                                                    mppool=mypool,
                                                    ra_dec_point_sources=None, overwrite=False)

                ##############
                ## Fit the normalized spectrum of the star everywhere by modulating the continuum
                # This does a few things:
                # - subtracts the starlight everywhere as kind of fancy high-pass filter. New files saved in [utils_dir]/starsub1d.
                # - Updates the bad pixel map in the original data object with a sigma clipping
                dataobj_list = compute_starlight_subtraction(cleaned_cal_files, utils_dir, crds_dir, combined_star_func=combined_star_func,
                                              coords_offset=(poly_p_RA,poly_p_dec), mppool=mypool)

                ##############
                ## Interpolation the wavelength dimension onto a regular wavelength grid and return combined dataset object
                regwvs_combdataobj = get_combined_regwvs(dataobj_list,
                                                         mask_charge_transfer_radius=mask_charge_transfer_radius,
                                                         use_starsub1d=True)
                regwvs_combdataobj.set_coords2ifu()


                # Fit a model PSF (WebbPSF) to the combined point cloud of dataobj_list
                # Save output as fitpsf_filename
                init_centroid = (0,0)
                ann_width = None
                padding = 0.0
                sector_area = None
                debug_init,debug_end = None,None
                # debug_init,debug_end = 1000,1100
                aper_radius = 0.15

                if 1:#len(glob(cube_filename)) !=1:
                    # Load the webbPSF model (or compute if it does not yet exist)
                    webbpsf_reload = regwvs_combdataobj.reload_webbpsf_model()
                    if webbpsf_reload is None:
                        webbpsf_reload = regwvs_combdataobj.compute_webbpsf_model(
                            wv_sampling=regwvs_combdataobj.wv_sampling,
                            image_mask=None,
                            pixelscale=0.1, oversample=10,
                            parallelize=False, mppool=mypool,
                            save_utils=True)
                    wpsfs, wpsfs_header, wepsfs, webbpsf_wvs, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale = webbpsf_reload
                    webbpsf_X = np.tile(webbpsf_X[None, :, :], (wepsfs.shape[0], 1, 1))
                    webbpsf_Y = np.tile(webbpsf_Y[None, :, :], (wepsfs.shape[0], 1, 1))

                    flux_cube,fluxerr_cube,ra_grid, dec_grid = \
                        build_cube(regwvs_combdataobj, # combined point cloud
                                   wepsfs, webbpsf_X, webbpsf_Y, # webbPSF model for flux extraction
                                   ra_vec, dec_vec, # spatial sampling of final cube
                                   out_filename=cube_filename,linear_interp=True,mppool=None,aper_radius=aper_radius,
                                   debug_init=debug_init,debug_end=debug_end)  # min max wavelength indices for partial extraction

                if 1: # plotting
                    fontsize=12

                    if debug_init is not None or debug_end is not None:
                        cube_filename = cube_filename.replace(".fits", "_from{0}to{1}.fits".format(debug_init, debug_end))
                    hdulist = fits.open(cube_filename)
                    flux_cube = hdulist[0].data
                    fluxerr_cube = hdulist['FLUXERR_CUBE'].data
                    ra_grid = hdulist['RA'].data
                    dec_grid = hdulist['DEC'].data
                    wv_sampling = hdulist['WAVE'].data
                    hdulist.close()

                    flux_im = np.nanmean(flux_cube, axis=0)
                    r_grid = np.sqrt(ra_grid**2+dec_grid**2)
                    sep_min,sep_max =  0.4,0.45
                    where_annulus = np.where((r_grid>sep_min)*(r_grid<sep_max))

                    flux_calib_filename = os.path.join(output_dir,"jw03399001001_" + splitbasename[1] + "_" + splitbasename[3] +
                                                       "_flux_calib_IWA{0:.2f}_OWA{1:.2f}.txt".format(0, aper_radius))
                    flux_calib_poly_coefs = np.loadtxt(flux_calib_filename, delimiter=' ')
                    print("Load flux calibration", flux_calib_poly_coefs)



                    plt.figure(1)
                    plt.title("PSF-subtracted image " + detector)
                    plt.imshow(flux_im, origin="lower")
                    # plt.clim([0,5e-11])

                    now = datetime.datetime.now()
                    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

                    out_filename = os.path.join(output_dir, formatted_datetime+"_" + targetname+"_"+filter+"_"+detector+ "_HPF_cube_ish.png")
                    print("Saving " + out_filename)
                    plt.savefig(out_filename, dpi=300)

                    plt.figure(2)
                    # plt.plot(wv_sampling, (flux_cube[:, where_annulus[0], where_annulus[1]]))
                    residuals_mJy = 1e9*(flux_cube[:, where_annulus[0], where_annulus[1]]) * np.polyval(flux_calib_poly_coefs, wv_sampling)[:,None]
                    plt.plot(wv_sampling, residuals_mJy,linewidth=1,linestyle=":")
                    plt.plot(wv_sampling, np.nanstd(residuals_mJy,axis=1),linewidth=5,linestyle="-")
                    plt.plot(wv_sampling, np.nanstd(residuals_mJy,axis=1),linewidth=5,linestyle="-")
                    plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
                    plt.ylabel("mJy", fontsize=fontsize)
                    plt.legend()

                    out_filename = os.path.join(output_dir, formatted_datetime+"_" + targetname+"_"+filter+"_"+detector+ "_HPF_cube_ish_spectra.png")
                    print("Saving " + out_filename)
                    plt.savefig(out_filename, dpi=300)
                    # plt.show()

exit()

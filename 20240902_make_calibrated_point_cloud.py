import os
import time
import jwst
import multiprocessing as mp
import numpy as np
import datetime
# Print out what pipeline version we're using
print('JWST pipeline version',jwst.__version__)
from breads.jwst_tools.reduction_utils import find_files_to_process
from breads.jwst_tools.reduction_utils import run_stage1,run_stage2
from breads.jwst_tools.reduction_utils import run_coordinate_recenter
from breads.jwst_tools.reduction_utils import run_noise_clean
from breads.jwst_tools.reduction_utils import compute_normalized_stellar_spectrum
from breads.jwst_tools.reduction_utils import compute_starlight_subtraction
from breads.jwst_tools.reduction_utils import get_combined_regwvs
from breads.jwst_tools.reduction_utils import get_2D_point_cloud_interpolator
from breads.jwst_tools.reduction_utils import save_combined_regwvs

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patheffects as PathEffects

import mkl
mkl.set_num_threads(1)

if __name__ == "__main__":

    numthreads = 16

    # Main directory for the data and reduced products
    targetdir = "/stow/jruffio/data/JWST/nirspec/PSF_calib_3399/"
    # Output PSF models
    out_PSF_models = "/stow/jruffio/data/JWST/nirspec/PSF_calib_3399/combined_2d_point_cloud/"
    if not os.path.exists(out_PSF_models):
        os.makedirs(out_PSF_models)
    # directory where to look for the uncal files
    uncaldir = os.path.join(targetdir,"20240830_mast/03399")
    # Filename filter for glob
    uncal_filename_filter = "jw0339900*001_0310*_000*_nrs*_uncal.fits"

    crds_dir = "/stow/jruffio/data/JWST/crds_cache/"
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data-1.3.0"

    # if "03102" in file:
    #     stage1dir = os.path.join(stage1dir, "G140H_stage1")
    # elif "03102" in file:
    #     stage1dir = os.path.join(stage1dir, "G235H_stage1")
    # elif "03102" in file:
    #     stage1dir = os.path.join(stage1dir, "G395H_stage1")

    # Target J1757132;  Ks = 11.15
    # Target HD1634665;  Ks = 6.33
    # for targetname in ["HD1634665","J1757132"]:
    for targetname in ["J1757132"]:

        if targetname == "J1757132":
            obsnum = "obsnum01"
            mask_charge_transfer_radius = None
            model_charge_transfer=False
            IWA=0.1
        elif targetname == "HD1634665":
            obsnum = "obsnum02"
            mask_charge_transfer_radius = 0.16
            model_charge_transfer=True
            IWA=0.3

        for detector in ["nrs1","nrs2"]:
            for filter in ["G140H","G235H","G395H"]:

                if filter == "G140H":
                    filename_filter = "jw0339900"+obsnum[-1::]+"001_03102_000*_"+detector+"_uncal.fits"
                elif filter == "G235H":
                    filename_filter = "jw0339900"+obsnum[-1::]+"001_03104_000*_"+detector+"_uncal.fits"
                elif filter == "G395H":
                    filename_filter = "jw0339900"+obsnum[-1::]+"001_03106_000*_"+detector+"_uncal.fits"

                uncal_files = find_files_to_process(os.path.join(uncaldir, obsnum), filetype=filename_filter)
                stage1_outdir = os.path.join(targetdir, targetname, filter + "_stage1")
                stage2_outdir = os.path.join(targetdir,targetname, filter+"_stage2")
                utils_before_cleaning_dir = os.path.join(targetdir,targetname, filter+"_utils_before_cleaning")
                stage1_clean_outdir = os.path.join(targetdir,targetname, filter+"_stage1_cleaned")
                stage2_clean_outdir = os.path.join(targetdir,targetname, filter+"_stage2_cleaned")
                utils_dir = os.path.join(targetdir,targetname, filter+"_utils")

                ##############
                ## run_stage1
                rate_files = run_stage1(uncal_files, stage1_outdir, overwrite=False,maximum_cores="1")

                ##############
                ## run_stage2
                cal_files = run_stage2(rate_files, stage2_outdir, skip_cubes=True, overwrite=False)

                ##############
                ## Recenter coordinate system
                # First pass at computing the centroid
                poly_p_RA,poly_p_dec = run_coordinate_recenter(cal_files,utils_before_cleaning_dir,crds_dir,
                                         init_centroid = (0,0),wv_sampling=None, N_wvs_nodes=40,
                                         mask_charge_transfer_radius = mask_charge_transfer_radius,
                                         IWA=IWA,OWA=1.0,
                                         debug_init=None,debug_end=None,
                                         numthreads = numthreads,
                                         save_plots=True,
                                         overwrite=False,
                                         filename_suffix = "_webbpsf_init")
                ## Rerun the centroid calculation using the last one as the starting point, see init_centroid
                poly_p_RA,poly_p_dec = run_coordinate_recenter(cal_files,utils_before_cleaning_dir,crds_dir,
                                         init_centroid = (poly_p_RA[-1],poly_p_dec[-1]),wv_sampling=None, N_wvs_nodes=40,
                                         mask_charge_transfer_radius = mask_charge_transfer_radius,
                                         IWA=IWA,OWA=1.0,
                                         debug_init=None,debug_end=None,
                                         numthreads = numthreads,
                                         save_plots=True,
                                         overwrite=False,
                                         filename_suffix = "_webbpsf")

                ##############
                ## remove 1/f noise in rate files
                new_rate_files = run_noise_clean(rate_files, stage2_outdir, stage1_clean_outdir,crds_dir,
                                              N_nodes=40,
                                              model_charge_transfer=model_charge_transfer, utils_dir=utils_before_cleaning_dir,
                                              coords_offset=(poly_p_RA,poly_p_dec), overwrite=False)

                ##############
                ## run_stage2 with cleaned rate files
                cleaned_cal_files = run_stage2(new_rate_files, stage2_clean_outdir, skip_cubes=True, overwrite=False)

                mypool = mp.Pool(processes=numthreads)

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
                                                         use_starsub1d=False)
                # # regwvs_combdataobj_starsub1d = get_combined_regwvs(dataobj_list, use_starsub1d=True)
                out_filename = os.path.join(out_PSF_models,targetname+"_"+filter+"_"+detector+"_2d_point_cloud.fits")
                save_combined_regwvs(regwvs_combdataobj, out_filename)

                ##############
                ## Generate 2D interpolator object at a given wavelength (here the median wavelength)
                wv0 = np.nanmedian(regwvs_combdataobj.wavelengths)
                pointcloud_interp = get_2D_point_cloud_interpolator(regwvs_combdataobj,wv0)

                ##############
                ## Example for how to evaluate the interpolator object
                dra = 0.01
                ddec = 0.01
                ra_vec = np.arange(-2., 2., dra)
                dec_vec = np.arange(-2., 2., ddec)
                ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)

                myinterpim = pointcloud_interp(ra_grid, dec_grid)

                ##############
                ## Plot the PSF at this wavelength and save a QL
                fontsize=12
                rad = 2 # arcsec

                myextent = [ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2., dec_vec[-1] + ddec / 2.]
                plt.imshow(np.log10(myinterpim), interpolation="nearest", origin="lower", extent=myextent)
                # plt.clim([0,np.nanmax(myinterpim)/4.0])
                txt = plt.text(0.03, 0.99,
                               "log10(inteprolated flux) \nCentroid corrected \n$\lambda$={0:0.3f} $\mu$m".format(wv0),
                               fontsize=fontsize, ha='left',
                               va='top', transform=plt.gca().transAxes, color="black")
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                for x in np.arange(-rad, rad, 0.1):
                    plt.plot([x, x], [-rad, rad], color="grey", linewidth=1, alpha=0.5)
                    plt.plot([-rad, rad], [x, x], color="grey", linewidth=1, alpha=0.5)
                plt.xlim([-rad, rad])
                plt.ylim([-rad, rad])
                plt.gca().set_aspect('equal')
                plt.xlabel("$\Delta$RA (as)", fontsize=fontsize)
                plt.ylabel("$\Delta$Dec (as)", fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.gca().invert_xaxis()

                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

                out_filename = os.path.join(out_PSF_models, formatted_datetime+"_" + targetname+"_"+filter+"_"+detector+"_2D_point_cloud_QL.png")
                print("Saving " + out_filename)
                plt.savefig(out_filename, dpi=300)
                # plt.show()

exit()

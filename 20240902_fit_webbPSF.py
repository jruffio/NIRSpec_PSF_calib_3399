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
from breads.jwst_tools.reduction_utils import run_stage1,run_stage2
from breads.jwst_tools.reduction_utils import run_coordinate_recenter
from breads.jwst_tools.reduction_utils import run_noise_clean
from breads.jwst_tools.reduction_utils import compute_normalized_stellar_spectrum
from breads.jwst_tools.reduction_utils import compute_starlight_subtraction
from breads.jwst_tools.reduction_utils import get_combined_regwvs
from breads.jwst_tools.reduction_utils import get_2D_point_cloud_interpolator
from breads.jwst_tools.reduction_utils import save_combined_regwvs
from breads.instruments.jwstnirspec_cal import fitpsf

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

    # Target J1757132;  Ks = 11.15
    # Target HD1634665;  Ks = 6.33
    for targetname in ["HD1634665","J1757132"]:
    # for targetname in ["J1757132"]:

        if targetname == "J1757132":
            obsnum = "obsnum01"
            mask_charge_transfer_radius = None
            model_charge_transfer=False
            IWA,OWA=0.1,0.5
            flux4plotting = 0.5 #Jy
            # https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/1757132_stiswfc_005.fits
            calspec_file = os.path.join(output_dir,"1757132_stiswfc_005.fits")
        elif targetname == "HD1634665":
            obsnum = "obsnum02"
            mask_charge_transfer_radius = 0.16
            model_charge_transfer=True
            IWA,OWA=0.3,0.5
            flux4plotting = 10 #Jy
            # https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/hd163466_stis_006.fits
            calspec_file = os.path.join(output_dir, "hd163466_stis_006.fits")

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
                fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_fitpsf" + filename_suffix + ".fits")

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
                                                         use_starsub1d=False)
                # # regwvs_combdataobj_starsub1d = get_combined_regwvs(dataobj_list, use_starsub1d=True)

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

                # Fit a model PSF (WebbPSF) to the combined point cloud of dataobj_list
                # Save output as fitpsf_filename
                init_centroid = (0,0)
                ann_width = None
                padding = 0.0
                sector_area = None
                debug_init,debug_end = None,None
                # debug_init,debug_end = 1000,1100
                orginal_bad_pixels = copy(regwvs_combdataobj.bad_pixels)
                where_center_disk = regwvs_combdataobj.where_point_source((0.0, 0.0), IWA)
                regwvs_combdataobj.bad_pixels[where_center_disk] = np.nan

                if len(glob(fitpsf_filename)) !=1:
                    fitpsf(regwvs_combdataobj, wepsfs, webbpsf_X, webbpsf_Y, out_filename=fitpsf_filename, IWA=0.0, OWA=OWA,
                           mppool=mypool, init_centroid=init_centroid, ann_width=ann_width, padding=padding,
                           sector_area=sector_area, RDI_folder_suffix=filename_suffix,
                           rotate_psf=regwvs_combdataobj.east2V2_deg,
                           flipx=True, psf_spaxel_area=(wpsf_pixelscale) ** 2, debug_init=debug_init, debug_end=debug_end)

                with fits.open(fitpsf_filename) as hdulist:
                    _bestfit_paras = hdulist[0].data


                ##############
                ## Generate 2D interpolator object at a given wavelength (here the median wavelength)
                wv_sampling = regwvs_combdataobj.wv_sampling
                # wv0 = np.nanmedian(regwvs_combdataobj.wavelengths)
                # wv0_id = np.argmin(np.abs(wv_sampling-wv0))
                wv0_id = 1050
                wv0 = wv_sampling[1050]
                regwvs_combdataobj.bad_pixels = orginal_bad_pixels
                pointcloud_interp = get_2D_point_cloud_interpolator(regwvs_combdataobj,wv0)

                ##############
                ## Example for how to evaluate the interpolator object
                dra = 0.01
                ddec = 0.01
                ra_vec = np.arange(-2., 2., dra)
                dec_vec = np.arange(-2., 2., ddec)
                ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)


                # Save plot
                if 1:
                    color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
                    fontsize = 12
                    rad = 0.75 # arcsec

                    plt.figure(figsize=(12, 6))
                    print(_bestfit_paras.shape)

                    a0, a, xc, yc, th = _bestfit_paras[0,wv0_id,:]

                    wX, wY, wZ = webbpsf_X[wv0_id, :, :].ravel(), webbpsf_Y[wv0_id, :, :].ravel(),wepsfs[wv0_id, :, :].ravel()
                    wX, wY = rotate_coordinates(wX, wY, -regwvs_combdataobj.east2V2_deg, flipx=True)

                    wherepsffinite = np.where(np.isfinite(wZ))
                    wX, wY, wZ = wX[wherepsffinite], wY[wherepsffinite], wZ[wherepsffinite]
                    webbpsf_interp = LinearNDInterpolator((wX, wY), wZ, fill_value=0.0)

                    myinterpim = pointcloud_interp(ra_grid, dec_grid)
                    best_fit_model = a * webbpsf_interp(ra_grid - xc, dec_grid - yc)

                    plt.figure(figsize=(12,6))
                    gs = gridspec.GridSpec(2,3, height_ratios=[0.05,1], width_ratios=[1,1,1])
                    gs.update(left=0.05, right=0.95, bottom=0.19, top=0.85, wspace=0.0, hspace=0.0)
                    myextent = [ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2., dec_vec[-1] + ddec / 2.]

                    plt.subplot(gs[1, 0])
                    plt1 = plt.imshow(myinterpim*1e9, interpolation="nearest", origin="lower", extent=myextent)
                    plt.clim([0,flux4plotting])
                    txt = plt.text(0.03, 0.99,
                                   "flux \n$\lambda$={0:0.3f} $\mu$m".format(wv0),
                                   fontsize=fontsize, ha='left',
                                   va='top', transform=plt.gca().transAxes, color="black")
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                    cbax = plt.subplot(gs[0, 0])
                    cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')

                    plt.subplot(gs[1, 1])
                    plt1 = plt.imshow(best_fit_model*1e9, interpolation="nearest", origin="lower", extent=myextent)
                    plt.clim([0,flux4plotting])
                    txt = plt.text(0.03, 0.99,
                                   "best fit model \n$\lambda$={0:0.3f} $\mu$m".format(wv0),
                                   fontsize=fontsize, ha='left',
                                   va='top', transform=plt.gca().transAxes, color="black")
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                    cbax = plt.subplot(gs[0,1])
                    cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')

                    plt.subplot(gs[1, 2])
                    plt1 = plt.imshow((myinterpim-best_fit_model)*1e9, interpolation="nearest", origin="lower", extent=myextent)
                    plt.clim([-flux4plotting/2.0,flux4plotting/2.0])
                    txt = plt.text(0.03, 0.99,
                                   "residuals \n$\lambda$={0:0.3f} $\mu$m".format(wv0),
                                   fontsize=fontsize, ha='left',
                                   va='top', transform=plt.gca().transAxes, color="black")
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                    cbax = plt.subplot(gs[0, 2])
                    cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')

                    for k in range(3):
                        plt.subplot(gs[0, k])
                        cb.set_label(r'Flux (Jy)', labelpad=5, fontsize=fontsize)
                        # cb.set_ticks([0, 50,100,150,200])  # xticks[1::]
                        plt.gca().tick_params(axis='x', labelsize=fontsize)
                        plt.gca().tick_params(axis='y', labelsize=fontsize)

                    for k in range(3):
                        plt.subplot(gs[1, k])
                        circle = plt.Circle((0, 0), IWA, facecolor='#FFFFFF00', edgecolor='red')#,zorder=0
                        plt.gca().add_patch(circle)
                        circle = plt.Circle((0, 0), OWA, facecolor='#FFFFFF00', edgecolor='red')#,zorder=0
                        plt.gca().add_patch(circle)
                        for x in np.arange(-rad, rad, 0.1):
                            plt.plot([x, x], [-rad, rad], color="grey", linewidth=1, alpha=0.5)
                            plt.plot([-rad, rad], [x, x], color="grey", linewidth=1, alpha=0.5)
                        plt.xlim([-rad, rad])
                        plt.ylim([-rad, rad])
                        plt.gca().set_aspect('equal')
                        plt.xlabel("$\Delta$RA (as)", fontsize=fontsize)
                        plt.gca().tick_params(axis='x', labelsize=fontsize)
                        if k==0:
                            plt.ylabel("$\Delta$Dec (as)", fontsize=fontsize)
                            plt.gca().tick_params(axis='y', labelsize=fontsize)
                        else:
                            plt.yticks([])
                        plt.gca().invert_xaxis()
                    # plt.show()

                    plt.tight_layout()

                    now = datetime.datetime.now()
                    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

                    out_filename = os.path.join(output_dir, formatted_datetime+"_" + targetname+"_"+filter+"_"+detector + "_webbpsf_fit.png")
                    print("Saving " + out_filename)
                    plt.savefig(out_filename, dpi=300)

                    plt.figure(figsize=(12, 10))
                    plt.subplot(4, 1, 1)
                    wv_sampling_hd = np.arange(wv_sampling[0],wv_sampling[-1],np.nanmedian(wv_sampling)/20000)
                    plt.plot(wv_sampling_hd, combined_star_func(wv_sampling_hd), linestyle="--", color=color_list[2],
                             label="Continuum normalized star", linewidth=1)

                    plt.xlim([wv_sampling[0], wv_sampling[-1]])
                    plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
                    plt.ylabel("Normalized flux", fontsize=fontsize)
                    plt.gca().tick_params(axis='x', labelsize=fontsize)
                    plt.gca().tick_params(axis='y', labelsize=fontsize)
                    plt.legend(loc="upper right")



                    plt.subplot(4, 1, 2)
                    hdulist = fits.open(calspec_file,ignore_missing_simple=True)
                    stis_table = Table(fits.getdata(calspec_file,1))
                    stis_wvs =  (np.array(stis_table["WAVELENGTH"]) *u.Angstrom).to(u.um).value # angstroms -> mum
                    stis_spec = np.array(stis_table["FLUX"]) * u.erg /u.s/u.cm**2/u.Angstrom # erg s-1 cm-2 A-1
                    stis_spec = stis_spec.to(u.W*u.m**-2/u.um)
                    stis_spec_Fnu = stis_spec*(stis_wvs*u.um)**2/const.c # from Flambda back to Fnu
                    stis_spec_Fnu = stis_spec_Fnu.to(u.MJy).value
                    plt.plot(stis_wvs, stis_spec_Fnu * 1e9, linestyle=":", color="black", label="CALSPEC", linewidth=2)

                    interp_stis_spec = interp1d(stis_wvs,stis_spec_Fnu,bounds_error=False,fill_value=np.nan)(wv_sampling)
                    flux_calib = interp_stis_spec / _bestfit_paras[0, :, 0]
                    wherefinite = np.where(np.isfinite(flux_calib))
                    flux_calib_poly_coefs = np.polyfit(wv_sampling[wherefinite],flux_calib[wherefinite] ,deg=1)
                    print("flux calibration", flux_calib_poly_coefs)
                    flux_calib_filename = os.path.join(output_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_flux_calib_IWA{0}_OWA{1}.txt".format(IWA,OWA))
                    np.savetxt(flux_calib_filename, [flux_calib_poly_coefs], delimiter=' ')
                    if len(glob(flux_calib_filename)) == 1:
                        flux_calib_poly_coefs = np.loadtxt(flux_calib_filename, delimiter=' ')
                        print("Load flux calibration", flux_calib_poly_coefs)


                    plt.plot(wv_sampling, _bestfit_paras[0, :, 0] * 1e9, linestyle="-", color=color_list[0],
                             label="Fixed centroid", linewidth=1,alpha=0.2)
                    plt.plot(wv_sampling, _bestfit_paras[0, :, 1] * 1e9, linestyle="--", color=color_list[2],
                             label="Free centroid", linewidth=1,alpha=0.2)

                    plt.plot(wv_sampling, _bestfit_paras[0, :, 0] * 1e9*np.polyval(flux_calib_poly_coefs, wv_sampling), linestyle="-", color=color_list[0],
                             label="Fixed centroid (flux calibrated)", linewidth=1)
                    plt.plot(wv_sampling, _bestfit_paras[0, :, 1] * 1e9*np.polyval(flux_calib_poly_coefs, wv_sampling), linestyle="--", color=color_list[2],
                             label="Free centroid (flux calibrated)", linewidth=1)

                    plt.xlim([wv_sampling[0], wv_sampling[-1]])
                    plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
                    plt.ylabel("Flux density (Jy)", fontsize=fontsize)
                    plt.gca().tick_params(axis='x', labelsize=fontsize)
                    plt.gca().tick_params(axis='y', labelsize=fontsize)
                    plt.legend(loc="upper right")

                    plt.subplot(4, 1, 3)
                    plt.plot(wv_sampling, _bestfit_paras[0, :, 2], label="bestfit centroid")
                    plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
                    plt.ylabel("$\Delta$RA (as)", fontsize=fontsize)
                    plt.gca().tick_params(axis='x', labelsize=fontsize)
                    plt.gca().tick_params(axis='y', labelsize=fontsize)
                    plt.legend(loc="upper right")

                    plt.subplot(4, 1, 4)
                    plt.plot(wv_sampling, _bestfit_paras[0, :, 3], label="bestfit centroid")
                    plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
                    plt.ylabel("$\Delta$Dec (as)", fontsize=fontsize)
                    plt.gca().tick_params(axis='x', labelsize=fontsize)
                    plt.gca().tick_params(axis='y', labelsize=fontsize)

                    plt.tight_layout()

                    now = datetime.datetime.now()
                    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

                    out_filename = os.path.join(output_dir, formatted_datetime+"_" + targetname+"_"+filter+"_"+detector+ "_spectra.png")
                    print("Saving " + out_filename)
                    plt.savefig(out_filename, dpi=300)
                    # plt.show()

exit()

#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

from IPython.display import Latex, Math

import numpy as np
from celerite.modeling import Model
from scipy.signal import savgol_filter

import exoplanet as xo
from exoplanet.gp import terms, GP
from exoplanet.gp.terms import TermSum
from exoplanet.utils import eval_in_model
from exoplanet.sampling import PyMC3Sampler
from exoplanet.utils import get_samples_from_trace
import corner
import pymc3 as pm
from astropy.io import fits
import theano.tensor as tt
import theano
import pandas as pd
import math
import astropy
import os
import scipy
from matplotlib import cm

# import ipywidgets
#
# from tqdm import tqdm

from astroquery.mast import Observations, Catalogs

from astropy.stats import LombScargle

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

np.random.seed(42)

from scipy.interpolate import splrep, splev

import lightkurve as lk

from scipy import optimize









def get_fwhm(x_in,y_in):
    x_interp = np.linspace(np.min(x_in),np.max(x_in),1000)
    y_interp = np.interp(x_interp, x_in, y_in)

    half = np.max(y_in) / 2.0
    signs = np.sign(np.add(y_interp, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    where_zero_crossings = np.where(zero_crossings)[0]

    x1 = np.mean(x_interp[where_zero_crossings[0]:where_zero_crossings[0] + 1])
    x2 = np.mean(x_interp[where_zero_crossings[1]:where_zero_crossings[1] + 1])

    return x2 - x1
def aflare1(t, tpeak, fwhm, ampl):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Use this function for fitting classical flares with most curve_fit
    tools.
    Note: this model assumes the flux before the flare is zero centered
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1.,
                                (t > tpeak)],
                            [lambda x: (_fr[0]+                       # 0th order
                                        _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                        _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                        _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                        _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                             lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                        _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                            ) * np.abs(ampl) # amplitude

    return flare
def aflare1_upper(t, tpeak, fwhm, ampl):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Use this function for fitting classical flares with most curve_fit
    tools.
    Note: this model assumes the flux before the flare is zero centered
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053+0.008, -0.175084+0.032, -2.24588+0.039, -1.12498+0.016]
    _fd = [0.689008+0.0008, -1.60053+0.003, 0.302963+0.0009, -0.278318+0.0007]

    flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1.,
                                (t > tpeak)],
                            [lambda x: (_fr[0]+                       # 0th order
                                        _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                        _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                        _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                        _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                             lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                        _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                            ) * np.abs(ampl) # amplitude

    return flare
def aflare1_lower(t, tpeak, fwhm, ampl):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Use this function for fitting classical flares with most curve_fit
    tools.
    Note: this model assumes the flux before the flare is zero centered
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053-0.008, -0.175084-0.032, -2.24588-0.039, -1.12498-0.016]
    _fd = [0.689008-0.0008, -1.60053-0.003, 0.302963-0.0009, -0.278318-0.0007]

    flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1.,
                                (t > tpeak)],
                            [lambda x: (_fr[0]+                       # 0th order
                                        _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                        _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                        _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                        _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                             lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                        _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                            ) * np.abs(ampl) # amplitude

    return flare
def aflare(t, p):
    """
    This is the Analytic Flare Model from the flare-morphology paper.
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Note: this model assumes the flux before the flare is zero centered
    Note: many sub-flares can be modeled by this method by changing the
    number of parameters in "p". As a result, this routine may not work
    for fitting with methods like scipy.optimize.curve_fit, which require
    a fixed number of free parameters. Instead, for fitting a single peak
    use the aflare1 method.
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    p : 1-d array
        p == [tpeak, fwhm (units of time), amplitude (units of flux)] x N
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    """
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    #Nflare = int( np.floor( (len(p)/3.0) ) )
    Nflare = 2

    flare = np.zeros_like(t)
    # compute the flare model for each flare
    for i in range(Nflare):
        outm = np.piecewise(t, [(t<= p[0+i*3]) * (t-p[0+i*3])/p[1+i*3] > -1.,
                                (t > p[0+i*3])],
                            [lambda x: (_fr[0]+                             # 0th order
                                        _fr[1]*((x-p[0+i*3])/p[1+i*3])+     # 1st order
                                        _fr[2]*((x-p[0+i*3])/p[1+i*3])**2.+  # 2nd order
                                        _fr[3]*((x-p[0+i*3])/p[1+i*3])**3.+  # 3rd order
                                        _fr[4]*((x-p[0+i*3])/p[1+i*3])**4. ),# 4th order
                             lambda x: (_fd[0]*np.exp( ((x-p[0+i*3])/p[1+i*3])*_fd[1] ) +
                                        _fd[2]*np.exp( ((x-p[0+i*3])/p[1+i*3])*_fd[3] ))]
                            ) * p[2+i*3] # amplitude
        flare = flare + outm

    return flare
def keplerflux_to_keplermag(keplerflux, f12=1.74e5):
    '''This converts the Kepler flux in electrons/sec to Kepler magnitude.

    The kepler mag/flux relation is::

        fkep = (10.0**(-0.4*(kepmag - 12.0)))*f12
        f12 = 1.74e5 # electrons/sec

    Parameters
    ----------

    keplerflux : float or array-like
        The flux value(s) to convert to magnitudes.

    f12 : float
        The flux value in the Kepler band corresponding to Kepler mag = 12.0.

    Returns
    -------

    np.array
        Magnitudes in the Kepler band corresponding to the input `keplerflux`
        flux value(s).

    '''

    kepmag = 12.0 - 2.5*np.log10(keplerflux/f12)
    return kepmag
def fit_star_flares(x,residuals,flare_df):

    points_after_flare = 2
    points_before_flare = 2

    paf = [0,1,2,3,4,5,6,7,8]
    pbf = [0,1,2,3]

    for i,f in flare_df.iterrows():

        if (i == 6) or (i == 9):
            continue
        else:
            print(i)

            x_flare = x[int(f.istart)-points_before_flare:int(f.istop+points_after_flare)]
            y_flare = residuals[int(f.istart)-points_before_flare:int(f.istop+points_after_flare)]

            x_fine = np.linspace(np.min(x_flare), np.max(x_flare), 100)

            current_best_pars = [[0,0,0],[0,0,0]]
            current_best_fit = [[0,0,0,0],[0,0,0,0]]
            current_best_plot = [[],[]]

            for a in range(len(paf)):
                for b in range(len(pbf)):
                    x_flare_test = x[int(f.istart) - pbf[b]:int(f.istop + paf[a])]
                    y_flare_test = residuals[int(f.istart) - pbf[b]:int(f.istop + paf[a])]

                    x_fine_test = np.linspace(np.min(x_flare_test), np.max(x_flare_test), 100)

                    guess_peak = x_flare_test[np.where(y_flare_test == np.nanmax(y_flare_test))[0][0]]
                    guess_fwhm = 0.03
                    guess_ampl = y_flare_test[np.where(y_flare_test == np.nanmax(y_flare_test))[0][0]]

                    #import pdb; pdb.set_trace()

                    popt, pcov = optimize.curve_fit(aflare1, x_flare_test, y_flare_test, p0 = (guess_peak,guess_fwhm,guess_ampl))

                    last_fit = aflare1(x_flare_test, *current_best_pars[0])
                    current_fit = aflare1(x_flare_test, *popt)

                    var_last = np.var(y_flare_test - last_fit)
                    var_current = np.var(y_flare_test - current_fit)

                    chi2_last = np.sum((y_flare_test - last_fit)**2)
                    chi2_current = np.sum((y_flare_test - current_fit)**2)

                    # if var_current < var_last:
                    if chi2_current < chi2_last:
                        current_best_pars = [popt, pcov]
                        current_best_fit = [x_flare_test, current_fit]
                        current_best_plot = [x_fine, aflare1(x_fine, *popt)]

            flare_profile_fit = current_best_plot


            # # normalize time array to set t=0 at maximum flux for this flare
            # where_max = np.where(y_flare == np.max(y_flare))[0][0]
            # t_max = x_flare[where_max]
            # f_max = y_flare[where_max]
            #
            # # get FWHM of flare
            # fwhm = get_fwhm(x_flare, y_flare)*0.65
            #
            # # generate simple flare profile based on given max value
            # flare_profile = aflare1(x_fine,t_max,fwhm,f_max)
            eq_dur = np.trapz(flare_profile_fit[1], x=flare_profile_fit[0])
            eq_dur *= 86400 # convert days to seconds

            L_star = 1.2  # solar luminosity
            L_star *= 3.827e33  # convert to erg/s
            flare_energy = L_star * eq_dur

            plt.title(r'Equivalent_Duration = ' + str(np.round(eq_dur,2)) + ' (sec)\nFlare Energy = ' + str(np.round(flare_energy,2)) + 'erg s$^{-1}$')
            plt.scatter(x_flare, y_flare + 1, c='black')
            # plt.plot(np.array(x_fine), flare_profile, c='red', label='parameters given')
            plt.plot(flare_profile_fit[0], flare_profile_fit[1] + 1, c='blue', label='parameters fit')
            plt.legend(loc='upper right')
            plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/flare_'+str(i)+'.pdf')
            plt.close()
            #plt.show()

            #import pdb; pdb.set_trace()
def fit_star_flares_norm(x, norm_flux, flare_df):
    norm_flux -= 1
    points_after_flare = 5
    points_before_flare = 2

    paf = [0,1, 2, 3, 4, 5, 6, 7, 8]
    pbf = [0,1, 2, 3]

    for i, f in flare_df.iterrows():

        if (i == 6):
            continue
        else:
            print(i)

            x_flare = x[int(f.istart) - points_before_flare:int(f.istop + points_after_flare)]
            y_flare = norm_flux[int(f.istart) - points_before_flare:int(f.istop + points_after_flare)]

            x_fine = np.linspace(np.min(x_flare), np.max(x_flare), 100)

            current_best_pars = [[0, 0, 0], [0, 0, 0]]
            current_best_fit = [[0, 0, 0, 0], [0, 0, 0, 0]]
            current_best_plot = [[], []]

            for a in range(len(paf)):
                for b in range(len(pbf)):
                    x_flare_test = x[int(f.istart) - pbf[b]:int(f.istop + paf[a])]
                    y_flare_test = norm_flux[int(f.istart) - pbf[b]:int(f.istop + paf[a])]

                    x_fine_test = np.linspace(np.min(x_flare_test), np.max(x_flare_test), 100)

                    #import pdb; pdb.set_trace

                    guess_peak = x_flare_test[np.where(y_flare_test == np.nanmax(y_flare_test))[0][0]] - 0.02
                    guess_fwhm = 0.03
                    guess_ampl = y_flare_test[np.where(y_flare_test == np.nanmax(y_flare_test))[0][0]]

                    #import pdb; pdb.set_trace()

                    popt, pcov = optimize.curve_fit(aflare1, x_flare_test, y_flare_test, p0=(guess_peak, guess_fwhm, guess_ampl))

                    last_fit = aflare1(x_flare_test, *current_best_pars[0])
                    current_fit = aflare1(x_flare_test, *popt)


                    chi2_last = np.sum((y_flare_test - last_fit) ** 2)
                    chi2_current = np.sum((y_flare_test - current_fit) ** 2)

                    # if var_current < var_last:
                    if chi2_current < chi2_last:
                        current_best_pars = [popt, pcov]
                        current_best_fit = [x_flare_test, current_fit]
                        current_best_plot = [x_fine, aflare1(x_fine, *popt)]

            flare_profile_fit = current_best_plot

            # normalize time array to set t=0 at maximum flux for this flare
            # where_max = np.where(y_flare == np.max(y_flare))[0][0]
            # t_max = x_flare[where_max]
            # f_max = y_flare[where_max]

            # get FWHM of flare
            # fwhm = get_fwhm(x_flare, y_flare) * 0.65
            #
            # # generate simple flare profile based on given max value
            # flare_profile = aflare1(x_fine, t_max, fwhm, f_max)

            eq_dur = np.trapz(flare_profile_fit[1]+1, x=flare_profile_fit[0])
            eq_dur *= 86400 # convert days to seconds

            L_star = 1.2 # solar luminosity
            L_star *= 3.827e33 # convert to erg/s
            flare_energy = L_star*eq_dur

            plt.title(r'Equivalent_Duration = ' + str(np.round(eq_dur,2)) + ' (sec)\nFlare Energy = ' + str(np.round(flare_energy,2)) + 'erg s$^{-1}$')
            plt.scatter(x_flare, y_flare+1, c='black')
            # plt.plot(np.array(x_fine), flare_profile, c='red', label='parameters given')
            plt.plot(flare_profile_fit[0], flare_profile_fit[1]+1, c='blue', label='parameters fit')
            plt.legend(loc='upper right')
            plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/flare_norm_' + str(i) + '.pdf')
            plt.close()
            # plt.show()

            # import pdb; pdb.set_trace()


make_plots = False
if make_plots == True:
    fit_star_flares(x,residuals,flare_df)
    fit_star_flares_norm(x, norm_flux_scaled, flare_df)

# for i,f in flare_df.iterrows():
#
#     if (i == 6) or (i == 9):
#         continue
#     else:
#         print(i)
#
#         x_flare = x[int(f.istart)-points_before_flare:int(f.istop+points_after_flare)]
#         y_flare = residuals[int(f.istart)-points_before_flare:int(f.istop+points_after_flare)]
#
#         x_fine = np.linspace(np.min(x_flare), np.max(x_flare), 100)
#
#         # popt, _ = curve_fit(objective, x, y)
#         # # summarize the parameter values
#         # a, b = popt
#
#         # plt.plot(x_fine,y_fine,c='blue')
#         # plt.show()
#         # plt.scatter(x_flare, y_flare, c='black')
#         # plt.show()
#
#
#         # x1 = 0
#         # x2 = np.max(x_flare) - np.min(x_flare)
#         # divs = np.linspace(x1, x2, 8)
#         # div_shift = divs - np.diff(divs)[0]
#
#         current_best_pars = [[0,0,0],[0,0,0]]
#         current_best_fit = [[0,0,0,0],[0,0,0,0]]
#         current_best_plot = [[],[]]
#         current_best_pars_upper = [[0, 0, 0], [0, 0, 0]]
#         current_best_fit_upper = [[0, 0, 0, 0], [0, 0, 0, 0]]
#         current_best_plot_upper = [[], []]
#         current_best_pars_lower = [[0, 0, 0], [0, 0, 0]]
#         current_best_fit_lower = [[0, 0, 0, 0], [0, 0, 0, 0]]
#         current_best_plot_lower = [[], []]
#         for a in range(len(paf)):
#             for b in range(len(pbf)):
#                 x_flare_test = x[int(f.istart) - pbf[b]:int(f.istop + paf[a])]
#                 y_flare_test = residuals[int(f.istart) - pbf[b]:int(f.istop + paf[a])]
#
#                 x_fine_test = np.linspace(np.min(x_flare_test), np.max(x_flare_test), 100)
#
#                 popt, pcov = optimize.curve_fit(aflare1, x_flare_test, y_flare_test)
#                 popt_upper, pcov_upper = optimize.curve_fit(aflare1_upper, x_flare_test, y_flare_test)
#                 popt_lower, pcov_lower = optimize.curve_fit(aflare1_lower, x_flare_test, y_flare_test)
#
#                 last_fit = aflare1(x_flare_test, *current_best_pars[0])
#                 last_fit_upper = aflare1_upper(x_flare_test, *current_best_pars_upper[0])
#                 last_fit_lower = aflare1_lower(x_flare_test, *current_best_pars_lower[0])
#                 current_fit = aflare1(x_flare_test, *popt)
#                 current_fit_upper = aflare1_upper(x_flare_test, *popt_upper)
#                 current_fit_lower = aflare1_lower(x_flare_test, *popt_lower)
#
#                 var_last = np.var(y_flare_test - last_fit)
#                 var_last_upper = np.var(y_flare_test - last_fit_upper)
#                 var_last_lower = np.var(y_flare_test - last_fit_lower)
#                 var_current = np.var(y_flare_test - current_fit)
#                 var_current_upper = np.var(y_flare_test - current_fit_upper)
#                 var_current_lower = np.var(y_flare_test - current_fit_lower)
#
#                 chi2_last = np.sum((y_flare_test - last_fit)**2)
#                 chi2_last_upper = np.sum((y_flare_test - last_fit_upper)**2)
#                 chi2_last_lower = np.sum((y_flare_test - last_fit_lower)**2)
#                 chi2_current = np.sum((y_flare_test - current_fit)**2)
#                 chi2_current_upper = np.sum((y_flare_test - current_fit_upper)**2)
#                 chi2_current_lower = np.sum((y_flare_test - current_fit_lower)**2)
#
#                 if chi2_current < chi2_last:
#                     current_best_pars = [popt, pcov]
#                     current_best_fit = [x_flare_test, current_fit]
#                     current_best_plot = [x_fine, aflare1(x_fine, *popt)]
#                 if chi2_current_upper < chi2_last_upper:
#                     current_best_pars_upper = [popt_upper, pcov_upper]
#                     current_best_fit_upper = [x_flare_test, current_fit_upper]
#                     current_best_plot_upper = [x_fine, aflare1_upper(x_fine, *popt_upper)]
#                 if chi2_current_lower < chi2_last_lower:
#                     current_best_pars_lower = [popt_lower, pcov_lower]
#                     current_best_fit_lower = [x_flare_test, current_fit_lower]
#                     current_best_plot_lower = [x_fine, aflare1_lower(x_fine, *popt_lower)]
#
#         flare_profile_fit = current_best_plot
#         flare_profile_fit_upper = current_best_plot_upper
#         flare_profile_fit_lower = current_best_plot_lower
#
#
#         # normalize time array to set t=0 at maximum flux for this flare
#         where_max = np.where(y_flare == np.max(y_flare))[0][0]
#         t_max = x_flare[where_max]
#         f_max = y_flare[where_max]
#
#         # get FWHM of flare
#         fwhm = get_fwhm(x_flare, y_flare)*0.65
#
#         # generate simple flare profile based on given max value
#         flare_profile = aflare1(x_fine,t_max,fwhm,f_max)
#
#         plt.scatter(x_flare, y_flare, c='black')
#         plt.plot(np.array(x_fine), flare_profile, c='red', label='parameters given')
#         plt.plot(flare_profile_fit[0], flare_profile_fit[1], c='blue', label='parameters fit')
#         plt.fill_between(flare_profile_fit_upper[0], flare_profile_fit_upper[1], y2=flare_profile_fit_lower[1], color='blue', alpha=0.30)
#         plt.legend(loc='upper right')
#         # plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/flare_'+str(i)+'.pdf')
#         # plt.close()
#         plt.show()
#
#         import pdb; pdb.set_trace()

# ------------------  TEST METHODS ON SYNTHETIC FLARES --------------------- #

def create_synthetic(cadence, stddev, set_cadence, given_amplitude=1000, given_fwhm=0.001,  fixed_amplitude=False, fixed_fwhm=False, ask_input=False):
    print('\nGenerating Synthetic Flares...\n')
    shrink_value = 1./50
    end_day = 500
    days_apart = 5

    x_synth = np.arange(0,end_day+days_apart,cadence) # cadence in days
    t_max = np.arange(days_apart,end_day+days_apart,days_apart) # 500 flares spaced 2 days apart
    t_max = t_max + np.random.uniform(-0.05,0.05,len(t_max))
    y_synth = np.random.normal(0,stddev*shrink_value,len(x_synth))
    y_synth_noscatter = np.zeros_like(x_synth)


    if fixed_amplitude == True:
        if ask_input == True:
            amplitude_input = np.float(input(r'Amplitude in multiples of stddev ($\sigma$): '))
        else:
            amplitude_input = given_amplitude
        amplitude = np.zeros_like(t_max) + (amplitude_input * stddev)
    else:
        amplitude = np.random.uniform(3,1000,len(t_max)) * stddev # np.linspace(3,20000,100) * stddev


    if fixed_fwhm == True:
        if ask_input == True:
            fwhm_input = np.float(input('FWHM (~0.002-0.02): '))
        else:
            fwhm_input = given_fwhm
        fwhm = np.zeros_like(t_max) + fwhm_input
    else:
        # fwhm = np.ones_like(amplitude) * np.random.uniform(0.001,0.02,len(t_max))# days
        fwhm = np.ones_like(amplitude) * np.random.uniform(0.5, set_cadence, len(t_max))*(1./60.)*(1./24.) # days
        np.random.shuffle(fwhm)

    flare_window = []

    for a in range(len(amplitude)):
        #print(a)
        flare_synth_a_noscatter = aflare1(x_synth, t_max[a], fwhm[a], amplitude[a])
        flare_synth_a = flare_synth_a_noscatter + np.random.normal(0,stddev*shrink_value,len(x_synth))

        y_synth_noscatter += flare_synth_a_noscatter
        y_synth += flare_synth_a

        flare_window.append([t_max[a] - 0.05, t_max[a] + 0.2])

        # where_flare = np.where(flare_synth_a_noscatter > 0.005)[0]
        #
        # plt.plot(x_synth[where_flare], np.zeros_like(x_synth[where_flare]), c='black', alpha=0.4, label='synthetic quiescence no scatter')
        # plt.scatter(x_synth[where_flare], y_synth[where_flare], c='black', s=np.pi*(0.5)**2, label='synthetic quiescence')
        # plt.scatter(x_synth[where_flare], flare_synth_a[where_flare], c='#ff0056', s=np.pi*(0.5)**2, label='synthetic flare')
        # plt.plot(x_synth[where_flare], flare_synth_a_noscatter[where_flare], c='#b300b3', alpha=0.4, label='synthetic flare no scatter')
        # plt.legend(loc='upper right')
        # # plt.savefig( '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/_synthetic_flare_' + str(a) + '.pdf')
        # # plt.close()
        # plt.show()

        #import pdb; pdb.set_trace()

    flare_properties = {"tpeak":t_max,
                  "amplitude":amplitude,
                  "fwhm":fwhm,
                  }
    return x_synth, y_synth, y_synth_noscatter, flare_window, flare_properties
stddev = 0.001
cadence = 1.*(1./1440.) # 2 minutes converted to days

#x_synth, y_synth, y_synth_noscatter, flare_window, flare_properties = create_synthetic(cadence, stddev)

# for v in range(len(flare_window)):
#
#     print(v)
#
#     where_start = np.int(np.floor(np.random.uniform(0,16,1)))
#
#     x_downsample = x_synth[where_start::15]
#     y_downsample = y_synth[where_start::15]
#     y_noscatter_downsample = y_synth_noscatter[where_start::15]
#
#     current_best_pars = [[0, 0, 0], [0, 0, 0]]
#     current_best_fit = [[0, 0, 0, 0], [0, 0, 0, 0]]
#     #current_best_plot = [[], []]
#     t_starts = [0] + flare_window[v][0]
#     t_ends = [0] + flare_window[v][1] #-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] + flare_window[v][1]
#     for a in range(len(t_starts)):
#         for b in range(len(t_ends)):
#
#             window = np.where((x_downsample >= t_starts[a]) & (x_downsample <= t_ends[b]))[0]
#             x_window = x_downsample[window]
#             y_window = y_downsample[window]
#             y_noscatter_window = y_noscatter_downsample[window]
#
#             x_fine_test = np.linspace(np.min(x_window), np.max(x_window), 100)
#
#             guess_peak = x_window[np.where(y_window == np.max(y_window))[0][0]]
#             guess_fwhm = 0.01
#             guess_ampl = y_window[np.where(y_window == np.max(y_window))[0][0]]
#
#             try:
#                 popt, pcov = optimize.curve_fit(aflare1, x_window, y_window, p0 = (guess_peak,guess_fwhm,guess_ampl)) #diag=(1./x_window.mean(),1./y_window.mean()))
#             except:
#                 continue
#
#             #print(v)
#
#             last_fit = aflare1(x_window, *current_best_pars[0])
#             current_fit = aflare1(x_window, *popt)
#
#             var_last = np.var(y_window - last_fit)
#             var_current = np.var(y_window - current_fit)
#
#             chi2_last = np.sum((y_window - last_fit) ** 2)
#             chi2_current = np.sum((y_window - current_fit) ** 2)
#
#             # if var_current < var_last:
#             if chi2_current < chi2_last:
#                 current_best_pars = [popt, pcov]
#                 current_best_fit = [x_window, current_fit]
#                 #current_best_plot = [x_fine, aflare1(x_fine, *popt)]
#
#     # flare_profile_fit = current_best_plot
#
#
#
#     window = np.where((x_downsample >= flare_window[v][0]) & (x_downsample <= flare_window[v][1]))[0]
#     x_window = x_downsample[window]
#     y_window = y_downsample[window]
#     y_noscatter_window = y_noscatter_downsample[window]
#
#     # try:
#     #popt, pcov = optimize.curve_fit(aflare1, x_window, y_window)
#     #popt_noscatter, pcov_noscatter = optimize.curve_fit(aflare1, x_window, y_noscatter_window)
#     # except:
#     #     continue
#     # else:
#     x_fit = np.linspace(flare_window[v][0], flare_window[v][1], 300)
#     y_fit = aflare1(x_fit, *current_best_pars[0]) #popt)
#     #y_fit_noscatter = aflare1(x_fit, *popt_noscatter)
#
#     window_hicad = np.where((x_synth >= flare_window[v][0]) & (x_synth <= flare_window[v][1]))[0]
#     plt.scatter(x_synth[window_hicad], y_synth_noscatter[window_hicad], c='#ff5050', s=np.pi * (1) ** 2, alpha=1.0, label='true flare 2min cadence')
#     plt.scatter(x_window, y_window, c='black', s=np.pi * (2) ** 2, label='noisy flare 30min cadence')
#     plt.plot(x_fit, y_fit, c='#0099cc', lw=2.5, label='fit to downsampled noisy flare')
#     #plt.plot(x_fit, y_fit_noscatter, c='red', label='fit to downsampled noscatter flare')
#     plt.legend(loc='upper right')
#     plt.ylabel('Flux Residuals (ppt)')
#     plt.xlabel('Time (days)')
#     plt.xlim([np.min(x_window), np.max(x_window)])
#     plt.ylim([-stddev, np.max([np.max(y_synth[window_hicad]), np.max(y_fit),np.max(y_window)]) * 1.15])
#     plt.tight_layout()
#     plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/synthetic_flare_' + str(v) + '.pdf')
#     plt.close()
#     # plt.show()
#     #
#     # import pdb; pdb.set_trace()

# def gauss_fit(dat, hist_dat,x_factor):
#     mu, sigma = scipy.stats.norm.fit(hist_dat[0])
#     mu, sigma = scipy.stats.norm.fit(dat)
#     x_gauss = np.linspace(-x_factor * sigma, x_factor * sigma, 1000)
#     y_gauss = scipy.stats.norm.pdf(x_gauss, mu, sigma)
#
#     return x_gauss, y_gauss, sigma
# def cauchy_fit(dat, hist_dat, x_factor):
#
#     pars = scipy.stats.cauchy.fit(dat)
#     #x_cauchy = np.linspace(np.min(hist_dat[1]),np.max(hist_dat[1]), 5000)
#     x_cauchy = np.linspace(-x_factor*pars[1],x_factor*pars[1],5000)
#     y_cauchy = scipy.stats.cauchy.pdf(x_cauchy,*pars)
#
#
#
#     return x_cauchy,y_cauchy,pars,pars[1]

def plot_stat_hist(t_peak_opt,fwhm_opt,ampl_opt,fwhm_true,ampl_true,save_as):
    font_size = 'medium'
    nbins = 100
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'
    x_factor = 20.

    dat1 = np.array(t_peak_opt)*24*60
    dat2 = np.array(fwhm_opt)*24*60
    dat3 = ampl_opt

    fig = plt.figure(1, figsize=(14, 4), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(131)
    #ax1.hist(dat1, color=t_peak_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat1 = ax1.hist(dat1, color=t_peak_color, bins=nbins, weights=np.ones(len(dat1))/len(dat1)) #, edgecolor='#000000', linewidth=1.2)
    ax1.hist(dat1, color='#000000', bins=nbins, linewidth=1.2, histtype='step', weights=np.ones(len(dat1))/len(dat1))

    x_gauss1,y_gauss1, sigma1 = gauss_fit(dat1,hist_dat1,x_factor)
    #ax1.plot(x_gauss1, y_gauss1, color='orange', lw=1.5)

    mult_fact1 = np.max(hist_dat1[0])*0.1
    x_cauchy1,y_cauchy1,cauchy_pars1,sigma1 = cauchy_fit(dat1,hist_dat1,x_factor)
    #ax1.plot(x_cauchy1, y_cauchy1*mult_fact1 , color='orange', lw=1.5)

    ax1.plot([0,0],[0,np.max(hist_dat1[0])*10], '--',  color='#000000', lw=1) #, label="Rotation Model")
    #ax1.set_xlim(np.min(hist_dat1[1]), np.max(hist_dat1[1]))
    ax1.set_xlim(-10, 10)
    ax1.set_ylim([0, np.max(hist_dat1[0]) * 1.15])
    #plt.legend(fontsize=10, loc="upper left")
    ax1.set_xlabel("Difference From True Peak Time (min)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Fraction of Total", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title("Peak Time ", fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    ax2 = fig.add_subplot(132)
    #ax2.hist(dat2, color=fwhm_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat2 = ax2.hist(dat2, color=fwhm_color, bins=nbins*6, weights=np.ones(len(dat2))/len(dat2)) #, edgecolor='#000000', linewidth=1.2)
    ax2.hist(dat2, color='#000000', bins=nbins*6, linewidth=1.2, histtype='step', weights=np.ones(len(dat2))/len(dat2))

    x_gauss2, y_gauss2, sigma2 = gauss_fit(dat2,hist_dat2, x_factor)
    #ax2.plot(x_gauss2, y_gauss2, color='orange', lw=1.5)

    mult_fact2 = np.max(hist_dat2[0]) * 0.1
    x_cauchy2, y_cauchy2,cauchy_pars2,sigma2 = cauchy_fit(dat2,hist_dat2,x_factor)
    #ax2.plot(x_cauchy2, y_cauchy2*mult_fact2, color='orange', lw=1.5)

    ax2.plot([0, 0], [0, np.max(hist_dat2[0]) * 10], '--', color='#000000', lw=1)
    # plt.plot([0,0],[0,] rot_model, color="C1", lw=1, label="Rotation Model")
    #ax2.set_xlim(np.min(hist_dat2[1]), np.max(hist_dat2[1]))
    ax2.set_xlim(-x_factor * sigma2, x_factor * sigma2)
    ax2.set_ylim([0, np.max(hist_dat2[0]) * 1.15])
    #plt.legend(fontsize=10, loc="upper left")
    ax2.set_xlabel("% Difference From True FWHM", fontsize=font_size, style='normal', family='sans-serif')
    #ax2.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_title("True FWHM: " + str(np.round(fwhm_true[0]*24*60,3)), fontsize=font_size, style='normal', family='sans-serif')
    ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    ax3 = fig.add_subplot(133)
    #ax3.hist(dat3, color=ampl_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat3 = ax3.hist(dat3, color=ampl_color, bins=nbins*4, weights=np.ones(len(dat3))/len(dat3)) #, edgecolor='#000000', linewidth=1.2)
    ax3.hist(dat3, color='#000000', bins=nbins*4, linewidth=1.2, histtype='step', weights=np.ones(len(dat3))/len(dat3))

    x_gauss3, y_gauss3, sigma3 = gauss_fit(dat3,hist_dat3, x_factor)
    #ax3.plot(x_gauss3, y_gauss3, color='orange', lw=1.5)

    mult_fact3 = np.max(hist_dat3[0]) * 0.1
    x_cauchy3, y_cauchy3,cauchy_pars3,sigma3 = cauchy_fit(dat3,hist_dat3,x_factor)
    #ax3.plot(x_cauchy3, y_cauchy3*mult_fact3, color='orange', lw=1.5)

    ax3.plot([0, 0], [0, np.max(hist_dat3[0]) * 10], '--', color='#000000', lw=1)
    # plt.plot([0,0],[0,] rot_model, color="C1", lw=1, label="Rotation Model")
    #ax3.set_xlim(np.min(hist_dat3[1]), np.max(hist_dat3[1]))
    ax3.set_xlim(-x_factor*2*sigma3, x_factor*2*sigma3)
    ax3.set_ylim([0, np.max(hist_dat3[0]) * 1.15])
    # plt.legend(fontsize=10, loc="upper left")
    ax3.set_xlabel("% Difference From True Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    #ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax3.set_title("True Amplitude: " + str(np.round(ampl_true[0],3)), fontsize=font_size, style='normal', family='sans-serif')
    ax3.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
    #import pdb;pdb.set_trace()
def plot_stat_hist2(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,flare_energy_opt,impulsiveness,if_downsample,save_as):
    font_size = 'medium'
    nbins = 100
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'
    eqdur_color = '#cc6600'
    energy_color = '#666699'
    impulsiveness_color = '#669999'
    x_factor = 20.

    dat1 = np.array(t_peak_opt)*24*60
    dat2 = np.array(fwhm_opt)*24*60
    dat3 = ampl_opt
    dat4 = eq_duration_opt
    dat5 = flare_energy_opt
    dat6 = impulsiveness

    fig = plt.figure(1, figsize=(15, 4*2.2), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(231)
    #ax1.hist(dat1, color=t_peak_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat1 = ax1.hist(dat1, color=t_peak_color, bins=nbins, weights=np.ones(len(dat1))/len(dat1)) #, edgecolor='#000000', linewidth=1.2)
    ax1.hist(dat1, color='#000000', bins=nbins, linewidth=1.2, histtype='step', weights=np.ones(len(dat1))/len(dat1))

    x_gauss1,y_gauss1, sigma1 = gauss_fit(dat1,hist_dat1,x_factor)
    #ax1.plot(x_gauss1, y_gauss1, color='orange', lw=1.5)

    x_cauchy1,y_cauchy1,cauchy_pars1,sigma1 = cauchy_fit(dat1,hist_dat1,x_factor)
    #ax1.plot(x_cauchy1, y_cauchy1*mult_fact1 , color='orange', lw=1.5)

    ax1.plot([0,0],[0,np.max(hist_dat1[0])*10], '--',  color='#000000', lw=1) #, label="Rotation Model")
    #ax1.set_xlim(np.min(hist_dat1[1]), np.max(hist_dat1[1]))
    if if_downsample == True:
        ax1.set_xlim(-10, 10)
    if if_downsample == False:
        ax1.set_xlim(-0.25, 0.25)
    ax1.set_ylim([0, np.max(hist_dat1[0]) * 1.15])
    #plt.legend(fontsize=10, loc="upper left")
    ax1.set_xlabel("Difference From True Peak Time (min)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Fraction of Total", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title("Peak Time ", fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax2 = fig.add_subplot(232)
    #ax2.hist(dat2, color=fwhm_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat2 = ax2.hist(dat2, color=fwhm_color, bins=nbins*6, weights=np.ones(len(dat2))/len(dat2)) #, edgecolor='#000000', linewidth=1.2)
    ax2.hist(dat2, color='#000000', bins=nbins*6, linewidth=1.2, histtype='step', weights=np.ones(len(dat2))/len(dat2))

    x_gauss2, y_gauss2, sigma2 = gauss_fit(dat2,hist_dat2, x_factor)
    #ax2.plot(x_gauss2, y_gauss2, color='orange', lw=1.5)

    mult_fact2 = np.max(hist_dat2[0]) * 0.1
    x_cauchy2, y_cauchy2,cauchy_pars2,sigma2 = cauchy_fit(dat2,hist_dat2,x_factor)
    #ax2.plot(x_cauchy2, y_cauchy2*mult_fact2, color='orange', lw=1.5)

    ax2.plot([0, 0], [0, np.max(hist_dat2[0]) * 10], '--', color='#000000', lw=1)
    # plt.plot([0,0],[0,] rot_model, color="C1", lw=1, label="Rotation Model")
    #ax2.set_xlim(np.min(hist_dat2[1]), np.max(hist_dat2[1]))
    ax2.set_xlim(-x_factor * sigma2, x_factor * sigma2)
    ax2.set_ylim([0, np.max(hist_dat2[0]) * 1.15])
    #plt.legend(fontsize=10, loc="upper left")
    ax2.set_xlabel("% Difference From True FWHM", fontsize=font_size, style='normal', family='sans-serif')
    #ax2.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_title("FWHM", fontsize=font_size, style='normal', family='sans-serif')
    ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax3 = fig.add_subplot(233)
    #ax3.hist(dat3, color=ampl_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    if if_downsample == True:
        hist_dat3 = ax3.hist(dat3, color=ampl_color, bins=nbins*4, weights=np.ones(len(dat3))/len(dat3)) #, edgecolor='#000000', linewidth=1.2)
        ax3.hist(dat3, color='#000000', bins=nbins*4, linewidth=1.2, histtype='step', weights=np.ones(len(dat3))/len(dat3))
    if if_downsample == False:
        hist_dat3 = ax3.hist(dat3, color=ampl_color, bins=nbins * 20, weights=np.ones(len(dat3)) / len(dat3))  # , edgecolor='#000000', linewidth=1.2)
        ax3.hist(dat3, color='#000000', bins=nbins * 20, linewidth=1.2, histtype='step', weights=np.ones(len(dat3)) / len(dat3))

    x_gauss3, y_gauss3, sigma3 = gauss_fit(dat3,hist_dat3, x_factor)
    #ax3.plot(x_gauss3, y_gauss3, color='orange', lw=1.5)

    mult_fact3 = np.max(hist_dat3[0]) * 0.1
    x_cauchy3, y_cauchy3,cauchy_pars3,sigma3 = cauchy_fit(dat3,hist_dat3,x_factor)
    #ax3.plot(x_cauchy3, y_cauchy3*mult_fact3, color='orange', lw=1.5)

    ax3.plot([0, 0], [0, np.max(hist_dat3[0]) * 10], '--', color='#000000', lw=1)
    # plt.plot([0,0],[0,] rot_model, color="C1", lw=1, label="Rotation Model")
    #ax3.set_xlim(np.min(hist_dat3[1]), np.max(hist_dat3[1]))
    ax3.set_xlim(-x_factor*2*sigma3, x_factor*2*sigma3)
    ax3.set_ylim([0, np.max(hist_dat3[0]) * 1.15])
    # plt.legend(fontsize=10, loc="upper left")
    ax3.set_xlabel("% Difference From True Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    #ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax3.set_title("Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    ax3.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax4 = fig.add_subplot(234)
    if if_downsample == True:
        hist_dat4 = ax4.hist(dat4, color=eqdur_color, bins=nbins, weights=np.ones(len(dat4)) / len(dat4))  # , edgecolor='#000000', linewidth=1.2)
        ax4.hist(dat4, color='#000000', bins=nbins, linewidth=1.2, histtype='step', weights=np.ones(len(dat4)) / len(dat4))
    if if_downsample == False:
        hist_dat4 = ax4.hist(dat4, color=eqdur_color, bins=nbins*4, weights=np.ones(len(dat4)) / len(dat4))  # , edgecolor='#000000', linewidth=1.2)
        ax4.hist(dat4, color='#000000', bins=nbins*4, linewidth=1.2, histtype='step', weights=np.ones(len(dat4)) / len(dat4))

    x_cauchy4, y_cauchy4, cauchy_pars4, sigma4 = cauchy_fit(dat4, hist_dat4, x_factor)

    ax4.plot([0, 0], [0, np.max(hist_dat4[0]) * 10], '--', color='#000000', lw=1)
    ax4.set_xlim(-x_factor * sigma4, x_factor * sigma4)
    ax4.set_ylim([0, np.max(hist_dat4[0]) * 1.15])
    # plt.legend(fontsize=10, loc="upper left")
    ax4.set_xlabel("% Difference From True Equivalent Duration", fontsize=font_size, style='normal', family='sans-serif')
    ax4.set_ylabel("Fraction of Total", fontsize=font_size, style='normal', family='sans-serif')
    ax4.set_title("Equivalent Duration", fontsize=font_size, style='normal',family='sans-serif')
    ax4.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax5 = fig.add_subplot(235)
    if if_downsample == True:
        hist_dat5 = ax5.hist(dat5, color=energy_color, bins=nbins, weights=np.ones(len(dat5)) / len(dat5))  # , edgecolor='#000000', linewidth=1.2)
        ax5.hist(dat5, color='#000000', bins=nbins, linewidth=1.2, histtype='step', weights=np.ones(len(dat5)) / len(dat5))
    if if_downsample == False:
        hist_dat5 = ax5.hist(dat5, color=energy_color, bins=nbins*4, weights=np.ones(len(dat5)) / len(dat5))  # , edgecolor='#000000', linewidth=1.2)
        ax5.hist(dat5, color='#000000', bins=nbins*4, linewidth=1.2, histtype='step', weights=np.ones(len(dat5)) / len(dat5))
    x_cauchy5, y_cauchy5, cauchy_pars5, sigma5 = cauchy_fit(dat5, hist_dat5, x_factor)
    ax5.plot([0, 0], [0, np.max(hist_dat5[0]) * 10], '--', color='#000000', lw=1)
    ax5.set_xlim(-x_factor/2. * sigma5, x_factor/2. * sigma5)
    ax5.set_ylim([0, np.max(hist_dat5[0]) * 1.15])
    # plt.legend(fontsize=10, loc="upper left")
    ax5.set_xlabel("% Difference From True Flare Energy", fontsize=font_size, style='normal',family='sans-serif')
    # ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax5.set_title("Flare Energy", fontsize=font_size, style='normal', family='sans-serif')
    ax5.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax6 = fig.add_subplot(236)
    hist_dat6 = ax6.hist(dat6, color=impulsiveness_color, bins=nbins, weights=np.ones(len(dat6)) / len(dat6))  # , edgecolor='#000000', linewidth=1.2)
    ax6.hist(dat6, color='#000000', bins=nbins, linewidth=1.2, histtype='step', weights=np.ones(len(dat6)) / len(dat6))
    ax6.plot([0, 0], [0, np.max(hist_dat6[0]) * 10], '--', color='#000000', lw=1)
    #ax6.set_xlim(-x_factor / 2. * sigma6, x_factor / 2. * sigma6)
    ax6.set_ylim([0, np.max(hist_dat6[0]) * 1.15])
    # plt.legend(fontsize=10, loc="upper left")
    ax6.set_xlabel("Distribution of Impulsiveness Index of Flares Tested", fontsize=font_size, style='normal', family='sans-serif')
    # ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax6.set_title("Impulsive Index", fontsize=font_size, style='normal', family='sans-serif')
    ax6.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)



    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
    #import pdb;pdb.set_trace()
def plot_stat_hist3(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,flare_energy_opt,impulsiveness, save_as):

    print('Plotting Simple Histograms...')

    font_size = 'medium'
    nbins = 'auto'
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'
    eqdur_color = '#cc6600'
    energy_color = '#666699'
    impulsiveness_color = '#669999'
    x_factor = 20.

    impulse_max_lim_factor = 0.50

    dat1 = np.array(t_peak_opt)*24*60
    dat2 = fwhm_opt
    dat3 = ampl_opt
    dat4 = eq_duration_opt
    dat5 = flare_energy_opt
    dat6 = impulsiveness

    fig = plt.figure(1, figsize=(15, 4*2.2), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(231)

    y_hist, bin_edges = np.histogram(dat1, bins='auto')
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat1, bins=bin_edges[where_within])

    hist_dat1 = ax1.hist(dat1, color=t_peak_color, bins=bin_edges) #, weights=np.ones(len(dat1))/len(dat1)) #, edgecolor='#000000', linewidth=1.2)
    ax1.hist(dat1, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat1))/len(dat1))
    ax1.plot([0,0],[0,np.max(hist_dat1[0])*10], '--',  color='#000000', lw=1) #, label="Rotation Model")

    ax1.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax1.set_ylim([0, np.max(hist_dat1[0]) * 1.10])
    #plt.legend(fontsize=10, loc="upper left")
    ax1.set_xlabel("Difference From True Peak Time (min)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Fraction of Total", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title("Peak Time ", fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax2 = fig.add_subplot(232)

    y_hist, bin_edges = np.histogram(dat2, bins='auto')
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat2, bins=bin_edges[where_within])

    #ax2.hist(dat2, color=fwhm_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat2 = ax2.hist(dat2, color=fwhm_color, bins=bin_edges) #, weights=np.ones(len(dat2))/len(dat2)) #, edgecolor='#000000', linewidth=1.2)
    ax2.hist(dat2, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat2))/len(dat2))
    ax2.plot([0, 0], [0, np.max(hist_dat2[0]) * 10], '--', color='#000000', lw=1)

    ax2.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax2.set_ylim([0, np.max(hist_dat2[0]) * 1.10])
    #plt.legend(fontsize=10, loc="upper left")
    ax2.set_xlabel("% Difference From True FWHM", fontsize=font_size, style='normal', family='sans-serif')
    #ax2.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_title("FWHM", fontsize=font_size, style='normal', family='sans-serif')
    ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax3 = fig.add_subplot(233)

    y_hist, bin_edges = np.histogram(dat3, bins='auto')
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat3, bins=bin_edges[where_within])

    hist_dat3 = ax3.hist(dat3, color=ampl_color, bins=bin_edges) #, weights=np.ones(len(dat3))/len(dat3)) #, edgecolor='#000000', linewidth=1.2)
    ax3.hist(dat3, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat3))/len(dat3))
    ax3.plot([0, 0], [0, np.max(hist_dat3[0]) * 10], '--', color='#000000', lw=1)

    ax3.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax3.set_ylim([0, np.max(hist_dat3[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax3.set_xlabel("% Difference From True Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    #ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax3.set_title("Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    ax3.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax4 = fig.add_subplot(234)

    y_hist, bin_edges = np.histogram(dat4, bins='auto')
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat4, bins=bin_edges[where_within])

    hist_dat4 = ax4.hist(dat4, color=eqdur_color, bins=bin_edges) #, weights=np.ones(len(dat4)) / len(dat4))  # , edgecolor='#000000', linewidth=1.2)
    ax4.hist(dat4, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat4)) / len(dat4))
    ax4.plot([0, 0], [0, np.max(hist_dat4[0]) * 10], '--', color='#000000', lw=1)

    ax4.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax4.set_ylim([0, np.max(hist_dat4[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax4.set_xlabel("% Difference From True Equivalent Duration", fontsize=font_size, style='normal', family='sans-serif')
    ax4.set_ylabel("Fraction of Total", fontsize=font_size, style='normal', family='sans-serif')
    ax4.set_title("Equivalent Duration", fontsize=font_size, style='normal',family='sans-serif')
    ax4.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax5 = fig.add_subplot(235)

    y_hist, bin_edges = np.histogram(dat5, bins='auto')
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat5, bins=bin_edges[where_within])

    hist_dat5 = ax5.hist(dat5, color=energy_color, bins=bin_edges) #, weights=np.ones(len(dat5)) / len(dat5))  # , edgecolor='#000000', linewidth=1.2)
    ax5.hist(dat5, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat5)) / len(dat5))
    ax5.plot([0, 0], [0, np.max(hist_dat5[0]) * 10], '--', color='#000000', lw=1)

    ax5.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax5.set_ylim([0, np.max(hist_dat5[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax5.set_xlabel("% Difference From True Flare Energy", fontsize=font_size, style='normal',family='sans-serif')
    # ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax5.set_title("Flare Energy", fontsize=font_size, style='normal', family='sans-serif')
    ax5.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax6 = fig.add_subplot(236)

    y_hist, bin_edges = np.histogram(dat6, bins='auto')
    where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
    y_hist, bin_edges = np.histogram(dat6, bins=bin_edges[where_within])

    hist_dat6 = ax6.hist(dat6, color=impulsiveness_color, bins=bin_edges) #, weights=np.ones(len(dat6)) / len(dat6))  # , edgecolor='#000000', linewidth=1.2)
    ax6.hist(dat6, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat6)) / len(dat6))
    ax6.plot([0, 0], [0, np.max(hist_dat6[0]) * 10], '--', color='#000000', lw=1)

    ax6.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
    ax6.set_ylim([0, np.max(hist_dat6[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax6.set_xlabel("Distribution of Impulsiveness Index of Flares Tested", fontsize=font_size, style='normal', family='sans-serif')
    # ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax6.set_title("Impulsive Index", fontsize=font_size, style='normal', family='sans-serif')
    ax6.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)



    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
    #import pdb;pdb.set_trace()
def plot_stat_hist4(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,flare_energy_opt,impulsiveness, save_as):

    print('Plotting Simple Histograms...')

    font_size = 'medium'
    nbins = 'auto'
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'
    eqdur_color = '#cc6600'
    energy_color = '#666699'
    impulsiveness_color = '#669999'
    x_factor = 20.
    bin_slice_factor = 3.

    impulse_max_lim_factor = 0.50

    dat1 = np.array(t_peak_opt)*24*60
    dat2 = fwhm_opt
    dat3 = ampl_opt
    dat4 = eq_duration_opt
    dat5 = flare_energy_opt
    dat6 = impulsiveness

    fig = plt.figure(1, figsize=(15, 4*2.2), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(231)

    y_hist, bin_edges = np.histogram(dat1, bins='auto')
    #bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    axis_spacing = np.arange(0, len(bin_edges), 1)
    if bin_slice_factor == 3:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)[0:-1]
    else:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)
    bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat1, bins=bin_edges[where_within])

    hist_dat1 = ax1.hist(dat1, color=t_peak_color, bins=bin_edges) #, weights=np.ones(len(dat1))/len(dat1)) #, edgecolor='#000000', linewidth=1.2)
    ax1.hist(dat1, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat1))/len(dat1))
    ax1.plot([0,0],[0,np.max(hist_dat1[0])*10], '--',  color='#000000', lw=1) #, label="Rotation Model")

    ax1.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax1.set_ylim([0, np.max(hist_dat1[0]) * 1.10])
    #plt.legend(fontsize=10, loc="upper left")
    ax1.set_xlabel("Difference From True Peak Time (min)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title("Peak Time ", fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax2 = fig.add_subplot(232)

    y_hist, bin_edges = np.histogram(dat2, bins='auto')
    # bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    axis_spacing = np.arange(0, len(bin_edges), 1)
    if bin_slice_factor == 3:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)[0:-1]
    else:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)
    bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat2, bins=bin_edges[where_within])

    #ax2.hist(dat2, color=fwhm_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat2 = ax2.hist(dat2, color=fwhm_color, bins=bin_edges) #, weights=np.ones(len(dat2))/len(dat2)) #, edgecolor='#000000', linewidth=1.2)
    ax2.hist(dat2, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat2))/len(dat2))
    ax2.plot([0, 0], [0, np.max(hist_dat2[0]) * 10], '--', color='#000000', lw=1)

    ax2.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax2.set_ylim([0, np.max(hist_dat2[0]) * 1.10])
    #plt.legend(fontsize=10, loc="upper left")
    ax2.set_xlabel("% Difference From True FWHM", fontsize=font_size, style='normal', family='sans-serif')
    #ax2.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_title("FWHM", fontsize=font_size, style='normal', family='sans-serif')
    ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax3 = fig.add_subplot(233)

    y_hist, bin_edges = np.histogram(dat3, bins='auto')
    #bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    axis_spacing = np.arange(0, len(bin_edges), 1)
    if bin_slice_factor == 3:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)[0:-1]
    else:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)
    bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat3, bins=bin_edges[where_within])

    hist_dat3 = ax3.hist(dat3, color=ampl_color, bins=bin_edges) #, weights=np.ones(len(dat3))/len(dat3)) #, edgecolor='#000000', linewidth=1.2)
    ax3.hist(dat3, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat3))/len(dat3))
    ax3.plot([0, 0], [0, np.max(hist_dat3[0]) * 10], '--', color='#000000', lw=1)

    ax3.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax3.set_ylim([0, np.max(hist_dat3[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax3.set_xlabel("% Difference From True Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    #ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax3.set_title("Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    ax3.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax4 = fig.add_subplot(234)

    y_hist, bin_edges = np.histogram(dat4, bins='auto')
    # bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    axis_spacing = np.arange(0, len(bin_edges), 1)
    if bin_slice_factor == 3:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)[0:-1]
    else:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)
    bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat4, bins=bin_edges[where_within])

    hist_dat4 = ax4.hist(dat4, color=eqdur_color, bins=bin_edges) #, weights=np.ones(len(dat4)) / len(dat4))  # , edgecolor='#000000', linewidth=1.2)
    ax4.hist(dat4, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat4)) / len(dat4))
    ax4.plot([0, 0], [0, np.max(hist_dat4[0]) * 10], '--', color='#000000', lw=1)

    ax4.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax4.set_ylim([0, np.max(hist_dat4[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax4.set_xlabel("% Difference From True Equivalent Duration", fontsize=font_size, style='normal', family='sans-serif')
    ax4.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax4.set_title("Equivalent Duration", fontsize=font_size, style='normal',family='sans-serif')
    ax4.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax5 = fig.add_subplot(235)

    y_hist, bin_edges = np.histogram(dat5, bins='auto')
    #bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    axis_spacing = np.arange(0, len(bin_edges), 1)
    if bin_slice_factor == 3:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)[0:-1]
    else:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
                                     1. / bin_slice_factor)
    bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    bin_width = np.diff(bin_edges)[0]
    where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    y_hist, bin_edges = np.histogram(dat5, bins=bin_edges[where_within])

    hist_dat5 = ax5.hist(dat5, color=energy_color, bins=bin_edges) #, weights=np.ones(len(dat5)) / len(dat5))  # , edgecolor='#000000', linewidth=1.2)
    ax5.hist(dat5, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat5)) / len(dat5))
    ax5.plot([0, 0], [0, np.max(hist_dat5[0]) * 10], '--', color='#000000', lw=1)

    ax5.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax5.set_ylim([0, np.max(hist_dat5[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax5.set_xlabel("% Difference From True Flare Energy", fontsize=font_size, style='normal',family='sans-serif')
    # ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax5.set_title("Flare Energy", fontsize=font_size, style='normal', family='sans-serif')
    ax5.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax6 = fig.add_subplot(236)

    y_hist, bin_edges = np.histogram(dat6, bins='auto')
    where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
    y_hist, bin_edges = np.histogram(dat6, bins=bin_edges[where_within])

    hist_dat6 = ax6.hist(dat6, color=impulsiveness_color, bins=bin_edges) #, weights=np.ones(len(dat6)) / len(dat6))  # , edgecolor='#000000', linewidth=1.2)
    ax6.hist(dat6, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat6)) / len(dat6))
    ax6.plot([0, 0], [0, np.max(hist_dat6[0]) * 10], '--', color='#000000', lw=1)

    ax6.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
    ax6.set_ylim([0, np.max(hist_dat6[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax6.set_xlabel("Distribution of Impulsiveness Index of Flares Tested", fontsize=font_size, style='normal', family='sans-serif')
    # ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax6.set_title("Impulsive Index", fontsize=font_size, style='normal', family='sans-serif')
    ax6.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)



    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
    #import pdb;pdb.set_trace()

def plot_hist_cant_fit(t_peak_cant_fit,fwhm_cant_fit,ampl_cant_fit,impulsiveness_cant_fit, save_as):

    print('Plotting Simple Histograms...')

    font_size = 'medium'
    nbins = 'auto'
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'
    eqdur_color = '#cc6600'
    energy_color = '#666699'
    impulsiveness_color = '#669999'
    x_factor = 20.
    bin_slice_factor = 3.

    impulse_max_lim_factor = 0.50

    dat1 = np.array(t_peak_cant_fit) #*24*60
    dat2 = np.array(fwhm_cant_fit)*24*60
    dat3 = np.array(ampl_cant_fit)
    dat4 = np.array(impulsiveness_cant_fit)

    fig = plt.figure(1, figsize=(15, 4), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(141)

    y_hist, bin_edges = np.histogram(dat1, bins='auto')
    #bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    # axis_spacing = np.arange(0, len(bin_edges), 1)
    # if bin_slice_factor == 3:
    #     new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
    #                                  1. / bin_slice_factor)[0:-1]
    # else:
    #     new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
    #                                  1. / bin_slice_factor)
    # bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    # bin_width = np.diff(bin_edges)[0]
    # where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    # y_hist, bin_edges = np.histogram(dat1, bins=bin_edges[where_within])

    hist_dat1 = ax1.hist(dat1, color=t_peak_color, bins=bin_edges) #, weights=np.ones(len(dat1))/len(dat1)) #, edgecolor='#000000', linewidth=1.2)
    ax1.hist(dat1, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat1))/len(dat1))
    #ax1.plot([0,0],[0,np.max(hist_dat1[0])*10], '--',  color='#000000', lw=1) #, label="Rotation Model")

    # ax1.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax1.set_ylim([0, np.max(hist_dat1[0]) * 1.10])
    #plt.legend(fontsize=10, loc="upper left")
    ax1.set_xlabel("Fractional Location of Peak Time in Window", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title("Peak Time", fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax2 = fig.add_subplot(142)

    y_hist, bin_edges = np.histogram(dat2, bins='auto')
    # bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    # axis_spacing = np.arange(0, len(bin_edges), 1)
    # if bin_slice_factor == 3:
    #     new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
    #                                  1. / bin_slice_factor)[0:-1]
    # else:
    #     new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
    #                                  1. / bin_slice_factor)
    # bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    # bin_width = np.diff(bin_edges)[0]
    # where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    # y_hist, bin_edges = np.histogram(dat2, bins=bin_edges[where_within])

    #ax2.hist(dat2, color=fwhm_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat2 = ax2.hist(dat2, color=fwhm_color, bins=bin_edges) #, weights=np.ones(len(dat2))/len(dat2)) #, edgecolor='#000000', linewidth=1.2)
    ax2.hist(dat2, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat2))/len(dat2))
    #ax2.plot([0, 0], [0, np.max(hist_dat2[0]) * 10], '--', color='#000000', lw=1)

    # ax2.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax2.set_ylim([0, np.max(hist_dat2[0]) * 1.10])
    #plt.legend(fontsize=10, loc="upper left")
    ax2.set_xlabel("FWHM (min)", fontsize=font_size, style='normal', family='sans-serif')
    #ax2.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_title("FWHM", fontsize=font_size, style='normal', family='sans-serif')
    ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax3 = fig.add_subplot(143)

    y_hist, bin_edges = np.histogram(dat3, bins='auto')
    #bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    # axis_spacing = np.arange(0, len(bin_edges), 1)
    # if bin_slice_factor == 3:
    #     new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
    #                                  1. / bin_slice_factor)[0:-1]
    # else:
    #     new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
    #                                  1. / bin_slice_factor)
    # bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    # bin_width = np.diff(bin_edges)[0]
    # where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    # y_hist, bin_edges = np.histogram(dat3, bins=bin_edges[where_within])

    hist_dat3 = ax3.hist(dat3, color=ampl_color, bins=bin_edges) #, weights=np.ones(len(dat3))/len(dat3)) #, edgecolor='#000000', linewidth=1.2)
    ax3.hist(dat3, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat3))/len(dat3))
    #ax3.plot([0, 0], [0, np.max(hist_dat3[0]) * 10], '--', color='#000000', lw=1)

    # ax3.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax3.set_ylim([0, np.max(hist_dat3[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax3.set_xlabel("Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    #ax3.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax3.set_title("Amplitude", fontsize=font_size, style='normal', family='sans-serif')
    ax3.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax4 = fig.add_subplot(144)

    y_hist, bin_edges = np.histogram(dat4, bins='auto')
    # # bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    # axis_spacing = np.arange(0, len(bin_edges), 1)
    # if bin_slice_factor == 3:
    #     new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
    #                                  1. / bin_slice_factor)[0:-1]
    # else:
    #     new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor,
    #                                  1. / bin_slice_factor)
    # bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    # bin_width = np.diff(bin_edges)[0]
    # where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    # y_hist, bin_edges = np.histogram(dat4, bins=bin_edges[where_within])

    hist_dat4 = ax4.hist(dat4, color=impulsiveness_color, bins=bin_edges) #, weights=np.ones(len(dat4)) / len(dat4))  # , edgecolor='#000000', linewidth=1.2)
    ax4.hist(dat4, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat4)) / len(dat4))
    #ax4.plot([0, 0], [0, np.max(hist_dat4[0]) * 10], '--', color='#000000', lw=1)

    # ax4.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax4.set_ylim([0, np.max(hist_dat4[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax4.set_xlabel("Impulsiveness", fontsize=font_size, style='normal', family='sans-serif')
    #ax4.set_ylabel("Fraction of Total", fontsize=font_size, style='normal', family='sans-serif')
    ax4.set_title("Impulsiveness", fontsize=font_size, style='normal',family='sans-serif')
    ax4.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)



    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
    #import pdb;pdb.set_trace()



def plot_stat_cum(t_peak_opt,fwhm_opt,ampl_opt,fwhm_true, ampl_true,save_as):
    font_size = 'medium'
    nbins = 100
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'

    fig = plt.figure(1, figsize=(14, 4), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(131)
    dat1 = np.array(t_peak_opt)*24*60
    #ax1.hist(dat1, color=t_peak_color, bins=nbins, edgecolor='#000000', linewidth=1.2, cumulative=True, density=True, histtype='step')
    hist_dat1 = ax1.hist(dat1, color=t_peak_color, bins=nbins, linewidth=1.5, cumulative=True, density=True, histtype='step')
    ax1.plot([0,0],[0,np.max(hist_dat1[0])*10], '--',  color='black', lw=1) #, label="Rotation Model")
    ax1.set_xlim(np.min(hist_dat1[1]), np.max(hist_dat1[1]))
    ax1.set_ylim([0, 1])
    #plt.legend(fontsize=10, loc="upper left")
    ax1.set_xlabel("Difference From True Peak Time (min)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Cumulative Fraction", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title("Peak Time ", fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    ax2 = fig.add_subplot(132)
    dat2 = np.array(fwhm_opt)*24*60
    hist_dat2 = ax2.hist(dat2, color=fwhm_color, bins=nbins, linewidth=1.5, cumulative=True, density=True, histtype='step')
    #ax2.plot([flare_properties['fwhm'][0], flare_properties['fwhm'][0]], [0, np.max(hist_dat2[0]) * 10], '--', color='black', lw=1)
    ax2.plot([0, 0], [0, np.max(hist_dat2[0]) * 10], '--', color='#000000', lw=1)
    ax2.set_xlim(np.min(hist_dat2[1]), np.max(hist_dat2[1]))
    ax2.set_ylim([0, 1])
    #plt.legend(fontsize=10, loc="upper left")
    ax2.set_xlabel("Difference From True FWHM (min)", fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_title("True FWHM: " + str(np.round(fwhm_true[0]*24*60,3)), fontsize=font_size, style='normal', family='sans-serif')
    ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    ax3 = fig.add_subplot(133)
    dat3 = ampl_opt
    hist_dat3 = ax3.hist(dat3, color=ampl_color, bins=nbins*2, linewidth=1.5, cumulative=True, density=True, histtype='step')
    # ax3.plot([test_amplitudes[s]*stddev, test_amplitudes[s]*stddev], [0, np.max(hist_dat3[0]) * 10], '--', color='black', lw=1)
    ax3.plot([0, 0], [0, np.max(hist_dat3[0]) * 10], '--', color='#000000', lw=1)
    ax3.set_xlim(np.min(hist_dat3[1]), np.max(hist_dat3[1]))
    ax3.set_ylim([0, 1])
    # plt.legend(fontsize=10, loc="upper left")
    ax3.set_xlabel("Difference From True Amplitude (ppt)", fontsize=font_size, style='normal', family='sans-serif')
    ax3.set_title("True Amplitude: " + str(np.round(ampl_true[0],3)), fontsize=font_size, style='normal', family='sans-serif')
    ax3.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()

def plot_test_fit(x_flare,y_flare,x_fit,y_fit,y_true,eq_dur,flare_energy,eq_dur_true,flare_energy_true,save_as):
    font_size = 'large'
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'

    max_amp = np.max([np.max(y_true), np.max(y_flare), np.max(y_fit)]) - 1.0
    max_gap = 0.05 * max_amp

    #import pdb; pdb.set_trace()

    fig = plt.figure(1, figsize=(7,5.5), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(111)
    #ax1.set_xlim([0, 1])
    ax1.set_xlabel("Time (days)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Flux (ppt)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title(r'Equivalent Duration = ' + str(np.round(eq_dur, 2)) + ' (sec) Flare Energy = ' + str('{:0.3e}'.format(flare_energy)) + 'erg s$^{-1}$\nTrue Equivalent Duration = ' + str(np.round(eq_dur_true, 2)) + ' (sec) True Flare Energy = ' + str('{:0.3e}'.format(flare_energy_true)) + 'erg s$^{-1}$', pad=10, fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    ax1.plot(x_fit, y_true, c='#000000', lw=0.5, label='True Flare')
    ax1.plot(x_fit, y_fit, c='blue', lw=2, label='Flare Fit')
    ax1.fill_between(x_fit, y_fit, y2=np.zeros_like(y_fit), color='blue', alpha=0.15)
    ax1.scatter(x_flare, y_flare, c='red', label='Test Flare')
    ax1.set_ylim([1.0 - max_gap, 1 + max_amp + max_gap])
    ax1.legend(fontsize=font_size, loc='upper right')
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()

def fit_statistics(cadence, stddev, n_reps = 100):

    #test_amplitudes = np.geomspace(10,20000,20)
    test_amplitudes = [500,1000,2000,5000,10000,20000,50000]

    for s in range(len(test_amplitudes)):

        t_peak_opt = []
        fwhm_opt = []
        ampl_opt = []

        t_peak_true = []
        fwhm_true = []
        ampl_true = []

        for rep in range(n_reps):

            x_synth, y_synth, y_synth_noscatter, flare_window, flare_properties = create_synthetic(cadence, stddev, given_amplitude=test_amplitudes[s],fixed_amplitude=True) #, fixed_fwhm=True)

            print('Generating Statistics for Amplitude ' + str(test_amplitudes[s]) + '...\nRep. ' + str(rep+1) + ':')
            for w in range(len(flare_window)):

                # if np.mod(w+1,100) == 0:
                #     print(w+1)

                t_peak_true.append(flare_properties["tpeak"][w])
                fwhm_true.append(flare_properties["fwhm"][w])
                ampl_true.append(flare_properties["amplitude"][w])

                where_start = np.int(np.floor(np.random.uniform(0, 16, 1)))

                x_downsample = x_synth[where_start::15]
                y_downsample = y_synth[where_start::15]
                # y_noscatter_downsample = y_synth_noscatter[where_start::15]

                window = np.where((x_downsample >= flare_window[w][0]) & (x_downsample <= flare_window[w][1]))[0]
                x_window = x_downsample[window]
                y_window = y_downsample[window]
                # y_noscatter_window = y_noscatter_downsample[window]

                guess_peak = x_window[np.where(y_window == np.max(y_window))[0][0]]
                guess_fwhm = 0.01
                guess_ampl = y_window[np.where(y_window == np.max(y_window))[0][0]]

                try:
                    popt, pcov = optimize.curve_fit(aflare1, x_window, y_window, p0=(guess_peak, guess_fwhm, guess_ampl)) # diag=(1./x_window.mean(),1./y_window.mean()))
                except:
                    continue

                if np.mod(w+1,20) == 0:
                    x_fit = np.linspace(np.min(x_window),np.max(x_window),500)
                    y_fit = aflare1(x_fit,*popt)
                    y_true = aflare1(x_fit,flare_properties["tpeak"][w],flare_properties["fwhm"][w],flare_properties["amplitude"][w])
                    if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/Amp_' + str(test_amplitudes[s])):
                        os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/Amp_' + str(test_amplitudes[s]))
                    save_as_test = 'Amp_' + str(test_amplitudes[s]) + '/' + str(rep+1) + '_' + str(w+1) + '.pdf'
                    plot_test_fit(x_window,y_window,x_fit,y_fit,y_true,save_as_test)

                t_peak_opt.append((popt[0] - flare_properties["tpeak"][w]))
                fwhm_opt.append((popt[1] - flare_properties["fwhm"][w])/flare_properties["tpeak"][w]*100)
                ampl_opt.append((popt[2] - flare_properties["amplitude"][w])/flare_properties["tpeak"][w]*100)

        save_as_hist = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_hist_ampl_' + str(test_amplitudes[s]) + '.pdf'
        plot_stat_hist(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_hist)
        # save_as_cum = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_cum_ampl_' + str(test_amplitudes[s]) + '.pdf'
        # plot_stat_cum(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_cum)
        # import pdb;pdb.set_trace()
def fit_statistics2(cadence, stddev, downsample = True,n_reps = 100):

    #test_amplitudes = np.geomspace(10,20000,20)
    test_amplitudes = [3,5,10,100,250,500,750,1000,1200,1500]

    for s in range(len(test_amplitudes)):

        t_peak_opt = []
        fwhm_opt = []
        ampl_opt = []
        eq_duration_opt = []
        energy_opt = []

        t_peak_true = []
        fwhm_true = []
        ampl_true = []
        eq_duration_true = []
        energy_true = []

        for rep in range(n_reps):

            x_synth, y_synth, y_synth_noscatter, flare_window, flare_properties = create_synthetic(cadence, stddev, given_amplitude=test_amplitudes[s],fixed_ampltude=True,fixed_fwhm=False) #, fixed_fwhm=True)

            print('Generating Statistics for Amplitude ' + str(test_amplitudes[s]) + '...\nRep. ' + str(rep+1) + ':')
            for w in range(len(flare_window)):

                # if np.mod(w+1,100) == 0:
                #     print(w+1)

                t_peak_true.append(flare_properties["tpeak"][w])
                fwhm_true.append(flare_properties["fwhm"][w])
                ampl_true.append(flare_properties["amplitude"][w])

                if downsample == True:
                    where_start = np.int(np.floor(np.random.uniform(0, 16, 1)))

                    x_downsample = x_synth[where_start::15]
                    y_downsample = y_synth[where_start::15]
                    # y_noscatter_downsample = y_synth_noscatter[where_start::15]
                if downsample == False:
                    x_downsample = x_synth[0::1]
                    y_downsample = y_synth[0::1]
                    # y_noscatter_downsample = y_synth_noscatter[0::1]

                window = np.where((x_downsample >= flare_window[w][0]) & (x_downsample <= flare_window[w][1]))[0]
                x_window = x_downsample[window]
                y_window = y_downsample[window]
                # y_noscatter_window = y_noscatter_downsample[window]

                guess_peak = x_window[np.where(y_window == np.max(y_window))[0][0]]
                guess_fwhm = 0.01
                guess_ampl = y_window[np.where(y_window == np.max(y_window))[0][0]]

                try:
                    popt, pcov = optimize.curve_fit(aflare1, x_window, y_window, p0=(guess_peak, guess_fwhm, guess_ampl)) # diag=(1./x_window.mean(),1./y_window.mean()))
                except:
                    continue

                x_fit = np.linspace(np.min(x_window), np.max(x_window), 5000)
                y_fit = aflare1(x_fit, *popt)
                y_true = aflare1(x_fit, flare_properties["tpeak"][w], flare_properties["fwhm"][w], flare_properties["amplitude"][w])

                L_star = 1.2  # solar luminosity
                L_star *= 3.827e33  # convert to erg/s

                eq_dur = np.trapz(y_fit, x=x_fit)
                eq_dur *= 86400  # convert days to seconds
                flare_energy = L_star * eq_dur

                eq_dur_true = np.trapz(y_true, x=x_fit)
                eq_dur_true *= (24*60*60)  # convert days to seconds
                flare_energy_true = L_star * eq_dur_true

                eq_duration_true.append(eq_dur_true)
                energy_true.append(flare_energy_true)

                eq_duration_opt.append((eq_dur - eq_dur_true)/eq_dur_true*100)
                energy_opt.append((flare_energy - flare_energy_true)/flare_energy_true*100)

                if np.mod(w+1,50) == 0:
                    if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/Amp_' + str(test_amplitudes[s])):
                        os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/Amp_' + str(test_amplitudes[s]))
                    save_as_test = 'Amp_' + str(test_amplitudes[s]) + '/' + str(rep+1) + '_' + str(w+1) + '.pdf'
                    plot_test_fit(x_window,y_window+1,x_fit,y_fit+1,y_true+1,eq_dur,flare_energy,eq_dur_true,flare_energy_true,save_as_test)

                t_peak_opt.append((popt[0] - flare_properties["tpeak"][w]))
                fwhm_opt.append((popt[1] - flare_properties["fwhm"][w])/flare_properties["tpeak"][w]*100)
                ampl_opt.append((popt[2] - flare_properties["amplitude"][w])/flare_properties["tpeak"][w]*100)

        save_as_hist = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_hist_ampl_' + str(test_amplitudes[s]) + '.pdf'
        #plot_stat_hist(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_hist)
        plot_stat_hist2(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,energy_opt,downsample,save_as_hist)
        # save_as_cum = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_cum_ampl_' + str(test_amplitudes[s]) + '.pdf'
        # plot_stat_cum(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_cum)
        # import pdb;pdb.set_trace()
def fit_statistics3(cadence, stddev, set_fixed_amplitude=False, downsample = True,n_reps = 100):


    if set_fixed_amplitude == True:
        test_amplitudes = [3,5,10,100,250,500,750,1000,1200,1500]

        for s in range(len(test_amplitudes)):

            t_peak_opt = []
            fwhm_opt = []
            ampl_opt = []
            eq_duration_opt = []
            energy_opt = []

            t_peak_true = []
            fwhm_true = []
            ampl_true = []
            eq_duration_true = []
            energy_true = []

            for rep in range(n_reps):

                x_synth, y_synth, y_synth_noscatter, flare_window, flare_properties = create_synthetic(cadence, stddev, given_amplitude=test_amplitudes[s],fixed_amplitude=True,fixed_fwhm=False) #, fixed_fwhm=True)

                print('Generating Statistics for Amplitude ' + str(test_amplitudes[s]) + '...\nRep. ' + str(rep+1) + ':')
                for w in range(len(flare_window)):

                    # if np.mod(w+1,100) == 0:
                    #     print(w+1)

                    t_peak_true.append(flare_properties["tpeak"][w])
                    fwhm_true.append(flare_properties["fwhm"][w])
                    ampl_true.append(flare_properties["amplitude"][w])
                    impulsive_index_true.append(flare_properties["amplitude"][w] / flare_properties["fwhm"][w])

                    if downsample == True:
                        where_start = np.int(np.floor(np.random.uniform(0, 16, 1)))

                        x_downsample = x_synth[where_start::15]
                        y_downsample = y_synth[where_start::15]
                        # y_noscatter_downsample = y_synth_noscatter[where_start::15]
                    if downsample == False:
                        x_downsample = x_synth[0::1]
                        y_downsample = y_synth[0::1]
                        # y_noscatter_downsample = y_synth_noscatter[0::1]

                    window = np.where((x_downsample >= flare_window[w][0]) & (x_downsample <= flare_window[w][1]))[0]
                    x_window = x_downsample[window]
                    y_window = y_downsample[window]
                    # y_noscatter_window = y_noscatter_downsample[window]

                    guess_peak = x_window[np.where(y_window == np.max(y_window))[0][0]]
                    guess_fwhm = 0.01
                    guess_ampl = y_window[np.where(y_window == np.max(y_window))[0][0]]

                    try:
                        popt, pcov = optimize.curve_fit(aflare1, x_window, y_window, p0=(guess_peak, guess_fwhm, guess_ampl)) # diag=(1./x_window.mean(),1./y_window.mean()))
                    except:
                        continue

                    x_fit = np.linspace(np.min(x_window), np.max(x_window), 5000)
                    y_fit = aflare1(x_fit, *popt)
                    y_true = aflare1(x_fit, flare_properties["tpeak"][w], flare_properties["fwhm"][w], flare_properties["amplitude"][w])

                    L_star = 1.2  # solar luminosity
                    L_star *= 3.827e33  # convert to erg/s

                    eq_dur = np.trapz(y_fit, x=x_fit)
                    eq_dur *= 86400  # convert days to seconds
                    flare_energy = L_star * eq_dur

                    eq_dur_true = np.trapz(y_true, x=x_fit)
                    eq_dur_true *= (24*60*60)  # convert days to seconds
                    flare_energy_true = L_star * eq_dur_true

                    eq_duration_true.append(eq_dur_true)
                    energy_true.append(flare_energy_true)

                    eq_duration_opt.append((eq_dur - eq_dur_true)/eq_dur_true*100)
                    energy_opt.append((flare_energy - flare_energy_true)/flare_energy_true*100)

                    if np.mod(w+1,100) == 0:
                        if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/Amp_' + str(test_amplitudes[s])):
                            os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/Amp_' + str(test_amplitudes[s]))
                        save_as_test = 'Amp_' + str(test_amplitudes[s]) + '/' + str(rep+1) + '_' + str(w+1) + '.pdf'
                        plot_test_fit(x_window,y_window+1,x_fit,y_fit+1,y_true+1,eq_dur,flare_energy,eq_dur_true,flare_energy_true,save_as_test)

                    t_peak_opt.append((popt[0] - flare_properties["tpeak"][w]))
                    fwhm_opt.append((popt[1] - flare_properties["fwhm"][w])/flare_properties["tpeak"][w]*100)
                    ampl_opt.append((popt[2] - flare_properties["amplitude"][w])/flare_properties["tpeak"][w]*100)

                    if np.mod(w + 1, 100) == 0:
                        save_as_hist = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_hist_ampl_' + str(test_amplitudes[s]) + '.pdf'
                        # plot_stat_hist(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_hist)
                        plot_stat_hist2(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, downsample, save_as_hist)

            save_as_hist = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_hist_ampl_' + str(test_amplitudes[s]) + '.pdf'
            #plot_stat_hist(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_hist)
            plot_stat_hist2(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,energy_opt,impulsive_index_true,downsample,save_as_hist)
            # save_as_cum = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_cum_ampl_' + str(test_amplitudes[s]) + '.pdf'
            # plot_stat_cum(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_cum)
            # import pdb;pdb.set_trace()
    if set_fixed_amplitude == False:

        t_peak_opt = []
        fwhm_opt = []
        ampl_opt = []
        eq_duration_opt = []
        energy_opt = []

        t_peak_true = []
        fwhm_true = []
        ampl_true = []
        eq_duration_true = []
        energy_true = []
        impulsive_index_true = []

        for rep in range(n_reps):

            x_synth, y_synth, y_synth_noscatter, flare_window, flare_properties = create_synthetic(cadence, stddev, fixed_amplitude=False, fixed_fwhm=False)  # , fixed_fwhm=True)

            print('Generating Flare Statistics...\nRep. ' + str(rep + 1) + ':')
            for w in range(len(flare_window)):

                # if np.mod(w+1,100) == 0:
                #     print(w+1)


                if downsample == True:
                    where_start = np.int(np.floor(np.random.uniform(0, 16, 1)))

                    x_downsample = x_synth[where_start::15]
                    y_downsample = y_synth[where_start::15]
                    # y_noscatter_downsample = y_synth_noscatter[where_start::15]
                if downsample == False:
                    x_downsample = x_synth[0::1]
                    y_downsample = y_synth[0::1]
                    # y_noscatter_downsample = y_synth_noscatter[0::1]

                window = np.where((x_downsample >= flare_window[w][0]) & (x_downsample <= flare_window[w][1]))[0]
                x_window = x_downsample[window]
                y_window = y_downsample[window]
                # y_noscatter_window = y_noscatter_downsample[window]

                guess_peak = x_window[np.where(y_window == np.max(y_window))[0][0]]
                guess_fwhm = 0.01
                guess_ampl = y_window[np.where(y_window == np.max(y_window))[0][0]]

                try:
                    popt, pcov = optimize.curve_fit(aflare1, x_window, y_window, p0=(
                    guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                except:
                    continue

                x_fit = np.linspace(np.min(x_window), np.max(x_window), 5000)
                y_fit = aflare1(x_fit, *popt)
                y_true = aflare1(x_fit, flare_properties["tpeak"][w], flare_properties["fwhm"][w], flare_properties["amplitude"][w])

                L_star = 1.2  # solar luminosity
                L_star *= 3.827e33  # convert to erg/s

                eq_dur = np.trapz(y_fit, x=x_fit)
                eq_dur *= 86400  # convert days to seconds
                flare_energy = L_star * eq_dur

                eq_dur_true = np.trapz(y_true, x=x_fit)
                eq_dur_true *= (24 * 60 * 60)  # convert days to seconds
                flare_energy_true = L_star * eq_dur_true

                eq_duration_true.append(eq_dur_true)
                energy_true.append(flare_energy_true)

                eq_duration_opt.append((eq_dur - eq_dur_true) / eq_dur_true * 100)
                energy_opt.append((flare_energy - flare_energy_true) / flare_energy_true * 100)

                if np.mod(w + 1, 100) == 0:
                    if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes/'):
                        os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes/')
                    save_as_test = 'All_Amplitudes/' + str(rep + 1) + '_' + str(w + 1) + '.pdf'
                    plot_test_fit(x_window, y_window + 1, x_fit, y_fit + 1, y_true + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)

                t_peak_opt.append((popt[0] - flare_properties["tpeak"][w]))
                fwhm_opt.append((popt[1] - flare_properties["fwhm"][w]) / flare_properties["fwhm"][w] * 100)
                ampl_opt.append((popt[2] - flare_properties["amplitude"][w]) / flare_properties["amplitude"][w] * 100)

                t_peak_true.append(flare_properties["tpeak"][w])
                fwhm_true.append(flare_properties["fwhm"][w])
                ampl_true.append(flare_properties["amplitude"][w])
                impulsive_index_true.append(flare_properties["amplitude"][w] / flare_properties["fwhm"][w])

                if np.mod(w + 1, 100) == 0:
                    save_as_hist = 'All_Amplitudes/fit_stat_hist_all_amplitudes.pdf'
                    # plot_stat_hist(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_hist)
                    plot_stat_hist3(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, save_as_hist)

                    save_as_hist_2D_fwhm = 'All_Amplitudes/fit_stat_hist_2D_fwhm_sum.pdf'
                    sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                  ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                                  stddev, property_to_sort='fwhm')
                    save_as_hist_2D_fwhm = 'All_Amplitudes/fit_stat_hist_2D_fwhm_cum.pdf'
                    sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                   ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                                   save_as_hist_2D_fwhm,
                                   stddev, property_to_sort='fwhm')
                    save_as_hist_2D_ampl = 'All_Amplitudes/fit_stat_hist_2D_ampl_sum.pdf'
                    sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                   ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_ampl,
                                   stddev, property_to_sort='amplitude')
                    save_as_hist_2D_ampl = 'All_Amplitudes/fit_stat_hist_2D_ampl_cum.pdf'
                    sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                   ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                                   save_as_hist_2D_ampl, stddev, property_to_sort='amplitude')


        # save_as_hist = 'All_Amplitudes/fit_stat_hist_all_amplitudes.pdf'
        # plot_stat_hist3(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, save_as_hist)
        #
        # save_as_hist_2D = 'All_Amplitudes/fit_stat_hist_2D.pdf'
        # sort_property2(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
        #               ampl_true, eq_duration_true, energy_true, impulsive_index_true, save_as_hist_2D, property_to_sort='fwhm')
        # import pdb;pdb.set_trace()
def fit_statistics4(cadence, stddev, set_fixed_amplitude=False, downsample = True, set_cadence = 30, n_reps = 100):


    if set_fixed_amplitude == True:
        test_amplitudes = [3,5,10,100,250,500,750,1000,1200,1500]

        for s in range(len(test_amplitudes)):

            t_peak_opt = []
            fwhm_opt = []
            ampl_opt = []
            eq_duration_opt = []
            energy_opt = []

            t_peak_true = []
            fwhm_true = []
            ampl_true = []
            eq_duration_true = []
            energy_true = []

            for rep in range(n_reps):

                x_synth, y_synth, y_synth_noscatter, flare_window, flare_properties = create_synthetic(cadence, stddev, given_amplitude=test_amplitudes[s],fixed_amplitude=True,fixed_fwhm=False) #, fixed_fwhm=True)

                print('Generating Statistics for Amplitude ' + str(test_amplitudes[s]) + '...\nRep. ' + str(rep+1) + ':')
                for w in range(len(flare_window)):

                    # if np.mod(w+1,100) == 0:
                    #     print(w+1)

                    t_peak_true.append(flare_properties["tpeak"][w])
                    fwhm_true.append(flare_properties["fwhm"][w])
                    ampl_true.append(flare_properties["amplitude"][w])
                    impulsive_index_true.append(flare_properties["amplitude"][w] / flare_properties["fwhm"][w])

                    if downsample == True:
                        where_start = np.int(np.floor(np.random.uniform(0, 16, 1)))

                        x_downsample = x_synth[where_start::15]
                        y_downsample = y_synth[where_start::15]
                        # y_noscatter_downsample = y_synth_noscatter[where_start::15]
                    if downsample == False:
                        x_downsample = x_synth[0::1]
                        y_downsample = y_synth[0::1]
                        # y_noscatter_downsample = y_synth_noscatter[0::1]

                    window = np.where((x_downsample >= flare_window[w][0]) & (x_downsample <= flare_window[w][1]))[0]
                    x_window = x_downsample[window]
                    y_window = y_downsample[window]
                    # y_noscatter_window = y_noscatter_downsample[window]

                    guess_peak = x_window[np.where(y_window == np.max(y_window))[0][0]]
                    guess_fwhm = 0.01
                    guess_ampl = y_window[np.where(y_window == np.max(y_window))[0][0]]

                    try:
                        popt, pcov = optimize.curve_fit(aflare1, x_window, y_window, p0=(guess_peak, guess_fwhm, guess_ampl)) # diag=(1./x_window.mean(),1./y_window.mean()))
                    except:
                        continue

                    x_fit = np.linspace(np.min(x_window), np.max(x_window), 5000)
                    y_fit = aflare1(x_fit, *popt)
                    y_true = aflare1(x_fit, flare_properties["tpeak"][w], flare_properties["fwhm"][w], flare_properties["amplitude"][w])

                    L_star = 1.2  # solar luminosity
                    L_star *= 3.827e33  # convert to erg/s

                    eq_dur = np.trapz(y_fit, x=x_fit)
                    eq_dur *= 86400  # convert days to seconds
                    flare_energy = L_star * eq_dur

                    eq_dur_true = np.trapz(y_true, x=x_fit)
                    eq_dur_true *= (24*60*60)  # convert days to seconds
                    flare_energy_true = L_star * eq_dur_true

                    eq_duration_true.append(eq_dur_true)
                    energy_true.append(flare_energy_true)

                    eq_duration_opt.append((eq_dur - eq_dur_true)/eq_dur_true*100)
                    energy_opt.append((flare_energy - flare_energy_true)/flare_energy_true*100)

                    if np.mod(w+1,100) == 0:
                        if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/Amp_' + str(test_amplitudes[s])):
                            os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/Amp_' + str(test_amplitudes[s]))
                        save_as_test = 'Amp_' + str(test_amplitudes[s]) + '/' + str(rep+1) + '_' + str(w+1) + '.pdf'
                        plot_test_fit(x_window,y_window+1,x_fit,y_fit+1,y_true+1,eq_dur,flare_energy,eq_dur_true,flare_energy_true,save_as_test)

                    t_peak_opt.append((popt[0] - flare_properties["tpeak"][w]))
                    fwhm_opt.append((popt[1] - flare_properties["fwhm"][w])/flare_properties["tpeak"][w]*100)
                    ampl_opt.append((popt[2] - flare_properties["amplitude"][w])/flare_properties["tpeak"][w]*100)

                    if np.mod(w + 1, 100) == 0:
                        save_as_hist = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_hist_ampl_' + str(test_amplitudes[s]) + '.pdf'
                        # plot_stat_hist(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_hist)
                        plot_stat_hist2(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, downsample, save_as_hist)

            save_as_hist = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_hist_ampl_' + str(test_amplitudes[s]) + '.pdf'
            #plot_stat_hist(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_hist)
            plot_stat_hist2(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,energy_opt,impulsive_index_true,downsample,save_as_hist)
            # save_as_cum = 'Amp_' + str(test_amplitudes[s]) + '/fit_stat_cum_ampl_' + str(test_amplitudes[s]) + '.pdf'
            # plot_stat_cum(t_peak_opt, fwhm_opt, ampl_opt, fwhm_true, ampl_true, save_as_cum)
            # import pdb;pdb.set_trace()
    if set_fixed_amplitude == False:

        t_peak_opt = []
        fwhm_opt = []
        ampl_opt = []
        eq_duration_opt = []
        energy_opt = []

        t_peak_true = []
        fwhm_true = []
        ampl_true = []
        eq_duration_true = []
        energy_true = []
        impulsive_index_true = []

        t_peak_cant_fit = []
        fwhm_cant_fit = []
        ampl_cant_fit = []
        impulsive_index_cant_fit = []

        for rep in range(n_reps):

            x_synth, y_synth, y_synth_noscatter, flare_window, flare_properties = create_synthetic(cadence, stddev, set_cadence, fixed_amplitude=False, fixed_fwhm=False)  # , fixed_fwhm=True)

            print('Generating Flare Statistics...\nRep. ' + str(rep + 1) + ':')
            for w in range(len(flare_window)):

                # if np.mod(w+1,100) == 0:
                #     print(w+1)


                if downsample == True:
                    where_start = np.int(np.floor(np.random.uniform(0, set_cadence+1, 1)))

                    x_downsample = x_synth[where_start::set_cadence]
                    y_downsample = y_synth[where_start::set_cadence]
                    # y_noscatter_downsample = y_synth_noscatter[where_start::15]
                if downsample == False:
                    x_downsample = x_synth[0::1]
                    y_downsample = y_synth[0::1]
                    # y_noscatter_downsample = y_synth_noscatter[0::1]

                window = np.where((x_downsample >= flare_window[w][0]) & (x_downsample <= flare_window[w][1]))[0]
                x_window = x_downsample[window]
                y_window = y_downsample[window]
                # y_noscatter_window = y_noscatter_downsample[window]

                guess_peak = x_window[np.where(y_window == np.max(y_window))[0][0]]
                guess_fwhm = 0.01
                guess_ampl = y_window[np.where(y_window == np.max(y_window))[0][0]]

                try:
                    popt, pcov = optimize.curve_fit(aflare1, x_window, y_window, p0=(guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                except:
                    t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                    fwhm_cant_fit.append(flare_properties["fwhm"][w])
                    ampl_cant_fit.append(flare_properties["amplitude"][w])
                    impulsive_index_cant_fit.append(flare_properties["amplitude"][w] / flare_properties["fwhm"][w])
                    continue

                x_fit = np.linspace(np.min(x_window), np.max(x_window), 5000)
                y_fit = aflare1(x_fit, *popt)
                y_true = aflare1(x_fit, flare_properties["tpeak"][w], flare_properties["fwhm"][w], flare_properties["amplitude"][w])

                L_star = 1.2  # solar luminosity
                L_star *= 3.827e33  # convert to erg/s

                eq_dur = np.trapz(y_fit, x=x_fit)
                eq_dur *= 86400  # convert days to seconds
                flare_energy = L_star * eq_dur

                eq_dur_true = np.trapz(y_true, x=x_fit)
                eq_dur_true *= (24 * 60 * 60)  # convert days to seconds
                flare_energy_true = L_star * eq_dur_true

                eq_duration_true.append(eq_dur_true)
                energy_true.append(flare_energy_true)

                eq_duration_opt.append((eq_dur - eq_dur_true) / eq_dur_true * 100)
                energy_opt.append((flare_energy - flare_energy_true) / flare_energy_true * 100)

                if np.mod(w + 1, 100) == 0:
                    if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_cadence)) + 'min/'):
                        os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_cadence)) + 'min/')
                    save_as_test = 'All_Amplitudes_' + str(int(set_cadence)) + 'min/' + str(rep + 1) + '_' + str(w + 1) + '.pdf'
                    plot_test_fit(x_window, y_window + 1, x_fit, y_fit + 1, y_true + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)

                t_peak_opt.append((popt[0] - flare_properties["tpeak"][w]))
                fwhm_opt.append((popt[1] - flare_properties["fwhm"][w]) / flare_properties["fwhm"][w] * 100)
                ampl_opt.append((popt[2] - flare_properties["amplitude"][w]) / flare_properties["amplitude"][w] * 100)

                t_peak_true.append(flare_properties["tpeak"][w])
                fwhm_true.append(flare_properties["fwhm"][w])
                ampl_true.append(flare_properties["amplitude"][w])
                impulsive_index_true.append(flare_properties["amplitude"][w] / flare_properties["fwhm"][w])

                if np.mod(w + 1, 100) == 0:
                    save_as_hist = 'All_Amplitudes_' + str(int(set_cadence)) + 'min/fit_stat_hist_all_amplitudes.pdf'
                    plot_stat_hist4(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, save_as_hist)

                    save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_cadence)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                    sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                  ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                                  stddev, set_cadence, property_to_sort='fwhm')
                    save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_cadence)) + 'min/fit_stat_hist_2D_fwhm_cum.pdf'
                    sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                   ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                                   save_as_hist_2D_fwhm,
                                   stddev, set_cadence, property_to_sort='fwhm')
                    save_as_hist_2D_ampl = 'All_Amplitudes_' + str(int(set_cadence)) + 'min/fit_stat_hist_2D_ampl_sum.pdf'
                    sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                   ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_ampl,
                                   stddev, set_cadence, property_to_sort='amplitude')
                    save_as_hist_2D_ampl = 'All_Amplitudes_' + str(int(set_cadence)) + 'min/fit_stat_hist_2D_ampl_cum.pdf'
                    sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                   ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                                   save_as_hist_2D_ampl, stddev, set_cadence, property_to_sort='amplitude')

                    if len(fwhm_cant_fit) > 0:
                        save_as_cant_fit = 'All_Amplitudes_' + str(int(set_cadence)) + 'min/fit_stat_hist_cant_fit.pdf'
                        plot_hist_cant_fit(t_peak_cant_fit, fwhm_cant_fit, ampl_cant_fit, impulsive_index_cant_fit, save_as=save_as_cant_fit)


        # save_as_hist = 'All_Amplitudes/fit_stat_hist_all_amplitudes.pdf'
        # plot_stat_hist3(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, save_as_hist)
        #
        # save_as_hist_2D = 'All_Amplitudes/fit_stat_hist_2D.pdf'
        # sort_property2(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
        #               ampl_true, eq_duration_true, energy_true, impulsive_index_true, save_as_hist_2D, property_to_sort='fwhm')
        # import pdb;pdb.set_trace()


def sort_property(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,flare_energy_opt,t_peak_true,fwhm_true,ampl_true,eq_duration_true,flare_energy_true,impulsiveness,save_as,property_to_sort='fwhm'):

    opt_list = [t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,flare_energy_opt]

    if property_to_sort == 'fwhm':

        fwhm_min = 0.5*(1./60.)*(1./24.)
        fwhm_max = 30.*(1./60.)*(1./24.)
        fwhm_grid = np.linspace(fwhm_min, fwhm_max, 51)

        fwhm_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(fwhm_grid)-1):
            # In each iteration, add an empty list to the main list
            fwhm_slots.append([])

        for grid_index in range(1,len(fwhm_grid)):
            for true_index in range(0,len(fwhm_true)):
                if (fwhm_true[true_index] > fwhm_grid[grid_index-1]) and (fwhm_true[true_index] <= fwhm_grid[grid_index]):
                    fwhm_slots[grid_index-1].append(true_index)

        #fwhm_opt = fwhm_opt*24*60



        y_hist, bin_edges = np.histogram(fwhm_opt, bins='fd')
        Z = np.zeros((len(fwhm_grid)-1, len(bin_edges)-1))

        for slot_index in range(len(fwhm_slots)):
            y_hist, bin_edges = np.histogram(np.array(fwhm_opt)[fwhm_slots[slot_index]], bins=bin_edges)
            Z[slot_index, :] = y_hist

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'

        fig = plt.figure(num=None, figsize=(6, 5), facecolor='w', dpi=135)
        ax = fig.add_subplot(111)
        # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
        # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
        ax.set_ylabel('FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
        ax.set_xlabel('% Difference from True FWHM', fontsize=font_size, style=font_style, family=font_family)
        #ax.set_ylim([min(phase_list), max(phase_list)])
        bin_width = np.diff(bin_edges)[0]
        xlim_mult = 10.
        ax.set_xlim(-xlim_mult * bin_width, xlim_mult * bin_width)
        # mx = max(Yin)
        # ax.set_xticks((2,4,6,8,10,12,14,16,18,20,22,24,26,28,30))
        X, Y = np.meshgrid(bin_edges[0:-1] + 0.5*np.diff(bin_edges), fwhm_grid[0:-1]*24*60)
        p = ax.pcolor(X, Y, Z, cmap=cm.bone, edgecolors='face', vmin=Z.min(), vmax=Z.max())
        # cbaxes = fig.add_axes([])
        cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
        cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
        # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)
        cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)

        ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#ffffff')

        plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
        plt.close()
        # plt.show()

        #import pdb; pdb.set_trace()
def sort_property2(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, save_as, property_to_sort='fwhm'):

    opt_list = [np.array(t_peak_opt)*24*60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time (min)','% Difference from True FWHM',
                    '% Difference from True Amplitude','% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    if property_to_sort == 'fwhm':

        fwhm_min = 0.5 * (1. / 60.) * (1. / 24.)
        fwhm_max = 30. * (1. / 60.) * (1. / 24.)
        fwhm_grid = np.linspace(fwhm_min, fwhm_max, 51)

        fwhm_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(fwhm_grid) - 1):
            # In each iteration, add an empty list to the main list
            fwhm_slots.append([])

        for grid_index in range(1, len(fwhm_grid)):
            for true_index in range(0, len(fwhm_true)):
                if (fwhm_true[true_index] > fwhm_grid[grid_index - 1]) and (fwhm_true[true_index] <= fwhm_grid[grid_index]):
                    fwhm_slots[grid_index - 1].append(true_index)

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'

        fig = plt.figure(figsize=(10, 5), facecolor='#000000') #, dpi=300)
        for prop in range(len(opt_list)):

            y_hist, bin_edges = np.histogram(opt_list[prop], bins='fd')
            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))

            for slot_index in range(len(fwhm_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax.set_ylabel('FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
            ax.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)
            ax.set_ylim([np.min(fwhm_grid[0:-1]*24*60), np.max(fwhm_grid[0:-1]*24*60)])
            bin_width = np.diff(bin_edges)[0]
            xlim_mult = 20.
            if prop == 5:
                ax.set_xlim(0, np.max(bin_edges))
            else:
                #ax.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax.set_xlim(-xlim_mult * bin_width, xlim_mult * bin_width)
            # mx = max(Yin)
            # ax.set_xticks((2,4,6,8,10,12,14,16,18,20,22,24,26,28,30))
            X, Y = np.meshgrid(bin_edges[0:-1] + 0.5 * np.diff(bin_edges), fwhm_grid[0:-1] * 24 * 60)
            p = ax.pcolor(X, Y, Z, cmap=cm.bone, edgecolors='face', vmin=Z.min(), vmax=Z.max())
            # cbaxes = fig.add_axes([])
            cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
            cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
            if prop != 5:
                ax.plot([0,0],[np.min(fwhm_grid*24*60),np.max(fwhm_grid*24*60)], '--', color='red',lw=1)
            # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)
            cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
            ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#ffffff')

        plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300)
        plt.close()
        # plt.show()

        # import pdb; pdb.set_trace()
def test_plot2():
    fig = plt.figure(figsize=(10, 6), facecolor='#ffffff') #, constrained_layout=False)

    #grid = plt.GridSpec(10, 16, hspace=1, wspace=1.5)
    for prop in range(6):

        if prop <= 2:
            plot_bott = 0.55
            plot_top = 0.98
            plot_left = 0.05 + 0.30 * prop + 0.02*prop
            plot_right = 0.05 + 0.30 * (prop + 1) + 0.02*prop
        if prop > 2:
            plot_bott = 0.05
            plot_top = 0.50
            plot_left = 0.05 + 0.30 * (prop-3) + 0.02*(prop-3)
            plot_right = 0.05 + 0.30 * ((prop-3) + 1) + 0.02*(prop-3)
        gs1 = fig.add_gridspec(nrows=6, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott, wspace=0.05, hspace=0.05)
        ax1 = fig.add_subplot(gs1[2:6, 0:4])
        ax2 = fig.add_subplot(gs1[0:2, 0:4],xticklabels=[], sharey=ax1)
        ax3 = fig.add_subplot(gs1[2:6, 4:6],yticklabels=[], sharex=ax1)
        # ax = fig.add_subplot(grid[2+shift_v:5+shift_v, 0+shift_h:3+shift_h])
        # ax_x_hist = fig.add_subplot(grid[0+shift_v:2+shift_v, 0+shift_h:3+shift_h], xticklabels=[], sharey=ax)
        # ax_y_hist = fig.add_subplot(grid[2+shift_v:5+shift_v, 3+shift_h:5+shift_h], yticklabels=[], sharex=ax)
        # print([2+shift_v,5+shift_v, 3+shift_h,5+shift_h])
    plt.tight_layout()
    plt.show()
def sum_cols(Z_2D):
    n_cols = len(Z_2D[0, :])
    col_sums = []
    for col in range(n_cols):
        tot_in_col = np.sum(Z_2D[:,col])
        col_sums.append(tot_in_col)
    return col_sums
def sum_rows(Z_2D):
    n_rows = len(Z_2D[:, 0])
    row_sums = []
    for row in range(n_rows):
        tot_in_row = np.sum(Z_2D[row, :])
        row_sums.append(tot_in_row)
    return row_sums
def cum_cols(Z_2D):
    n_cols = len(Z_2D[0, :])
    col_cums = []
    col_tots = []
    for col in range(n_cols):
        tot_in_col = np.sum(Z_2D[:,col])
        col_tots.append(tot_in_col)
        col_cums.append(np.sum(col_tots)/np.sum(Z_2D))
    return col_cums

def sort_property3(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, save_as, property_to_sort='fwhm'):

    opt_list = [np.array(t_peak_opt)*24*60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time (min)','% Difference from True FWHM',
                    '% Difference from True Amplitude','% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    if property_to_sort == 'fwhm':

        fwhm_min = 0.5 * (1. / 60.) * (1. / 24.)
        fwhm_max = 30. * (1. / 60.) * (1. / 24.)
        fwhm_grid = np.linspace(fwhm_min, fwhm_max, 51)

        fwhm_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(fwhm_grid) - 1):
            # In each iteration, add an empty list to the main list
            fwhm_slots.append([])

        for grid_index in range(1, len(fwhm_grid)):
            for true_index in range(0, len(fwhm_true)):
                if (fwhm_true[true_index] > fwhm_grid[grid_index - 1]) and (fwhm_true[true_index] <= fwhm_grid[grid_index]):
                    fwhm_slots[grid_index - 1].append(true_index)

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005
        rect_main = [left, bottom, width, height]
        rect_hist_top = [left, bottom + height + spacing, width, 0.2]
        rect_hist_side = [left + width + spacing, bottom, 0.2, height]


        fig = plt.figure(figsize=(10, 5), facecolor='#000000') #, dpi=300)
        grid = plt.GridSpec(8, 12, hspace=0.2, wspace=0.2)
        for prop in range(len(opt_list)):

            ax = fig.add_subplot(grid[0:-2, 0:-2])
            ax_y_hist = fig.add_subplot(grid[0, -1], xticklabels=[], sharey=ax)
            ax_x_hist = fig.add_subplot(grid[-1, -1], yticklabels=[], sharex=ax)

            y_hist, bin_edges = np.histogram(opt_list[prop], bins='fd')
            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))

            for slot_index in range(len(fwhm_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            #ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax.set_ylabel('FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
            ax.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)
            ax.set_ylim([np.min(fwhm_grid[0:-1]*24*60), np.max(fwhm_grid[0:-1]*24*60)])
            bin_width = np.diff(bin_edges)[0]
            xlim_mult = 20.
            if prop == 5:
                ax.set_xlim(0, np.max(bin_edges))
            else:
                #ax.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax.set_xlim(-xlim_mult * bin_width, xlim_mult * bin_width)
            # mx = max(Yin)
            # ax.set_xticks((2,4,6,8,10,12,14,16,18,20,22,24,26,28,30))
            X, Y = np.meshgrid(bin_edges[0:-1] + 0.5 * np.diff(bin_edges), fwhm_grid[0:-1] * 24 * 60)
            p = ax.pcolor(X, Y, Z, cmap=cm.bone, edgecolors='face', vmin=Z.min(), vmax=Z.max())
            # cbaxes = fig.add_axes([])
            cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
            cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
            if prop != 5:
                ax.plot([0,0],[np.min(fwhm_grid*24*60),np.max(fwhm_grid*24*60)], '--', color='red',lw=1)
            # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)
            cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
            ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#ffffff')

        plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300)
        plt.close()
        # plt.show()

        # import pdb; pdb.set_trace()
def sort_property4(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, save_as, property_to_sort='fwhm'):

    opt_list = [np.array(t_peak_opt)*24*60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time (min)','% Difference from True FWHM',
                    '% Difference from True Amplitude','% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    if property_to_sort == 'fwhm':

        fwhm_min = 0.5 * (1. / 60.) * (1. / 24.)
        fwhm_max = 30. * (1. / 60.) * (1. / 24.)
        fwhm_grid = np.linspace(fwhm_min, fwhm_max, 51)

        fwhm_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(fwhm_grid) - 1):
            # In each iteration, add an empty list to the main list
            fwhm_slots.append([])

        for grid_index in range(1, len(fwhm_grid)):
            for true_index in range(0, len(fwhm_true)):
                if (fwhm_true[true_index] > fwhm_grid[grid_index - 1]) and (fwhm_true[true_index] <= fwhm_grid[grid_index]):
                    fwhm_slots[grid_index - 1].append(true_index)

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'

        fig = plt.figure(figsize=(10, 6), facecolor='#ffffff')
        for prop in range(len(opt_list)):

            if prop <= 2:
                plot_bott = 0.57
                plot_top = 0.98
                plot_left = 0.05 + 0.27 * prop + 0.05 * prop
                plot_right = 0.05 + 0.27 * (prop + 1) + 0.05 * prop
            if prop > 2:
                plot_bott = 0.06
                plot_top = 0.48
                plot_left = 0.05 + 0.27 * (prop - 3) + 0.05 * (prop - 3)
                plot_right = 0.05 + 0.27 * ((prop - 3) + 1) + 0.05 * (prop - 3)
            gs1 = fig.add_gridspec(nrows=6, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott, wspace=0.05, hspace=0.05)
            ax1 = fig.add_subplot(gs1[2:6, 0:4])
            ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[]) #, sharey=ax1)
            ax3 = fig.add_subplot(gs1[2:6, 4:6], yticklabels=[]) #, sharex=ax1)

            y_hist, bin_edges = np.histogram(opt_list[prop], bins='fd')
            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))

            for slot_index in range(len(fwhm_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            row_hist = sum_rows(Z)
            col_hist = sum_cols(Z)

            #ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax1.set_ylabel('FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
            ax1.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)
            ax1.set_ylim([np.min(fwhm_grid[0:-1]*24*60), np.max(fwhm_grid[0:-1]*24*60)])
            bin_width = np.diff(bin_edges)[0]
            xlim_mult = 14.
            if (prop == 0) or (prop == 1):
                ax1.set_xlim(-xlim_mult * bin_width, xlim_mult * bin_width)
                ax2.set_xlim(-xlim_mult * bin_width, xlim_mult * bin_width)
            if (prop == 2) or (prop == 3) or (prop == 4):
                ax1.set_xlim(-(100. + bin_width), xlim_mult * bin_width)
                ax2.set_xlim(-(100. + bin_width), xlim_mult * bin_width)
            if prop == 5:
                ax1.set_xlim(0, 0.66*np.max(bin_edges))
                ax2.set_xlim(0, 0.66*np.max(bin_edges))

            # mx = max(Yin)
            # ax.set_xticks((2,4,6,8,10,12,14,16,18,20,22,24,26,28,30))
            #X, Y = np.meshgrid(bin_edges[0:-1] + 0.5 * np.diff(bin_edges), fwhm_grid[0:-1] * 24 * 60)
            X, Y = np.meshgrid(bin_edges[0:-1], fwhm_grid[0:-1] * 24 * 60)
            p = ax1.pcolor(X, Y, Z, cmap=cm.BuPu, edgecolors='face', vmin=Z.min(), vmax=Z.max())
            # cbaxes = fig.add_axes([])
            cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
            # if (prop == 2) or (prop == 5):
            #     cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
            cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
            # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)

            ax2.bar(bin_edges[:-1], col_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge")
            ax3.barh(fwhm_grid[:-1], row_hist, height=np.diff(fwhm_grid), color='None', edgecolor="black", align="edge")
            ax3.set_ylim([np.min(fwhm_grid), np.max(fwhm_grid)])
            ax3.set_xlim([0, np.max(row_hist) * 1.10])

            if prop != 5:
                ax1.plot([0,0],[np.min(fwhm_grid*24*60),np.max(fwhm_grid*24*60)],  color='#ff0066',lw=1)
                ax2.plot([0,0], [np.min(col_hist), np.max(col_hist)], color='#ff0066', lw=1)

            ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)
            ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)
            ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)

        plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300, rasterized=True)
        plt.close()
        # plt.show()

        # import pdb; pdb.set_trace()
def sort_property5(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, downsample, save_as, stddev, property_to_sort='fwhm'):

    opt_list = [np.array(t_peak_opt)*24*60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time (min)','% Difference from True FWHM',
                    '% Difference from True Amplitude','% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    if property_to_sort == 'fwhm':

        print('Plotting FWHM sort...')

        fwhm_min = 0.5 * (1. / 60.) * (1. / 24.)
        fwhm_max = 30. * (1. / 60.) * (1. / 24.)
        fwhm_grid = np.linspace(fwhm_min, fwhm_max, 51)

        fwhm_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(fwhm_grid) - 1):
            # In each iteration, add an empty list to the main list
            fwhm_slots.append([])

        for grid_index in range(1, len(fwhm_grid)):
            for true_index in range(0, len(fwhm_true)):
                if (fwhm_true[true_index] > fwhm_grid[grid_index - 1]) and (fwhm_true[true_index] <= fwhm_grid[grid_index]):
                    fwhm_slots[grid_index - 1].append(true_index)

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'
        impulse_max_lim_factor = 0.50

        fig = plt.figure(figsize=(10, 6), facecolor='#ffffff') #, dpi=300)
        for prop in range(len(opt_list)):
            #print(prop)

            if prop <= 2:
                plot_bott = 0.57
                plot_top = 0.98
                plot_left = 0.05 + 0.27 * prop + 0.05 * prop
                plot_right = 0.05 + 0.27 * (prop + 1) + 0.05 * prop
            if prop > 2:
                plot_bott = 0.06
                plot_top = 0.48
                plot_left = 0.05 + 0.27 * (prop - 3) + 0.05 * (prop - 3)
                plot_right = 0.05 + 0.27 * ((prop - 3) + 1) + 0.05 * (prop - 3)
            gs1 = fig.add_gridspec(nrows=6, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs1[2:6, 0:4])
            ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[]) #, sharey=ax1)
            ax3 = fig.add_subplot(gs1[2:6, 4:6], yticklabels=[]) #, sharex=ax1)

            y_hist, bin_edges = np.histogram(opt_list[prop], bins='auto')
            if prop != 5:
                bin_width = np.diff(bin_edges)[0]
                xlim_mult = 14.
                where_within = np.where((bin_edges >= -xlim_mult * bin_width) & (bin_edges <= xlim_mult * bin_width))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
            if prop == 5:
                where_within = np.where(bin_edges <= impulse_max_lim_factor*np.max(bin_edges))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))

            for slot_index in range(len(fwhm_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            row_hist = sum_rows(Z)
            col_hist = sum_cols(Z)

            print(len(bin_edges))
            #import pdb; pdb.set_trace()

            #ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax1.set_ylabel('FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
            ax1.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)

            X, Y = np.meshgrid(bin_edges[0:-1], (fwhm_grid[0:-1] * 24 * 60))
            p = ax1.pcolor(X, Y, Z, cmap=cm.BuPu, edgecolors='face', vmin=Z.min(), vmax=Z.max(),rasterized=True)
            #import pdb; pdb.set_trace()
            # cbaxes = fig.add_axes([])
            cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
            # if (prop == 2) or (prop == 5):
            #     cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
            cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
            # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)

            ax2.bar(bin_edges[:-1], col_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge",rasterized=True)
            ax3.barh(fwhm_grid[:-1], row_hist, height=np.diff(fwhm_grid), color='None', edgecolor="black", align="edge",rasterized=True)

            if prop != 5:
                ax1.plot([0,0],[np.min(fwhm_grid*24*60),np.max(fwhm_grid*24*60)],  color='#ff0066',lw=1)
                ax2.plot([0,0], [0, np.max(col_hist)*1.10], color='#ff0066', lw=1)

            ax3.set_ylim([np.min(fwhm_grid), np.max(fwhm_grid)])
            ax3.set_xlim([0, np.max(row_hist) * 1.10])
            ax1.set_ylim([np.min(fwhm_grid[0:-1] * 24 * 60), np.max(fwhm_grid[0:-1] * 24 * 60)])

            if (prop == 0) or (prop == 1):
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            if (prop == 2) or (prop == 3) or (prop == 4):
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            if prop == 5:
                ax1.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
                ax2.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))

            ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)
            ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)
            ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)

        print('Attempting To Save...')
        #plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300,rasterized=True)
        plt.close()
        # plt.show()

        # import pdb; pdb.set_trace()

    if property_to_sort == 'amplitude':

        print('Plotting Amplitude sort...')

        ampl_min = 3 * stddev
        ampl_max = 1000 * stddev
        ampl_grid = np.linspace(ampl_min, ampl_max, 51)

        ampl_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(ampl_grid) - 1):
            # In each iteration, add an empty list to the main list
            ampl_slots.append([])

        for grid_index in range(1, len(ampl_grid)):
            for true_index in range(0, len(ampl_true)):
                if (ampl_true[true_index] > ampl_grid[grid_index - 1]) and (ampl_true[true_index] <= ampl_grid[grid_index]):
                    ampl_slots[grid_index - 1].append(true_index)

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'
        impulse_max_lim_factor = 0.50

        fig = plt.figure(figsize=(10, 6), facecolor='#ffffff', dpi=300)
        for prop in range(len(opt_list)):
            #print(prop)

            if prop <= 2:
                plot_bott = 0.57
                plot_top = 0.98
                plot_left = 0.05 + 0.27 * prop + 0.05 * prop
                plot_right = 0.05 + 0.27 * (prop + 1) + 0.05 * prop
            if prop > 2:
                plot_bott = 0.06
                plot_top = 0.48
                plot_left = 0.05 + 0.27 * (prop - 3) + 0.05 * (prop - 3)
                plot_right = 0.05 + 0.27 * ((prop - 3) + 1) + 0.05 * (prop - 3)
            gs1 = fig.add_gridspec(nrows=6, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs1[2:6, 0:4])
            ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[]) #, sharey=ax1)
            ax3 = fig.add_subplot(gs1[2:6, 4:6], yticklabels=[]) #, sharex=ax1)

            y_hist, bin_edges = np.histogram(opt_list[prop], bins='auto')
            if prop != 5:
                bin_width = np.diff(bin_edges)[0]
                xlim_mult = 20.
                where_within = np.where((bin_edges >= -xlim_mult * bin_width) & (bin_edges <= xlim_mult * bin_width))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
            if prop == 5:
                where_within = np.where(bin_edges <= impulse_max_lim_factor*np.max(bin_edges))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
            Z = np.zeros((len(ampl_grid) - 1, len(bin_edges) - 1))

            for slot_index in range(len(ampl_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[ampl_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            #import pdb; pdb.set_trace()

            row_hist = sum_rows(Z)
            col_hist = sum_cols(Z)

            print(len(bin_edges))

            #ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax1.set_ylabel(r'Amplitude (F$_{flare}$/F$_{quiescent}$)', fontsize=font_size, style=font_style, family=font_family)
            ax1.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)

            X, Y = np.meshgrid(bin_edges[0:-1], ampl_grid[0:-1])
            p = ax1.pcolor(X, Y, Z, cmap=cm.BuPu, edgecolors='face', vmin=Z.min(), vmax=Z.max(),rasterized=True)
            # cbaxes = fig.add_axes([])
            cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
            # if (prop == 2) or (prop == 5):
            #     cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
            cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
            # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)

            ax2.bar(bin_edges[:-1], col_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge",rasterized=True)
            ax3.barh(ampl_grid[:-1], row_hist, height=np.diff(ampl_grid), color='None', edgecolor="black", align="edge",rasterized=True)

            if prop != 5:
                ax1.plot([0,0],[np.min(ampl_grid),np.max(ampl_grid)],  color='#ff0066',lw=1)
                ax2.plot([0,0], [0, np.max(col_hist)*1.10], color='#ff0066', lw=1)

            ax3.set_ylim([np.min(ampl_grid), np.max(ampl_grid)])
            ax3.set_xlim([0, np.max(row_hist) * 1.10])
            ax1.set_ylim([np.min(ampl_grid[0:-1]), np.max(ampl_grid[0:-1])])

            if (prop == 0) or (prop == 1):
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            if (prop == 2) or (prop == 3) or (prop == 4):
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            if prop == 5:
                ax1.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
                ax2.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))

            ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)
            ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)
            ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000', length=0)

        print('Attempting To Save...')
        #plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300, rasterized=True)
        plt.close()
        # plt.show()

        # import pdb; pdb.set_trace()
def sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, plot_type, save_as, stddev, set_cadence, property_to_sort='fwhm'):
    opt_list = [np.array(t_peak_opt) * 24 * 60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time (min)', '% Difference from True FWHM',
                    '% Difference from True Amplitude', '% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    bin_slice_factor = 3.

    if property_to_sort == 'fwhm':

        print('Plotting FWHM sort...')

        fwhm_min = 0.5 * (1. / 60.) * (1. / 24.)
        fwhm_max = set_cadence * (1. / 60.) * (1. / 24.)
        fwhm_grid = np.linspace(fwhm_min, fwhm_max, 61)

        fwhm_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(fwhm_grid) - 1):
            # In each iteration, add an empty list to the main list
            fwhm_slots.append([])

        for grid_index in range(1, len(fwhm_grid)):
            for true_index in range(0, len(fwhm_true)):
                if (fwhm_true[true_index] > fwhm_grid[grid_index - 1]) and (
                        fwhm_true[true_index] <= fwhm_grid[grid_index]):
                    fwhm_slots[grid_index - 1].append(true_index)

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'
        impulse_max_lim_factor = 0.50
        xlim_mult = 20.

        fig = plt.figure(figsize=(10, 6), facecolor='#ffffff')  # , dpi=300)
        for prop in range(len(opt_list)):
            # print(prop)

            if prop <= 2:
                plot_bott = 0.57
                plot_top = 0.98
                plot_left = 0.05 + 0.27 * prop + 0.05 * prop
                plot_right = 0.05 + 0.27 * (prop + 1) + 0.05 * prop
            if prop > 2:
                plot_bott = 0.06
                plot_top = 0.48
                plot_left = 0.05 + 0.27 * (prop - 3) + 0.05 * (prop - 3)
                plot_right = 0.05 + 0.27 * ((prop - 3) + 1) + 0.05 * (prop - 3)
            gs1 = fig.add_gridspec(nrows=6, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott,
                                   wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs1[2:6, 0:4])
            ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[])  # , sharey=ax1)
            ax3 = fig.add_subplot(gs1[2:6, 4:6], yticklabels=[])  # , sharex=ax1)

            y_hist, bin_edges = np.histogram(opt_list[prop], bins='auto')
            # bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges)+0.5*np.diff(bin_edges)[0], 0.5*np.diff(bin_edges)[0])
            axis_spacing = np.arange(0,len(bin_edges),1)
            if bin_slice_factor == 3:
                new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)[0:-1]
            else:
                new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)
            bin_edges = np.interp(new_axis_spacing,axis_spacing,bin_edges)
            #bin_edges = np.interp(np.arange())

            #import pdb; pdb.set_trace()

            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
            for slot_index in range(len(fwhm_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            row_hist = sum_rows(Z)
            col_hist1 = sum_cols(Z)
            cum_hist1 = cum_cols(Z)

            if prop != 5:
                bin_width = np.diff(bin_edges)[0]
                where_within = np.where((bin_edges >= -xlim_mult * bin_width) & (bin_edges <= xlim_mult * bin_width))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
            if prop == 5:
                where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])

            col_hist = np.array(col_hist1)[where_within[:-1]]
            cum_hist = np.array(cum_hist1)[where_within[:-1]]

            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
            for slot_index in range(len(fwhm_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            print(len(y_hist))
            # import pdb; pdb.set_trace()

            # ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax1.set_ylabel('True FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
            ax1.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)

            X, Y = np.meshgrid(bin_edges[0:-1], (fwhm_grid[0:-1] * 24 * 60))
            p = ax1.pcolor(X, Y, Z, cmap=cm.BuPu, edgecolors='face', vmin=Z.min(), vmax=Z.max(), rasterized=True)
            # import pdb; pdb.set_trace()
            # cbaxes = fig.add_axes([])
            cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
            # if (prop == 2) or (prop == 5):
            #     cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
            cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
            # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)

            if plot_type == 'cumulative':
                ax2.bar(np.array(bin_edges[:-1]), cum_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge", rasterized=True)
            if plot_type == 'sum':
                #import pdb; pdb.set_trace()
                ax2.bar(np.array(bin_edges[:-1]), col_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge", rasterized=True)
            ax3.barh(fwhm_grid[:-1], row_hist, height=np.diff(fwhm_grid), color='None', edgecolor="black", align="edge", rasterized=True)

            if prop != 5:
                ax1.plot([0, 0], [np.min(fwhm_grid * 24 * 60), np.max(fwhm_grid * 24 * 60)], color='#ff0066', lw=1)
                if plot_type == 'cumulative':
                    ax2.plot([0, 0], [0, 1.0], color='#ff0066', lw=1)
                    ax2.set_ylim([0, 1.0])
                if plot_type == 'sum':
                    ax2.plot([0, 0], [0, np.max(col_hist) * 1.10], color='#ff0066', lw=1)
                    ax2.set_ylim([0, np.max(col_hist) * 1.10])

            ax3.set_ylim([np.min(fwhm_grid), np.max(fwhm_grid)])
            ax3.set_xlim([0, np.max(row_hist) * 1.10])
            ax1.set_ylim([np.min(fwhm_grid[0:-1] * 24 * 60), np.max(fwhm_grid[0:-1] * 24 * 60)])

            if (prop == 0) or (prop == 1):
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            if (prop == 2) or (prop == 3) or (prop == 4):
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            if prop == 5:
                ax1.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
                ax2.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))

            ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)
            ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)
            ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)

        print('Attempting To Save...')
        # plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300, rasterized=True)
        plt.close()
        # plt.show()

        # import pdb; pdb.set_trace()

    if property_to_sort == 'amplitude':

        print('Plotting Amplitude sort...')

        ampl_min = 3 * stddev
        ampl_max = 1000 * stddev
        ampl_grid = np.linspace(ampl_min, ampl_max, 51)

        ampl_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(ampl_grid) - 1):
            # In each iteration, add an empty list to the main list
            ampl_slots.append([])

        for grid_index in range(1, len(ampl_grid)):
            for true_index in range(0, len(ampl_true)):
                if (ampl_true[true_index] > ampl_grid[grid_index - 1]) and (
                        ampl_true[true_index] <= ampl_grid[grid_index]):
                    ampl_slots[grid_index - 1].append(true_index)

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'
        impulse_max_lim_factor = 0.50
        xlim_mult = 20.

        fig = plt.figure(figsize=(10, 6), facecolor='#ffffff', dpi=300)
        for prop in range(len(opt_list)):
            # print(prop)

            if prop <= 2:
                plot_bott = 0.57
                plot_top = 0.98
                plot_left = 0.05 + 0.27 * prop + 0.05 * prop
                plot_right = 0.05 + 0.27 * (prop + 1) + 0.05 * prop
            if prop > 2:
                plot_bott = 0.06
                plot_top = 0.48
                plot_left = 0.05 + 0.27 * (prop - 3) + 0.05 * (prop - 3)
                plot_right = 0.05 + 0.27 * ((prop - 3) + 1) + 0.05 * (prop - 3)
            gs1 = fig.add_gridspec(nrows=6, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott,
                                   wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs1[2:6, 0:4])
            ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[])  # , sharey=ax1)
            ax3 = fig.add_subplot(gs1[2:6, 4:6], yticklabels=[])  # , sharex=ax1)

            y_hist, bin_edges = np.histogram(opt_list[prop], bins='auto')
            #bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
            axis_spacing = np.arange(0, len(bin_edges), 1)
            if bin_slice_factor == 3:
                new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)[0:-1]
            else:
                new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)
            bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)

            Z = np.zeros((len(ampl_grid) - 1, len(bin_edges) - 1))
            for slot_index in range(len(ampl_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[ampl_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            row_hist = sum_rows(Z)
            col_hist1 = sum_cols(Z)
            cum_hist1 = cum_cols(Z)

            if prop != 5:
                bin_width = np.diff(bin_edges)[0]
                where_within = np.where((bin_edges >= -xlim_mult * bin_width) & (bin_edges <= xlim_mult * bin_width))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
            if prop == 5:
                where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])

            col_hist = np.array(col_hist1)[where_within[:-1]]
            cum_hist = np.array(cum_hist1)[where_within[:-1]]

            Z = np.zeros((len(ampl_grid) - 1, len(bin_edges) - 1))
            for slot_index in range(len(ampl_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[ampl_slots[slot_index]], bins=bin_edges)
                Z[slot_index, :] = y_hist

            print(len(y_hist))

            # ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax1.set_ylabel(r'True Amplitude (F$_{flare}$/F$_{quiescent}$ - 1)', fontsize=font_size, style=font_style,
                               family=font_family)
            ax1.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)

            X, Y = np.meshgrid(bin_edges[0:-1], ampl_grid[0:-1])
            p = ax1.pcolor(X, Y, Z, cmap=cm.BuPu, edgecolors='face', vmin=Z.min(), vmax=Z.max(), rasterized=True)
            # cbaxes = fig.add_axes([])
            cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
            # if (prop == 2) or (prop == 5):
            #     cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
            cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
            # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)

            if plot_type == 'cumulative':
                ax2.bar(np.array(bin_edges[:-1]), cum_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge", rasterized=True)
            if plot_type == 'sum':
                #import pdb; pdb.set_trace()
                ax2.bar(np.array(bin_edges[:-1]), col_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge", rasterized=True)
            ax3.barh(ampl_grid[:-1], row_hist, height=np.diff(ampl_grid), color='None', edgecolor="black", align="edge", rasterized=True)

            if prop != 5:
                ax1.plot([0, 0], [np.min(ampl_grid), np.max(ampl_grid)], color='#ff0066', lw=1)
                if plot_type == 'cumulative':
                    ax2.plot([0, 0], [0, 1.0], color='#ff0066', lw=1)
                    ax2.set_ylim([0, 1.0])
                if plot_type == 'sum':
                    ax2.plot([0, 0], [0, np.max(col_hist) * 1.10], color='#ff0066', lw=1)
                    ax2.set_ylim([0, np.max(col_hist) * 1.10])

            ax3.set_ylim([np.min(ampl_grid), np.max(ampl_grid)])
            ax3.set_xlim([0, np.max(row_hist) * 1.10])
            ax1.set_ylim([np.min(ampl_grid[0:-1]), np.max(ampl_grid[0:-1])])

            if (prop == 0) or (prop == 1):
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            if (prop == 2) or (prop == 3) or (prop == 4):
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            if prop == 5:
                ax1.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
                ax2.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))

            ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)
            ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)
            ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)

        print('Attempting To Save...')
        # plt.tight_layout()
        plt.savefig(
            '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,
            dpi=300, rasterized=True)
        plt.close()
        # plt.show()

        # import pdb; pdb.set_trace()


fit_statistics4(cadence, stddev, set_fixed_amplitude=False, downsample=True, set_cadence=50, n_reps=100)



# def plot_2D():
#     phase_grid = np.arange(0., np.max(phase_list) + 0.01, 0.01)
#     # print(phase_grid)
#
#     for ii in range(len(phase_list)):
#         diff_array = np.abs(phase_grid - phase_list[ii])
#         where_close = np.where(diff_array == np.min(diff_array))[0][0]
#         # print(where_close)
#
#     Z = np.zeros((len(phase_grid), len(new_fluxes_list_chr[0])))
#     for profile_n in range(len(new_wavs_list)):
#         diff_array = np.abs(phase_grid - phase_list[profile_n])
#         where_close = np.where(diff_array == np.min(diff_array))[0][0]
#
#         profile_difference = median_profile - new_fluxes_list_chr[profile_n]
#         Z[where_close, :] = profile_difference
#
#     Z_unbinned = np.zeros((len(phase_grid), len(fluxes_list_chr[0])))
#
#     font_size = 'small'
#     font_style = 'normal'
#     font_family = 'sans-serif'
#
#     fig = plt.figure(num=None, figsize=(3.5, 5), facecolor='w', dpi=135)
#     ax = fig.add_subplot(111)
#     # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
#     # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
#     ax.set_ylabel('Rotation Phase', fontsize=font_size, style=font_style, family=font_family)
#     ax.set_xlabel('Wavelength (nm)', fontsize=font_size, style=font_style, family=font_family)
#     ax.set_ylim([min(phase_list), max(phase_list)])
#     ax.set_xlim([xmin / 10., xmax / 10.])
#     # mx = max(Yin)
#     # ax.set_xticks((2,4,6,8,10,12,14,16,18,20,22,24,26,28,30))
#     X, Y = np.meshgrid(spec_wavs / 10., phase_grid)
#     p = ax.pcolor(X, Y, Z_unbinned, cmap=cm.bone, edgecolors='face', vmin=Z_unbinned.min(),
#                   vmax=Z_unbinned.max())  # vmax=abs(Z.max()))
#     # cbaxes = fig.add_axes([])
#
#     cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
#     cb.set_label(label='Flux Difference From Median', fontsize=font_size, style=font_style, family=font_family)
#     # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)
#     cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
#
#     ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#ffffff')
#
#     top_ax = ax.secondary_xaxis('top', functions=(wav_to_rv, rv_to_wav))
#     # top_ax.set_xticks(rv_values)
#     top_ax.set_xlabel(r'RV (km s$^{-1}$)', fontsize=font_size, style=font_style, family=font_family)
#     top_ax.tick_params(axis='x', labelsize=font_size, direction='in', color='#ffffff')
#
#     # fig.savefig(savehere + target + '_2D-period.pdf',format='pdf',bbox_inches='tight')
#     # close()









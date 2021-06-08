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

#np.random.seed(42)
np.random.seed()

from scipy.interpolate import splrep, splev

import lightkurve as lk

from scipy import optimize

import sys, os
import colour # https://www.colour-science.org/installation-guide/

# from numba import jit


def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def choose_cmap():
    colors1 = [(102/255, 0/255, 204/255), (255/255, 128/255, 0/255), (0/255, 153/255, 153/255)]
    colors2 = [(204/255, 0/255, 102/255), (51/255, 51/255, 204/255), (153/255, 204/255, 0/255)]
    colors3 = [(128/255, 0/255, 64/255),(51/255, 51/255, 204/255),(0/255, 255/255, 153/255)]
    colors4 = [(255/255, 255/255, 255/255),(0/255, 255/255, 204/255),(0/255, 153/255, 204/255),(0/255, 153/255, 255/255),(102/255, 0/255, 204/255)]
    colors5 = [(255/255, 255/255, 255/255),(153/255, 255/255, 153/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255)]
    colors6 = [(255/255, 255/255, 255/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255),(0/255, 0/255, 77/255)]
    colors7 = [(255 / 255, 255 / 255, 255 / 255), (255 / 255, 204 / 255, 0 / 255), (255 / 255, 0 / 255, 102 / 255), (134 / 255, 0 / 255, 179 / 255), (0 / 255, 0 / 255, 77 / 255)]
    colors8 = [(255 / 255, 255 / 255, 255 / 255), (255 / 255, 0 / 255, 102 / 255), (153 / 255, 0 / 255, 204 / 255), (0 / 255, 0 / 255, 77 / 255)]
    colors9 = [(255/255, 255/255, 255/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255)]


    position = [0, 0.5, 1]
    position2 = [0, 0.25, 0.5, 0.75, 1]
    position2_2 = [0, 0.10, 0.5, 0.75, 1]
    position3 = [0, 1./3., 2./3., 1]
    mycolormap = make_cmap(colors7, position=position2_2)

    return mycolormap



def get_fwhm(x_in,y_in,do_interp=True):
    if do_interp == True:
        x_interp = np.linspace(np.min(x_in),np.max(x_in),1000)
        y_interp = np.interp(x_interp, x_in, y_in)

        half = np.max(y_in) / 2.0
        signs = np.sign(np.add(y_interp, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        where_zero_crossings = np.where(zero_crossings)[0]

        x1 = np.mean(x_interp[where_zero_crossings[0]:where_zero_crossings[0] + 1])
        x2 = np.mean(x_interp[where_zero_crossings[1]:where_zero_crossings[1] + 1])
    else:
        half = np.max(y_in) / 2.0
        signs = np.sign(np.add(y_in, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        where_zero_crossings = np.where(zero_crossings)[0]

        x1 = np.mean(x_in[where_zero_crossings[0]:where_zero_crossings[0] + 1])
        x2 = np.mean(x_in[where_zero_crossings[1]:where_zero_crossings[1] + 1])

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
def jflare1(x, tpeak, j_fwhm, dav_fwhm, gauss_ampl, decay_ampl):
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    g_profile = np.abs(gauss_ampl) * np.exp(-((x - tpeak) ** 2) / (j_fwhm ** 2))

    # decay_start = tpeak - tdiff
    decay_start = x[np.min(np.where(np.abs(g_profile - decay_ampl) == np.min(np.abs(g_profile - decay_ampl)))[0])]
    if decay_start > tpeak:
        decay_start = tpeak - (decay_start - tpeak)

    d_profile = (_fd[0] * np.exp(((x - decay_start) / dav_fwhm) * _fd[1]) +
                 _fd[2] * np.exp(((x - decay_start) / dav_fwhm) * _fd[3])) * np.abs(decay_ampl)

    # set padding
    g_profile[x < 0.1] = 0
    g_profile[x > 0.9] = 0
    d_profile[x > 0.9] = 0
    d_profile[x < decay_start] = 0

    c_profile = np.convolve(g_profile, d_profile, 'same')

    # print(x[np.where(g_profile == np.max(g_profile))[0]])
    # print(x[np.where(c_profile == np.max(c_profile))[0]])

    return g_profile, d_profile, c_profile, decay_start

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






cadence = 1./24./60./60. # seconds converted to days  #1.*(1./1440.) # minutes converted to days
stddev = 0.001 * (1. / 50)

def create_single_synthetic(cadence, max_fwhm):
    np.random.seed()

    # print('\nGenerating Synthetic Flares...\n')
    std = 0.001 * (1. / 50)

    x_synth = np.arange(0, 3, cadence)  # cadence in days

    t_max = 0.5
    t_max = t_max + np.random.uniform(-0.05, 0.05, 1)
    fwhm = np.random.uniform(0.5, max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # days
    amplitude = np.random.uniform(0.05,1.0,1)
    # amplitude = np.random.uniform(3, 1000, 1) * stddev

    y_synth = np.random.normal(0, std, len(x_synth))
    y_synth_noscatter = np.zeros_like(x_synth)

    flare_synth_a_noscatter = aflare1(x_synth, t_max, fwhm, amplitude)
    flare_synth_a = flare_synth_a_noscatter + np.random.normal(0, std, len(x_synth))

    y_synth_noscatter += flare_synth_a_noscatter
    y_synth += flare_synth_a

    # y_synth = aflare1(x_synth, t_max, fwhm, amplitude)

    # where_start = np.where(y_synth > 0)[0][0]
    # where_end = np.where(y_synth > 0.0025)[0][-1]
    #
    # plt.title('Rise Time = ' + str(np.round((t_max_in - x_synth[where_start])*60*24,2))+' min\nFall Time = ' + str(np.round((x_synth[where_end] - t_max_in)*60*24,2))+' min')
    # plt.plot([t_max_in,t_max_in], [0,np.max(y_synth*1.1)], c='grey', lw=0.6)
    # plt.plot([0, 2], [0, 0], c='grey', lw=0.6)
    # plt.plot([x_synth[where_start], x_synth[where_start]], [0, np.max(y_synth * 1.2)], c='green', lw=0.6)
    # plt.plot([x_synth[where_end], x_synth[where_end]], [0, np.max(y_synth * 1.2)], c='red', lw=0.6)
    # plt.plot(x_synth, y_synth, c='black', label='fwhm = ' + str(np.round(fwhm_in * 60 * 24, 1)) + ' min')
    # #plt.xlim(x_synth[where_start]-0.05,x_synth[where_end]+0.05)
    # plt.xlim(0.1,0.5)
    # plt.ylim(-0.05,np.max(y_synth*1.1))
    # plt.xlabel('Time (days)')
    # plt.legend(loc='upper right')
    # plt.savefig( '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/_synthetic_flare_' + str(a) + '.pdf')
    # plt.close()
    # plt.show()

    #import pdb; pdb.set_trace()

    flare_properties = {"tpeak": t_max,
                        "amplitude": amplitude,
                        "fwhm": fwhm,
                        }
    return x_synth, y_synth, y_synth_noscatter, flare_properties
def create_single_synthetic_jflare1(cadence, max_fwhm=10):
    np.random.seed()

    std = 0.001 * (1. / 50)

    window = 1. #days
    x_synth = np.arange(0, window, cadence)  # cadence in days

    gauss_tpeak = 0.5*window # peak time
    #gauss_tpeak = gauss_tpeak + np.random.uniform(-0.05, 0.05, 1)[0]

    gauss_ampl_j = 1.0
    decay_ampl_j = 1.0 # np.random.uniform(0.01,1.0,1)
    decay_fwhm_j = np.random.uniform(1./60., max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    # gauss_fwhm_j = np.random.uniform(1./60., max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    gauss_fwhm_j = decay_fwhm_j

    # compute the convolution
    g_profile, d_profile, synth_flare, decay_start = jflare1(x_synth, gauss_tpeak, gauss_fwhm_j, decay_fwhm_j, gauss_ampl_j, decay_ampl_j)

    # normalize the flare
    synth_flare/=np.max(synth_flare)

    # add scatter to the flare
    synth_flare_scatter = synth_flare + np.random.normal(-std, std, len(synth_flare))

    # determine which element in the array the flare peaks
    where_max = np.where(synth_flare == np.max(synth_flare))[0]
    # determine flare fwhm
    fwhm = get_fwhm(x_synth, synth_flare, do_interp=False)
    # set flare amplitude after normalization
    amplitude = np.random.uniform(0.05, 1, 1)

    synth_flare *= amplitude
    synth_flare_scatter *= amplitude

    #import pdb; pdb.set_trace()

    flare_properties = {"tpeak": x_synth[where_max],
                        "amplitude": amplitude,
                        "fwhm": fwhm,
                        }

    return x_synth, synth_flare_scatter, synth_flare, flare_properties
def create_single_synthetic_jflare1_equation(cadence, max_fwhm=10):
    np.random.seed()

    std = 0.001 * (1. / 50)

    window = 1. #days
    x_synth = np.arange(0, window, cadence)  # cadence in days

    gauss_tpeak = 0.5*window # peak time
    #gauss_tpeak = gauss_tpeak + np.random.uniform(-0.05, 0.05, 1)[0]

    gauss_ampl_j = 1.0
    decay_ampl_j = 1.0 # np.random.uniform(0.01,1.0,1)
    decay_fwhm_j = np.random.uniform(0.5, max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    gauss_fwhm_j = np.random.uniform(0.5, max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days

    g_profile = np.abs(gauss_ampl_j) * np.exp(-((x_synth - gauss_tpeak) ** 2) / (gauss_fwhm_j** 2))

    decay_start = x_synth[np.min(np.where(np.abs(g_profile - decay_ampl_j) == np.min(np.abs(g_profile - decay_ampl_j)))[0])]
    if decay_start > gauss_tpeak:
        decay_start = gauss_tpeak - (decay_start - gauss_tpeak)

    A = gauss_ampl_j
    B = gauss_tpeak
    C = gauss_fwhm_j
    K = decay_ampl_j
    J = decay_fwhm_j
    H = decay_start
    D = 0.689008
    E = -1.60053
    F = 0.302963
    G = -0.278318

    # compute the flare
    synth_flare = 0.5 * np.sqrt(np.pi) * A * C * K * (D * (scipy.special.erf(B / C + (C * E) / (2. * J)) - scipy.special.erf(B / C + (C * E) / (2. * J) - x_synth / C)) * np.exp(
        (E * (4. * J * (B - H + x_synth) + C ** 2. * E)) / (4. * J ** 2.)) + F * (scipy.special.erf(B / C + (C * G) / (2. * J)) - scipy.special.erf(B / C + (C * G) / (2. * J) - x_synth / C)) * np.exp(
        (G * (4 * J * (B - H + x_synth) + C ** 2. * G)) / (4. * J ** 2.)))
    # normalize the flare
    synth_flare/=np.max(synth_flare)

    # add scatter to the flare
    synth_flare_scatter = synth_flare + np.random.normal(-std, std, len(synth_flare))

    # determine which element in the array the flare peaks
    where_max = np.where(synth_flare == np.max(synth_flare))[0]
    # determine flare fwhm
    fwhm = get_fwhm(x_synth, synth_flare, do_interp=False)
    # set flare amplitude after normalization
    amplitude = np.random.uniform(0.01, 1, 1)

    synth_flare *= amplitude
    synth_flare_scatter *= amplitude

    flare_properties = {"tpeak": x_synth[where_max],
                        "amplitude": amplitude,
                        "fwhm": fwhm,
                        }

    return x_synth, synth_flare_scatter, synth_flare, flare_properties


# @jit(nopython = True)
def test_single_synthetic_jflare1(cadence, max_fwhm=10):
    np.random.seed()

    std = 0.001 * (1. / 50)

    window = 1. #days
    x_synth = np.arange(0, window, cadence)  # cadence in days

    gauss_tpeak = 0.5*window # peak time
    #gauss_tpeak = gauss_tpeak + np.random.uniform(-0.05, 0.05, 1)[0]

    gauss_ampl_j = 1.0
    decay_ampl_j = 1.0 # np.random.uniform(0.01,1.0,1)
    # decay_fwhm_j = 1.0/24./60.#np.random.uniform(0.5, max_fwhm, 1) #* (1. / 60.) * (1. / 24.)  # minutes to days # 0.004
    # gauss_fwhm_j = 1.0/24./60. #np.random.uniform(0.5, max_fwhm, 1) #* (1. / 60.) * (1. / 24.)  # minutes to days # 0.004
    #
    # g_profile, d_profile, synth_flare, decay_start = jflare1(x_synth, gauss_tpeak, gauss_fwhm_j, decay_fwhm_j, gauss_ampl_j, decay_ampl_j)


    peak_location = []
    peak_value = []
    input_fwhm_value = []
    output_fwhm_value = []
    test_fwhms = np.arange(0.5,5.5,0.5)
    for beep in range(len(test_fwhms)):
        print((beep+1)/len(test_fwhms))
        decay_fwhm_j = test_fwhms[beep] / 24. / 60.  # np.random.uniform(0.5, max_fwhm, 1) #* (1. / 60.) * (1. / 24.)  # minutes to days # 0.004
        gauss_fwhm_j = test_fwhms[beep] / 24. / 60.  # np.random.uniform(0.5, max_fwhm, 1) #* (1. / 60.) * (1. / 24.)  # minutes to days # 0.004

        g_profile, d_profile, synth_flare, decay_start = jflare1(x_synth, gauss_tpeak, gauss_fwhm_j, decay_fwhm_j, gauss_ampl_j, decay_ampl_j)

        synth_flare/=np.max(synth_flare)

        plt.plot(x_synth, synth_flare, label = 'FWHM: ' + str(test_fwhms[beep]))

        where_max = np.where(synth_flare == np.max(synth_flare))[0]
        peak_location.append(x_synth[where_max])
        peak_value.append(synth_flare[where_max])
        input_fwhm_value.append(test_fwhms[beep])

        fwhm = get_fwhm(x_synth, synth_flare, do_interp=False)

        output_fwhm_value.append(fwhm) # days to minutes

    # plt.scatter(input_fwhm_value,peak_value)
    # plt.xlabel('Input FWHM Value')
    # plt.ylabel('Peak Value')
    # plt.show()
    #
    # plt.scatter(input_fwhm_value, output_fwhm_value)
    # plt.xlabel('Input FWHM Value')
    # plt.ylabel('Output FWHm Value')
    # plt.show()
    #
    # plt.scatter(input_fwhm_value, peak_location)
    # plt.xlabel('Input FWHM Value')
    # plt.ylabel('Peak Time')
    # plt.show()



    #import pdb; pdb.set_trace()


    # synth_flare_scatter = synth_flare + np.random.normal(-std, std, len(synth_flare))




    # plt.plot(x_synth,synth_flare)
    #plt.savefig('/Users/lbiddle/Desktop/testflare_j.pdf')
    plt.legend(loc='upper right')
    plt.show()

    #import pdb; pdb.set_trace()

    # # where_start = np.where(synth_flare > 0)[0][0]
    # # where_end = np.where(synth_flare > 0.000025)[0][-1]
    #
    # # plt.title('Rise Time = ' + str(np.round((t_max_in - x_synth[where_start])*60*24,2))+' min\nFall Time = ' + str(np.round((x_synth[where_end] - t_max_in)*60*24,2))+' min')
    # # plt.plot([t_max_in,t_max_in], [0,np.max(synth_flare*1.1)], c='grey', lw=0.6)
    # # plt.plot([0, 2], [0, 0], c='grey', lw=0.6)
    # # plt.plot([x_synth[where_start], x_synth[where_start]], [0, np.max(synth_flare * 1.2)], c='green', lw=0.6)
    # # plt.plot([x_synth[where_end], x_synth[where_end]], [0, np.max(synth_flare * 1.2)], c='red', lw=0.6)
    # plt.plot(x_synth, synth_flare, c='black') #, label='fwhm = ' + str(np.round(fwhm_in * 60 * 24, 1)) + ' min')
    # # plt.xlim(x_synth[where_start]-0.05,x_synth[where_end]+0.05)
    # #plt.xlim(0.45,0.55)
    # # plt.ylim(-0.05,np.max(synth_flare*1.1))
    # plt.xlabel('Time (days)')
    # # plt.legend(loc='upper right')
    # # plt.savefig( '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/_synthetic_flare_' + str(a) + '.pdf')
    # # plt.savefig('/Users/lbiddle/Desktop/testflare_j.pdf')
    # # plt.close()
    # plt.show()
    #
    # # import pdb; pdb.set_trace()
    # #
    # # # flare_properties = {"tpeak": t_max,
    # # #                     "amplitude": amplitude,
    # # #                     "fwhm": fwhm,
    # # #                     }
    # # return x_synth, synth_flare, synth_flare_scatter, flare_properties
# test_single_synthetic_jflare1(cadence, max_fwhm=10)
def test_single_synthetic_jflare1_equation(cadence, max_fwhm=10):
    np.random.seed()

    std = 0.001 * (1. / 50)

    window = 1. #days
    x_synth = np.arange(0, window, cadence)  # cadence in days

    gauss_tpeak = 0.5*window # peak time
    #gauss_tpeak = gauss_tpeak + np.random.uniform(-0.05, 0.05, 1)[0]

    gauss_ampl_j = 1.0
    decay_ampl_j = 1.0 # np.random.uniform(0.01,1.0,1)
    # decay_fwhm_j = 1.0/24./60.#np.random.uniform(0.5, max_fwhm, 1) #* (1. / 60.) * (1. / 24.)  # minutes to days # 0.004
    # gauss_fwhm_j = 1.0/24./60. #np.random.uniform(0.5, max_fwhm, 1) #* (1. / 60.) * (1. / 24.)  # minutes to days # 0.004
    #
    # g_profile, d_profile, synth_flare, decay_start = jflare1(x_synth, gauss_tpeak, gauss_fwhm_j, decay_fwhm_j, gauss_ampl_j, decay_ampl_j)


    peak_location = []
    peak_value = []
    input_fwhm_value = []
    output_fwhm_value = []
    test_fwhms = np.arange(0.5,5.5,0.5)
    for beep in range(len(test_fwhms)):
        print((beep+1)/len(test_fwhms))
        decay_fwhm_j = test_fwhms[beep] / 24. / 60.  # np.random.uniform(0.5, max_fwhm, 1) #* (1. / 60.) * (1. / 24.)  # minutes to days # 0.004
        gauss_fwhm_j = test_fwhms[beep] / 24. / 60.  # np.random.uniform(0.5, max_fwhm, 1) #* (1. / 60.) * (1. / 24.)  # minutes to days # 0.004

        g_profile = np.abs(gauss_ampl_j) * np.exp(-((x_synth - gauss_tpeak) ** 2) / (gauss_fwhm_j** 2))

        decay_start = x_synth[np.min(np.where(np.abs(g_profile - decay_ampl_j) == np.min(np.abs(g_profile - decay_ampl_j)))[0])]
        if decay_start > gauss_tpeak:
            decay_start = gauss_tpeak - (decay_start - gauss_tpeak)

        A = gauss_ampl_j
        B = gauss_tpeak
        C = gauss_fwhm_j
        K = decay_ampl_j
        J = decay_fwhm_j
        H = decay_start
        D = 0.689008
        E = -1.60053
        F = 0.302963
        G = -0.278318

        synth_flare = 0.5 * np.sqrt(np.pi) * A * C * K * (D * (scipy.special.erf(B / C + (C * E) / (2. * J)) - scipy.special.erf(B / C + (C * E) / (2. * J) - x_synth / C)) * np.exp(
            (E * (4. * J * (B - H + x_synth) + C ** 2. * E)) / (4. * J ** 2.)) + F * (scipy.special.erf(B / C + (C * G) / (2. * J)) - scipy.special.erf(B / C + (C * G) / (2. * J) - x_synth / C)) * np.exp(
            (G * (4 * J * (B - H + x_synth) + C ** 2. * G)) / (4. * J ** 2.)))

        synth_flare/=np.max(synth_flare)

        plt.plot(x_synth, synth_flare, label = 'FWHM: ' + str(test_fwhms[beep]))

        where_max = np.where(synth_flare == np.max(synth_flare))[0]
        peak_location.append(x_synth[where_max])
        peak_value.append(synth_flare[where_max])
        input_fwhm_value.append(test_fwhms[beep])

        fwhm = get_fwhm(x_synth, synth_flare, do_interp=False)

        output_fwhm_value.append(fwhm) # days to minutes

    # plt.scatter(input_fwhm_value,peak_value)
    # plt.xlabel('Input FWHM Value')
    # plt.ylabel('Peak Value')
    # plt.show()
    #
    # plt.scatter(input_fwhm_value, output_fwhm_value)
    # plt.xlabel('Input FWHM Value')
    # plt.ylabel('Output FWHm Value')
    # plt.show()
    #
    # plt.scatter(input_fwhm_value, peak_location)
    # plt.xlabel('Input FWHM Value')
    # plt.ylabel('Peak Time')
    # plt.show()



    #import pdb; pdb.set_trace()


    # synth_flare_scatter = synth_flare + np.random.normal(-std, std, len(synth_flare))




    # plt.plot(x_synth,synth_flare)
    #plt.savefig('/Users/lbiddle/Desktop/testflare_j.pdf')
    plt.legend(loc='upper right')
    plt.show()

    #import pdb; pdb.set_trace()

    # # where_start = np.where(synth_flare > 0)[0][0]
    # # where_end = np.where(synth_flare > 0.000025)[0][-1]
    #
    # # plt.title('Rise Time = ' + str(np.round((t_max_in - x_synth[where_start])*60*24,2))+' min\nFall Time = ' + str(np.round((x_synth[where_end] - t_max_in)*60*24,2))+' min')
    # # plt.plot([t_max_in,t_max_in], [0,np.max(synth_flare*1.1)], c='grey', lw=0.6)
    # # plt.plot([0, 2], [0, 0], c='grey', lw=0.6)
    # # plt.plot([x_synth[where_start], x_synth[where_start]], [0, np.max(synth_flare * 1.2)], c='green', lw=0.6)
    # # plt.plot([x_synth[where_end], x_synth[where_end]], [0, np.max(synth_flare * 1.2)], c='red', lw=0.6)
    # plt.plot(x_synth, synth_flare, c='black') #, label='fwhm = ' + str(np.round(fwhm_in * 60 * 24, 1)) + ' min')
    # # plt.xlim(x_synth[where_start]-0.05,x_synth[where_end]+0.05)
    # #plt.xlim(0.45,0.55)
    # # plt.ylim(-0.05,np.max(synth_flare*1.1))
    # plt.xlabel('Time (days)')
    # # plt.legend(loc='upper right')
    # # plt.savefig( '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/_synthetic_flare_' + str(a) + '.pdf')
    # # plt.savefig('/Users/lbiddle/Desktop/testflare_j.pdf')
    # # plt.close()
    # plt.show()
    #
    # # import pdb; pdb.set_trace()
    # #
    # # # flare_properties = {"tpeak": t_max,
    # # #                     "amplitude": amplitude,
    # # #                     "fwhm": fwhm,
    # # #                     }
    # # return x_synth, synth_flare, synth_flare_scatter, flare_properties
# test_single_synthetic_jflare1_equation(cadence, max_fwhm=10)



def plot_single_synthetic(fwhm_in = 80):

    x_synth = np.linspace(0,2.0,200000)
    t_max_in = 0.25
    fwhm_in = fwhm_in*(1./60.)*(1./24.) #minutes to days
    amplitude_in = 1
    y_synth = aflare1(x_synth, t_max_in, fwhm_in, amplitude_in)

    where_start = np.where(y_synth > 0)[0][0]
    where_end = np.where(y_synth > 0.0025)[0][-1]

    plt.title('Rise Time = ' + str(np.round((t_max_in - x_synth[where_start])*60*24,2))+' min\nFall Time = ' + str(np.round((x_synth[where_end] - t_max_in)*60*24,2))+' min')
    plt.plot([t_max_in,t_max_in], [0,np.max(y_synth*1.1)], c='grey', lw=0.6)
    plt.plot([0, 2], [0, 0], c='grey', lw=0.6)
    plt.plot([x_synth[where_start], x_synth[where_start]], [0, np.max(y_synth * 1.2)], c='green', lw=0.6)
    plt.plot([x_synth[where_end], x_synth[where_end]], [0, np.max(y_synth * 1.2)], c='red', lw=0.6)
    plt.plot(x_synth, y_synth, c='black', label='fwhm = ' + str(np.round(fwhm_in * 60 * 24, 1)) + ' min')
    #plt.xlim(x_synth[where_start]-0.05,x_synth[where_end]+0.05)
    plt.xlim(0.1,0.5)
    plt.ylim(-0.05,np.max(y_synth*1.1))
    plt.xlabel('Time (days)')
    plt.legend(loc='upper right')
    # plt.savefig( '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/_synthetic_flare_' + str(a) + '.pdf')
    # plt.close()
    plt.show()

    #import pdb; pdb.set_trace()
#create_single_synthetic()

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

def plot_test_fit(x_flare,y_flare,x_fit,y_fit,y_true,flare_id_time,flare_id_flux,x_template,y_template,eq_dur,flare_energy,eq_dur_true,flare_energy_true,save_as):
    font_size = 'large'
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'

    max_amp = np.max([np.max(y_true), np.max(y_flare), np.max(y_fit), np.max(y_template)]) - 1.0
    max_gap = 0.05 * max_amp

    #import pdb; pdb.set_trace()

    fig = plt.figure(1, figsize=(7,5.5), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(111)
    ax1.set_xlim([np.min(flare_id_time)-0.005, np.max(flare_id_time)+0.01])
    ax1.set_xlabel("Time (days)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Flux (ppt)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title(r'Equivalent Duration = ' + str(np.round(eq_dur, 2)) + ' (sec) Flare Energy = ' + str('{:0.3e}'.format(flare_energy)) + 'erg s$^{-1}$\nTrue Equivalent Duration = ' + str(np.round(eq_dur_true, 2)) + ' (sec) True Flare Energy = ' + str('{:0.3e}'.format(flare_energy_true)) + 'erg s$^{-1}$', pad=10, fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    ax1.plot(x_fit, y_true, c='#000000', lw=0.5, label='Dav Flare w/ Template Params')
    ax1.plot(x_fit, y_fit, c='blue', lw=2, label='Flare Fit')
    if len(flare_id_time) > 0:
        where_under = np.where((x_fit >= np.min(flare_id_time)) & (x_fit <= np.max(flare_id_time)))[0]
        ax1.fill_between(x_fit[where_under], y_fit[where_under], y2=np.zeros_like(y_fit[where_under]), color='blue', alpha=0.25)
    else:
        ax1.fill_between(x_fit, y_fit, y2=np.zeros_like(y_fit), color='#006699', alpha=0.15)
    ax1.scatter(x_flare, y_flare, c='red', s=np.pi*(3)**2, label='Test Flare')
    ax1.scatter(flare_id_time, flare_id_flux, c='#00cc00', s=np.pi*(2)**2, label='Identified Flare')
    ax1.plot(x_template,y_template, c='orange', lw=0.75, label='True Flare Template')
    ax1.set_ylim([1.0 - max_gap, 1 + max_amp + max_gap])
    ax1.legend(fontsize=font_size, loc='upper right')
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
def plot_cant_find(x_flare,y_flare,save_as):
    font_size = 'large'
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'

    max_amp = np.max(y_flare)
    max_gap = 0.05 * max_amp

    #import pdb; pdb.set_trace()

    fig = plt.figure(1, figsize=(7,5.5), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(111)
    ax1.set_xlim([np.min(x_flare)-0.005, np.max(x_flare)+0.01])
    ax1.set_xlabel("Time (days)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Flux (ppt)", fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    ax1.scatter(x_flare, y_flare, c='red', s=np.pi*(3)**2, label='Test Flare')
    #ax1.set_ylim([max_gap, max_amp + max_gap])
    ax1.legend(fontsize=font_size, loc='upper right')
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()


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
def plot_stat_hist4(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,flare_energy_opt,impulsiveness, hist_inclusion, bin_slice_factor, save_as):
    hist_inclusion = 5000
    print('Plotting Simple Histograms...')

    font_size = 'medium'
    nbins = 'auto'
    t_peak_color = '#006699'
    fwhm_color = '#990033'
    ampl_color = '#669900'
    eqdur_color = '#cc6600'
    energy_color = '#666699'
    impulsiveness_color = '#669999'
    x_factor = hist_inclusion * bin_slice_factor

    impulse_max_lim_factor = 1.0 #0.50

    dat1 = np.array(t_peak_opt)*24*60
    dat2 = fwhm_opt
    dat3 = ampl_opt
    dat4 = eq_duration_opt
    dat5 = flare_energy_opt
    dat6 = impulsiveness

    #import pdb; pdb.set_trace()

    fig = plt.figure(1, figsize=(15, 4*2.2), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(231)

    y_hist, bin_edges = np.histogram(dat1, bins='auto')
    #bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges) + 0.5 * np.diff(bin_edges)[0], 0.5 * np.diff(bin_edges)[0])
    axis_spacing = np.arange(0, len(bin_edges), 1)
    bin_slice_factor_ax1 = 3
    if bin_slice_factor_ax1 == 3:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor_ax1,
                                     1. / bin_slice_factor_ax1)[0:-1]
    else:
        new_axis_spacing = np.arange(np.min(axis_spacing), np.max(axis_spacing) + 1. / bin_slice_factor_ax1,
                                     1. / bin_slice_factor_ax1)
    bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
    bin_width = np.diff(bin_edges)[0]
    # where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    where_hist_max = np.where(y_hist == np.max(y_hist))[0]
    if len(where_hist_max) > 0:
        where_hist_max = int(np.mean(where_hist_max))
    where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (x_factor * bin_width)) & (
            bin_edges <= bin_edges[where_hist_max] + (x_factor * bin_width)))[0]
    y_hist, bin_edges = np.histogram(dat1, bins=bin_edges[where_within])

    hist_dat1 = ax1.hist(dat1, color=t_peak_color, bins=bin_edges) #, weights=np.ones(len(dat1))/len(dat1)) #, edgecolor='#000000', linewidth=1.2)
    ax1.hist(dat1, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat1))/len(dat1))
    ax1.plot([0,0],[0,np.max(hist_dat1[0])*10], '--',  color='#000000', lw=1) #, label="Rotation Model")

    # ax1.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax1.set_xlim(np.min(bin_edges),np.max(bin_edges))
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
    #where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    where_hist_max = np.where(y_hist == np.max(y_hist))[0]
    if len(where_hist_max) > 0:
        where_hist_max = int(np.mean(where_hist_max))
    where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (x_factor * bin_width)) & (
                bin_edges <= bin_edges[where_hist_max] + (x_factor * bin_width)))[0]

    #import pdb; pdb.set_trace()

    y_hist, bin_edges = np.histogram(dat2, bins=bin_edges[where_within])

    #ax2.hist(dat2, color=fwhm_color, bins=nbins, edgecolor='#000000', linewidth=1.2)
    hist_dat2 = ax2.hist(dat2, color=fwhm_color, bins=bin_edges) #, weights=np.ones(len(dat2))/len(dat2)) #, edgecolor='#000000', linewidth=1.2)
    ax2.hist(dat2, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat2))/len(dat2))
    ax2.plot([0, 0], [0, np.max(hist_dat2[0]) * 10], '--', color='#000000', lw=1)

    #ax2.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
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
    #where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    where_hist_max = np.where(y_hist == np.max(y_hist))[0]
    if len(where_hist_max) > 0:
        where_hist_max = int(np.mean(where_hist_max))
    where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (x_factor * bin_width)) & (
                bin_edges <= bin_edges[where_hist_max] + (x_factor * bin_width)))[0]
    y_hist, bin_edges = np.histogram(dat3, bins=bin_edges[where_within])

    hist_dat3 = ax3.hist(dat3, color=ampl_color, bins=bin_edges) #, weights=np.ones(len(dat3))/len(dat3)) #, edgecolor='#000000', linewidth=1.2)
    ax3.hist(dat3, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat3))/len(dat3))
    ax3.plot([0, 0], [0, np.max(hist_dat3[0]) * 10], '--', color='#000000', lw=1)

    # ax3.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax3.set_xlim(np.min(bin_edges), np.max(bin_edges))
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
    # where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    where_hist_max = np.where(y_hist == np.max(y_hist))[0]
    if len(where_hist_max) > 0:
        where_hist_max = int(np.mean(where_hist_max))
    where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (x_factor * bin_width)) & (
                bin_edges <= bin_edges[where_hist_max] + (x_factor * bin_width)))[0]
    y_hist, bin_edges = np.histogram(dat4, bins=bin_edges[where_within])

    hist_dat4 = ax4.hist(dat4, color=eqdur_color, bins=bin_edges) #, weights=np.ones(len(dat4)) / len(dat4))  # , edgecolor='#000000', linewidth=1.2)
    ax4.hist(dat4, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat4)) / len(dat4))
    ax4.plot([0, 0], [0, np.max(hist_dat4[0]) * 10], '--', color='#000000', lw=1)

    #ax4.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax4.set_xlim(np.min(bin_edges), np.max(bin_edges))
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
    #where_within = np.where((bin_edges >= -x_factor * bin_width) & (bin_edges <= x_factor * bin_width))[0]
    where_hist_max = np.where(y_hist == np.max(y_hist))[0]
    if len(where_hist_max) > 0:
        where_hist_max = int(np.mean(where_hist_max))
    where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (x_factor * bin_width)) & (
                bin_edges <= bin_edges[where_hist_max] + (x_factor * bin_width)))[0]
    y_hist, bin_edges = np.histogram(dat5, bins=bin_edges[where_within])

    hist_dat5 = ax5.hist(dat5, color=energy_color, bins=bin_edges) #, weights=np.ones(len(dat5)) / len(dat5))  # , edgecolor='#000000', linewidth=1.2)
    ax5.hist(dat5, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat5)) / len(dat5))
    ax5.plot([0, 0], [0, np.max(hist_dat5[0]) * 10], '--', color='#000000', lw=1)

    # ax5.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax5.set_xlim(np.min(bin_edges), np.max(bin_edges))
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

    print("Plotting Can't Fit Histograms...")

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
    ax1.set_xlabel("Fractional Location of Peak Time Between Points", fontsize=font_size, style='normal', family='sans-serif')
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
    ax2.set_xlabel("Fraction of Lightcurve Cadence", fontsize=font_size, style='normal', family='sans-serif')
    #ax2.set_xlabel("FWHM", fontsize=font_size, style='normal', family='sans-serif')
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
def plot_hist_cant_find(t_peak_cant_find,fwhm_cant_find,ampl_cant_find,impulsiveness_cant_find, save_as):

    print("Plotting Can't Fit Histograms...")

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

    dat1 = np.array(t_peak_cant_find) #*24*60
    dat2 = np.array(fwhm_cant_find)*24*60
    dat3 = np.array(ampl_cant_find)
    dat4 = np.array(impulsiveness_cant_find)

    fig = plt.figure(1, figsize=(15, 4), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(141)

    try:
        y_hist, bin_edges = np.histogram(dat1, bins='auto')
    except:
        print('hist issue')
        import pdb; pdb.set_trace()

    hist_dat1 = ax1.hist(dat1, color=t_peak_color, bins=bin_edges) #, weights=np.ones(len(dat1))/len(dat1)) #, edgecolor='#000000', linewidth=1.2)
    ax1.hist(dat1, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat1))/len(dat1))
    #ax1.plot([0,0],[0,np.max(hist_dat1[0])*10], '--',  color='#000000', lw=1) #, label="Rotation Model")

    # ax1.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax1.set_ylim([0, np.max(hist_dat1[0]) * 1.10])
    #plt.legend(fontsize=10, loc="upper left")
    ax1.set_xlabel("Fractional Location of Peak Time Between Points", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_title("Peak Time", fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax2 = fig.add_subplot(142)

    y_hist, bin_edges = np.histogram(dat2, bins='auto')
    hist_dat2 = ax2.hist(dat2, color=fwhm_color, bins=bin_edges) #, weights=np.ones(len(dat2))/len(dat2)) #, edgecolor='#000000', linewidth=1.2)
    ax2.hist(dat2, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat2))/len(dat2))
    #ax2.plot([0, 0], [0, np.max(hist_dat2[0]) * 10], '--', color='#000000', lw=1)

    # ax2.set_xlim(-x_factor*bin_width,x_factor*bin_width)
    ax2.set_ylim([0, np.max(hist_dat2[0]) * 1.10])
    #plt.legend(fontsize=10, loc="upper left")
    ax2.set_xlabel("Fraction of Lightcurve Cadence", fontsize=font_size, style='normal', family='sans-serif')
    # ax2.set_xlabel("FWHM", fontsize=font_size, style='normal', family='sans-serif')
    #ax2.set_ylabel("Counts", fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_title("FWHM", fontsize=font_size, style='normal', family='sans-serif')
    ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


    ax3 = fig.add_subplot(143)

    y_hist, bin_edges = np.histogram(dat3, bins='auto')
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


def sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, plot_type, save_as, stddev, lc_cadence, max_fwhm, hist_inclusion, bin_slice_factor, property_to_sort='fwhm'):

    opt_list = [np.array(t_peak_opt) * 24 * 60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    # true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time (min)', '% Difference from True FWHM',
                    '% Difference from True Amplitude', '% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    #bin_slice_factor = 1.

    if property_to_sort == 'fwhm':

        print('Plotting FWHM sort...')

        # fwhm_min = (1./60.) * (1. / 60.) * (1. / 24.)
        fwhm_min = 0.99 * np.min(fwhm_true)
        # fwhm_max = max_fwhm * (1. / 60.) * (1. / 24.)
        fwhm_max = 1.01 * np.max(fwhm_true)
        #import pdb; pdb.set_trace()
        #fwhm_grid = np.linspace(fwhm_min, fwhm_max, 101)
        #grid_spacing = ((lc_cadence * (1. / 60.) * (1. / 24.)) - fwhm_min)/100.
        grid_spacing = (fwhm_max - fwhm_min)/100
        #print('lc_cadence: ' + str(lc_cadence))
        #print('fwhm_min: ' + str(fwhm_min))
        #print('fwhm_max: ' + str(fwhm_max))
        #print('grid_spacing: ' + str(grid_spacing))
        fwhm_grid = np.arange(fwhm_min,fwhm_max+grid_spacing,grid_spacing)
        #print(fwhm_grid)
        #print('len(fwhm_grid): ' + str(len(fwhm_grid)))
        fwhm_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(fwhm_grid) - 1):
            # In each iteration, add an empty list to the main list
            fwhm_slots.append([])

        for grid_index in range(1, len(fwhm_grid)):
            for true_index in range(0, len(fwhm_true)):
                if (fwhm_true[true_index] > fwhm_grid[grid_index - 1]) and (fwhm_true[true_index] <= fwhm_grid[grid_index]):
                    fwhm_slots[grid_index - 1].append(true_index)

        #import pdb; pdb.set_trace()

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'
        impulse_max_lim_factor = 0.50
        xlim_mult = hist_inclusion * bin_slice_factor

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
            # print('bin_edges initial' + str(bin_edges))
            # print('y_hist initial' + str(y_hist))
            # bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges)+0.5*np.diff(bin_edges)[0], 0.5*np.diff(bin_edges)[0])
            if (prop != 5) and (bin_slice_factor > 1):
                axis_spacing = np.arange(0,len(bin_edges),1)
                if bin_slice_factor == 3:
                    new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)[0:-1]
                else:
                    new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)
                bin_edges = np.interp(new_axis_spacing,axis_spacing,bin_edges)
            #bin_edges = np.interp(np.arange())

            #import pdb; pdb.set_trace()

            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
            #print('creating first grid')
            for slot_index in range(len(fwhm_slots)):
                #print('bin_edges before fill' + str(bin_edges))
                #y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                #print('bin_edges after fill' + str(bin_edges))
                Z[slot_index, :] = y_hist
                #print('y_hist: ' + str(y_hist))

            row_hist = sum_rows(Z)
            #print('summed rows')
            col_hist1 = sum_cols(Z)
            #print('summed cols')
            if plot_type == 'cumulative':
                cum_hist1 = cum_cols(Z)

            if prop != 5:
                bin_width = np.diff(bin_edges)[0]
                #print('bin_width: ' + str(bin_width))
                where_within = np.where((bin_edges >= -xlim_mult * bin_width) & (bin_edges <= xlim_mult * bin_width))[0]
                #print('where_within: ' + str(where_within))
                #print('bin_edges[where_within]: ' + str(bin_edges[where_within]))
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
            if prop == 5:
                where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])

            col_hist = np.array(col_hist1)[where_within[:-1]]
            if plot_type == 'cumulative':
                cum_hist = np.array(cum_hist1)[where_within[:-1]]

            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
            #print('creating second grid')
            for slot_index in range(len(fwhm_slots)):
                #print('opt_list[prop]: ' + str(opt_list[prop]))
                #print('opt_list[prop][fwhm_slots[slot_index]]: ' + str(np.array(opt_list[prop])[fwhm_slots[slot_index]]))
                #print('bin_edges before: ' + str(bin_edges))
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                #print('bin_edges after: ' + str(bin_edges))
                Z[slot_index, :] = y_hist
               # print('y_hist: ' + str(y_hist))

                # if len(fwhm_slots[slot_index]) > 0:
                #     import pdb; pdb.set_trace()

            #print('Z: ' + str(Z))
            #print('len y_hist: ' + str(len(y_hist)))
            # import pdb; pdb.set_trace()

            # ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax1.set_ylabel('True FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
            ax1.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)

            X, Y = np.meshgrid(bin_edges[:-1], (fwhm_grid[:-1] * 24 * 60))
            p = ax1.pcolor(X, Y, Z, cmap=cm.BuPu, edgecolors='face', vmin=Z.min(), vmax=Z.max(), rasterized=True)
            #import pdb; pdb.set_trace()
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
                ax1.plot([np.min(bin_edges), np.max(bin_edges)], [lc_cadence, lc_cadence], color='#000000', alpha=0.2, lw=0.5)
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

            if prop == 2:
                #ax2.set_title(str(np.round(lc_cadence,2)) + ' min Cadence',fontsize=font_size, style=font_style, family=font_family)
                props = dict(boxstyle=None,facecolor=None, alpha=0.5)
                textstr = '\n'.join(('Cadence:', str(np.round(lc_cadence,2)),'min'))
                ax1.text(1.30, 1.25, textstr, transform=ax1.transAxes, fontsize='medium', style=font_style,
                         family=font_family, weight='heavy', verticalalignment='center', horizontalalignment='center') #,bbox=props)

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
        xlim_mult = 15. * bin_slice_factor

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
            if (prop != 5) and (bin_slice_factor > 1):
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
def sort_property7(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, plot_type, save_as, stddev, lc_cadence, max_fwhm, hist_inclusion, bin_slice_factor, property_to_sort='fwhm'):

    opt_list = [np.array(t_peak_opt) * 24 * 60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    # true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time (min)', '% Difference from True FWHM',
                    '% Difference from True Amplitude', '% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    #bin_slice_factor = 1.

    if property_to_sort == 'fwhm':

        print('Plotting FWHM sort...')

        # fwhm_min = (1./60.) * (1. / 60.) * (1. / 24.)
        fwhm_min = 0.99 * np.min(fwhm_true)
        # fwhm_max = max_fwhm * (1. / 60.) * (1. / 24.)
        fwhm_max = 1.01 * np.max(fwhm_true)
        grid_spacing = (fwhm_max - fwhm_min)/5000
        fwhm_grid = np.arange(fwhm_min,fwhm_max+grid_spacing,grid_spacing)
        #fwhm_grid = np.logspace(fwhm_min,fwhm_max, num=100, endpoint=True, base=10.0)
        #print(fwhm_grid)
        #print('len(fwhm_grid): ' + str(len(fwhm_grid)))
        fwhm_slots = []
        # Iterate over a sequence of numbers
        for slot_num in range(len(fwhm_grid) - 1):
            # In each iteration, add an empty list to the main list
            fwhm_slots.append([])

        for grid_index in range(1, len(fwhm_grid)):
            for true_index in range(0, len(fwhm_true)):
                if (fwhm_true[true_index] > fwhm_grid[grid_index - 1]) and (fwhm_true[true_index] <= fwhm_grid[grid_index]):
                    fwhm_slots[grid_index - 1].append(true_index)


        # import pdb; pdb.set_trace()

        font_size = 'small'
        font_style = 'normal'
        font_family = 'sans-serif'
        impulse_max_lim_factor = 1.0 #0.50
        xlim_mult = hist_inclusion * bin_slice_factor

        fig = plt.figure(figsize=(10, 6), facecolor='#ffffff')  # , dpi=300)
        for prop in range(len(opt_list)):

            print(prop)

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
            # print('bin_edges initial' + str(bin_edges))
            # print('y_hist initial' + str(y_hist))
            # bin_edges = np.arange(np.min(bin_edges), np.max(bin_edges)+0.5*np.diff(bin_edges)[0], 0.5*np.diff(bin_edges)[0])
            if prop == 0:
                bin_slice_factor = 3
            else:
                bin_slice_factor = 1
            if (prop != 5) and (bin_slice_factor > 1):
                axis_spacing = np.arange(0,len(bin_edges),1)
                if bin_slice_factor == 3:
                    new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)[0:-1]
                else:
                    new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)
                bin_edges = np.interp(new_axis_spacing,axis_spacing,bin_edges)
            #bin_edges = np.interp(np.arange())

            #import pdb; pdb.set_trace()

            Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
            #print('creating first grid')
            for slot_index in range(len(fwhm_slots)):
                #print('bin_edges before fill' + str(bin_edges))
                #y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                #print('bin_edges after fill' + str(bin_edges))
                Z[slot_index, :] = y_hist
                #print('y_hist: ' + str(y_hist))


            row_hist = sum_rows(Z)
            #print('summed rows')
            col_hist1 = sum_cols(Z)
            #print('summed cols')
            if plot_type == 'cumulative':
                cum_hist1 = cum_cols(Z)

            if prop != 5:
                bin_width = np.diff(bin_edges)[0]

                where_hist_max = np.where(col_hist1 == np.max(col_hist1))[0]
                if len(where_hist_max) > 0:
                    where_hist_max = int(np.mean(where_hist_max))

                # print('bin_width: ' + str(bin_width))
                # where_within = np.where((bin_edges >= -xlim_mult * bin_width) & (bin_edges <= xlim_mult * bin_width))[0]
                # if prop == 0:
                # where_within = np.where((bin_edges >= np.min(bin_edges)+0.30*(np.max(bin_edges) - np.min(bin_edges))) & (bin_edges <= np.max(bin_edges) - 0.10*(np.max(bin_edges) - np.min(bin_edges))))[0]
                # if (prop == 1) or (prop == 2):
                #     where_within = np.where((bin_edges >= np.min(bin_edges) + 0.0 * (np.max(bin_edges) - np.min(bin_edges))) & (bin_edges <= np.max(bin_edges) - 0.4 * (np.max(bin_edges) - np.min(bin_edges))))[0]
                # if prop == 0:
                #     where_within = np.where((bin_edges >= np.min(bin_edges) + 0.0 * (np.max(bin_edges) - np.min(bin_edges))) & (bin_edges <= np.max(bin_edges) - 0.0 * (np.max(bin_edges) - np.min(bin_edges))))[0]
                # else:
                #     #import pdb; pdb.set_trace()
                where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (xlim_mult * bin_width)) & (bin_edges <= bin_edges[where_hist_max] + (xlim_mult * bin_width)))[0]

                    # import pdb; pdb.set_trace()

                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
            if prop == 5:
                where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])

            if len(where_within) > 0:

                col_hist = np.array(col_hist1)[where_within[:-1]]
                if plot_type == 'cumulative':
                    cum_hist = np.array(cum_hist1)[where_within[:-1]]

                Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
                #print('creating second grid')
                for slot_index in range(len(fwhm_slots)):
                    #print('opt_list[prop]: ' + str(opt_list[prop]))
                    #print('opt_list[prop][fwhm_slots[slot_index]]: ' + str(np.array(opt_list[prop])[fwhm_slots[slot_index]]))
                    #print('bin_edges before: ' + str(bin_edges))
                    if len(fwhm_slots[slot_index]) > 0:
                        y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                        # print('bin_edges after: ' + str(bin_edges))
                        Z[slot_index, :] = y_hist

                # print(Z[-1, :])
            # import pdb; pdb.set_trace()
            else:
                print('')
                print('prop: ' + str(prop))
                print('poopie')
                print('')

                plt.close()
                break

                # col_hist = np.array(col_hist1)
                # if plot_type == 'cumulative':
                #     cum_hist = np.array(cum_hist1)
                # Z = np.zeros((len(fwhm_grid),len(bin_edges)))



               # print('y_hist: ' + str(y_hist))

                # if len(fwhm_slots[slot_index]) > 0:
                #     import pdb; pdb.set_trace()

            #print('Z: ' + str(Z))
            #print('len y_hist: ' + str(len(y_hist)))
            # import pdb; pdb.set_trace()

            # ax = fig.add_subplot(2,3,prop+1)
            # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
            # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
            if (prop == 0) or (prop == 3):
                ax1.set_ylabel('True FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
            ax1.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)

            X, Y = np.meshgrid(bin_edges, (fwhm_grid * 24 * 60))
            mycolormap = choose_cmap()
            p = ax1.pcolor(X, Y, Z, cmap=mycolormap, edgecolors='face', vmin=Z.min(), vmax=Z.max(), rasterized=True) # cm.BuPu
            #import pdb; pdb.set_trace()
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
                ax1.plot([np.min(bin_edges), np.max(bin_edges)], [lc_cadence, lc_cadence], color='#000000', alpha=0.2, lw=0.5)
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

            if prop == 2:
                #ax2.set_title(str(np.round(lc_cadence,2)) + ' min Cadence',fontsize=font_size, style=font_style, family=font_family)
                props = dict(boxstyle=None,facecolor=None, alpha=0.5)
                if lc_cadence >= 1:
                    textstr = '\n'.join(('Cadence:', str(np.round(lc_cadence,2)),'min'))
                if lc_cadence < 1:
                    textstr = '\n'.join(('Cadence:', str(np.round(lc_cadence*60, 2)), 'sec'))
                ax1.text(1.30, 1.25, textstr, transform=ax1.transAxes, fontsize='medium', style=font_style,
                         family=font_family, weight='heavy', verticalalignment='center', horizontalalignment='center') #,bbox=props)

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
        xlim_mult = 15. * bin_slice_factor

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
            if (prop != 5) and (bin_slice_factor > 1):
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


from altaipony.flarelc import FlareLightCurve
from altaipony import lcio


def fit_statistics5(cadence, downsample = True, set_lc_cadence = 30, set_max_fwhm = 120, hist_inclusion=15, bin_slice_factor=1., n_reps = 100):


    # initialize parameter arrays
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

    t_peak_cant_find = []
    fwhm_cant_find = []
    ampl_cant_find = []
    impulsive_index_cant_find = []

    for rep in range(n_reps):

        if (np.mod(rep+1,250) == 0) or (rep == 0):
            print('Generating Flare Statistics...\nCad. ' + str(np.round(set_lc_cadence, 1)) +'    Rep. ' + str(rep + 1))
            print(' ')

        x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic(cadence, max_fwhm=set_max_fwhm)


        if downsample == True:
            where_start = np.int(np.floor(np.random.uniform(0, set_lc_cadence+1, 1)))

            x_flare = x_synth[where_start::int(set_lc_cadence)]
            y_flare = y_synth[where_start::int(set_lc_cadence)]
            # y_noscatter_downsample = y_synth_noscatter[where_start::15]
        if downsample == False:
            x_flare = x_synth[0::1]
            y_flare = y_synth[0::1]
            # y_noscatter_downsample = y_synth_noscatter[0::1]

        guess_peak = x_flare[np.where(y_flare == np.max(y_flare))[0][0]]
        guess_fwhm = 0.01
        guess_ampl = y_flare[np.where(y_flare == np.max(y_flare))[0][0]]

        try:
            popt, pcov = optimize.curve_fit(aflare1, x_flare, y_flare, p0=(guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
        except:

            for bep in range(len(x_flare)):
                if (flare_properties["tpeak"][0] > x_flare[bep]) and (flare_properties["tpeak"][0] < x_flare[bep+1]):
                    t_peak_frac = (flare_properties["tpeak"][0] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
                    break
                if flare_properties["tpeak"][0] == x_flare[bep]:
                    t_peak_frac = 0
                    break
            # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
            t_peak_cant_fit.append(t_peak_frac)
            fwhm_cant_fit.append(flare_properties["fwhm"][0] / np.float(set_lc_cadence))
            ampl_cant_fit.append(flare_properties["amplitude"][0])
            impulsive_index_cant_fit.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])
            continue


        # flare duration
        flc = FlareLightCurve(time=x_flare, flux=y_flare, flux_err=np.zeros_like(y_flare)+1e-4,detrended_flux=y_flare,detrended_flux_err=np.zeros_like(y_flare)+1e-4)
        try:
            flc = flc.find_flares()
        except:
            flare_time = []
            flare_flux = []
            continue
        #flc.flares.to_csv('flares_' + targ + '.csv', index=False)
        if len(flc.flares) > 0:
            flare_time = flc.time[flc.flares['istart'][0]-1:flc.flares['istop'][0] + 1] #flc.time[f.istart:f.istop + 1]
            flare_flux = flc.flux[flc.flares['istart'][0]-1:flc.flares['istop'][0] + 1]

        else:
            flare_time = []
            flare_flux = []

        #import pdb; pdb.set_trace()

        # energy and equivalent_duration
        if len(flare_time) > 0:
            x_fit = np.linspace(np.min(flare_time), np.max(flare_time), 2500)
            y_fit = aflare1(x_fit, *popt)
            y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
        else:
            x_fit = np.linspace(np.min(x_flare), np.max(x_flare), 5000)
            y_fit = aflare1(x_fit, *popt)
            y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])

            for bep in range(len(x_flare)):
                if (flare_properties["tpeak"][0] > x_flare[bep]) and (flare_properties["tpeak"][0] < x_flare[bep+1]):
                    t_peak_frac = (flare_properties["tpeak"][0] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
                    break
                if flare_properties["tpeak"][0] == x_flare[bep]:
                    t_peak_frac = 0
                    break

            t_peak_cant_find.append(t_peak_frac)
            fwhm_cant_find.append(flare_properties["fwhm"][0] / np.float(set_lc_cadence))
            ampl_cant_find.append(flare_properties["amplitude"][0])
            impulsive_index_cant_find.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])

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

        # for the plot
        x_fit = np.linspace(np.min(x_flare), np.max(x_flare), 5000)
        y_fit = aflare1(x_fit, *popt)
        y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])


        if set_lc_cadence - int(set_lc_cadence) > 0:
            is_half = 'yes'
        else:
            is_half = 'no'


        if (np.mod(rep + 1, 2500) == 0) or (rep == 0):
            if is_half == 'yes':
                # if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/'):
                #     os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/')
                remainder = set_lc_cadence - int(set_lc_cadence)
                if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/'):
                    os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/')
                #save_as_test = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/' + str(rep + 1) + '.pdf'
                save_as_test = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/' + str(rep + 1) + '.pdf'
            else:
                if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/'):
                    os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/')
                save_as_test = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/' + str(rep + 1) + '.pdf'

            try:
                plot_test_fit(x_flare, y_flare + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)
            except:
                continue

        if (np.mod(rep + 1, 2500) == 0) or (rep == 0):
            if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/For_Animation/'):
                os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/For_Animation/')


        t_peak_opt.append((popt[0] - flare_properties["tpeak"][0]))
        fwhm_opt.append((popt[1] - flare_properties["fwhm"][0]) / flare_properties["fwhm"][0] * 100)
        ampl_opt.append((popt[2] - flare_properties["amplitude"][0]) / flare_properties["amplitude"][0] * 100)

        t_peak_true.append(flare_properties["tpeak"][0])
        fwhm_true.append(flare_properties["fwhm"][0])
        ampl_true.append(flare_properties["amplitude"][0])
        impulsive_index_true.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])


        if (np.mod(rep + 1, 1000) == 0) or (np.mod(rep + 1, n_reps) == 0):
            if is_half == 'yes':
                # save_as_hist = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/fit_stat_hist.pdf'
                remainder = set_lc_cadence - int(set_lc_cadence)
                save_as_hist = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/fit_stat_hist.pdf'
            else:
                save_as_hist = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist.pdf'
            plot_stat_hist4(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, hist_inclusion, bin_slice_factor, save_as_hist)

            if np.mod(rep + 1, n_reps) == 0:
                if is_half == 'yes':
                    #save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                    remainder = set_lc_cadence - int(set_lc_cadence)
                    save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                else:
                    save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                              ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                              stddev, set_lc_cadence, set_max_fwhm, hist_inclusion, bin_slice_factor,  property_to_sort='fwhm')
            if np.mod(rep + 1, n_reps) == 0:
                if is_half == 'yes':
                    remainder = set_lc_cadence - int(set_lc_cadence)
                    save_as_hist_2D_fwhm = 'For_Animation/fit_stat_hist_2D_fwhm_sum_' + str(int(set_lc_cadence)) + '+' + str(remainder) + '.pdf'
                else:
                    save_as_hist_2D_fwhm = 'For_Animation/fit_stat_hist_2D_fwhm_sum_' + str(int(set_lc_cadence)) + '.pdf'
                sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                               ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                               stddev, set_lc_cadence, set_max_fwhm, hist_inclusion, bin_slice_factor, property_to_sort='fwhm')
                # if (rep > 0) and (np.mod(rep + 1, n_reps) == 0): # int(0.5*n_reps)) == 0):
                #     save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_fwhm_cum.pdf'
                #     sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                    ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                #                    save_as_hist_2D_fwhm, stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor,  property_to_sort='fwhm')
                # save_as_hist_2D_ampl = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_ampl_sum.pdf'
                # sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_ampl,
                #                stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor,  property_to_sort='amplitude')
                # if (rep > 0) and (np.mod(rep + 1, n_reps) == 0): # int(0.5*n_reps)) == 0):
                #     save_as_hist_2D_ampl = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_ampl_cum.pdf'
                #     sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                    ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                #                    save_as_hist_2D_ampl, stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor, property_to_sort='amplitude')

            if len(fwhm_cant_fit) > 0:
                if is_half == 'yes':
                    # save_as_cant_fit = 'All_Amplitudes_' + str(np.round(set_lc_cadence, 1)) + 'min/fit_stat_hist_cant_fit.pdf'
                    remainder = set_lc_cadence - int(set_lc_cadence)
                    save_as_cant_fit = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/fit_stat_hist_cant_fit.pdf'

                else:
                    save_as_cant_fit = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_cant_fit.pdf'
                plot_hist_cant_fit(t_peak_cant_fit, fwhm_cant_fit, ampl_cant_fit, impulsive_index_cant_fit, save_as=save_as_cant_fit)

            if len(fwhm_cant_find) > 0:
                if is_half == 'yes':
                    #save_as_cant_find = 'All_Amplitudes_' + str(np.round(set_lc_cadence, 1)) + 'min/fit_stat_hist_cant_find.pdf'
                    remainder = set_lc_cadence - int(set_lc_cadence)
                    save_as_cant_find = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/fit_stat_hist_cant_find.pdf'
                else:
                    save_as_cant_find = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_cant_find.pdf'
                plot_hist_cant_find(t_peak_cant_find, fwhm_cant_find, ampl_cant_find, impulsive_index_cant_find, save_as=save_as_cant_find)
def fit_statistics6(cadence, template = 1, downsample = True, set_lc_cadence = 30, set_max_fwhm = 120, hist_inclusion=15, bin_slice_factor=1., n_reps = 100):


    # initialize parameter arrays
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

    t_peak_cant_find = []
    fwhm_cant_find = []
    ampl_cant_find = []
    impulsive_index_cant_find = []

    for rep in range(n_reps):

        print(rep+1)

        if (np.mod(rep+1,250) == 0) or (rep == 0):
            print('Generating Flare Statistics...\nCad. ' + str(np.round(set_lc_cadence, 1)) +'    Rep. ' + str(rep + 1))
            print(' ')

        # Davenport et al. (2014)
        if template == 1:
            x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic(cadence, max_fwhm=set_max_fwhm)
        # Jackman et al. (2018) -  numpy.convolve method
        if template == 2:
            x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic_jflare1(cadence, max_fwhm=set_max_fwhm)
        # Jackman et al. (2018) -  straight calculus method
        if template == 3:
            x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic_jflare1_equation(cadence, max_fwhm=set_max_fwhm)


        if downsample == True:

            cadence_bench = (set_lc_cadence)*60  # to put in terms of seconds because finest sampling done with 1 sec cadence

            where_start = np.int(np.floor(np.random.uniform(0, cadence_bench+1, 1)))

            x_flare = x_synth[where_start::int(cadence_bench)]
            y_flare = y_synth[where_start::int(cadence_bench)]
            # y_noscatter_downsample = y_synth_noscatter[where_start::15]
        if downsample == False:
            x_flare = x_synth[0::1]
            y_flare = y_synth[0::1]
            # y_noscatter_downsample = y_synth_noscatter[0::1]

        guess_peak = x_flare[np.where(y_flare == np.max(y_flare))[0][0]]
        guess_fwhm = 0.01
        guess_ampl = y_flare[np.where(y_flare == np.max(y_flare))[0][0]]

        try:
            popt, pcov = optimize.curve_fit(aflare1, x_flare, y_flare, p0=(guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
        except:

            # for bep in range(len(x_flare)):
            #     if (flare_properties["tpeak"][0] > x_flare[bep]) and (flare_properties["tpeak"][0] < x_flare[bep+1]):
            #         t_peak_frac = (flare_properties["tpeak"][0] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
            #         break
            #     if flare_properties["tpeak"][0] == x_flare[bep]:
            #         t_peak_frac = 0
            #         break
            # # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
            # t_peak_cant_fit.append(t_peak_frac)
            # fwhm_cant_fit.append(flare_properties["fwhm"][0] / (np.float(set_lc_cadence)/24./60.))
            # ampl_cant_fit.append(flare_properties["amplitude"][0])
            # impulsive_index_cant_fit.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])
            for bep in range(len(x_flare)):
                if (flare_properties["tpeak"] > x_flare[bep]) and (flare_properties["tpeak"] < x_flare[bep+1]):
                    t_peak_frac = (flare_properties["tpeak"] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
                    break
                if flare_properties["tpeak"] == x_flare[bep]:
                    t_peak_frac = [0]
                    break
            # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
            t_peak_cant_fit.append(t_peak_frac)
            fwhm_cant_fit.append(flare_properties["fwhm"] / (np.float(set_lc_cadence)/24./60.))
            ampl_cant_fit.append(flare_properties["amplitude"])
            impulsive_index_cant_fit.append(flare_properties["amplitude"] / flare_properties["fwhm"])
            continue


        # flare duration
        flc = FlareLightCurve(time=x_flare, flux=y_flare, flux_err=np.zeros_like(y_flare)+1e-4,detrended_flux=y_flare,detrended_flux_err=np.zeros_like(y_flare)+1e-4)
        try:
            flc = flc.find_flares()
        except:
            flare_time = []
            flare_flux = []
            continue
        #flc.flares.to_csv('flares_' + targ + '.csv', index=False)
        if len(flc.flares) > 0:
            flare_time = flc.time[flc.flares['istart'][0]-1:flc.flares['istop'][0] + 1] #flc.time[f.istart:f.istop + 1]
            flare_flux = flc.flux[flc.flares['istart'][0]-1:flc.flares['istop'][0] + 1]

        else:
            flare_time = []
            flare_flux = []

        #import pdb; pdb.set_trace()

        # energy and equivalent_duration
        if len(flare_time) > 0:
            x_fit = np.linspace(np.min(flare_time), np.max(flare_time), 10000)
            y_fit = aflare1(x_fit, *popt)
            #y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
            y_true = aflare1(x_fit, flare_properties["tpeak"], flare_properties["fwhm"],flare_properties["amplitude"])
        else:
            x_fit = np.linspace(np.min(x_flare), np.max(x_flare), 10000)
            y_fit = aflare1(x_fit, *popt)
            #y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
            y_true = aflare1(x_fit, flare_properties["tpeak"], flare_properties["fwhm"], flare_properties["amplitude"])

            # for bep in range(len(x_flare)):
            #     if (flare_properties["tpeak"][0] > x_flare[bep]) and (flare_properties["tpeak"][0] < x_flare[bep+1]):
            #         t_peak_frac = (flare_properties["tpeak"][0] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
            #         break
            #     if flare_properties["tpeak"][0] == x_flare[bep]:
            #         t_peak_frac = 0
            #         break
            #
            # t_peak_cant_find.append(t_peak_frac)
            # fwhm_cant_find.append(flare_properties["fwhm"][0] / (np.float(set_lc_cadence)/24./60.))
            # ampl_cant_find.append(flare_properties["amplitude"][0])
            # impulsive_index_cant_find.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])
            for bep in range(len(x_flare)):
                if (flare_properties["tpeak"] > x_flare[bep]) and (flare_properties["tpeak"] < x_flare[bep+1]):
                    t_peak_frac = (flare_properties["tpeak"] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
                    break
                if flare_properties["tpeak"][0] == x_flare[bep]:
                    t_peak_frac = [0]
                    break

            t_peak_cant_find.append(t_peak_frac)
            fwhm_cant_find.append(flare_properties["fwhm"] / (np.float(set_lc_cadence)/24./60.))
            ampl_cant_find.append(flare_properties["amplitude"])
            impulsive_index_cant_find.append(flare_properties["amplitude"] / flare_properties["fwhm"])

        L_star = 1.2  # solar luminosity
        L_star *= 3.827e33  # convert to erg/s

        eq_dur = np.trapz(y_fit, x=x_fit)
        eq_dur *= 86400  # convert days to seconds
        flare_energy = L_star * eq_dur

        #eq_dur_true = np.trapz(y_true, x=x_fit)
        eq_dur_true = np.trapz(y_synth, x=x_synth)
        eq_dur_true *= (24 * 60 * 60)  # convert days to seconds
        flare_energy_true = L_star * eq_dur_true

        eq_duration_true.append(eq_dur_true)
        energy_true.append(flare_energy_true)

        eq_duration_opt.append((eq_dur - eq_dur_true) / eq_dur_true * 100)
        energy_opt.append((flare_energy - flare_energy_true) / flare_energy_true * 100)

        # for the plot
        x_fit = np.linspace(np.min(x_flare), np.max(x_flare), 10000)
        y_fit = aflare1(x_fit, *popt)
        # y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
        y_true = aflare1(x_fit, flare_properties["tpeak"], flare_properties["fwhm"], flare_properties["amplitude"])



        if set_lc_cadence - int(set_lc_cadence) > 0:
            is_half = 'yes'
        else:
            is_half = 'no'


        if (np.mod(rep + 1, 2500) == 0) or (rep == 0):
            if is_half == 'yes':
                # if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/'):
                #     os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/')
                remainder = set_lc_cadence - int(set_lc_cadence)
                if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/'):
                    os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/')
                #save_as_test = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/' + str(rep + 1) + '.pdf'
                save_as_test = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/' + str(rep + 1) + '.pdf'
            else:
                if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/'):
                    os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/')
                save_as_test = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/' + str(rep + 1) + '.pdf'

            try:
                plot_test_fit(x_flare, y_flare + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1, x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)
                # plot_test_fit(x_synth, y_synth + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)

            except:
                continue

        if (np.mod(rep + 1, 2500) == 0) or (rep == 0):
            if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/For_Animation/'):
                os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/For_Animation/')


        # t_peak_opt.append((popt[0] - flare_properties["tpeak"][0]))
        # fwhm_opt.append((popt[1] - flare_properties["fwhm"][0]) / flare_properties["fwhm"][0] * 100)
        # ampl_opt.append((popt[2] - flare_properties["amplitude"][0]) / flare_properties["amplitude"][0] * 100)
        #
        # t_peak_true.append(flare_properties["tpeak"][0])
        # fwhm_true.append(flare_properties["fwhm"][0])
        # ampl_true.append(flare_properties["amplitude"][0])
        # impulsive_index_true.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])
        t_peak_opt.append((popt[0] - flare_properties["tpeak"])[0])
        fwhm_opt.append((popt[1] - flare_properties["fwhm"]) / flare_properties["fwhm"] * 100)
        ampl_opt.append(((popt[2] - flare_properties["amplitude"]) / flare_properties["amplitude"])[0] * 100)

        t_peak_true.append(flare_properties["tpeak"])
        fwhm_true.append(flare_properties["fwhm"])
        ampl_true.append(flare_properties["amplitude"])
        impulsive_index_true.append((flare_properties["amplitude"] / flare_properties["fwhm"])[0])


        if (np.mod(rep + 1, 1000) == 0) or (np.mod(rep + 1, n_reps) == 0):
            if is_half == 'yes':
                # save_as_hist = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/fit_stat_hist.pdf'
                remainder = set_lc_cadence - int(set_lc_cadence)
                save_as_hist = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/fit_stat_hist.pdf'
            else:
                save_as_hist = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist.pdf'
            plot_stat_hist4(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, hist_inclusion, bin_slice_factor, save_as_hist)

            if np.mod(rep + 1, n_reps) == 0:
                if is_half == 'yes':
                    #save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                    remainder = set_lc_cadence - int(set_lc_cadence)
                    save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                else:
                    save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                              ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                              stddev, set_lc_cadence, set_max_fwhm, hist_inclusion, bin_slice_factor,  property_to_sort='fwhm')
            if np.mod(rep + 1, n_reps) == 0:
                if is_half == 'yes':
                    remainder = set_lc_cadence - int(set_lc_cadence)
                    save_as_hist_2D_fwhm = 'For_Animation/fit_stat_hist_2D_fwhm_sum_' + str(int(set_lc_cadence)) + '+' + str(remainder) + '.pdf'
                else:
                    save_as_hist_2D_fwhm = 'For_Animation/fit_stat_hist_2D_fwhm_sum_' + str(int(set_lc_cadence)) + '.pdf'
                sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                               ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                               stddev, set_lc_cadence, set_max_fwhm, hist_inclusion, bin_slice_factor, property_to_sort='fwhm')
                # if (rep > 0) and (np.mod(rep + 1, n_reps) == 0): # int(0.5*n_reps)) == 0):
                #     save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_fwhm_cum.pdf'
                #     sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                    ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                #                    save_as_hist_2D_fwhm, stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor,  property_to_sort='fwhm')
                # save_as_hist_2D_ampl = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_ampl_sum.pdf'
                # sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_ampl,
                #                stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor,  property_to_sort='amplitude')
                # if (rep > 0) and (np.mod(rep + 1, n_reps) == 0): # int(0.5*n_reps)) == 0):
                #     save_as_hist_2D_ampl = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_ampl_cum.pdf'
                #     sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                    ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                #                    save_as_hist_2D_ampl, stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor, property_to_sort='amplitude')

            if len(fwhm_cant_fit) > 0:
                if is_half == 'yes':
                    # save_as_cant_fit = 'All_Amplitudes_' + str(np.round(set_lc_cadence, 1)) + 'min/fit_stat_hist_cant_fit.pdf'
                    remainder = set_lc_cadence - int(set_lc_cadence)
                    save_as_cant_fit = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/fit_stat_hist_cant_fit.pdf'

                else:
                    save_as_cant_fit = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_cant_fit.pdf'
                plot_hist_cant_fit(t_peak_cant_fit, fwhm_cant_fit, ampl_cant_fit, impulsive_index_cant_fit, save_as=save_as_cant_fit)

            if len(fwhm_cant_find) > 0:
                if is_half == 'yes':
                    #save_as_cant_find = 'All_Amplitudes_' + str(np.round(set_lc_cadence, 1)) + 'min/fit_stat_hist_cant_find.pdf'
                    remainder = set_lc_cadence - int(set_lc_cadence)
                    save_as_cant_find = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(remainder) + 'min/fit_stat_hist_cant_find.pdf'
                else:
                    save_as_cant_find = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_cant_find.pdf'
                plot_hist_cant_find(t_peak_cant_find, fwhm_cant_find, ampl_cant_find, impulsive_index_cant_find, save_as=save_as_cant_find)
def fit_statistics7(cadence, template = 1, downsample = True, set_lc_cadence = 30, set_max_fwhm = 120, hist_inclusion=15, bin_slice_factor=1., n_reps = 100):


    # initialize parameter arrays
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

    t_peak_cant_find = []
    fwhm_cant_find = []
    ampl_cant_find = []
    impulsive_index_cant_find = []

    for rep in range(n_reps):

        print(rep+1)

        if set_lc_cadence - int(set_lc_cadence) > 0:
            is_half = 'yes'
            remainder = set_lc_cadence - int(set_lc_cadence)
        else:
            is_half = 'no'

        if (np.mod(rep+1,250) == 0) or (rep == 0):
            print('Generating Flare Statistics...\nCad. ' + str(np.round(set_lc_cadence, 1)) +'    Rep. ' + str(rep + 1))
            print(' ')

        # Davenport et al. (2014)
        if template == 1:
            x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic(cadence, max_fwhm=set_max_fwhm)
        # Jackman et al. (2018) -  numpy.convolve method
        if template == 2:
            x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic_jflare1(cadence, max_fwhm=set_max_fwhm)
        # Jackman et al. (2018) -  straight calculus method
        if template == 3:
            x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic_jflare1_equation(cadence, max_fwhm=set_max_fwhm)



        if downsample == True:

            cadence_bench = (set_lc_cadence)*60  # to put in terms of seconds because finest sampling done with 1 sec cadence

            where_start = np.int(np.floor(np.random.uniform(0, cadence_bench+1, 1)))

            x_flare = x_synth[where_start::int(cadence_bench)]
            y_flare = y_synth[where_start::int(cadence_bench)]
            # y_noscatter_downsample = y_synth_noscatter[where_start::15]
        if downsample == False:
            x_flare = x_synth[0::1]
            y_flare = y_synth[0::1]
            # y_noscatter_downsample = y_synth_noscatter[0::1]

        # guess_peak = x_flare[np.where(y_flare == np.max(y_flare))[0][0]]
        # guess_fwhm = 15. * (1./60.) * (1./24.)
        # guess_ampl = y_flare[np.where(y_flare == np.max(y_flare))[0][0]]
        #
        # try:
        #     popt, pcov = optimize.curve_fit(aflare1, x_flare, y_flare, p0=(guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
        # except:
        #     print('first guess fail')
        #     guess_peak = x_flare[np.where(y_flare == np.max(y_flare))[0][0]]
        #     guess_fwhm = 5 * (1./60.) * (1./24.)
        #     guess_ampl = y_flare[np.where(y_flare == np.max(y_flare))[0][0]]
        #
        #     try:
        #         popt, pcov = optimize.curve_fit(aflare1, x_flare, y_flare, p0=(guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
        #
        #     except:
        #         print('second guess fail')
        #
        #         guess_peak = x_flare[np.where(y_flare == np.max(y_flare))[0][0]]
        #         guess_fwhm = 5 * (1./60.) * (1./24.)
        #         guess_ampl = y_flare[np.where(y_flare == np.max(y_flare))[0][0]] * 1.5
        #
        #         try:
        #             popt, pcov = optimize.curve_fit(aflare1, x_flare, y_flare, p0=(guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
        #
        #         except:
        #             print('third guess fail')
        #     # for bep in range(len(x_flare)):
        #     #     if (flare_properties["tpeak"][0] > x_flare[bep]) and (flare_properties["tpeak"][0] < x_flare[bep+1]):
        #     #         t_peak_frac = (flare_properties["tpeak"][0] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
        #     #         break
        #     #     if flare_properties["tpeak"][0] == x_flare[bep]:
        #     #         t_peak_frac = 0
        #     #         break
        #     # # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
        #     # t_peak_cant_fit.append(t_peak_frac)
        #     # fwhm_cant_fit.append(flare_properties["fwhm"][0] / (np.float(set_lc_cadence)/24./60.))
        #     # ampl_cant_fit.append(flare_properties["amplitude"][0])
        #     # impulsive_index_cant_fit.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])
        #             for bep in range(len(x_flare)):
        #                 if (flare_properties["tpeak"] > x_flare[bep]) and (flare_properties["tpeak"] < x_flare[bep+1]):
        #                     t_peak_frac = (flare_properties["tpeak"] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
        #                     break
        #                 if flare_properties["tpeak"] == x_flare[bep]:
        #                     t_peak_frac = [0]
        #                     break
        #             # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
        #             t_peak_cant_fit.append(t_peak_frac)
        #             fwhm_cant_fit.append(flare_properties["fwhm"] / (np.float(set_lc_cadence)/24./60.))
        #             ampl_cant_fit.append(flare_properties["amplitude"])
        #             impulsive_index_cant_fit.append(flare_properties["amplitude"] / flare_properties["fwhm"])
        #             continue


        # flare duration
        flc = FlareLightCurve(time=x_flare, flux=y_flare, flux_err=np.zeros_like(y_flare)+1e-4,detrended_flux=y_flare,detrended_flux_err=np.zeros_like(y_flare)+1e-4)
        try:
            flc = flc.find_flares()
        except:
            flare_time = []
            flare_flux = []
            # continue
        #flc.flares.to_csv('flares_' + targ + '.csv', index=False)
        if len(flc.flares) > 0:
            flare_time = flc.time[flc.flares['istart'][0]-1:flc.flares['istop'][0] + 1] #flc.time[f.istart:f.istop + 1]
            flare_flux = flc.flux[flc.flares['istart'][0]-1:flc.flares['istop'][0] + 1]

        else:
            flare_time = []
            flare_flux = []

        #import pdb; pdb.set_trace()

        if len(flare_time) > 0:
            guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
            guess_fwhm = 15. * (1. / 60.) * (1. / 24.)
            guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]

            try:
                popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
            except:
                print('first guess fail')
                guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                guess_fwhm = 5 * (1. / 60.) * (1. / 24.)
                guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]

                try:
                    popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                    guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))

                except:
                    print('second guess fail')

                    guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                    guess_fwhm = 5 * (1. / 60.) * (1. / 24.)
                    guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] * 1.5

                    try:
                        popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                        guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))

                    except:
                        print('third guess fail')

                        for bep in range(len(flare_time)):
                            if (flare_properties["tpeak"] > flare_time[bep]) and (
                                    flare_properties["tpeak"] < flare_time[bep + 1]):
                                t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (flare_time[bep + 1] - flare_time[bep])
                                break
                            if flare_properties["tpeak"] == flare_time[bep]:
                                t_peak_frac = [0]
                                break
                        # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                        t_peak_cant_fit.append(t_peak_frac)
                        fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                        # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                        ampl_cant_fit.append(flare_properties["amplitude"])
                        impulsive_index_cant_fit.append(flare_properties["amplitude"] / flare_properties["fwhm"])
                        # continue

        if len(flare_time) > 0:
            x_fit = np.linspace(np.min(flare_time), np.max(flare_time), 10000)
            y_fit = aflare1(x_fit, *popt)
            #y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
            y_true = aflare1(x_fit, flare_properties["tpeak"], flare_properties["fwhm"],flare_properties["amplitude"])
        else:
            # x_fit = np.linspace(np.min(x_flare), np.max(x_flare), 10000)
            # y_fit = aflare1(x_fit, *popt)
            # #y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
            # y_true = aflare1(x_fit, flare_properties["tpeak"], flare_properties["fwhm"], flare_properties["amplitude"])

            # for bep in range(len(x_flare)):
            #     if (flare_properties["tpeak"][0] > x_flare[bep]) and (flare_properties["tpeak"][0] < x_flare[bep+1]):
            #         t_peak_frac = (flare_properties["tpeak"][0] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
            #         break
            #     if flare_properties["tpeak"][0] == x_flare[bep]:
            #         t_peak_frac = 0
            #         break
            #
            # t_peak_cant_find.append(t_peak_frac)
            # fwhm_cant_find.append(flare_properties["fwhm"][0] / (np.float(set_lc_cadence)/24./60.))
            # ampl_cant_find.append(flare_properties["amplitude"][0])
            # impulsive_index_cant_find.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])
            for bep in range(len(x_flare)):
                if (flare_properties["tpeak"] > x_flare[bep]) and (flare_properties["tpeak"] < x_flare[bep+1]):
                    t_peak_frac = (flare_properties["tpeak"] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
                    break
                if flare_properties["tpeak"][0] == x_flare[bep]:
                    t_peak_frac = [0]
                    break

            t_peak_cant_find.append(t_peak_frac)
            fwhm_cant_find.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
            # fwhm_cant_find.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
            ampl_cant_find.append(flare_properties["amplitude"])
            impulsive_index_cant_find.append(flare_properties["amplitude"] / flare_properties["fwhm"])


            if np.random.uniform(0,1,1) <= 0.10:

                if is_half == 'yes':
                    if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder, 2)) + 'min/flares_cant_find/'):
                        os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder, 2)) + 'min/flares_cant_find/')
                    save_as_flare_cant_find = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder, 2)) + 'min/flares_cant_find/' + str(rep+1) + '.pdf'
                else:
                    if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/flares_cant_find/'):
                        os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/flares_cant_find/')
                    save_as_flare_cant_find = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/flares_cant_find/' + str(rep+1) + '.pdf'

                plot_cant_find(x_flare, y_flare, save_as_flare_cant_find)

            continue


        L_star = 1.2  # solar luminosity
        L_star *= 3.827e33  # convert to erg/s

        eq_dur = np.trapz(y_fit, x=x_fit)
        eq_dur *= 86400  # convert days to seconds
        flare_energy = L_star * eq_dur

        #eq_dur_true = np.trapz(y_true, x=x_fit)
        eq_dur_true = np.trapz(y_synth, x=x_synth)
        eq_dur_true *= (24 * 60 * 60)  # convert days to seconds
        flare_energy_true = L_star * eq_dur_true

        eq_duration_true.append(eq_dur_true)
        energy_true.append(flare_energy_true)

        eq_duration_opt.append((eq_dur - eq_dur_true) / eq_dur_true * 100)
        energy_opt.append((flare_energy - flare_energy_true) / flare_energy_true * 100)

        # for the plot
        x_fit = np.linspace(np.min(x_flare), np.max(x_flare), 10000)
        y_fit = aflare1(x_fit, *popt)
        # y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
        y_true = aflare1(x_fit, flare_properties["tpeak"], flare_properties["fwhm"], flare_properties["amplitude"])



        # if set_lc_cadence - int(set_lc_cadence) > 0:
        #     is_half = 'yes'
        #     remainder = set_lc_cadence - int(set_lc_cadence)
        # else:
        #     is_half = 'no'


        if (np.mod(rep + 1, 2500) == 0) or (rep == 0):
            if is_half == 'yes':
                # if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/'):
                #     os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/')
                if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/'):
                    os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/')
                #save_as_test = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/' + str(rep + 1) + '.pdf'
                save_as_test = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/' + str(rep + 1) + '.pdf'
            else:
                if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/'):
                    os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/')
                save_as_test = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/' + str(rep + 1) + '.pdf'

            try:
                plot_test_fit(x_flare, y_flare + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1, x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)
                # plot_test_fit(x_synth, y_synth + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)

            except:
                continue

        if (np.mod(rep + 1, 2500) == 0) or (rep == 0):
            if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/For_Animation/'):
                os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/For_Animation/')


        # t_peak_opt.append((popt[0] - flare_properties["tpeak"][0]))
        # fwhm_opt.append((popt[1] - flare_properties["fwhm"][0]) / flare_properties["fwhm"][0] * 100)
        # ampl_opt.append((popt[2] - flare_properties["amplitude"][0]) / flare_properties["amplitude"][0] * 100)
        #
        # t_peak_true.append(flare_properties["tpeak"][0])
        # fwhm_true.append(flare_properties["fwhm"][0])
        # ampl_true.append(flare_properties["amplitude"][0])
        # impulsive_index_true.append(flare_properties["amplitude"][0] / flare_properties["fwhm"][0])
        t_peak_opt.append((popt[0] - flare_properties["tpeak"])[0])
        fwhm_opt.append((popt[1] - flare_properties["fwhm"]) / flare_properties["fwhm"] * 100)
        ampl_opt.append(((popt[2] - flare_properties["amplitude"]) / flare_properties["amplitude"])[0] * 100)

        t_peak_true.append(flare_properties["tpeak"])
        fwhm_true.append(flare_properties["fwhm"])
        ampl_true.append(flare_properties["amplitude"])
        impulsive_index_true.append((flare_properties["amplitude"] / flare_properties["fwhm"])[0])



        #
        #
        # where_id_max = np.where(flare_flux == np.max(flare_flux))[0]
        # flare_id_max_time = flare_time[where_id_max]
        #
        # if flare_id_max_time < popt[0]:
        #     n_points_in_rise = len(flare_flux[::where_id_max])
        #     n_points_in_decay = len(flare_flux[where_id_max+1::])
        # if flare_id_max_time > popt[0]:
        #     n_points_in_rise = len(flare_flux[::where_id_max]) - 1
        #     n_points_in_decay = len(flare_flux[where_id_max::])
        #
        #

        if np.mod(rep + 1, 50) == 0:
            if is_half == 'yes':
                # save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
            else:
                save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
            sort_property7(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                           ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                           stddev, set_lc_cadence, set_max_fwhm, hist_inclusion, bin_slice_factor,
                           property_to_sort='fwhm')

            # if is_half == 'yes':
            #     # save_as_hist = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/fit_stat_hist.pdf'
            #     remainder = set_lc_cadence - int(set_lc_cadence)
            #     save_as_hist = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/fit_stat_hist.pdf'
            # else:
            #     save_as_hist = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist.pdf'
            # plot_stat_hist4(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, hist_inclusion, bin_slice_factor, save_as_hist)



        if (np.mod(rep + 1, 50) == 0) or (np.mod(rep + 1, n_reps) == 0):
            if is_half == 'yes':
                # save_as_hist = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/fit_stat_hist.pdf'
                save_as_hist = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/fit_stat_hist.pdf'
            else:
                save_as_hist = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist.pdf'
            plot_stat_hist4(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true, hist_inclusion, bin_slice_factor, save_as_hist)

            if (np.mod(rep + 1, n_reps) == 0):
                if is_half == 'yes':
                    #save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(np.round(set_lc_cadence,1)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                    save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                else:
                    save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_fwhm_sum.pdf'
                sort_property7(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                              ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                              stddev, set_lc_cadence, set_max_fwhm, hist_inclusion, bin_slice_factor,  property_to_sort='fwhm')
            do_animation = False
            if do_animation == True:
                if np.mod(rep + 1, n_reps) == 0:
                    if is_half == 'yes':
                        save_as_hist_2D_fwhm = 'For_Animation/fit_stat_hist_2D_fwhm_sum_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + '.pdf'
                    else:
                        save_as_hist_2D_fwhm = 'For_Animation/fit_stat_hist_2D_fwhm_sum_' + str(int(set_lc_cadence)) + '.pdf'
                    sort_property7(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                                   ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_fwhm,
                                   stddev, set_lc_cadence, set_max_fwhm, hist_inclusion, bin_slice_factor, property_to_sort='fwhm')
                # if (rep > 0) and (np.mod(rep + 1, n_reps) == 0): # int(0.5*n_reps)) == 0):
                #     save_as_hist_2D_fwhm = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_fwhm_cum.pdf'
                #     sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                    ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                #                    save_as_hist_2D_fwhm, stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor,  property_to_sort='fwhm')
                # save_as_hist_2D_ampl = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_ampl_sum.pdf'
                # sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'sum', save_as_hist_2D_ampl,
                #                stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor,  property_to_sort='amplitude')
                # if (rep > 0) and (np.mod(rep + 1, n_reps) == 0): # int(0.5*n_reps)) == 0):
                #     save_as_hist_2D_ampl = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_2D_ampl_cum.pdf'
                #     sort_property6(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true, fwhm_true,
                #                    ampl_true, eq_duration_true, energy_true, impulsive_index_true, 'cumulative',
                #                    save_as_hist_2D_ampl, stddev, set_lc_cadence, set_max_fwhm, bin_slice_factor, property_to_sort='amplitude')

            if len(fwhm_cant_fit) > 0:
                if is_half == 'yes':
                    # save_as_cant_fit = 'All_Amplitudes_' + str(np.round(set_lc_cadence, 1)) + 'min/fit_stat_hist_cant_fit.pdf'
                    save_as_cant_fit = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/fit_stat_hist_cant_fit.pdf'

                else:
                    save_as_cant_fit = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_cant_fit.pdf'
                plot_hist_cant_fit(t_peak_cant_fit, fwhm_cant_fit, ampl_cant_fit, impulsive_index_cant_fit, save_as=save_as_cant_fit)

            if len(fwhm_cant_find) > 0:
                if is_half == 'yes':
                    #save_as_cant_find = 'All_Amplitudes_' + str(np.round(set_lc_cadence, 1)) + 'min/fit_stat_hist_cant_find.pdf'
                    save_as_cant_find = 'All_Amplitudes_' + str(int(set_lc_cadence)) + '+' + str(np.round(remainder,2)) + 'min/fit_stat_hist_cant_find.pdf'
                else:
                    save_as_cant_find = 'All_Amplitudes_' + str(int(set_lc_cadence)) + 'min/fit_stat_hist_cant_find.pdf'
                plot_hist_cant_find(t_peak_cant_find, fwhm_cant_find, ampl_cant_find, impulsive_index_cant_find, save_as=save_as_cant_find)



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

lc_cadence = [30,20,10,5,2,1,0.5,10./60.] # [10./60., 2, 30] # [2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.25,12.5,12.75,13,13.25,13.5,13.75,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20] #[10,10.5,11,11.5,12,12.25,12.5,12.75,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20]
for this_cad in range(len(lc_cadence)):
    if lc_cadence[this_cad] <= 10:
        # fit_statistics5(cadence, downsample=True, set_lc_cadence=lc_cadence[this_cad], set_max_fwhm = 1.25*lc_cadence[this_cad], hist_inclusion=15., bin_slice_factor=3., n_reps=30000)
        fit_statistics7(cadence, template=2, downsample=True, set_lc_cadence=lc_cadence[this_cad], set_max_fwhm = 15., hist_inclusion=25., bin_slice_factor=1, n_reps=15000)
    if lc_cadence[this_cad] > 10:
        # fit_statistics5(cadence, downsample=True, set_lc_cadence=lc_cadence[this_cad], set_max_fwhm = 1.25*lc_cadence[this_cad], hist_inclusion=5., bin_slice_factor=3., n_reps=30000)
        fit_statistics7(cadence, template=2, downsample=True, set_lc_cadence=lc_cadence[this_cad], set_max_fwhm = 15., hist_inclusion=25., bin_slice_factor=1, n_reps=15000)

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









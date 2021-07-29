#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import matplotlib as mpl

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
from astropy.convolution import Gaussian2DKernel, convolve
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
import scipy.stats as st

import sys, os, shutil
import colour # https://www.colour-science.org/installation-guide/



def clean_directory(save_directory):
        save_dir = '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_directory + '/'
        if os.path.exists(save_dir):
            for files in os.listdir(save_dir):
                path = os.path.join(save_dir, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)

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
    # colors1 = [(102/255, 0/255, 204/255), (255/255, 128/255, 0/255), (0/255, 153/255, 153/255)]
    # colors2 = [(204/255, 0/255, 102/255), (51/255, 51/255, 204/255), (153/255, 204/255, 0/255)]
    # colors3 = [(128/255, 0/255, 64/255),(51/255, 51/255, 204/255),(0/255, 255/255, 153/255)]
    # colors4 = [(255/255, 255/255, 255/255),(0/255, 255/255, 204/255),(0/255, 153/255, 204/255),(0/255, 153/255, 255/255),(102/255, 0/255, 204/255)]
    # colors5 = [(255/255, 255/255, 255/255),(153/255, 255/255, 153/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255)]
    # colors6 = [(255/255, 255/255, 255/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255),(0/255, 0/255, 77/255)]
    colors7 = [(255 / 255, 255 / 255, 255 / 255), (255 / 255, 204 / 255, 0 / 255), (255 / 255, 0 / 255, 102 / 255), (134 / 255, 0 / 255, 179 / 255), (0 / 255, 0 / 255, 77 / 255)]
    # colors8 = [(255 / 255, 255 / 255, 255 / 255), (255 / 255, 0 / 255, 102 / 255), (153 / 255, 0 / 255, 204 / 255), (0 / 255, 0 / 255, 77 / 255)]
    # colors9 = [(255/255, 255/255, 255/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255)]
    colors10 = [(0 / 255, 255 / 255, 204 / 255), (255 / 255, 204 / 255, 0 / 255), (255 / 255, 0 / 255, 102 / 255), (134 / 255, 0 / 255, 179 / 255), (0 / 255, 0 / 255, 77 / 255)]
    colors11 = [(0 / 255, 102 / 255, 153 / 255), (255 / 255, 204 / 255, 0 / 255), (255 / 255, 0 / 255, 102 / 255), (134 / 255, 0 / 255, 179 / 255), (0 / 255, 0 / 255, 77 / 255)]


    # position = [0, 0.5, 1]
    # position2 = [0, 0.25, 0.5, 0.75, 1]
    position2_2 = [0, 0.25, 0.5, 0.75, 1]
    # position3 = [0, 1./3., 2./3., 1]
    mycolormap = make_cmap(colors11, position=position2_2)

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
    # np.random.seed()

    # print('\nGenerating Synthetic Flares...\n')
    #std = 0.001 * (1. / 50)
    std = 0.0025

    x_synth = np.arange(0, 3, cadence)  # cadence in days

    t_max = 0.5
    t_max = t_max + np.random.uniform(-0.05, 0.05, 1)
    fwhm = np.random.uniform(0.5, max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # days
    amplitude = np.random.uniform(0.05,1.0,1)
    # amplitude = np.random.uniform(3, 1000, 1) * stddev

    # y_synth = np.random.normal(0, std, len(x_synth))
    # y_synth_noscatter = np.zeros_like(x_synth)
    #
    # flare_synth_a_noscatter = aflare1(x_synth, t_max, fwhm, amplitude)
    # flare_synth_a = flare_synth_a_noscatter + np.random.normal(0, std, len(x_synth))
    #
    # y_synth_noscatter += flare_synth_a_noscatter
    # y_synth += flare_synth_a

    y_synth = aflare1(x_synth, t_max, fwhm, amplitude)

    y_synth_scatter = []
    for f in range(len(y_synth)):
        y_synth_scatter.append(np.random.normal(y_synth[f], std, 1)[0])
    y_synth_scatter = np.array(y_synth_scatter)

    # create 'uncertainties' on the lightcurve
    y_synth_err = np.random.normal(std + 0.10 * std, 0.10 * std, len(y_synth))

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
    return x_synth, y_synth_scatter, y_synth_err, y_synth, flare_properties
def create_single_synthetic_jflare1(cadence, max_fwhm=10):
    # np.random.seed()

    #std = 0.001 * (1. / 50)
    std = 0.0025

    window = 1. #days
    x_synth = np.arange(0, window, cadence)  # cadence in days

    gauss_tpeak = 0.5*window # peak time
    #gauss_tpeak = gauss_tpeak + np.random.uniform(-0.05, 0.05, 1)[0]

    gauss_ampl_j = 1.0
    decay_ampl_j = np.random.uniform(0.6,1.0,1)
    # decay_fwhm_j = np.random.uniform(1./60., max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    # gauss_fwhm_j = decay_fwhm_j

    # gauss_fwhm_j = np.random.uniform(1. / 60., max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    # decay_fwhm_j = np.random.uniform((1. / 3.) * gauss_fwhm_j, gauss_fwhm_j, 1)

    decay_fwhm_j = np.random.uniform(1. / 60., max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    gauss_fwhm_j = np.random.uniform((1. / 3.) * decay_fwhm_j, decay_fwhm_j, 1)

    # compute the convolution
    g_profile, d_profile, synth_flare, decay_start = jflare1(x_synth, gauss_tpeak, gauss_fwhm_j, decay_fwhm_j, gauss_ampl_j, decay_ampl_j)

    # normalize the flare
    synth_flare/=np.max(synth_flare)

    # add scatter to the flare
    #synth_flare_scatter = synth_flare + np.random.normal(std, 0.10*std, len(synth_flare))

    synth_flare_scatter = []
    for f in range(len(synth_flare)):
        synth_flare_scatter.append(np.random.normal(synth_flare[f], std, 1)[0])

    # create 'uncertainties' on the lightcurve
    synth_flare_err = np.random.normal(std+0.10*std, 0.10*std, len(synth_flare))

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

    return x_synth, synth_flare_scatter, synth_flare_err, synth_flare, flare_properties
def create_single_synthetic_jflare1_equation(cadence, max_fwhm=10):
    # np.random.seed()

    #std = 0.001 * (1. / 50)
    std = 0.0025

    window = 1. #days
    x_synth = np.arange(0, window, cadence)  # cadence in days

    gauss_tpeak = 0.5*window # peak time
    #gauss_tpeak = gauss_tpeak + np.random.uniform(-0.05, 0.05, 1)[0]

    gauss_ampl_j = 1.0
    decay_ampl_j = np.random.uniform(0.6,1.0,1)
    # decay_fwhm_j = np.random.uniform(1./60., max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    # gauss_fwhm_j = decay_fwhm_j

    # gauss_fwhm_j = np.random.uniform(1. / 60., max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    # decay_fwhm_j = np.random.uniform((1./3.)*gauss_fwhm_j, gauss_fwhm_j, 1)

    decay_fwhm_j = np.random.uniform(1. / 60., max_fwhm, 1) * (1. / 60.) * (1. / 24.)  # minutes to days
    gauss_fwhm_j = np.random.uniform((1. / 3.) * decay_fwhm_j, decay_fwhm_j, 1)

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
    #synth_flare_scatter = synth_flare + np.random.normal(-std, std, len(synth_flare))

    synth_flare_scatter = []
    for f in range(len(synth_flare)):
        synth_flare_scatter.append(np.random.normal(synth_flare[f], std, 1)[0])

    # create 'uncertainties' on the lightcurve
    synth_flare_err = np.random.normal(std + 0.10 * std, 0.10 * std, len(synth_flare))

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

    return x_synth, synth_flare_scatter, synth_flare_err, synth_flare, flare_properties


# @jit(nopython = True)
def test_single_synthetic_jflare1(cadence, max_fwhm=10):
    # np.random.seed()

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
    # np.random.seed()

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

def plot_test_fit(x_flare,y_flare,y_flare_err,x_fit,y_fit,y_true,flare_id_time,flare_id_flux,x_template,y_template,eq_dur,flare_energy,eq_dur_true,flare_energy_true,save_as):
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
    ax1.set_title('Equivalent Duration = ' + str(np.round(eq_dur, 2))  + ' sec' + '\nTrue Equivalent Duration = ' + str(np.round(eq_dur_true, 2)) + ' sec', pad=10, fontsize=font_size, style='normal', family='sans-serif')
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    ax1.plot(x_fit, y_true, c='#000000', lw=0.5, label='Dav Flare w/ Template Params')
    ax1.plot(x_fit, y_fit, c='blue', lw=2, label='Flare Fit')
    if len(flare_id_time) > 0:
        where_under = np.where((x_fit >= np.min(flare_id_time)) & (x_fit <= np.max(flare_id_time)))[0]
        ax1.fill_between(x_fit[where_under], y_fit[where_under], y2=np.zeros_like(y_fit[where_under]), color='blue', alpha=0.25)
    else:
        ax1.fill_between(x_fit, y_fit, y2=np.zeros_like(y_fit), color='#006699', alpha=0.15)
    ax1.scatter(x_flare, y_flare, c='red', s=np.pi*(3)**2, label='Test Flare')
    ax1.errorbar(x_flare, y_flare, yerr=y_flare_err, fmt='None', ecolor='red', elinewidth=0.5, capsize=0.5, capthick=0.5)
    ax1.scatter(flare_id_time, flare_id_flux, c='#00cc00', s=np.pi*(2)**2, label='Identified Flare')
    ax1.plot(x_template,y_template, c='orange', lw=0.75, label='True Flare Template')
    ax1.set_ylim([1.0 - max_gap, 1 + max_amp + max_gap])
    ax1.legend(fontsize=font_size, loc='upper right')
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
def plot_cant_find(x_flare,y_flare,y_flare_err,save_as):
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
    ax1.errorbar(x_flare, y_flare, yerr=y_flare_err, fmt='None', ecolor='red', elinewidth=0.5, capsize=0.5, capthick=0.5)
    #ax1.set_ylim([max_gap, max_amp + max_gap])
    ax1.legend(fontsize=font_size, loc='upper right')
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
def plot_test_fit_check(x_flare,y_flare,y_flare_err,x_fit,y_fit,y_true,flare_id_time,flare_id_flux,x_template,y_template,eq_dur,flare_energy,eq_dur_true,flare_energy_true,save_as):
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
    ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    ax1.plot(x_fit, y_true, c='#000000', lw=0.5, label='Dav Flare w/ Template Params')
    ax1.plot(x_fit, y_fit, c='blue', lw=2, label='Flare Fit')
    if len(flare_id_time) > 0:
        where_under = np.where((x_fit >= np.min(flare_id_time)) & (x_fit <= np.max(flare_id_time)))[0]
        ax1.fill_between(x_fit[where_under], y_fit[where_under], y2=np.zeros_like(y_fit[where_under]), color='blue', alpha=0.25)
    else:
        ax1.fill_between(x_fit, y_fit, y2=np.zeros_like(y_fit), color='#006699', alpha=0.15)
    ax1.scatter(x_flare, y_flare, c='red', s=np.pi*(3)**2, label='Test Flare')
    ax1.errorbar(x_flare, y_flare, yerr=y_flare_err, fmt='None', ecolor='red', elinewidth=0.5, capsize=0.5, capthick=0.5)
    ax1.scatter(flare_id_time, flare_id_flux, c='#00cc00', s=np.pi*(2)**2, label='Identified Flare')
    ax1.plot(x_template,y_template, c='orange', lw=0.75, label='True Flare Template')
    ax1.set_ylim([1.0 - max_gap, 1 + max_amp + max_gap])
    ax1.legend(fontsize=font_size, loc='upper right')
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()

def quick_test_plot(any_x,any_y,label_x,label_y, y_axis_label, x_axis_range, y_axis_range, save_as):
    font_size = 'large'

    #mycolormap = choose_cmap()
    # colors = mycolormap(range(len(any_x)))
    # colors = ['#b30047', '#ff3300', '#00cc99', '#3366ff']
    if len(any_x) > 1:
        from matplotlib import cm
        evenly_spaced_interval = np.linspace(0, 1, len(any_x))
        colors = [cm.rainbow(smoosh) for smoosh in evenly_spaced_interval]
    if len(any_x) == 1:
        colors = ['#000000']

    #import pdb; pdb.set_trace()

    fig = plt.figure(1, figsize=(7,5.5), facecolor="#ffffff")  # , dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(x_axis_range)
    ax.set_ylim(y_axis_range)
    ax.set_xlabel(label_x, fontsize=font_size, style='normal', family='sans-serif')
    ax.set_ylabel(y_axis_label, fontsize=font_size, style='normal', family='sans-serif')
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    for v in range(len(any_x)):
        #ax = fig.add_subplot(1, len(any_x), v+1)

        ax.plot(any_x[v], any_y[v], c=colors[v], lw=1, label=label_y[v])

        #ax.fill_between(x_fit, y_fit, y2=np.zeros_like(y_fit), color='#006699', alpha=0.15)
        #ax.scatter(x_flare, y_flare, c='red', s=np.pi*(3)**2, label='Test Flare')
        #ax.errorbar(x_flare, y_flare, yerr=y_flare_err, fmt='None', ecolor='red', elinewidth=0.5, capsize=0.5, capthick=0.5)
        #a.scatter(flare_id_time, flare_id_flux, c='#00cc00', s=np.pi*(2)**2, label='Identified Flare')
        #ax1.plot(x_template,y_template, c='orange', lw=0.75, label='True Flare Template')
        #ax.set_ylim([1.0 - max_gap, 1 + max_amp + max_gap])
    ax.legend(fontsize=font_size, loc='upper right')
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()


def check_fitpars(popts,true_properties):

    weird_props = {"tpeak":0,
                   "fwhm":0,
                   "amplitude":0,
                   }
    # if ((popts[0] - true_properties["tpeak"]) / true_properties["tpeak"]) * 100:
    #     weird_props["tpeak"] = popts[0]
    #     print('')
    #     print('real tpeak: ' + str(true_properties["tpeak"]))
    #     print('fit tpeak: ' + str(popts[0]))
    #     print('')
    if ((popts[1] - true_properties["fwhm"]) / true_properties["fwhm"]) * 100 < -100:
        weird_props["fwhm"] = popts[1]
        print('')
        print('real fwhm: ' + str(true_properties["fwhm"]))
        print('fit fwhm: ' + str(popts[1]))
        print('')
    if ((np.abs(popts[2]) - true_properties["amplitude"]) / true_properties["amplitude"]) * 100 < -100:
        weird_props["amplitude"] = popts[2]
        print('')
        print('real amp: ' + str(true_properties["amplitude"]))
        print('fit amp: ' + str(popts[2]))
        print('')

    # if (weird_props["tpeak"] != 0) or (weird_props["fwhm"] != 0) or (weird_props["amplitude"] != 0):
    #     if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Cadences/weird_flare_fit_parameters/'):
    #         os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Cadences/weird_flare_fit_parameters/')

    return weird_props


def plot_stat_hist4(t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,flare_energy_opt,impulsiveness, hist_inclusion, bin_slice_factor, lc_cadence, save_as):
    hist_inclusion = 10000
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
    x_factor_ax1 = hist_inclusion * bin_slice_factor_ax1
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
    where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (x_factor_ax1 * bin_width)) & (
            bin_edges <= bin_edges[where_hist_max] + (x_factor_ax1 * bin_width)))[0]
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


def plot_hist_cant_fit(t_peak_cant_fit,fwhm_cant_fit,ampl_cant_fit,impulsiveness_cant_fit,cadence_cant_fit,save_as):

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
    dat5 = np.array(cadence_cant_fit)

    fig = plt.figure(1, figsize=(15, 4), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(151)

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


    ax2 = fig.add_subplot(152)

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


    ax3 = fig.add_subplot(153)

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


    ax4 = fig.add_subplot(154)

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



    ax5 = fig.add_subplot(155)

    y_hist, bin_edges = np.histogram(dat5, bins='auto')
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

    hist_dat5 = ax5.hist(dat5, color=impulsiveness_color, bins=bin_edges)  # , weights=np.ones(len(dat4)) / len(dat4))  # , edgecolor='#000000', linewidth=1.2)
    ax5.hist(dat5, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step')  # , weights=np.ones(len(dat4)) / len(dat4))
    # ax4.plot([0, 0], [0, np.max(hist_dat4[0]) * 10], '--', color='#000000', lw=1)

    # ax4.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax5.set_ylim([0, np.max(hist_dat5[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax5.set_xlabel("Cadence (min)", fontsize=font_size, style='normal', family='sans-serif')
    # ax4.set_ylabel("Fraction of Total", fontsize=font_size, style='normal', family='sans-serif')
    ax5.set_title("Cadence", fontsize=font_size, style='normal', family='sans-serif')
    ax5.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)



    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
    #import pdb;pdb.set_trace()
def plot_hist_cant_find(t_peak_cant_find,fwhm_cant_find,ampl_cant_find,impulsiveness_cant_find,cadence_cant_find,save_as):

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
    dat5 = np.array(cadence_cant_find)

    fig = plt.figure(1, figsize=(15, 4), facecolor="#ffffff")  # , dpi=300)
    ax1 = fig.add_subplot(151)

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


    ax2 = fig.add_subplot(152)

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


    ax3 = fig.add_subplot(153)

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


    ax4 = fig.add_subplot(154)

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



    ax5 = fig.add_subplot(155)

    y_hist, bin_edges = np.histogram(dat5, bins='auto')
    hist_dat5 = ax5.hist(dat5, color=impulsiveness_color, bins=bin_edges) #, weights=np.ones(len(dat4)) / len(dat4))  # , edgecolor='#000000', linewidth=1.2)
    ax5.hist(dat5, color='#000000', bins=bin_edges, linewidth=1.2, histtype='step') #, weights=np.ones(len(dat4)) / len(dat4))
    #ax4.plot([0, 0], [0, np.max(hist_dat4[0]) * 10], '--', color='#000000', lw=1)

    # ax4.set_xlim(np.min(bin_edges), x_factor * bin_width)
    ax5.set_ylim([0, np.max(hist_dat5[0]) * 1.10])
    # plt.legend(fontsize=10, loc="upper left")
    ax5.set_xlabel("Cadence (min)", fontsize=font_size, style='normal', family='sans-serif')
    #ax4.set_ylabel("Fraction of Total", fontsize=font_size, style='normal', family='sans-serif')
    ax5.set_title("Cadence", fontsize=font_size, style='normal',family='sans-serif')
    ax5.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)



    plt.tight_layout()
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()
    #plt.show()
    #import pdb;pdb.set_trace()


def sort_property7(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, plot_type, save_as, stddev, lc_cadence, max_fwhm, hist_inclusion, bin_slice_factor, previous_vmax, property_to_sort='fwhm'):

    opt_list = [np.array(t_peak_opt) * 24 * 60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    # true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time (min)', '% Difference from True FWHM',
                    '% Difference from True Amplitude', '% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    #bin_slice_factor = 1.

    current_vmax = previous_vmax
    redo = False

    if property_to_sort == 'fwhm':

        print('Plotting FWHM sort...')

        # fwhm_min = (1./60.) * (1. / 60.) * (1. / 24.)
        fwhm_min = 0.99 * np.min(fwhm_true)
        # fwhm_max = max_fwhm * (1. / 60.) * (1. / 24.)
        fwhm_max = 1.01 * np.max(fwhm_true)
        grid_spacing = (fwhm_max - fwhm_min)/2000
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
                xlim_mult = hist_inclusion * bin_slice_factor
            else:
                bin_slice_factor = 1
                xlim_mult = hist_inclusion * bin_slice_factor
            if (prop == 1) or (prop == 2) and (lc_cadence < 10):
                bin_slice_factor = 3
                xlim_mult = hist_inclusion * bin_slice_factor

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

            if Z.max() > current_vmax:
                print('Z.max(): ' + str(Z.max()))
                print('current_vmax: ' + str(current_vmax))
                current_vmax = Z.max()
                redo = True

            mycolormap = choose_cmap()
            p = ax1.pcolor(X, Y, Z, cmap=mycolormap, edgecolors='face', vmin=Z.min(), vmax=current_vmax, rasterized=True) # cm.BuPu
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
                #ax1.plot([0, 0], [np.min(fwhm_grid * 24 * 60), np.max(fwhm_grid * 24 * 60)], color='#ff0066', lw=1)
                ax1.plot([0, 0], [0, np.max(fwhm_grid * 24 * 60)], color='#ff0066', lw=1)
                ax1.plot([np.min(bin_edges), np.max(bin_edges)], [lc_cadence, lc_cadence], color='#000000', alpha=0.2, lw=0.5)
                if plot_type == 'cumulative':
                    ax2.plot([0, 0], [0, 1.0], color='#ff0066', lw=1)
                    ax2.set_ylim([0, 1.0])
                if plot_type == 'sum':
                    ax2.plot([0, 0], [0, np.max(col_hist) * 1.10], color='#ff0066', lw=1)
                    ax2.set_ylim([0, np.max(col_hist) * 1.10])

            #ax3.set_ylim([np.min(fwhm_grid), np.max(fwhm_grid)])
            ax3.set_ylim([0, np.max(fwhm_grid)])
            ax3.set_xlim([0, np.max(row_hist) * 1.10])
            #ax1.set_ylim([np.min(fwhm_grid[0:-1] * 24 * 60), np.max(fwhm_grid[0:-1] * 24 * 60)])
            ax1.set_ylim([0, np.max(fwhm_grid[0:-1] * 24 * 60)])

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

            if (redo == True) and (prop == 5):

                plt.close()

                print('Adjusting Colorbars...')

                fig = plt.figure(figsize=(10, 6), facecolor='#ffffff')  # , dpi=300)

                for prop2 in range(len(opt_list)):

                    print(prop2)

                    if prop2 <= 2:
                        plot_bott = 0.57
                        plot_top = 0.98
                        plot_left = 0.05 + 0.27 * prop2 + 0.05 * prop2
                        plot_right = 0.05 + 0.27 * (prop2 + 1) + 0.05 * prop2
                    if prop2 > 2:
                        plot_bott = 0.06
                        plot_top = 0.48
                        plot_left = 0.05 + 0.27 * (prop2 - 3) + 0.05 * (prop2 - 3)
                        plot_right = 0.05 + 0.27 * ((prop2 - 3) + 1) + 0.05 * (prop2 - 3)
                    gs1 = fig.add_gridspec(nrows=6, ncols=6, left=plot_left, right=plot_right, top=plot_top,
                                           bottom=plot_bott,
                                           wspace=0, hspace=0)
                    ax1 = fig.add_subplot(gs1[2:6, 0:4])
                    ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[])  # , sharey=ax1)
                    ax3 = fig.add_subplot(gs1[2:6, 4:6], yticklabels=[])  # , sharex=ax1)

                    y_hist, bin_edges = np.histogram(opt_list[prop2], bins='auto')

                    if prop2 == 0:
                        bin_slice_factor = 3
                        xlim_mult = hist_inclusion * bin_slice_factor
                    else:
                        bin_slice_factor = 1
                        xlim_mult = hist_inclusion * bin_slice_factor
                    if (prop2 == 1) or (prop2 == 2) and (lc_cadence < 10):
                        bin_slice_factor = 3
                        xlim_mult = hist_inclusion * bin_slice_factor

                    if (prop2 != 5) and (bin_slice_factor > 1):
                        axis_spacing = np.arange(0, len(bin_edges), 1)
                        if bin_slice_factor == 3:
                            new_axis_spacing = np.arange(np.min(axis_spacing),
                                                         np.max(axis_spacing) + 1. / bin_slice_factor,
                                                         1. / bin_slice_factor)[0:-1]
                        else:
                            new_axis_spacing = np.arange(np.min(axis_spacing),
                                                         np.max(axis_spacing) + 1. / bin_slice_factor,
                                                         1. / bin_slice_factor)
                        bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
                    # bin_edges = np.interp(np.arange())

                    # import pdb; pdb.set_trace()

                    Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
                    # print('creating first grid')
                    for slot_index in range(len(fwhm_slots)):
                        # print('bin_edges before fill' + str(bin_edges))
                        # y_hist, bin_edges = np.histogram(np.array(opt_list[prop2])[fwhm_slots[slot_index]], bins=bin_edges)
                        y_hist, bin_edges = np.histogram(np.array(opt_list[prop2])[fwhm_slots[slot_index]],
                                                         bins=bin_edges)
                        # print('bin_edges after fill' + str(bin_edges))
                        Z[slot_index, :] = y_hist
                        # print('y_hist: ' + str(y_hist))

                    row_hist = sum_rows(Z)
                    # print('summed rows')
                    col_hist1 = sum_cols(Z)
                    # print('summed cols')
                    if plot_type == 'cumulative':
                        cum_hist1 = cum_cols(Z)

                    if prop2 != 5:
                        bin_width = np.diff(bin_edges)[0]

                        where_hist_max = np.where(col_hist1 == np.max(col_hist1))[0]
                        if len(where_hist_max) > 0:
                            where_hist_max = int(np.mean(where_hist_max))
                        where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (xlim_mult * bin_width)) & (
                                    bin_edges <= bin_edges[where_hist_max] + (xlim_mult * bin_width)))[0]


                        y_hist, bin_edges = np.histogram(opt_list[prop2], bins=bin_edges[where_within])
                    if prop2 == 5:
                        where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
                        y_hist, bin_edges = np.histogram(opt_list[prop2], bins=bin_edges[where_within])

                    if len(where_within) > 0:

                        col_hist = np.array(col_hist1)[where_within[:-1]]
                        if plot_type == 'cumulative':
                            cum_hist = np.array(cum_hist1)[where_within[:-1]]

                        Z = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
                        # print('creating second grid')
                        for slot_index in range(len(fwhm_slots)):
                            # print('opt_list[prop]: ' + str(opt_list[prop]))
                            # print('opt_list[prop][fwhm_slots[slot_index]]: ' + str(np.array(opt_list[prop])[fwhm_slots[slot_index]]))
                            # print('bin_edges before: ' + str(bin_edges))
                            if len(fwhm_slots[slot_index]) > 0:
                                y_hist, bin_edges = np.histogram(np.array(opt_list[prop2])[fwhm_slots[slot_index]],
                                                                 bins=bin_edges)
                                # print('bin_edges after: ' + str(bin_edges))
                                Z[slot_index, :] = y_hist

                        # print(Z[-1, :])
                    # import pdb; pdb.set_trace()
                    else:
                        print('')
                        print('prop2: ' + str(prop2))
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

                    # print('Z: ' + str(Z))
                    # print('len y_hist: ' + str(len(y_hist)))
                    # import pdb; pdb.set_trace()

                    # ax = fig.add_subplot(2,3,prop+1)
                    # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
                    # ax.set_title('Original', fontsize=font_size, style=font_style, family=font_family)
                    if (prop2 == 0) or (prop2 == 3):
                        ax1.set_ylabel('True FWHM (min)', fontsize=font_size, style=font_style, family=font_family)
                    ax1.set_xlabel(x_label_list[prop2], fontsize=font_size, style=font_style, family=font_family)

                    X, Y = np.meshgrid(bin_edges, (fwhm_grid * 24 * 60))

                    mycolormap = choose_cmap()

                    p = ax1.pcolor(X, Y, Z, cmap=mycolormap, edgecolors='face', vmin=Z.min(), vmax=current_vmax,
                                   rasterized=True)  # cm.BuPu
                    # import pdb; pdb.set_trace()
                    # cbaxes = fig.add_axes([])
                    cb = fig.colorbar(p)  # , ticks=linspace(0,abs(Z).max(),10))
                    # if (prop == 2) or (prop == 5):
                    #     cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
                    cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
                    # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)

                    if plot_type == 'cumulative':
                        ax2.bar(np.array(bin_edges[:-1]), cum_hist, width=np.diff(bin_edges), color='None',
                                edgecolor="black", align="edge", rasterized=True)
                    if plot_type == 'sum':
                        # import pdb; pdb.set_trace()
                        ax2.bar(np.array(bin_edges[:-1]), col_hist, width=np.diff(bin_edges), color='None',
                                edgecolor="black", align="edge", rasterized=True)
                    ax3.barh(fwhm_grid[:-1], row_hist, height=np.diff(fwhm_grid), color='None', edgecolor="black",
                             align="edge", rasterized=True)

                    if prop2 != 5:
                        # ax1.plot([0, 0], [np.min(fwhm_grid * 24 * 60), np.max(fwhm_grid * 24 * 60)], color='#ff0066', lw=1)
                        ax1.plot([0, 0], [0, np.max(fwhm_grid * 24 * 60)], color='#ff0066', lw=1)
                        ax1.plot([np.min(bin_edges), np.max(bin_edges)], [lc_cadence, lc_cadence], color='#000000',
                                 alpha=0.2, lw=0.5)
                        if plot_type == 'cumulative':
                            ax2.plot([0, 0], [0, 1.0], color='#ff0066', lw=1)
                            ax2.set_ylim([0, 1.0])
                        if plot_type == 'sum':
                            ax2.plot([0, 0], [0, np.max(col_hist) * 1.10], color='#ff0066', lw=1)
                            ax2.set_ylim([0, np.max(col_hist) * 1.10])

                    # ax3.set_ylim([np.min(fwhm_grid), np.max(fwhm_grid)])
                    ax3.set_ylim([0, np.max(fwhm_grid)])
                    ax3.set_xlim([0, np.max(row_hist) * 1.10])
                    # ax1.set_ylim([np.min(fwhm_grid[0:-1] * 24 * 60), np.max(fwhm_grid[0:-1] * 24 * 60)])
                    ax1.set_ylim([0, np.max(fwhm_grid[0:-1] * 24 * 60)])

                    if (prop2 == 0) or (prop2 == 1):
                        ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                        ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
                    if (prop2 == 2) or (prop2 == 3) or (prop2 == 4):
                        ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                        ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
                    if prop2 == 5:
                        ax1.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
                        ax2.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))

                    if prop2 == 2:
                        # ax2.set_title(str(np.round(lc_cadence,2)) + ' min Cadence',fontsize=font_size, style=font_style, family=font_family)
                        props = dict(boxstyle=None, facecolor=None, alpha=0.5)
                        if lc_cadence >= 1:
                            textstr = '\n'.join(('Cadence:', str(np.round(lc_cadence, 2)), 'min'))
                        if lc_cadence < 1:
                            textstr = '\n'.join(('Cadence:', str(np.round(lc_cadence * 60, 2)), 'sec'))
                        ax1.text(1.30, 1.25, textstr, transform=ax1.transAxes, fontsize='medium', style=font_style,
                                 family=font_family, weight='heavy', verticalalignment='center',
                                 horizontalalignment='center')  # ,bbox=props)

                    ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                                    color='#000000',
                                    length=0)
                    ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                                    color='#000000',
                                    length=0)
                    ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                                    color='#000000',
                                    length=0)


        print('Attempting To Save...')
        # plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300, rasterized=True)
        plt.close()
        # plt.show()

        # import pdb; pdb.set_trace()

    return current_vmax

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
def sort_property8(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true, impulsiveness, plot_type, save_as, stddev, lc_cadence, max_fwhm, hist_inclusion, bin_slice_factor_in, rep_in, previous_vmax, property_to_sort='fwhm'):
    opt_list = [np.array(t_peak_opt) * 24 * 60, fwhm_opt, ampl_opt, eq_duration_opt, flare_energy_opt, impulsiveness]
    # true_list = [t_peak_true, fwhm_true, ampl_true, eq_duration_true, flare_energy_true]
    x_label_list = ['Difference From True Peak Time / Cadence', '% Difference from True FWHM',
                    '% Difference from True Amplitude', '% Difference from True Equivalent Duration',
                    '% Difference from True Flare Energy', 'Impulsive Index']

    #bin_slice_factor = 1.

    current_vmax = previous_vmax
    redo = False

    if property_to_sort == 'fwhm':

        print('Plotting 2D Histogram...')

        fwhm_min = 0. # np.min(fwhm_true)
        fwhm_max = np.max(fwhm_true)
        fwhm_max = 20.
        grid_spacing = (fwhm_max - fwhm_min)/1000
        fwhm_grid = np.arange(fwhm_min,fwhm_max+grid_spacing,grid_spacing)

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
        xlim_mult = hist_inclusion * bin_slice_factor_in

        fig = plt.figure(figsize=(10, 9), facecolor='#ffffff')  # , dpi=300)
        for prop in range(len(opt_list)):

            print(prop)

            if prop <= 2:
                plot_bott = 0.57
                plot_top = 0.98
                plot_left = 0.06 + 0.26 * prop + 0.06 * prop
                plot_right = 0.06 + 0.26 * (prop + 1) + 0.06 * prop
            if prop > 2:
                plot_bott = 0.06
                plot_top = 0.48
                plot_left = 0.06 + 0.26 * (prop - 3) + 0.06 * (prop - 3)
                plot_right = 0.06 + 0.26 * ((prop - 3) + 1) + 0.06 * (prop - 3)
            gs1 = fig.add_gridspec(nrows=12, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs1[2:12, 0:4])
            ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[])  # , sharey=ax1)
            ax3 = fig.add_subplot(gs1[2:12, 4:6], yticklabels=[])  # , sharex=ax1)

            y_hist, bin_edges = np.histogram(opt_list[prop], bins='auto')

            if (prop > 2):
                bin_slice_factor = 1
                xlim_mult = hist_inclusion * bin_slice_factor
            else:
                bin_slice_factor = bin_slice_factor_in
                xlim_mult = hist_inclusion * bin_slice_factor

            if (prop != 5) and (bin_slice_factor > 1):
                axis_spacing = np.arange(0,len(bin_edges),1)
                if bin_slice_factor == 3:
                    new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)[0:-1]
                else:
                    new_axis_spacing = np.arange(np.min(axis_spacing),np.max(axis_spacing)+1./bin_slice_factor,1./bin_slice_factor)
                bin_edges = np.interp(new_axis_spacing,axis_spacing,bin_edges)

            #import pdb; pdb.set_trace()

            print('made it here: 1')

            Z1 = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
            print(np.shape(Z1))
            for slot_index in range(len(fwhm_slots)):
                y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                Z1[slot_index, :] = y_hist

            print('made it here: 2')


            row_hist = sum_rows(Z1)
            #print('summed rows')
            col_hist1 = sum_cols(Z1)
            #print('summed cols')
            if plot_type == 'cumulative':
                cum_hist1 = cum_cols(Z1)

            if prop != 5:
                bin_width = np.diff(bin_edges)[0]

                where_hist_max = np.where(col_hist1 == np.max(col_hist1))[0]
                if len(where_hist_max) > 0:
                    where_hist_max = int(np.mean(where_hist_max))

                if prop > 2:
                    where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (xlim_mult * bin_width)) & (bin_edges <= bin_edges[where_hist_max] + (xlim_mult * bin_width)))[0]

                if prop == 0:
                    if np.max(bin_edges) < 2:
                        where_within = np.where((bin_edges >= -1.2) & (bin_edges <= np.max(bin_edges)))[0]
                    else:
                        where_within = np.where((bin_edges >= -1.2) & (bin_edges <= 2))[0]

                if prop == 1:
                    # where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (xlim_mult * bin_width)) & (bin_edges <= -10))[0]
                    # where_within = np.where((bin_edges >= -70) & (bin_edges <= -10))[0]
                    if np.max(bin_edges) < 5:
                        where_within = np.where((bin_edges >= np.min(bin_edges)) & (bin_edges <= np.max(bin_edges)))[0]
                    else:
                        where_within = np.where((bin_edges >= np.min(bin_edges)) & (bin_edges <= 5))[0]

                if prop == 2:
                    # where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (xlim_mult * bin_width)) & (bin_edges <= -10))[0]
                    # where_within = np.where((bin_edges >= -70) & (bin_edges <= -10))[0]
                    where_within = np.where((bin_edges >= np.min(bin_edges)) & (bin_edges <= bin_edges[where_hist_max] + (xlim_mult * bin_width)))[0]
                # import pdb; pdb.set_trace()

                print('made it here: 2.25')
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])
                print('made it here: 2.75')
            if prop == 5:
                where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
                y_hist, bin_edges = np.histogram(opt_list[prop], bins=bin_edges[where_within])

            print('made it here: 3')

            if len(where_within) > 0:

                col_hist = np.array(col_hist1)[where_within[:-1]]
                if plot_type == 'cumulative':
                    cum_hist = np.array(cum_hist1)[where_within[:-1]]

                Z2 = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
                for slot_index in range(len(fwhm_slots)):

                    if len(fwhm_slots[slot_index]) > 0:
                        y_hist, bin_edges = np.histogram(np.array(opt_list[prop])[fwhm_slots[slot_index]], bins=bin_edges)
                        Z2[slot_index, :] = y_hist

                print('made it here: 4')

            # import pdb; pdb.set_trace()
            else:
                print('')
                print('prop: ' + str(prop))
                print('poopie')
                print('')

                plt.close()
                break

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
                ax1.set_ylabel('True FWHM / Cadence', fontsize=font_size, style=font_style, family=font_family)
            ax1.set_xlabel(x_label_list[prop], fontsize=font_size, style=font_style, family=font_family)

            X, Y = np.meshgrid(bin_edges, fwhm_grid)

            if Z2.max() > current_vmax:
                print('Z.max(): ' + str(Z2.max()))
                print('current_vmax: ' + str(current_vmax))
                current_vmax = Z2.max()
                redo = True
            print('made it here: 4.25')
            mycolormap = choose_cmap()
            print('made it here: 4.75')
            if current_vmax > 1:
                bounds = np.arange(0, current_vmax + 1, 1)
                cmap = plt.get_cmap(mycolormap, current_vmax+1)
                norm = mpl.colors.BoundaryNorm(bounds, mycolormap.N)
                p = ax1.pcolor(X, Y, Z2, cmap=cmap, norm=norm, edgecolors='face', vmin=0,
                               vmax=current_vmax, rasterized=True) # cm.BuPu
                #import pdb; pdb.set_trace()
                # cbaxes = fig.add_axes([])
                cb = fig.colorbar(p, cmap=cmap, norm=norm, format='%1i')
                if current_vmax >= 10:
                    tick_locs = (np.arange(0, current_vmax, int(current_vmax / 10)) + 0.5)
                    cb.set_ticks(tick_locs)
                    cb.set_ticklabels(np.arange(0, current_vmax, int(current_vmax / 10)))
                else:
                    tick_locs = (np.arange(current_vmax) + 0.5)
                    cb.set_ticks(tick_locs)
                    cb.set_ticklabels(np.arange(current_vmax))
            else:
                p = ax1.pcolor(X, Y, Z2, cmap=mycolormap, edgecolors='face', vmin=Z2.min(), vmax=current_vmax, rasterized=True)
                cb = fig.colorbar(p)

            # if (prop == 2) or (prop == 5):
            #     cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
            cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
            # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)


            print('made it here: 5')

            if plot_type == 'cumulative':
                ax2.bar(np.array(bin_edges[:-1]), cum_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge", rasterized=True)
            if plot_type == 'sum':
                #import pdb; pdb.set_trace()
                ax2.bar(np.array(bin_edges[:-1]), col_hist, width=np.diff(bin_edges), color='None', edgecolor="black", align="edge", rasterized=True)
            ax3.barh(fwhm_grid[:-1], row_hist, height=np.diff(fwhm_grid), color='None', edgecolor="black", align="edge", rasterized=True)

            if prop != 5:

                ax1.plot([0, 0], [0, np.max(fwhm_grid)], color='#ff0066', lw=1)

                if plot_type == 'cumulative':
                    ax2.plot([0, 0], [0, 1.0], color='#ff0066', lw=1)
                    ax2.set_ylim([0, 1.0])
                if plot_type == 'sum':
                    ax2.plot([0, 0], [0, np.max(col_hist) * 1.10], color='#ff0066', lw=1)
                    ax2.set_ylim([0, np.max(col_hist) * 1.10])

            #ax3.set_ylim([np.min(fwhm_grid), np.max(fwhm_grid)])
            ax3.set_ylim([0, np.max(fwhm_grid)])
            ax3.set_xlim([0, np.max(row_hist) * 1.10])
            #ax1.set_ylim([np.min(fwhm_grid[0:-1] * 24 * 60), np.max(fwhm_grid[0:-1] * 24 * 60)])
            ax1.set_ylim([0, np.max(fwhm_grid[0:-1])])

            if prop != 5:
                ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
            else:
                ax1.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
                ax2.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))

            if prop == 2:
                total_included = np.sum(row_hist)
                frac_found = np.float(total_included)/np.float(rep_in)
                textstr = '\n'.join(('Fraction of', 'Flares Found:', str(np.round(frac_found,2))))
                ax1.text(1.35, 1.12, textstr, transform=ax1.transAxes, fontsize='small', style=font_style,
                         family=font_family, weight='heavy', verticalalignment='center', horizontalalignment='center')

            ax1.ticklabel_format(axis='y', style='scientific')
            ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)
            ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)
            ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#000000',
                            length=0)

            print('made it here: 6')


            # -------------------------------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------------------------


            if (redo == True) and (prop == 5):

                plt.close()

                print('Adjusting Colorbars...')

                fig = plt.figure(figsize=(10, 9), facecolor='#ffffff')  # , dpi=300)

                for prop2 in range(len(opt_list)):

                    print(prop2)

                    if prop2 <= 2:
                        plot_bott = 0.57
                        plot_top = 0.98
                        plot_left = 0.06 + 0.26 * prop2 + 0.06 * prop2
                        plot_right = 0.06 + 0.26 * (prop2 + 1) + 0.06 * prop2
                    if prop2 > 2:
                        plot_bott = 0.06
                        plot_top = 0.48
                        plot_left = 0.06 + 0.26 * (prop2 - 3) + 0.06 * (prop2 - 3)
                        plot_right = 0.06 + 0.26 * ((prop2 - 3) + 1) + 0.06 * (prop2 - 3)
                    # gs1 = fig.add_gridspec(nrows=6, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott, wspace=0, hspace=0)
                    # ax1 = fig.add_subplot(gs1[2:6, 0:4])
                    # ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[])  # , sharey=ax1)
                    # ax3 = fig.add_subplot(gs1[2:6, 4:6], yticklabels=[])  # , sharex=ax1)
                    gs1 = fig.add_gridspec(nrows=12, ncols=6, left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott, wspace=0, hspace=0)
                    ax1 = fig.add_subplot(gs1[2:12, 0:4])
                    ax2 = fig.add_subplot(gs1[0:2, 0:4], xticklabels=[])  # , sharey=ax1)
                    ax3 = fig.add_subplot(gs1[2:12, 4:6], yticklabels=[])  # , sharex=ax1)

                    y_hist, bin_edges = np.histogram(opt_list[prop2], bins='auto')

                    # if (prop2 == 0) or (prop2 == 1):
                    #     bin_slice_factor = 3
                    #     xlim_mult = hist_inclusion * bin_slice_factor
                    # else:
                    #     if prop2 == 2:
                    #         bin_slice_factor = 2
                    #         xlim_mult = hist_inclusion * bin_slice_factor
                    #     else:
                    #         bin_slice_factor = bin_slice_factor_in
                    #         xlim_mult = hist_inclusion * bin_slice_factor

                    if (prop2 > 2):
                        bin_slice_factor = 1
                        xlim_mult = hist_inclusion * bin_slice_factor
                    else:
                        bin_slice_factor = bin_slice_factor_in
                        xlim_mult = hist_inclusion * bin_slice_factor

                    print('made it here: 7')

                    if (prop2 != 5) and (bin_slice_factor > 1):
                        axis_spacing = np.arange(0, len(bin_edges), 1)
                        if bin_slice_factor == 3:
                            new_axis_spacing = np.arange(np.min(axis_spacing),
                                                         np.max(axis_spacing) + 1. / bin_slice_factor,
                                                         1. / bin_slice_factor)[0:-1]
                        else:
                            new_axis_spacing = np.arange(np.min(axis_spacing),
                                                         np.max(axis_spacing) + 1. / bin_slice_factor,
                                                         1. / bin_slice_factor)
                        bin_edges = np.interp(new_axis_spacing, axis_spacing, bin_edges)
                    # bin_edges = np.interp(np.arange())

                    # import pdb; pdb.set_trace()

                    Z1 = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
                    # print('creating first grid')
                    for slot_index in range(len(fwhm_slots)):
                        y_hist, bin_edges = np.histogram(np.array(opt_list[prop2])[fwhm_slots[slot_index]],bins=bin_edges)
                        Z1[slot_index, :] = y_hist
                        # print('y_hist: ' + str(y_hist))

                    print('made it here: 8')

                    row_hist = sum_rows(Z1)
                    # print('summed rows')
                    col_hist1 = sum_cols(Z1)
                    # print('summed cols')
                    if plot_type == 'cumulative':
                        cum_hist1 = cum_cols(Z1)

                    if prop2 != 5:
                        bin_width = np.diff(bin_edges)[0]

                        where_hist_max = np.where(col_hist1 == np.max(col_hist1))[0]
                        if len(where_hist_max) > 0:
                            where_hist_max = int(np.mean(where_hist_max))

                        if prop2 > 2:
                            where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (xlim_mult * bin_width)) & (bin_edges <= bin_edges[where_hist_max] + (xlim_mult * bin_width)))[0]

                        if prop2 == 0:
                            if np.max(bin_edges) < 2:
                                where_within = np.where((bin_edges >= -1.2) & (bin_edges <= np.max(bin_edges)))[0]
                            else:
                                where_within = np.where((bin_edges >= -1.2) & (bin_edges <= 2))[0]

                        if prop2 == 1:
                            if np.max(bin_edges) < 5:
                                where_within = np.where((bin_edges >= np.min(bin_edges)) & (bin_edges <= np.max(bin_edges)))[0]
                            else:
                                where_within = np.where((bin_edges >= np.min(bin_edges)) & (bin_edges <= 5))[0]

                        if prop2 == 2:
                            # where_within = np.where((bin_edges >= bin_edges[where_hist_max] - (xlim_mult * bin_width)) & (bin_edges <= -10))[0]
                            # where_within = np.where((bin_edges >= -70) & (bin_edges <= -10))[0]
                            where_within = np.where((bin_edges >= np.min(bin_edges)) & (bin_edges <= bin_edges[where_hist_max] + (xlim_mult * bin_width)))[0]


                        print('made it here: 8.25')
                        y_hist, bin_edges = np.histogram(opt_list[prop2], bins=bin_edges[where_within])
                        print('made it here: 8.75')
                    if prop2 == 5:
                        where_within = np.where(bin_edges <= impulse_max_lim_factor * np.max(bin_edges))[0]
                        y_hist, bin_edges = np.histogram(opt_list[prop2], bins=bin_edges[where_within])

                    if len(where_within) > 0:

                        col_hist = np.array(col_hist1)[where_within[:-1]]
                        if plot_type == 'cumulative':
                            cum_hist = np.array(cum_hist1)[where_within[:-1]]

                        Z2 = np.zeros((len(fwhm_grid) - 1, len(bin_edges) - 1))
                        for slot_index in range(len(fwhm_slots)):
                            if len(fwhm_slots[slot_index]) > 0:
                                y_hist, bin_edges = np.histogram(np.array(opt_list[prop2])[fwhm_slots[slot_index]],bins=bin_edges)
                                Z2[slot_index, :] = y_hist

                    print('made it here: 9')


                    if (prop2 == 0) or (prop2 == 3):
                        ax1.set_ylabel('True FWHM / Cadence', fontsize=font_size, style=font_style, family=font_family)
                    ax1.set_xlabel(x_label_list[prop2], fontsize=font_size, style=font_style, family=font_family)

                    X, Y = np.meshgrid(bin_edges, fwhm_grid)

                    mycolormap = choose_cmap()

                    if current_vmax > 1:
                        bounds = np.arange(0, current_vmax + 1, 1)
                        cmap = plt.get_cmap(mycolormap, current_vmax+1)
                        norm = mpl.colors.BoundaryNorm(bounds, mycolormap.N)
                        p = ax1.pcolor(X, Y, Z2, cmap=cmap, norm=norm, edgecolors='face',
                                       vmin=0, vmax=current_vmax, rasterized=True)  # cm.BuPu
                        # import pdb; pdb.set_trace()
                        # cbaxes = fig.add_axes([])
                        cb = fig.colorbar(p, cmap=cmap, norm=norm, format='%1i')  # , ticks=linspace(0,abs(Z).max(),10))
                        if current_vmax >= 10:
                            tick_locs = (np.arange(0,current_vmax,int(current_vmax/10)) + 0.5)
                            cb.set_ticks(tick_locs)
                            cb.set_ticklabels(np.arange(0, current_vmax, int(current_vmax / 10)))
                        else:
                            tick_locs = (np.arange(current_vmax) + 0.5)
                            cb.set_ticks(tick_locs)
                            cb.set_ticklabels(np.arange(current_vmax))
                    else:
                        p = ax1.pcolor(X, Y, Z2, cmap=mycolormap, edgecolors='face', vmin=Z2.min(), vmax=current_vmax, rasterized=True)
                        cb = fig.colorbar(p)
                    # if (prop == 2) or (prop == 5):
                    #     cb.set_label(label='Counts', fontsize=font_size, style=font_style, family=font_family)
                    cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)
                    # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)

                    if plot_type == 'cumulative':
                        ax2.bar(np.array(bin_edges[:-1]), cum_hist, width=np.diff(bin_edges), color='None',
                                edgecolor="black", align="edge", rasterized=True)
                    if plot_type == 'sum':
                        # import pdb; pdb.set_trace()
                        ax2.bar(np.array(bin_edges[:-1]), col_hist, width=np.diff(bin_edges), color='None',
                                edgecolor="black", align="edge", rasterized=True)
                    ax3.barh(fwhm_grid[:-1], row_hist, height=np.diff(fwhm_grid), color='None', edgecolor="black",
                             align="edge", rasterized=True)

                    print('made it here: 10')

                    if prop2 != 5:
                        # ax1.plot([0, 0], [np.min(fwhm_grid * 24 * 60), np.max(fwhm_grid * 24 * 60)], color='#ff0066', lw=1)
                        ax1.plot([0, 0], [0, np.max(fwhm_grid)], color='#ff0066', lw=1)

                        if plot_type == 'cumulative':
                            ax2.plot([0, 0], [0, 1.0], color='#ff0066', lw=1)
                            ax2.set_ylim([0, 1.0])
                        if plot_type == 'sum':
                            ax2.plot([0, 0], [0, np.max(col_hist) * 1.10], color='#ff0066', lw=1)
                            ax2.set_ylim([0, np.max(col_hist) * 1.10])

                    # ax3.set_ylim([np.min(fwhm_grid), np.max(fwhm_grid)])
                    ax3.set_ylim([0, np.max(fwhm_grid)])
                    ax3.set_xlim([0, np.max(row_hist) * 1.10])
                    # ax1.set_ylim([np.min(fwhm_grid[0:-1] * 24 * 60), np.max(fwhm_grid[0:-1] * 24 * 60)])
                    ax1.set_ylim([0, np.max(fwhm_grid[0:-1])])

                    if prop2 != 5:
                        ax1.set_xlim(np.min(bin_edges), np.max(bin_edges))
                        ax2.set_xlim(np.min(bin_edges), np.max(bin_edges))
                    else:
                        ax1.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))
                        ax2.set_xlim(0, impulse_max_lim_factor * np.max(bin_edges))

                    print('made it here: 11')

                    if prop2 == 2:
                        total_included = np.sum(row_hist)
                        frac_found = np.float(total_included) / np.float(rep_in)
                        textstr = '\n'.join(('Fraction of', 'Flares Found:', str(np.round(frac_found, 2))))
                        ax1.text(1.35, 1.12, textstr, transform=ax1.transAxes, fontsize='small', style=font_style,
                                 family=font_family, weight='heavy', verticalalignment='center',
                                 horizontalalignment='center')

                    ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                                    color='#000000', length=0)
                    ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                                    color='#000000', length=0)
                    ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                                    color='#000000', length=0)

            else:
                continue


        print('Saving 2D Histogram...')
        # plt.tight_layout()
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300, rasterized=True)
        plt.close()

    return current_vmax

def plot_corner(fitpars, fitlabs, save_as):
    font_size = 'small'
    font_style = 'normal'
    font_family = 'sans-serif'
    # impulse_max_lim_factor = 1.0  # 0.50
    # xlim_mult = hist_inclusion * bin_slice_factor_in

    fig = plt.figure(figsize=(10, 9), facecolor='#ffffff')  # , dpi=300)
    gs1 = fig.add_gridspec(nrows=len(fitpars), ncols=len(fitpars), wspace=0, hspace=0)
    # left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott,

    col_history = []
    row_history = []
    xlim_history = []

    for col in range(len(fitpars)):
        for row in range(len(fitpars)):

            # if row > col:
            #     break

            # col_history.append(col)
            # row_history.append(row)

            if row < col:

                ax = fig.add_subplot(gs1[col, row])
                xplot = fitpars[row]
                yplot = fitpars[col]
                xlabel = fitlabs[row]
                ylabel = fitlabs[col]

                col_history.append(col)
                row_history.append(row)

                astro_smooth = True
                # ax.scatter(xplot, yplot, s=np.pi*1.5**2, color='#000000', alpha=0.2, rasterized=True)
                H, xedges, yedges = np.histogram2d(xplot, yplot, bins=(15, 15))
                # maxplot = np.where(H == H.max())
                xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])
                if astro_smooth:
                    kernel = Gaussian2DKernel(x_stddev=1,y_stddev=0.69)
                    H = convolve(H, kernel)
                clevels = ax.contour(xmesh, ymesh, H.T, lw=1, cmap='winter')
                # Identify points within contours
                if (col == 1) and (row == 0):
                    p0 = clevels.collections[0].get_paths()[0]
                    vert = p0.vertices
                    xx = vert[:, 0]
                    yy = vert[:, 1]
                    # print(vert)
                    # print(clevels.levels)
                    #import pdb; pdb.set_trace()
                p = clevels.collections[0].get_paths()
                inside = np.full_like(xplot, False, dtype=bool)
                for level in p:
                    inside |= level.contains_points(list(zip(*(xplot, yplot))))
                ax.plot(np.array(xplot)[~inside], np.array(yplot)[~inside], 'x', color='#000000', alpha=0.2, rasterized=True)

                # ax.set_ylim(np.min(yplot), np.max(yplot))
                # ax.set_xlim(np.min(xplot), np.max(xplot))

                maxplot = np.where(H == H.max())
                #import pdb; pdb.set_trace()
                max_x = xplot[maxplot[0][0]]
                max_y = yplot[maxplot[1][0]]
                xlim_min = max_x - 2*np.std(xplot)
                xlim_max = max_x + 2*np.std(xplot)
                ylim_min = max_y - 2*np.std(yplot)
                ylim_max = max_y + 2*np.std(yplot)

                ax.plot([0, 0], [ylim_min, ylim_max], lw=0.5, color='red')
                ax.plot([xlim_min, xlim_max], [0, 0], lw=0.5, color='red')

                ax.set_xlim(xlim_min, xlim_max)
                ax.set_ylim(ylim_min, ylim_max)

                xlim_history.append([xlim_min, xlim_max])


                if col < len(fitpars) - 1:
                    ax.set_xticklabels([])
                else:
                    # x_increment = (np.max(fitpars[col]) - np.min(fitpars[col])) / 4.
                    # xlabs = np.arange(np.min(fitpars[col]),np.max(fitpars[col]) + x_increment,x_increment)
                    # ax.set_xticklabels(xlabs)
                    ax.set_xlabel(xlabel, fontsize=font_size, style=font_style, family=font_family)

                if row > 0:
                    ax.set_yticklabels([])
                else:
                    # y_increment = (np.max(fitpars[row]) - np.min(fitpars[row])) / 4.
                    # ylabs = np.arange(np.min(fitpars[row]), np.max(fitpars[row]) + y_increment, y_increment)
                    # ax.set_yticklabels(ylabs)
                    ax.set_ylabel(ylabel, fontsize=font_size, style=font_style, family=font_family)

                ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                               color='#000000', length=1)

                #ax.text(0.5, 0.8, str(col) + ', ' + str(row), horizontalalignment='center', verticalalignment = 'center', transform = ax.transAxes)

    for col in range(len(fitpars)):
        for row in range(len(fitpars)):

            if row == col:

                ax = fig.add_subplot(gs1[col, row])
                xplot = fitpars[row]
                yplot = fitpars[col]
                xlabel = fitlabs[row]
                ylabel = fitlabs[col]

                hist_counts = ax.hist(xplot, edgecolor='#000000',color='grey', bins='auto')

                ax.plot([0, 0], [0, np.max(hist_counts[0])*1.1], lw=0.5, color='red')
                #ax.set_xlim(np.min(hist_counts[1]), np.max(hist_counts[1]))
                ax.set_ylim(0, np.max(hist_counts[0])*1.1)

                # hist_wheremax = np.where(hist_counts[0] == np.max(hist_counts[0]))[0]
                # max_x = np.max(yplot)
                # xlim_min = max_x - 2*np.std(yplot)
                # xlim_max = max_x + 2*np.std(yplot)

                where_col = np.where(np.array(col_history) == col)[0]

                change_flag = 0
                try:
                    xlim_min = xlim_history[where_col[0]][0]
                except:
                    where_row = np.where(np.array(row_history) == row)[0]
                    # print(row_history)
                    # print(where_row)
                    # print(np.array(xlim_history))
                    xlim_min = xlim_history[where_row[0]][0]
                    xlim_max = xlim_history[where_row[0]][1]
                    ax.set_xlim(xlim_min, xlim_max)
                    change_flag = 1

                if change_flag != 1:
                    xlim_max = xlim_history[where_col[0]][1]
                    ax.set_xlim(xlim_min, xlim_max)

                if col < len(fitpars)-1:
                    ax.set_xticklabels([])
                else:
                    # x_increment = (np.max(fitpars[col]) - np.min(fitpars[col])) / 4.
                    # xlabs = np.arange(np.min(fitpars[col]),np.max(fitpars[col]) + x_increment,x_increment)
                    # ax.set_xticklabels(xlabs)
                    ax.set_xlabel(xlabel, fontsize=font_size, style=font_style, family=font_family)

                if row > 0:
                    ax.set_yticklabels([])
                else:
                    # y_increment = (np.max(fitpars[row]) - np.min(fitpars[row])) / 4.
                    # ylabs = np.arange(np.min(fitpars[row]), np.max(fitpars[row]) + y_increment, y_increment)
                    # ax.set_yticklabels(ylabs)
                    ax.set_ylabel(ylabel, fontsize=font_size, style=font_style, family=font_family)

                ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                                color='#000000', length=1)

    print('Saving Cornor Plot...')
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300, rasterized=True)
    plt.close()
def plot_corner2(fitpars, fitlabs, flare_template, save_as):
    font_size = 'small'
    font_style = 'normal'
    font_family = 'sans-serif'
    # impulse_max_lim_factor = 1.0  # 0.50
    # xlim_mult = hist_inclusion * bin_slice_factor_in

    fig = plt.figure(figsize=(12, 11), facecolor='#ffffff')  # , dpi=300)
    gs1 = fig.add_gridspec(nrows=len(fitpars), ncols=len(fitpars), wspace=0, hspace=0)
    # left=plot_left, right=plot_right, top=plot_top, bottom=plot_bott,

    columns = np.arange(0,len(fitpars)+1,1)
    # rows = []
    # for thiscol, bloop in enumerate(columns):
    #     rows.append(list(columns[bloop:-1]))
    # rows = rows[:-1]
    columns = columns[:-1]
    #import pdb; pdb.set_trace()

    imp_ind_max = 0.15

    for icol,thiscol in enumerate(columns):
        rows = columns[thiscol:]
        for irow,thisrow in enumerate(rows):

            col = thisrow
            row = thiscol

            #import pdb; pdb.set_trace()

            ax = fig.add_subplot(gs1[col, row])
            xplot = fitpars[row]
            yplot = fitpars[col]
            xlabel = fitlabs[row]
            ylabel = fitlabs[col]

            if len(np.shape(yplot)) > 1:
                yplot = np.array(yplot).ravel()
            if len(np.shape(xplot)) > 1:
                xplot = np.array(xplot).ravel()

            #import pdb; pdb.set_trace()

            numbinsx = 40
            numbinsy = 40

            if row == col:

                if flare_template == 2:
                    # xlim_min = -100
                    # xlim_max = 150
                    if row == 0:
                        # numbinsx = 30
                        xlim_min = -3
                        xlim_max = 2.5
                    if row == 1:
                        xlim_min = -100
                        xlim_max = 10
                    if row == 2:
                        xlim_min = -100
                        xlim_max = 350
                    if row == 3:
                        xlim_min = -100
                        xlim_max = 100
                    if row == 4:
                        xlim_min = 0
                        xlim_max = 40
                    if row == 5:
                        xlim_min = 0
                        xlim_max = imp_ind_max
                    if row == 6:
                        xlim_min = 0
                        xlim_max = np.max(fitpars[-1])
                if flare_template == 3:
                    # xlim_min = -100
                    # xlim_max = 150
                    if row == 0:
                        # numbinsx = 30
                        xlim_min = -3
                        xlim_max = 2.5
                    if row == 1:
                        xlim_min = -100
                        xlim_max = 10
                    if row == 2:
                        xlim_min = -100
                        xlim_max = 350
                    if row == 3:
                        xlim_min = -100
                        xlim_max = 100
                    if row == 4:
                        xlim_min = 0
                        xlim_max = 40
                    if row == 5:
                        xlim_min = 0
                        xlim_max = imp_ind_max
                    if row == 6:
                        xlim_min = 0
                        xlim_max = np.max(fitpars[-1])
                if flare_template == 1:
                    # xlim_min = -100
                    # xlim_max = 100
                    if row == 0:
                        # numbinsx = 30
                        xlim_min = -0.5
                        xlim_max = 1
                    if row == 1:
                        xlim_min = -15
                        xlim_max = 150
                    if row == 2:
                        xlim_min = -100
                        xlim_max = 100
                    if row == 3:
                        xlim_min = -100
                        xlim_max = 100
                    if row == 4:
                        xlim_min = 0
                        xlim_max = 40
                    if row == 5:
                        xlim_min = 0
                        xlim_max = imp_ind_max
                    if row == 6:
                        xlim_min = 0
                        xlim_max = np.max(fitpars[-1])

                where1 = np.where((np.array(xplot) >= xlim_min) & (np.array(xplot) <= xlim_max))[0]
                # yplotx = np.array(yplot)[where1]
                xplotx = np.array(xplot)[where1]
                # where2 = np.where((np.array(yplot) >= ylim_min) & (np.array(yplot) <= ylim_max))[0]
                # xplot = np.array(xplot)[where2]
                # yplot = np.array(yplot)[where2]

                deltaX = (xlim_max - xlim_min) / numbinsx
                # deltaY = (ylim_max - ylim_min) / numbins
                xmin = xlim_min - 1 * deltaX
                xmax = xlim_max + 1 * deltaX
                # ymin = ylim_min - 1 * deltaY
                # ymax = ylim_max + 1 * deltaY
                xx = np.arange(xmin, xmax + deltaX, deltaX)
                # yy = np.arange(ymin, ymax + deltaY, deltaY)

                hist_counts = ax.hist(xplotx, edgecolor='#000000', color='grey', bins=xx)

                ax.plot([0, 0], [0, np.max(hist_counts[0]) * 1.1], lw=0.5, color='red')

                # hist_wheremax = np.where(hist_counts[0] == np.max(hist_counts[0]))[0]
                # if len(hist_wheremax) > 1:
                #     max_x = np.mean(hist_counts[1][hist_wheremax])
                # else:
                #     try:
                #         max_x = np.mean(hist_counts[1][hist_wheremax[0]:hist_wheremax[0]+1])
                #     except:
                #         import pdb; pdb.set_trace()
                # xlim_min = max_x - 2*np.std(xplot)
                # xlim_max = max_x + 2*np.std(xplot)
                ylim_min = 0
                ylim_max = np.max(hist_counts[0]) * 1.1

                # if flare_template == 2:
                #     xlim_min = -100
                #     xlim_max = 150
                #     if row == 0:
                #         xlim_min = -2.5
                #         xlim_max = 5
                # if flare_template == 1:
                #     xlim_min = -100
                #     xlim_max = 50
                #     if row == 0:
                #         xlim_min = -2.5
                #         xlim_max = 5
                #
                # if xlim_min < np.min(hist_counts[1]):
                #     xlim_min = np.min(hist_counts[1])
                # if xlim_max > np.max(hist_counts[1]):
                #     xlim_max = np.max(hist_counts[1])

                if xlim_min < np.min(hist_counts[1]):
                    xlim_min = np.min(hist_counts[1])
                if xlim_max > np.max(hist_counts[1]):
                    xlim_max = np.max(hist_counts[1])

                ax.set_xlim(xlim_min, xlim_max)
                ax.set_ylim(ylim_min, ylim_max)

            if col > row:

                astro_smooth = False
                gauss_kde_contour = False
                regular = True
                # ax.scatter(xplot, yplot, s=np.pi * 1.75 ** 2, color='#000000', alpha=0.2, rasterized=True)
                if regular == True:
                    # deltaX = (np.max(xplot) - np.min(xplot)) / 50
                    # deltaY = (np.max(yplot) - np.min(yplot)) / 50
                    # xmin = np.min(xplot) - 1*deltaX
                    # xmax = np.max(xplot) + 1*deltaX
                    # ymin = np.min(yplot) - 1*deltaY
                    # ymax = np.max(yplot) + 1*deltaY
                    # xx = np.arange(xmin, xmax + deltaX, deltaX)
                    # yy = np.arange(ymin, ymax + deltaY, deltaY)

                    if flare_template == 2:
                        # ylim_min = -100
                        # ylim_max = 150
                        # if ylim_min < np.min(yplot):
                        #     ylim_min = np.min(yplot)
                        # if ylim_max > np.max(yplot):
                        #     ylim_max = np.max(yplot)
                        # if col == 4:
                        #     if np.max(yplot) < 40:
                        #         ylim_max = np.max(yplot)
                        #     else:
                        #         ylim_max = 40
                        if col == 1:
                            ylim_min = -100
                            ylim_max = 10
                        if col == 2:
                            ylim_min = -100
                            ylim_max = 350
                        if col == 3:
                            ylim_min = -100
                            ylim_max = 100
                        if col == 4:
                            ylim_min = 0
                            ylim_max = 40
                        if col == 5:
                            ylim_min = 0
                            ylim_max = imp_ind_max
                        if col == 6:
                            ylim_min = 0
                            ylim_max = np.max(fitpars[-1])

                    if flare_template == 3:
                        if col == 1:
                            ylim_min = -100
                            ylim_max = 10
                        if col == 2:
                            ylim_min = -100
                            ylim_max = 350
                        if col == 3:
                            ylim_min = -100
                            ylim_max = 100
                        if col == 4:
                            ylim_min = 0
                            ylim_max = 40
                        if col == 5:
                            ylim_min = 0
                            ylim_max = imp_ind_max
                        if col == 6:
                            ylim_min = 0
                            ylim_max = np.max(fitpars[-1])

                    if flare_template == 1:
                        # ylim_min = -20
                        # ylim_max = 20
                        # if ylim_min < np.min(yplot):
                        #     ylim_min = np.min(yplot)
                        # if ylim_max > np.max(yplot):
                        #     ylim_max = np.max(yplot)
                        if col == 4:
                            ylim_min = 0
                            ylim_max = 40
                        if col == 2:
                            ylim_max = 100 # 0.1
                            ylim_min = -100 # -0.1
                        if col == 3:
                            ylim_max = 100 # 0.1
                            ylim_min = -100 # -0.1
                        if col == 1:
                            ylim_max = 150 # 0.1
                            ylim_min = -15 # -0.1
                        if col == 5:
                            ylim_max = imp_ind_max
                            ylim_min = 0
                        if col == 6:
                            ylim_min = 0
                            ylim_max = 30

                    # if row == 0:
                    #     numbinsx = 30
                    #     numbinsy = 30

                    where1 = np.where((np.array(xplot) >= xlim_min) & (np.array(xplot) <= xlim_max))[0]
                    yplot = np.array(yplot)[where1]
                    xplot = np.array(xplot)[where1]
                    zplot = np.array(fitpars[-1])[where1]
                    where2 = np.where((np.array(yplot) >= ylim_min) & (np.array(yplot) <= ylim_max))[0]
                    xplot = np.array(xplot)[where2]
                    yplot = np.array(yplot)[where2]
                    zplot = np.array(zplot)[where2]

                    deltaX = (xlim_max - xlim_min) / numbinsx
                    deltaY = (ylim_max - ylim_min) / numbinsy
                    xmin = xlim_min - 1 * deltaX
                    xmax = xlim_max + 1 * deltaX
                    ymin = ylim_min - 1 * deltaY
                    ymax = ylim_max + 1 * deltaY
                    # xx = np.arange(xmin, xmax + deltaX, deltaX)
                    # yy = np.arange(ymin, ymax + deltaY, deltaY)
                    # testbins = 100
                    ax.scatter(xplot, yplot, s=np.pi*1.5**2, c=zplot, cmap='rainbow', vmin=np.min(zplot), vmax=np.max(zplot), alpha=1.0, rasterized=True)
                    # pcb = ax.colorbar(pcm, ax=axs[1:, :], location='right', shrink=0.6)
                    # H, xedges, yedges = np.histogram2d(xplot, yplot, density=True, bins=[xx,yy])
                    # xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])
                    #clevels_regular = ax.contour(xmesh, ymesh, H.T, lw=0.5, colors='green')  # , cmap='winter')

                    #mycolormap = choose_cmap()
                    # bounds = np.arange(0, H.max() + 1, 1)
                    #cmap = plt.get_cmap(mycolormap, H.max() + 1)
                    #pcolor = ax.pcolormesh(xmesh, ymesh, H.T, cmap = 'binary')
                    #cb = ax.colorbar(p)

                if gauss_kde_contour == True:
                    deltaX = (np.max(xplot) - np.min(xplot)) / 10
                    deltaY = (np.max(yplot) - np.min(yplot)) / 10
                    xmin = np.min(xplot) - 1*deltaX
                    xmax = np.max(xplot) + 1*deltaX
                    ymin = np.min(yplot) - 1*deltaY
                    ymax = np.max(yplot) + 1*deltaY
                    #print(xmin, xmax, ymin, ymax)  # Create meshgrid
                    xx, yy = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    values = np.vstack([xplot, yplot])
                    kernel1 = st.gaussian_kde(values) #, bw_method = 'silverman')
                    #kernel2 = st.gaussian_kde(values, bw_method = 'silverman')
                    f1 = np.reshape(kernel1(positions).T, xx.shape)
                    #f2 = np.reshape(kernel2(positions).T, xx.shape)
                    #cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
                    #ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
                    cset1 = ax.contour(xx, yy, f1, lw=0.25, colors='k')
                    #cset2 = ax.contour(xx, yy, f2, lw=0.25, colors='purple')
                    #print(cset.levels)
                    #ax.clabel(cset, inline=1, fmt = '%2.1f', fontsize=4)

                if astro_smooth == True:
                    deltaX = (np.max(xplot) - np.min(xplot)) / 100
                    deltaY = (np.max(yplot) - np.min(yplot)) / 100
                    xmin = np.min(xplot) - 200*deltaX
                    xmax = np.max(xplot) + 200*deltaX
                    ymin = np.min(yplot) - 200*deltaY
                    ymax = np.max(yplot) + 200*deltaY
                    xx = np.arange(xmin,xmax+deltaX,deltaX)
                    yy = np.arange(ymin,ymax+deltaY,deltaY)
                    # testbins = 100
                    H, xedges, yedges = np.histogram2d(xplot, yplot, density=True, bins = [xx, yy]) # bins=[testbins, testbins])
                    xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])
                    # testfactor = 0.10
                    kernel_astro = Gaussian2DKernel(x_stddev=5,y_stddev=5)
                    H = convolve(H, kernel_astro)
                    clevels_astro = ax.contour(xmesh, ymesh, H.T, lw=0.25, colors='k')  # , cmap='winter')
                #clevels = ax.contour(xmesh, ymesh, H.T, lw=0.5, colors='b') # , cmap='winter')
                # Identify points within contours
                # if (col == 1) and (row == 0):
                #     p0 = clevels.collections[0].get_paths()[0]
                #     vert = p0.vertices
                #     xx = vert[:, 0]
                #     yy = vert[:, 1]
                #     # print(vert)
                #     # print(clevels.levels)
                #     #import pdb; pdb.set_trace()
                # p = clevels.collections[0].get_paths()
                # inside = np.full_like(xplot, False, dtype=bool)
                # for level in p:
                #     inside |= level.contains_points(list(zip(*(xplot, yplot))))
                # ax.plot(np.array(xplot)[~inside], np.array(yplot)[~inside], 'x', color='#000000', alpha=0.2, rasterized=True)

                # import pdb; pdb.set_trace()

                ax.plot([0, 0], [ylim_min, ylim_max], lw=0.5, color='red')
                ax.plot([xlim_min, xlim_max], [0, 0], lw=0.5, color='red')

                ax.set_xlim(xlim_min, xlim_max)
                ax.set_ylim(ylim_min, ylim_max)


            if col < len(fitpars) - 1:
                ax.set_xticklabels([])
            if col == len(fitpars) - 1:
                # x_increment = (np.max(fitpars[col]) - np.min(fitpars[col])) / 4.
                # xlabs = np.arange(np.min(fitpars[col]),np.max(fitpars[col]) + x_increment,x_increment)
                # ax.set_xticklabels(xlabs)
                ax.set_xlabel(xlabel, fontsize=font_size, style=font_style, family=font_family)

            if row > 0:
                ax.set_yticklabels([])
            if row == 0:
                # y_increment = (np.max(fitpars[row]) - np.min(fitpars[row])) / 4.
                # ylabs = np.arange(np.min(fitpars[row]), np.max(fitpars[row]) + y_increment, y_increment)
                # ax.set_yticklabels(ylabs)
                ax.set_ylabel(ylabel, fontsize=font_size, style=font_style, family=font_family)
            # if row == col:
            #     ax.yaxis.tick_right()

            ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True,
                           color='#000000', length=1)


            #ax.text(0.5, 0.8, str(col) + ', ' + str(row), horizontalalignment='center', verticalalignment = 'center', transform = ax.transAxes)


    plt.tight_layout()
    print('Saving Cornor Plot...')
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as,dpi=300, rasterized=True)
    plt.close()


def dofit(flare_t, flare_f, flare_f_err, starting_popt, set_lc_cadence):

    potential_tpeaks = flare_t[np.where(flare_f == np.max(flare_f))[0][0]] + np.linspace(-1.25,0.25,40) * set_lc_cadence \
                       * (1. / 24.) * (1. / 60.)
    potential_tpeaks = potential_tpeaks[::-1]
    potential_fwhms = np.array([0.1,0.5,1.0,5.0,10.,20.,30.,40.,50.,60.,70.,80.,
                                90.,110.,130.,150.]) * (1. / 60.) * (1. / 24.)
    potential_amps =  np.linspace(0.50, 2.0, 20) * np.max(flare_f)

    chisqs = []
    bestpars = []

    initial_guess_tpeak = starting_popt[0]
    initial_guess_fwhm = starting_popt[1]
    initial_guess_ampl = starting_popt[2]

    initial_poptimum, initial_pcov = optimize.curve_fit(aflare1, flare_t, flare_f, p0=(initial_guess_tpeak, initial_guess_fwhm, initial_guess_ampl), sigma=flare_f_err)

    initial_fit = aflare1(flare_t, *initial_poptimum)
    initial_chisq = np.sum((((initial_fit - flare_f)**2)/flare_f)) #*(1./flare_f_err**2))
    chisqs.append(initial_chisq)


    initial_fit2 = aflare1(flare_t, *starting_popt)
    initial_chisq2 = np.sum((((initial_fit2 - flare_f) ** 2) / flare_f))  # *(1./flare_f_err**2))

    if initial_chisq < initial_chisq2:
        bestpars = initial_poptimum


    print('initial bestpars: ' + str(bestpars))

    for a_i in range(len(potential_amps)):
        for f_i in range(len(potential_fwhms)):
            for t_i in range(len(potential_tpeaks)):
                guess_tpeak = potential_tpeaks[t_i]
                guess_fwhm = potential_fwhms[f_i]
                guess_ampl = potential_amps[a_i]

                try:
                    poptimum, pcov = optimize.curve_fit(aflare1, flare_t, flare_f, p0=(guess_tpeak, guess_fwhm, guess_ampl), sigma=flare_f_err)
                except:
                    continue
                else:

                    model = aflare1(flare_t, *poptimum)

                    if st.mode(model)[0][0] == 0.0:
                        continue
                    else:
                        chisq = np.sum((((model - flare_f)**2)/flare_f)) #*(1./flare_f_err**2))
                        chisqs.append(chisq)

                        if len(chisqs) > 1:
                            if chisq < np.min(chisqs):
                                print('yeaahhh buddy')
                                bestpars = poptimum
                                print('new bestpars: ' + str(bestpars))

    # if initial_poptimum == starting_popt:
    #     bestpars = []

    return bestpars, starting_popt


def do_bayes(results, results_labels, min_cadence, max_cadence, flare_template, save_as):

    # results = [t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, fwhm_true, impulsive_index_true, good_cadences]
    # results_labels = ['Difference From True\nPeak Time / Cadence', '% Difference from\nTrue FWHM',
    #                   '% Difference from\nTrue Amplitude', '% Difference from\nTrue Equivalent Duration',
    #                   'True FWHM\n/ Cadence', 'Impulsive Index\n(Amplitude/FWHM)', 'Cadence (min)']

    cadences = np.array(results[-1])
    eqdur_diffs = np.array(results[3])
    impulsive_indices = np.array(results[-2])


    # ------------- cadence fraction ------------- #

    d_cadence = 1. # minutes
    #cadence_bins = np.arange(min_cadence,max_cadence+d_cadence,d_cadence)
    cadence_bins = np.arange(0, max_cadence + d_cadence, d_cadence)

    P_cads = []

    for i_cad in range(len(cadence_bins)-1):

        where_cadence = np.where((cadences > cadence_bins[i_cad]) & (cadences <= cadence_bins[i_cad+1]))[0]

        num_cadence = np.float(len(where_cadence))
        tot_cadence = np.float(len(cadences))

        P_cadence = num_cadence/tot_cadence
        P_cads.append(P_cadence)


    # ------------- eq_dur fraction ------------- #

    d_eqdur = 1  # 10 seconds to minutes
    eqdur_bins = np.arange(-100, 100 + d_eqdur, d_eqdur)

    P_cads_given_eqdur = []

    P_eqdurs = []

    for i_eqdur in range(len(eqdur_bins) - 1):
        where_eqdur = np.where((eqdur_diffs > eqdur_bins[i_eqdur]) & (eqdur_diffs <= eqdur_bins[i_eqdur + 1]))[0]

        num_eqdur = np.float(len(where_eqdur))
        tot_eqdur = np.float(len(eqdur_diffs))

        P_eqdur = num_eqdur / tot_eqdur
        P_eqdurs.append(P_eqdur)


        # ------------- cad given eqdur ------------- #

        cads_per_eqdur = []

        for i_cad2 in range(len(cadence_bins)-1):

            eqdur_cads = cadences[where_eqdur]
            where_cadence2 = np.where((eqdur_cads >= cadence_bins[i_cad2]) & (eqdur_cads < cadence_bins[i_cad2+1]))[0]

            if num_eqdur == 0:
                cads_per_eqdur.append(0)
            else:
                cads_per_eqdur.append(np.float(len(where_cadence2))/num_eqdur)


        P_cads_given_eqdur.append(np.array(cads_per_eqdur))


    # ------------- eq_dur given cad ------------- #

    P_eqdur_given_cads = []

    for i_cad3 in range(len(cadence_bins) - 1):

        eqdur_per_cad = []

        for i_eqdur2 in range(len(eqdur_bins) - 1):

            if P_cads[i_cad3] == 0:
                P_Bayes = 0
            else:
                P_Bayes = (P_cads_given_eqdur[i_eqdur2][i_cad3] * P_eqdurs[i_eqdur2]) / P_cads[i_cad3]
                # if P_Bayes > 0:
                #     print(P_Bayes)

            eqdur_per_cad.append(P_Bayes)

        P_eqdur_given_cads.append(np.array(eqdur_per_cad))



    # import pdb; pdb.set_trace()

    print('Saving Bayes Table To CSV...')
    table_dat1 = []
    table_dat2 = []
    for mlem in range(len(P_eqdur_given_cads)):
        table_dat1.append(eqdur_bins[:-1])
        table_dat2.append(P_eqdur_given_cads[mlem])

    d = {'Cadence':cadence_bins[:-1],
         '%Diff From EqDur':table_dat1,
         'Likelihood Dist':table_dat2,
         }
    df = pd.DataFrame(data=d)
    df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Cadences/EqDur_ikelihoods.csv', index=False)

    # import pdb; pdb.set_trace()



    # test_diffs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    test_cad_diffs = np.arange(0,31,1)
    font_size = 'medium'
    loc = 0

    fig = plt.figure(1, figsize=(5, 3 * len(test_cad_diffs)-1), facecolor="#ffffff") #, dpi=300)

    for test_diff in test_cad_diffs[:-1]:

        # where_test = np.where(abs(eqdur_bins-test_diff) == np.min(abs(eqdur_bins-test_diff)))[0]
        where_test = np.where(abs(cadence_bins - test_diff) == np.min(abs(cadence_bins - test_diff)))[0][0]


        likelihood_dist = np.array(P_eqdur_given_cads)[where_test]

        loc += 1
        ax = fig.add_subplot(len(test_cad_diffs),1,loc)
        # where_values = np.where(likelihood_dist != 0)[0]
        # line_cadence = cadence_bins[where_values] + 0.5*np.diff(cadence_bins)[0]
        # line_likelihood = likelihood_dist[where_values]
        # line_cadence = np.concatenate((np.array([0]),np.array(line_cadence)))
        # line_likelihood = np.concatenate((np.array([0]), np.array(line_likelihood)))
        # # ax.plot(line_cadence, line_likelihood, color='#000000', lw=2)
        ax.bar(eqdur_bins[:-1], likelihood_dist, width=np.diff(eqdur_bins), color='#e6fff7',
               edgecolor='#000000', align='edge', lw=1, rasterized=True) #ccffef
        ax.set_ylim([0, 1.1])
        ax.set_xlim([np.min(eqdur_bins), np.max(eqdur_bins)]) #max_cadence])
        if loc == len(test_cad_diffs):
            ax.set_xlabel('% Difference from True Equivalent Duration (sec)', fontsize=font_size, style='normal', family='sans-serif')
            #ax.set_xlabel('Cadence (min)', fontsize=font_size, style='normal', family='sans-serif')
        ax.set_ylabel('Posterior Likelihood', fontsize=font_size, style='normal', family='sans-serif')
        #ax.set_title('Eq Dur:  ' + str(int(test_diff)) + ' – ' + str(int(test_diff+1)) + '% Diff From True Value', fontsize='large', style='normal', family='sans-serif')
        ax.set_title('Given Cadence:  ' + str(int(test_diff)) + ' – ' + str(int(test_diff+1)) + ' min', fontsize=font_size, style='normal', family='sans-serif')
        ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
        ax.tick_params(axis='x', length=0.5)

    plt.tight_layout()
    print('Saving Bayes Plot...')
    plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as, dpi=300)
    plt.close()


        #import pdb; pdb.set_trace()
def do_bayes2(bayes_dict, save_plot, save_as):

    given_variable = np.array(bayes_dict['givenvar'])[0].flatten()
    test_variable = np.array(bayes_dict['testvar'])[0].flatten()
    given_label = bayes_dict['givenlabel']
    test_label = bayes_dict['testlabel']
    test_save_label = bayes_dict['testsavelabel']
    given_save_label = bayes_dict['givensavelabel']

    min_given = bayes_dict['mingiven']
    max_given = bayes_dict['maxgiven']
    d_given = bayes_dict['dgiven']
    min_test = bayes_dict['mintest']
    max_test = bayes_dict['maxtest']
    d_test = bayes_dict['dtest']

    test_given_diffs = np.array(bayes_dict['testgivendiffs'])[0]




    #import pdb; pdb.set_trace()

    # ------------- given variable fraction ------------- #

    # d_given = 1. # minutes
    #cadence_bins = np.arange(min_cadence,max_cadence+d_cadence,d_cadence)
    given_bins = np.arange(min_given, max_given + d_given, d_given)

    P_givens = []

    for i_cad in range(len(given_bins)-1):

        where_given = np.where((given_variable > given_bins[i_cad]) & (given_variable <= given_bins[i_cad+1]))[0]

        num_given = np.float(len(where_given))
        tot_given = np.float(len(given_variable))

        P_given = num_given/tot_given
        P_givens.append(P_given)


    # ------------- test variable fraction ------------- #

    # d_test = 1  # 10 seconds to minutes
    test_bins = np.arange(min_test, max_test + d_test, d_test)

    P_givenvar_given_testvar = []

    P_tests = []

    for i_test in range(len(test_bins) - 1):
        where_test = np.where((test_variable > test_bins[i_test]) & (test_variable <= test_bins[i_test + 1]))[0]

        num_test = np.float(len(where_test))
        tot_test = np.float(len(test_variable))

        P_test = num_test / tot_test
        P_tests.append(P_test)


        # ------------- cad given eqdur ------------- #

        givens_per_test = []

        for i_cad2 in range(len(given_bins)-1):

            test_givens = np.array(given_variable)[where_test]
            where_test2 = np.where((test_givens >= given_bins[i_cad2]) & (test_givens < given_bins[i_cad2+1]))[0]

            if num_test == 0:
                givens_per_test.append(0)
            else:
                givens_per_test.append(np.float(len(where_test2))/num_test)

        P_givenvar_given_testvar.append(np.array(givens_per_test))


    # ------------- eq_dur given cad ------------- #

    P_testvar_given_givenvar = []

    for i_cad3 in range(len(given_bins) - 1):

        tests_per_given = []

        for i_test2 in range(len(test_bins) - 1):

            if P_givens[i_cad3] == 0:
                P_Bayes = 0
            else:
                P_Bayes = (P_givenvar_given_testvar[i_test2][i_cad3] * P_tests[i_test2]) / P_givens[i_cad3]
                # if P_Bayes > 0:
                #     print(P_Bayes)

            tests_per_given.append(P_Bayes)

        # if np.sum(tests_per_given) > 0:
        #     P_testvar_given_givenvar.append(np.array(tests_per_given)/np.sum(tests_per_given))
        # else:
        #     P_testvar_given_givenvar.append(np.array(tests_per_given))
        P_testvar_given_givenvar.append(np.array(tests_per_given))



    # import pdb; pdb.set_trace()

    print('Saving Bayes Table To CSV...')
    table_dat1 = []
    table_dat2 = []
    for mlem in range(len(P_testvar_given_givenvar)):
        table_dat1.append(np.array(test_bins[:-1]))
        table_dat2.append(np.array(P_testvar_given_givenvar[mlem]))
        #print(P_testvar_given_givenvar[mlem][P_testvar_given_givenvar[mlem] > 0])

    # print('test_variable: ' + str(test_variable))
    # print('min_test: ' + str(min_test))
    # print('max_test: ' + str(max_test))
    # print('d_test: ' + str(d_test))
    # import pdb; pdb.set_trace()
    # print('table_dat1: ' + str(table_dat1))
    # import pdb; pdb.set_trace()
    # print('table_dat2: ' + str(table_dat2))
    # import pdb; pdb.set_trace()

    d = {given_label : given_bins[:-1],
         test_label : table_dat1,
         'Likelihood Dist' : table_dat2,
         }
    df = pd.DataFrame(data=d)
    df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as + '_' + test_save_label + '_given_' + given_save_label + '.csv', index=False)

    # import pdb; pdb.set_trace()

    if save_plot == True:

        font_size = 'medium'
        loc = 0

        fig = plt.figure(1, figsize=(5, 3 * len(test_given_diffs)-1), facecolor="#ffffff") #, dpi=300)

        for test_diff in test_given_diffs[:-1]:

            # where_test = np.where(abs(eqdur_bins-test_diff) == np.min(abs(eqdur_bins-test_diff)))[0]
            where_test_given = np.where(abs(given_bins - test_diff) == np.min(abs(given_bins - test_diff)))[0][0]


            likelihood_dist = np.array(P_testvar_given_givenvar)[where_test_given]

            loc += 1
            ax = fig.add_subplot(len(test_given_diffs)-1,1,loc)
            # where_values = np.where(likelihood_dist != 0)[0]
            # line_cadence = cadence_bins[where_values] + 0.5*np.diff(cadence_bins)[0]
            # line_likelihood = likelihood_dist[where_values]
            # line_cadence = np.concatenate((np.array([0]),np.array(line_cadence)))
            # line_likelihood = np.concatenate((np.array([0]), np.array(line_likelihood)))
            # # ax.plot(line_cadence, line_likelihood, color='#000000', lw=2)
            ax.bar(test_bins[:-1], likelihood_dist, width=np.diff(test_bins), color='#e6fff7',
                   edgecolor='#000000', align='edge', lw=1, rasterized=True) #ccffef
            ax.set_ylim([0, 1.1])
            ax.set_xlim([np.min(test_bins), np.max(test_bins)]) #max_cadence])
            if loc == len(test_given_diffs)-1:
                ax.set_xlabel('% Difference from True Equivalent Duration', fontsize=font_size, style='normal', family='sans-serif')
                ax.set_xlabel(test_label, fontsize=font_size, style='normal', family='sans-serif')
                #ax.set_xlabel('Cadence (min)', fontsize=font_size, style='normal', family='sans-serif')
            ax.set_ylabel('Posterior Likelihood', fontsize=font_size, style='normal', family='sans-serif')
            #ax.set_title('Eq Dur:  ' + str(int(test_diff)) + ' – ' + str(int(test_diff+1)) + '% Diff From True Value', fontsize='large', style='normal', family='sans-serif')
            ax.set_title('Given ' + given_save_label + ':  ' + str(np.round(test_diff,4)) + '  –  ' + str(np.round(test_diff+np.diff(test_given_diffs)[0],4)), fontsize=font_size, style='normal', family='sans-serif')
            ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax.tick_params(axis='x', length=0.5)

        plt.tight_layout()
        print('Saving Bayes Plot...')
        plt.savefig('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as + '_' + test_save_label + '_given_' + given_save_label + '.pdf', dpi=300)
        plt.close()


        #import pdb; pdb.set_trace()

def get_t_fast_decay(x_synth,y_synth_noscatter):
    slope = np.diff(y_synth_noscatter)/np.diff(x_synth)
    slope /= np.abs(np.min(slope))
    slope_x = x_synth[:-1]

    where_range = np.where((slope_x >= 0.45) & (slope_x <= 0.60))[0]
    slope = slope[where_range]
    slope_x = slope_x[where_range]

    where_neg = np.where(slope < 0)[0]

    neg_slope = slope[where_neg]
    neg_slope_x = slope_x[where_neg]

    slope2 = neg_slope[1:]/neg_slope[:-1]
    slope2_x = neg_slope_x[:-1]
    nonan = np.where(np.isnan(slope2) == False)[0]
    slope2 = slope2[nonan]
    slope2_x = slope2_x[nonan]
    wheremin = np.where(slope2 == np.min(slope2))[0][0]
    try:
        where_past = np.where(slope2_x >= slope2_x[wheremin])
    except:
        print('issue with were_past')
        import pdb; pdb.set_trace()

    slope2 = slope2[where_past]
    slope2_x = slope2_x[where_past]

    slope4 = np.diff(slope2) / np.diff(slope2_x)
    slope4_x = slope2_x[1:]

    where1 = np.where(slope4 == np.max(slope4))[0]
    thres_x = [slope4_x[where1][0], slope4_x[where1][0]]
    thres_y = [np.min(slope), np.max(slope)]

    where2 = np.where(slope2 == np.min(slope2))[0]

    where0 = np.where(y_synth_noscatter == np.max(y_synth_noscatter))[0]

    fast_decay_start = x_synth[where0][0]
    fast_decay_end = slope4_x[where1][0]
    #fast_decay_end = slope2_x[where2][0]
    t_fast_decay = fast_decay_end - fast_decay_start

    return t_fast_decay, fast_decay_start, fast_decay_end


from altaipony.flarelc import FlareLightCurve
from altaipony import lcio


def fit_statistics8(cadence, template = 1, downsample = True, set_max_fwhm = 120, hist_inclusion=20, bin_slice_factor=1., check_in_rep=100, n_reps = 100, where_to_save = 'All_Cadences', do_clean_directory = False):

    if do_clean_directory == True:
        clean_directory(save_directory=where_to_save)

    # initialize parameter arrays
    t_peak_opt = []
    fwhm_opt = []
    ampl_opt = []
    eq_duration_opt = []
    energy_opt = []
    eq_duration_opt_noscatter = []
    energy_opt_noscatter = []
    t_peak_frac_opt = []
    best_fit_impulsive_index = []
    best_fit_impulsive_index_modified = []

    t_peak_true = []
    fwhm_true = []
    fwhm_true_over_cadence = []
    ampl_true = []
    eq_duration_true = []
    energy_true = []
    eq_duration_true_noscatter = []
    energy_true_noscatter = []
    impulsive_index_true = []
    impulsive_index_over_cadence = []
    impulsive_index_true_modified = []
    impulsive_index_modified_over_cadence = []
    eqdur_fit = []
    eqdur_true = []

    t_peak_cant_fit = []
    fwhm_cant_fit = []
    fwhm_over_cadence_cant_fit = []
    ampl_cant_fit = []
    impulsive_index_cant_fit = []
    cadence_cant_fit = []
    impulsive_index_cant_fit_modified = []
    t_peak_frac_opt_cant_fit = []
    t_start_cant_fit = []
    t_fast_decay_end_cant_fit = []
    eq_duration_true_noscatter_cant_fit = []

    t_peak_cant_find = []
    fwhm_over_cadence_cant_find = []
    fwhm_cant_find = []
    ampl_cant_find = []
    impulsive_index_cant_find = []
    cadence_cant_find = []
    impulsive_index_cant_find_modified = []
    t_peak_true_cant_find = []
    t_start_cant_find = []
    t_fast_decay_end_cant_find = []
    eq_duration_true_noscatter_cant_find = []

    t_peak_weird_fit = []
    fwhm_over_cadence_weird = []
    fwhm_weird_true = []
    fwhm_weird_fit = []
    ampl_weird_true = []
    ampl_weird_fit = []
    impulsive_index_weird_fit = []
    cadence_weird_fit = []
    impulsive_index_weird_fit_modified = []
    t_peak_frac_opt_weird_fit = []
    t_peak_true_cant_fit = []
    tpeak_weird_true = []
    t_start_weird = []
    t_fast_decay_end_weird = []
    eq_duration_true_weird_fit_noscatter = []

    cadences = []
    fwhms = []
    good_cadences = []

    t_start_time = []
    t_fast_decay_end = []

    all_synth_x = []
    all_synth_y = []
    all_synth_y_noscatter = []
    all_downsampled_x = []
    all_downsampled_y = []
    all_downsampled_y_noscatter = []

    global_vmax = 1.

    max_cadence = 30. # minutes
    min_cadence = 1./60. # minutes

    for rep in range(n_reps):

        set_lc_cadence = np.random.uniform(min_cadence,max_cadence,1)[0]
        print(str(rep+1) + '  |   cadence: ' + str(np.round(set_lc_cadence,2)) + ' min   |  template: ' + str(template))

        cadences.append(set_lc_cadence)

        if rep == 0:
            if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + where_to_save + '/'):
                os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + where_to_save + '/')
            if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + where_to_save + '/Figures/'):
                os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + where_to_save + '/Figures/')


        # Davenport et al. (2014)
        if template == 1:
            x_synth, y_synth, y_synth_err, y_synth_noscatter, flare_properties = create_single_synthetic(cadence, max_fwhm=set_max_fwhm)
        # Jackman et al. (2018) -  numpy.convolve method
        if template == 2:
            x_synth, y_synth, y_synth_err, y_synth_noscatter, flare_properties = create_single_synthetic_jflare1(cadence, max_fwhm=set_max_fwhm)
        # Jackman et al. (2018) -  straight calculus method
        if template == 3:
            x_synth, y_synth, y_synth_err, y_synth_noscatter, flare_properties = create_single_synthetic_jflare1_equation(cadence, max_fwhm=set_max_fwhm)



        if downsample == True:

            cadence_bench = (set_lc_cadence)*60  # to put in terms of seconds because finest sampling done with 1 sec cadence

            where_start = np.int(np.floor(np.random.uniform(0, cadence_bench+1, 1)))

            x_flare = x_synth[where_start::int(cadence_bench)]
            y_flare = y_synth[where_start::int(cadence_bench)]
            y_flare_err = y_synth_err[where_start::int(cadence_bench)]
            y_noscatter = y_synth_noscatter[where_start::int(cadence_bench)]
        if downsample == False:
            x_flare = x_synth[0::1]
            y_flare = y_synth[0::1]
            y_flare_err = y_synth_err[0::1]
            y_noscatter = y_synth_noscatter[0::1]


        # flare duration
        #flc = FlareLightCurve(time=x_flare, flux=y_flare, flux_err=np.zeros_like(y_flare)+1e-4,detrended_flux=y_flare,detrended_flux_err=np.zeros_like(y_flare)+1e-4)
        flc = FlareLightCurve(time=x_flare, flux=y_flare, flux_err=y_flare_err, detrended_flux=y_flare, detrended_flux_err=y_flare_err)

        #import pdb; pdb.set_trace()
        try:

            flc = flc.find_flares(N3=2) #, addtails=True) #, tailthreshdiff=1)
            #flc = flc.find_flares()  # , addtails=True) #, tailthreshdiff=1)
        except:
            print('flare-finding error')
            flare_time = []
            flare_flux = []
            flare_flux_err = []
            # continue
        #flc.flares.to_csv('flares_' + targ + '.csv', index=False)
        if len(flc.flares) > 0:
            if len(flc.flares) == 1:
                flare_time = flc.time[flc.flares['istart'][0]-2:flc.flares['istop'][0]+6] #flc.time[f.istart:f.istop + 1]
                flare_flux = flc.flux[flc.flares['istart'][0]-2:flc.flares['istop'][0]+6]
                flare_flux_err = flc.flux_err[flc.flares['istart'][0]-2:flc.flares['istop'][0]+6]

            if len(flc.flares) > 1:
                maxfluxes = []
                for blep in range(len(flc.flares)):
                    maxfluxes.append(np.max(flc.flux[flc.flares['istart'][blep]-2:flc.flares['istop'][blep]+6]))

                wheremaxflux = np.where(maxfluxes == np.max(maxfluxes))[0][0]

                flare_time = flc.time[flc.flares['istart'][wheremaxflux]-2:flc.flares['istop'][wheremaxflux]+6]  # flc.time[f.istart:f.istop + 1]
                flare_flux = flc.flux[flc.flares['istart'][wheremaxflux]-2:flc.flares['istop'][wheremaxflux]+6]
                flare_flux_err = flc.flux_err[flc.flares['istart'][wheremaxflux]-2:flc.flares['istop'][wheremaxflux]+6]



        else:
            flare_time = []
            flare_flux = []
            flare_flux_err = []

        # import pdb; pdb.set_trace()

        # Attemp guesses at flare fits. Try other guess variations if curve_fit fails
        if template == 1:

            if len(flare_time) > 0:
                guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                guess_fwhm = np.random.uniform(0.5, 60, 1)[0] * (1. / 60.) * (1. / 24.)
                guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]

                try:
                    popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                        guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                except:
                    print('first guess fail')
                    guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                    guess_fwhm = 1 * (1. / 60.) * (1. / 24.)
                    guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]

                    try:
                        popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                            guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))

                    except:
                        print('second guess fail')

                        guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                        guess_fwhm = 30 * (1. / 60.) * (1. / 24.)
                        guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]

                        try:
                            popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                                guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))

                        except:
                            print('third guess fail')

                            guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                            guess_fwhm = 10 * (1. / 60.) * (1. / 24.)
                            guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]

                            try:
                                popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                                    guess_peak, guess_fwhm,
                                    guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))

                            except:
                                print('fourth guess fail')

                                for bep in range(len(flare_time)):
                                    if (flare_properties["tpeak"] > flare_time[bep]) and (
                                            flare_properties["tpeak"] < flare_time[bep + 1]):
                                        t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                                                flare_time[bep + 1] - flare_time[bep])
                                        break
                                    if flare_properties["tpeak"] == flare_time[bep]:
                                        t_peak_frac = [0]
                                        break
                                # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                                t_peak_cant_fit.append(t_peak_frac)
                                fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                                # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                                ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                                impulsive_index_cant_fit.append(
                                    np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                                # continue

                                for bep in range(len(flare_time)):
                                    if (flare_properties["tpeak"] > flare_time[bep]) and (
                                            flare_properties["tpeak"] < flare_time[bep + 1]):
                                        t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                                                    flare_time[bep + 1] - flare_time[bep])
                                        break
                                    if flare_properties["tpeak"] == flare_time[bep]:
                                        t_peak_frac = [0]
                                        break
                                # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                                t_peak_cant_fit.append(t_peak_frac)
                                fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                                # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                                ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                                impulsive_index_cant_fit.append(np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                                cadence_cant_fit.append(set_lc_cadence)
                                continue
                # else:
                #     guess_peak += np.random.uniform(0.5, 1.0, 1)[0] * set_lc_cadence * (1. / 24.) * (1. / 60.)
                #     guess_fwhm += np.random.uniform(0.5, 5, 1)[0] * (1. / 60.) * (1. / 24.)
                #     guess_ampl -= np.abs(popt[2]) - (np.random.uniform(-0.5, 0.5, 1)[0] * np.abs(popt[2]))
                #
                #     try:
                #         popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #             guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #     except:
                #         print('first-second guess fail')
                #         guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + \
                #                      np.random.uniform(0.5, 1.0, 1)[0] * set_lc_cadence * (1. / 24.) * (1. / 60.)
                #         guess_fwhm = 1 * (1. / 60.) * (1. / 24.)
                #         guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - \
                #                      np.random.uniform(0.1, 0.4, 1)[0] * np.max(flare_flux)
                #
                #         try:
                #             popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                 guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #         except:
                #             print('second guess fail')
                #
                #             guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + \
                #                          np.random.uniform(0.5, 1.5, 1)[0] * set_lc_cadence * (1. / 24.) * (1. / 60.)
                #             guess_fwhm = 30 * (1. / 60.) * (1. / 24.)
                #             guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - \
                #                          np.random.uniform(0.1, 0.4, 1)[0] * np.max(flare_flux)
                #
                #             try:
                #                 popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                     guess_peak, guess_fwhm,
                #                     guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #             except:
                #                 print('third guess fail')
                #
                #                 guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                #                 guess_fwhm = 10 * (1. / 60.) * (1. / 24.)
                #                 guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]
                #
                #                 try:
                #                     popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                         guess_peak, guess_fwhm,
                #                         guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #                 except:
                #                     print('fourth guess fail')
                #
                #                     for bep in range(len(flare_time)):
                #                         if (flare_properties["tpeak"] > flare_time[bep]) and (
                #                                 flare_properties["tpeak"] < flare_time[bep + 1]):
                #                             t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                #                                     flare_time[bep + 1] - flare_time[bep])
                #                             break
                #                         if flare_properties["tpeak"] == flare_time[bep]:
                #                             t_peak_frac = [0]
                #                             break
                #                     # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                #                     t_peak_cant_fit.append(t_peak_frac)
                #                     fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                #                     # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                #                     ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                #                     impulsive_index_cant_fit.append(
                #                         np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                #                     # continue
                #
                #                     for bep in range(len(flare_time)):
                #                         if (flare_properties["tpeak"] > flare_time[bep]) and (
                #                                 flare_properties["tpeak"] < flare_time[bep + 1]):
                #                             t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                #                                     flare_time[bep + 1] - flare_time[bep])
                #                             break
                #                         if flare_properties["tpeak"] == flare_time[bep]:
                #                             t_peak_frac = [0]
                #                             break
                #                     # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                #                     t_peak_cant_fit.append(t_peak_frac)
                #                     fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                #                     # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                #                     ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                #                     impulsive_index_cant_fit.append(
                #                         np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                #                     continue

        if template == 2:
            # if len(flare_time) > 0:
            #     popt = dofit(flare_time, flare_flux, flare_flux_err, set_lc_cadence)
            #
            #     # Attempt guesses at flare fits. Try other guess variations if curve_fit fails
            #     if len(popt) == 0:
            #         for bep in range(len(flare_time)):
            #             if (flare_properties["tpeak"] > flare_time[bep]) and (
            #                     flare_properties["tpeak"] < flare_time[bep + 1]):
            #                 t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
            #                             flare_time[bep + 1] - flare_time[bep])
            #                 break
            #             if flare_properties["tpeak"] == flare_time[bep]:
            #                 t_peak_frac = [0]
            #                 break
            #         # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
            #         t_peak_cant_fit.append(t_peak_frac)
            #         fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
            #         # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
            #         ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
            #         impulsive_index_cant_fit.append(np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
            #         cadence_cant_fit.append(set_lc_cadence)
            #         continue
            if len(flare_time) > 0:
                guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + np.random.uniform(-1.25,0.25,1)[0]*set_lc_cadence*(1./24.)*(1./60.)
                guess_fwhm = np.random.uniform(0.5,60,1)[0] * (1. / 60.) * (1. / 24.)
                guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - np.random.uniform(-0.5,0.5,1)[0]*np.max(flare_flux)

                try:
                    popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                    guess_peak, guess_fwhm, guess_ampl), sigma=flare_flux_err)  # diag=(1./x_window.mean(),1./y_window.mean()))
                except:
                    print('first guess fail')
                    guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + np.random.uniform(-1.25,0.25,1)[0]*set_lc_cadence*(1./24.)*(1./60.)
                    guess_fwhm = 1 * (1. / 60.) * (1. / 24.)
                    guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - np.random.uniform(-0.1,0.1,1)[0]*np.max(flare_flux)

                    try:
                        popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                        guess_peak, guess_fwhm, guess_ampl), sigma=flare_flux_err)  # diag=(1./x_window.mean(),1./y_window.mean()))

                    except:
                        print('second guess fail')

                        guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + np.random.uniform(-1.25,0.25,1)[0]*set_lc_cadence*(1./24.)*(1./60.)
                        guess_fwhm = 30 * (1. / 60.) * (1. / 24.)
                        guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - np.random.uniform(-0.1,0.1,1)[0]*np.max(flare_flux)

                        try:
                            popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                                guess_peak, guess_fwhm, guess_ampl), sigma=flare_flux_err)  # diag=(1./x_window.mean(),1./y_window.mean()))

                        except:
                            print('third guess fail')

                            guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                            guess_fwhm = 10 * (1. / 60.) * (1. / 24.)
                            guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]

                            try:
                                popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                                    guess_peak, guess_fwhm, guess_ampl), sigma=flare_flux_err)  # diag=(1./x_window.mean(),1./y_window.mean()))

                            except:
                                print('fourth guess fail')

                                for bep in range(len(flare_time)):
                                    if (flare_properties["tpeak"] > flare_time[bep]) and (
                                            flare_properties["tpeak"] < flare_time[bep + 1]):
                                        t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                                                    flare_time[bep + 1] - flare_time[bep])
                                        break
                                    if flare_properties["tpeak"] == flare_time[bep]:
                                        t_peak_frac = [0]
                                        break
                                # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                                t_peak_cant_fit.append(t_peak_frac)
                                fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                                # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                                ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                                impulsive_index_cant_fit.append(np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                                # continue

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
                                ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                                impulsive_index_cant_fit.append(np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                                cadence_cant_fit.append(set_lc_cadence)
                                continue
                # else:
                #     guess_peak += np.random.uniform(0.5, 1.0, 1)[0]*set_lc_cadence*(1./24.)*(1./60.)
                #     guess_fwhm += np.random.uniform(0.5, 5, 1)[0] * (1. / 60.) * (1. / 24.)
                #     guess_ampl -= np.abs(popt[2]) - (np.random.uniform(0.05, 0.5, 1)[0] * np.abs(popt[2]))
                #
                #     try:
                #         popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #             guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #     except:
                #         print('first-second guess fail')
                #         guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + \
                #                      np.random.uniform(0.5, 1.0, 1)[0] * set_lc_cadence * (1. / 24.) * (1. / 60.)
                #         guess_fwhm = 1 * (1. / 60.) * (1. / 24.)
                #         guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - \
                #                      np.random.uniform(0.1, 0.4, 1)[0] * np.max(flare_flux)
                #
                #         try:
                #             popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                 guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #         except:
                #             print('second guess fail')
                #
                #             guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + \
                #                          np.random.uniform(0.5, 1.5, 1)[0] * set_lc_cadence * (1. / 24.) * (1. / 60.)
                #             guess_fwhm = 30 * (1. / 60.) * (1. / 24.)
                #             guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - \
                #                          np.random.uniform(0.1, 0.4, 1)[0] * np.max(flare_flux)
                #
                #             try:
                #                 popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                     guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #             except:
                #                 print('third guess fail')
                #
                #                 guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                #                 guess_fwhm = 10 * (1. / 60.) * (1. / 24.)
                #                 guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]
                #
                #                 try:
                #                     popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                         guess_peak, guess_fwhm,
                #                         guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #                 except:
                #                     print('fourth guess fail')
                #
                #                     for bep in range(len(flare_time)):
                #                         if (flare_properties["tpeak"] > flare_time[bep]) and (
                #                                 flare_properties["tpeak"] < flare_time[bep + 1]):
                #                             t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                #                                     flare_time[bep + 1] - flare_time[bep])
                #                             break
                #                         if flare_properties["tpeak"] == flare_time[bep]:
                #                             t_peak_frac = [0]
                #                             break
                #                     # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                #                     t_peak_cant_fit.append(t_peak_frac)
                #                     fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                #                     # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                #                     ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                #                     impulsive_index_cant_fit.append(
                #                         np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                #                     # continue
                #
                #                     for bep in range(len(flare_time)):
                #                         if (flare_properties["tpeak"] > flare_time[bep]) and (
                #                                 flare_properties["tpeak"] < flare_time[bep + 1]):
                #                             t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                #                                         flare_time[bep + 1] - flare_time[bep])
                #                             break
                #                         if flare_properties["tpeak"] == flare_time[bep]:
                #                             t_peak_frac = [0]
                #                             break
                #                     # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                #                     t_peak_cant_fit.append(t_peak_frac)
                #                     fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                #                     # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                #                     ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                #                     impulsive_index_cant_fit.append(
                #                         np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                #                     continue

        if template == 3:

            if len(flare_time) > 0:
                guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + np.random.uniform(-1.25,0.25,1)[0]*set_lc_cadence*(1./24.)*(1./60.)
                guess_fwhm = np.random.uniform(0.5,60,1)[0] * (1. / 60.) * (1. / 24.)
                guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - np.random.uniform(-0.5,0.5,1)[0]*np.max(flare_flux)

                try:
                    popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                    guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                except:
                    print('first guess fail')
                    guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + np.random.uniform(-1.25,0.25,1)[0]*set_lc_cadence*(1./24.)*(1./60.)
                    guess_fwhm = 1 * (1. / 60.) * (1. / 24.)
                    guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - np.random.uniform(-0.1,0.1,1)[0]*np.max(flare_flux)

                    try:
                        popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                        guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))

                    except:
                        print('second guess fail')

                        guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + np.random.uniform(-1.25,0.25,1)[0]*set_lc_cadence*(1./24.)*(1./60.)
                        guess_fwhm = 30 * (1. / 60.) * (1. / 24.)
                        guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - np.random.uniform(-0.1,0.1,1)[0]*np.max(flare_flux)

                        try:
                            popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                                guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))

                        except:
                            print('third guess fail')

                            guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                            guess_fwhm = 10 * (1. / 60.) * (1. / 24.)
                            guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]

                            try:
                                popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                                    guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))

                            except:
                                print('fourth guess fail')

                                for bep in range(len(flare_time)):
                                    if (flare_properties["tpeak"] > flare_time[bep]) and (
                                            flare_properties["tpeak"] < flare_time[bep + 1]):
                                        t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                                                    flare_time[bep + 1] - flare_time[bep])
                                        break
                                    if flare_properties["tpeak"] == flare_time[bep]:
                                        t_peak_frac = [0]
                                        break
                                # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                                t_peak_cant_fit.append(t_peak_frac)
                                fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                                # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                                ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                                impulsive_index_cant_fit.append(np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                                # continue

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
                                ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                                impulsive_index_cant_fit.append(np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                                cadence_cant_fit.append(set_lc_cadence)
                                continue
                # else:
                #     guess_peak += np.random.uniform(0.5, 1.0, 1)[0]*set_lc_cadence*(1./24.)*(1./60.)
                #     guess_fwhm += np.random.uniform(0.5, 5, 1)[0] * (1. / 60.) * (1. / 24.)
                #     guess_ampl -= np.abs(popt[2]) - (np.random.uniform(0.05, 0.5, 1)[0] * np.abs(popt[2]))
                #
                #     try:
                #         popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #             guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #     except:
                #         print('first-second guess fail')
                #         guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + \
                #                      np.random.uniform(0.5, 1.0, 1)[0] * set_lc_cadence * (1. / 24.) * (1. / 60.)
                #         guess_fwhm = 1 * (1. / 60.) * (1. / 24.)
                #         guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - \
                #                      np.random.uniform(0.1, 0.4, 1)[0] * np.max(flare_flux)
                #
                #         try:
                #             popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                 guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #         except:
                #             print('second guess fail')
                #
                #             guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]] + \
                #                          np.random.uniform(0.5, 1.5, 1)[0] * set_lc_cadence * (1. / 24.) * (1. / 60.)
                #             guess_fwhm = 30 * (1. / 60.) * (1. / 24.)
                #             guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]] - \
                #                          np.random.uniform(0.1, 0.4, 1)[0] * np.max(flare_flux)
                #
                #             try:
                #                 popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                     guess_peak, guess_fwhm, guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #             except:
                #                 print('third guess fail')
                #
                #                 guess_peak = flare_time[np.where(flare_flux == np.max(flare_flux))[0][0]]
                #                 guess_fwhm = 10 * (1. / 60.) * (1. / 24.)
                #                 guess_ampl = flare_flux[np.where(flare_flux == np.max(flare_flux))[0][0]]
                #
                #                 try:
                #                     popt, pcov = optimize.curve_fit(aflare1, flare_time, flare_flux, p0=(
                #                         guess_peak, guess_fwhm,
                #                         guess_ampl))  # diag=(1./x_window.mean(),1./y_window.mean()))
                #
                #                 except:
                #                     print('fourth guess fail')
                #
                #                     for bep in range(len(flare_time)):
                #                         if (flare_properties["tpeak"] > flare_time[bep]) and (
                #                                 flare_properties["tpeak"] < flare_time[bep + 1]):
                #                             t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                #                                     flare_time[bep + 1] - flare_time[bep])
                #                             break
                #                         if flare_properties["tpeak"] == flare_time[bep]:
                #                             t_peak_frac = [0]
                #                             break
                #                     # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                #                     t_peak_cant_fit.append(t_peak_frac)
                #                     fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                #                     # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                #                     ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                #                     impulsive_index_cant_fit.append(
                #                         np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                #                     # continue
                #
                #                     for bep in range(len(flare_time)):
                #                         if (flare_properties["tpeak"] > flare_time[bep]) and (
                #                                 flare_properties["tpeak"] < flare_time[bep + 1]):
                #                             t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                #                                         flare_time[bep + 1] - flare_time[bep])
                #                             break
                #                         if flare_properties["tpeak"] == flare_time[bep]:
                #                             t_peak_frac = [0]
                #                             break
                #                     # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
                #                     t_peak_cant_fit.append(t_peak_frac)
                #                     fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
                #                     # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
                #                     ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                #                     impulsive_index_cant_fit.append(
                #                         np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
                #                     continue

        # Flare Fitting
        if len(flare_time) > 0:
            x_fit = np.linspace(np.min(flare_time), np.max(flare_time), 10000)
            y_fit = aflare1(x_fit, *popt)
            #y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
            y_true = aflare1(x_fit, flare_properties["tpeak"], flare_properties["fwhm"],flare_properties["amplitude"])

        # Save Parameters of flares that couldn't not be found
        else:
            for bep in range(len(x_flare)):
                if (flare_properties["tpeak"] > x_flare[bep]) and (flare_properties["tpeak"] < x_flare[bep+1]):
                    t_peak_frac = (flare_properties["tpeak"] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
                    break
                if template != 3:
                    if flare_properties["tpeak"][0] == x_flare[bep]:
                        t_peak_frac = [0]
                        break
                else:
                    if flare_properties["tpeak"] == x_flare[bep]:
                        t_peak_frac = [0]
                        break

            t_peak_cant_find.append(t_peak_frac)
            fwhm_over_cadence_cant_find.append((flare_properties["fwhm"]*24.*60.) / np.float(set_lc_cadence))
            fwhm_cant_find.append(flare_properties["fwhm"] * 24. * 60.)
            ampl_cant_find.append(flare_properties["amplitude"])
            t_peak_true_cant_find.append(flare_properties["tpeak"])
            impulsive_index_cant_find.append(np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"]*24.*60.))
            cadence_cant_find.append(set_lc_cadence)

            t_fast_decay, fast_decay_start, fast_decay_end = get_t_fast_decay(x_synth, y_synth_noscatter)
            where_above = np.where(y_synth_noscatter > 0.001)[0]
            t_start = x_synth[where_above][0]
            t_rise = fast_decay_start - t_start
            symmetry_measure = t_rise / (t_fast_decay)
            impulsive_index_cant_find_modified.append(((np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"] * 24. * 60.))[0]) / symmetry_measure)

            t_start_cant_find.append(t_start)
            t_fast_decay_end_cant_find.append(t_fast_decay_end)

            eq_dur_true_noscatter_cant_find = np.trapz(y_synth_noscatter, x=x_synth)
            eq_dur_true_noscatter_cant_find *= (24 * 60 * 60)  # convert days to seconds

            eq_duration_true_noscatter_cant_find.append(eq_dur_true_noscatter_cant_find)

            # ----------
            print(' ')
            print('Saving The Flares The Altaipony Couldnt Find...')
            print(' ')
            cant_find_dict = {'Fraction between points of true peak time of flare': t_peak_cant_find,
                           'fwhm as a fraction of cadence': fwhm_over_cadence_cant_find,
                           'true equivalent duration (noscatter)': eq_duration_true_noscatter_cant_find,
                           'cadences (min)': cadence_cant_find,
                           'impulsive index': impulsive_index_cant_find,
                           'modified impulsive index': impulsive_index_cant_find_modified,
                           'true tpeak': t_peak_true_cant_find,
                           'true fwhm': fwhm_cant_find,
                           'true ampl': ampl_cant_find,
                           'true flare start': t_start_cant_find,
                           'fast-decay end': t_fast_decay_end_cant_find,
                           }
            save_as_cant_find = where_to_save
            df = pd.DataFrame(data=cant_find_dict)
            df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as_cant_find + '/Cant_Find.csv', index=False)
            # ----------

            if np.random.uniform(0,1,1) <= 0.05:

                if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + where_to_save + '/flares_cant_find/'):
                    os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + where_to_save + '/flares_cant_find/')
                save_as_flare_cant_find =  where_to_save + '/flares_cant_find/' + str(rep+1) + '_' + str(np.round(set_lc_cadence,2)) + '.pdf'

                plot_cant_find(x_flare, y_flare, y_flare_err, save_as_flare_cant_find)

            continue

        # Check for weird fit parameters
        weird_pars = check_fitpars(popt, flare_properties)
        if len(flare_time) > 0:

            if (weird_pars["tpeak"] != 0) or (weird_pars["fwhm"] != 0) or (weird_pars["amplitude"] != 0):
                if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + where_to_save + '/weird_flare_fit_parameters/'):
                    os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + where_to_save + '/weird_flare_fit_parameters/')

            if weird_pars["tpeak"] != 0:
                save_as_test = where_to_save + '/weird_flare_fit_parameters/tpeak_' + str(rep + 1) + '.pdf'
                plot_test_fit_check(x_flare, y_flare + 1, y_flare_err, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1,
                                    x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true,
                                    save_as_test)
            if weird_pars["fwhm"] != 0:
                save_as_test = where_to_save + '/weird_flare_fit_parameters/fwhm_' + str(rep + 1) + '.pdf'
                plot_test_fit_check(x_flare, y_flare + 1, y_flare_err, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1,
                                    x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true,
                                    save_as_test)
            if weird_pars["amplitude"] != 0:
                save_as_test = where_to_save + '/weird_flare_fit_parameters/amp_' + str(rep + 1) + '.pdf'
                plot_test_fit_check(x_flare, y_flare + 1, y_flare_err, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1,
                                    x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true,
                                    save_as_test)

        if (weird_pars["tpeak"] != 0) or (weird_pars["fwhm"] != 0) or (weird_pars["amplitude"] != 0):

            popt, previous_popt = dofit(flare_time, flare_flux, flare_flux_err, popt, set_lc_cadence)

            # The Flare Can't Be Fit
            if len(popt) == 0:
                for bep in range(len(flare_time)):
                    if (flare_properties["tpeak"] > flare_time[bep]) and (
                            flare_properties["tpeak"] < flare_time[bep + 1]):
                        t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                                flare_time[bep + 1] - flare_time[bep])
                        break
                    if flare_properties["tpeak"] == flare_time[bep]:
                        t_peak_frac = [0]
                        break
                t_peak_cant_fit.append(t_peak_frac)
                fwhm_cant_fit.append(flare_properties["fwhm"]*24.*60.)
                fwhm_over_cadence_cant_fit.append((flare_properties["fwhm"] * 24. * 60.) / np.float(set_lc_cadence))
                ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
                t_peak_true_cant_fit.append(np.abs(flare_properties["tpeak"]))
                impulsive_index_cant_fit.append(np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"]*24.*60.))
                cadence_cant_fit.append(set_lc_cadence)

                t_fast_decay, fast_decay_start, fast_decay_end = get_t_fast_decay(x_synth, y_synth_noscatter)
                where_above = np.where(y_synth_noscatter > 0.001)[0]
                t_start = x_synth[where_above][0]
                t_rise = fast_decay_start - t_start
                symmetry_measure = t_rise / (t_fast_decay)
                impulsive_index_cant_fit_modified.append(((np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"] * 24. * 60.))[0]) / symmetry_measure)

                for bep2 in range(len(flare_time) - 1):
                    if (popt[0] > flare_time[bep2]) and (popt[0] < flare_time[bep2 + 1]):
                        t_peak_frac2 = (popt[0] - flare_time[bep2]) / (flare_time[bep2 + 1] - flare_time[bep2])
                    if popt[0] == flare_time[bep2]:
                        t_peak_frac2 = 0
                    if popt[0] == flare_time[bep2 + 1]:
                        t_peak_frac2 = 1.
                t_peak_frac_opt_cant_fit.append(t_peak_frac2)

                t_start_cant_fit.append(t_start)
                t_fast_decay_end_cant_fit.append(t_fast_decay_end)

                eq_dur_true_noscatter_cant_fit = np.trapz(y_synth_noscatter, x=x_synth)
                eq_dur_true_noscatter_cant_fit *= (24 * 60 * 60)  # convert days to seconds

                eq_duration_true_noscatter_cant_fit.append(eq_dur_true_noscatter_cantfit)

                # ----------
                print('Saving The Flares The Code Couldnt Fit...')
                cant_fit_dict = {'Fraction between points of true peak time of flare': t_peak_cant_fit,
                                  'fwhm as a fraction of cadence': fwhm_over_cadence_cant_fit,
                                  'true equivalent duration (noscatter)': eq_duration_true_noscatter_cant_fit,
                                  'cadences (min)': cadence_cant_fit,
                                  'impulsive index': impulsive_index_cant_fit,
                                  'modified impulsive index': impulsive_index_cant_fit_modified,
                                  'true tpeak': t_peak_true_cant_fit,
                                  'true fwhm': fwhm_cant_fit,
                                  'true ampl': ampl_cant_fit,
                                  'true flare start': t_start_cant_fit,
                                  'fast-decay end': t_fast_decay_end_cant_fit,
                                  }
                save_as_cant_fit = where_to_save
                df = pd.DataFrame(data=cant_fit_dict)
                df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as_cant_fit + '/Cant_Fit.csv',index=False)
                # ----------

                continue
            else:
                y_fit = aflare1(x_fit, *popt)
                save_as_test = where_to_save + '/weird_flare_fit_parameters/amp_' + str(rep + 1) + '_redo.pdf'
                plot_test_fit_check(x_flare, y_flare + 1, y_flare_err, x_fit, y_fit + 1, y_true + 1, flare_time,
                                    flare_flux + 1, x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true,
                                    flare_energy_true, save_as_test)

                for bep in range(len(flare_time)):
                    if (flare_properties["tpeak"] > flare_time[bep]) and (
                            flare_properties["tpeak"] < flare_time[bep + 1]):
                        t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                                flare_time[bep + 1] - flare_time[bep])
                        break
                    if flare_properties["tpeak"] == flare_time[bep]:
                        t_peak_frac = [0]
                        break
                t_peak_weird_fit.append(t_peak_frac)
                fwhm_over_cadence_weird.append((flare_properties["fwhm"]*24.*60.) / np.float(set_lc_cadence))
                fwhm_weird_true.append(flare_properties["fwhm"] * 24. * 60.)
                fwhm_weird_fit.append(popt[1])
                ampl_weird_true.append(flare_properties["amplitude"])
                ampl_weird_fit.append(popt[2])
                tpeak_weird_true.append(flare_properties["tpeak"])
                impulsive_index_weird_fit.append(np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"]*24.*60.))
                cadence_weird_fit.append(set_lc_cadence)

                t_fast_decay, fast_decay_start, fast_decay_end = get_t_fast_decay(x_synth, y_synth_noscatter)
                where_above = np.where(y_synth_noscatter > 0.001)[0]
                t_start = x_synth[where_above][0]
                t_rise = fast_decay_start - t_start
                symmetry_measure = t_rise / (t_fast_decay)
                impulsive_index_weird_fit_modified.append(((np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"] * 24. * 60.))[0]) / symmetry_measure)

                for bep2 in range(len(flare_time) - 1):
                    if (popt[0] > flare_time[bep2]) and (popt[0] < flare_time[bep2 + 1]):
                        t_peak_frac2 = (popt[0] - flare_time[bep2]) / (flare_time[bep2 + 1] - flare_time[bep2])
                    if popt[0] == flare_time[bep2]:
                        t_peak_frac2 = 0
                    if popt[0] == flare_time[bep2 + 1]:
                        t_peak_frac2 = 1.
                t_peak_frac_opt_weird_fit.append(t_peak_frac2)

                t_start_weird.append(t_start)
                t_fast_decay_end_weird.append(t_fast_decay_end)

                eq_dur_true_weird_fit = np.trapz(y_synth_noscatter, x=x_synth)
                eq_dur_true_weird_fit *= (24 * 60 * 60)  # convert days to seconds

                eq_duration_true_weird_fit_noscatter.append(eq_dur_true_weird_fit)

                print('Saving The Flares That Produced A Weird Fit...')
                weird_fit_dict = {'Fraction between points of true peak time of flare': t_peak_weird_fit,
                                  'fwhm as a fraction of cadence': fwhm_over_cadence_weird,
                                  'percent difference from true fwhm': fwhm_weird_fit,
                                  'true equivalent duration (noscatter)': eq_duration_true_weird_fit_noscatter,
                                  'cadences (min)': cadence_weird_fit,
                                  'impulsive index': impulsive_index_weird_fit,
                                  'modified impulsive index': impulsive_index_weird_fit_modified,
                                  'true tpeak': tpeak_weird_true,
                                  'true fwhm': fwhm_weird_true,
                                  'true ampl': ampl_weird_true,
                                  'true flare start': t_start_weird,
                                  'fast-decay end': t_fast_decay_end_weird,
                                  'Fraction between points of best fit peak time': t_peak_frac_opt_weird_fit,
                                  }
                save_as_weird_fit = where_to_save
                df = pd.DataFrame(data=weird_fit_dict)
                df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as_weird_fit + '/Weird_Fit.csv',index=False)

                continue





        # Check FWHM fit value. It's the most prone to being flat out unphysical and wrong
        fwhm_percent_diff = (popt[1] - flare_properties["fwhm"]) / flare_properties["fwhm"] * 100

        diff_threshold = 5

        if fwhm_percent_diff > diff_threshold:

            save_as_test = where_to_save + '/potentially_problematic_fit_' + str(rep + 1) + '-initialfit.pdf'
            plot_test_fit(x_flare, y_flare + 1, y_flare_err, x_fit, y_fit + 1, y_true + 1, flare_time,
                          flare_flux + 1, x_synth, y_synth_noscatter + 1, eq_dur, flare_energy, eq_dur_true,
                          flare_energy_true, save_as_test)

            print('-------------------')
            print('retrying fit...')
            print('fwhm percent diff: ' + str(fwhm_percent_diff))
            popt, previous_popt = dofit(flare_time, flare_flux, flare_flux_err, popt, set_lc_cadence)

            if len(popt) > 0:
                new_fwhm_percent_diff = (popt[1] - flare_properties["fwhm"]) / flare_properties["fwhm"] * 100
                print(' ')
                print('new fwhm percent diff: ' + str(new_fwhm_percent_diff))
                print(' ')

                if np.abs(new_fwhm_percent_diff) < np.abs(fwhm_percent_diff):
                    print('adopted new fit parameters')
                    print(' ')
                    fwhm_percent_diff = new_fwhm_percent_diff
                    y_fit = aflare1(x_fit, *popt)

                    save_as_test = where_to_save + '/potentially_problematic_fit_' + str(rep + 1) + '-newfit.pdf'
                    plot_test_fit(x_flare, y_flare + 1, y_flare_err, x_fit, y_fit + 1, y_true + 1, flare_time,
                                  flare_flux + 1, x_synth, y_synth_noscatter + 1, eq_dur, flare_energy, eq_dur_true,
                                  flare_energy_true, save_as_test)
            else:
                popt = previous_popt
            print('-------------------')


        fwhm_opt.append(fwhm_percent_diff)
        print(' ')
        print('fwhm percent diff: ' + str(fwhm_percent_diff))
        print('max fwhm: ' + str(np.max(fwhm_opt)))
        print(' ')
        if fwhm_percent_diff > diff_threshold:
            print('woaahhh')
            print('true fwhm:' + str(flare_properties["fwhm"]))
            print('fit fwhm:' + str(popt[1]))
            print('len y_fit ' + str(len(y_fit)))
            wheremaxfit = np.where(y_fit == np.max(y_fit))[0]
            print('where max fit: ' + str(wheremaxfit))
            print('len where max fit ' + str(len(wheremaxfit)))

            # save_as_test = where_to_save + '/potentially_problematic_fit_' + str(rep + 1) + '-3.pdf'
            # plot_test_fit(x_flare, y_flare + 1, y_flare_err, x_fit, y_fit + 1, y_true + 1, flare_time,
            #               flare_flux + 1, x_synth, y_synth_noscatter + 1, eq_dur, flare_energy, eq_dur_true,
            #               flare_energy_true, save_as_test)


            if len(wheremaxfit) > 1:

                print('we got an issue')
                print(' ')

                for bep in range(len(flare_time)):
                    if (flare_properties["tpeak"] > flare_time[bep]) and (
                            flare_properties["tpeak"] < flare_time[bep + 1]):
                        t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
                                flare_time[bep + 1] - flare_time[bep])
                        break
                    if flare_properties["tpeak"] == flare_time[bep]:
                        t_peak_frac = [0]
                        break
                t_peak_weird_fit.append(t_peak_frac)
                fwhm_over_cadence_weird.append((flare_properties["fwhm"]*24.*60.) / np.float(set_lc_cadence))
                fwhm_weird_true.append(flare_properties["fwhm"] * 24. * 60.)
                fwhm_weird_fit.append(popt[1])
                ampl_weird_true.append(flare_properties["amplitude"])
                ampl_weird_fit.append(popt[2])
                tpeak_weird_true.append(flare_properties["tpeak"])
                impulsive_index_weird_fit.append(np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"]*24.*60.))
                cadence_weird_fit.append(set_lc_cadence)

                t_fast_decay, fast_decay_start, fast_decay_end = get_t_fast_decay(x_synth, y_synth_noscatter)
                where_above = np.where(y_synth_noscatter > 0.001)[0]
                t_start = x_synth[where_above][0]
                t_rise = fast_decay_start - t_start
                symmetry_measure = t_rise / (t_fast_decay)
                impulsive_index_weird_fit_modified.append(((np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"] * 24. * 60.))[0]) / symmetry_measure)

                for bep2 in range(len(flare_time) - 1):
                    if (popt[0] > flare_time[bep2]) and (popt[0] < flare_time[bep2 + 1]):
                        t_peak_frac2 = (popt[0] - flare_time[bep2]) / (flare_time[bep2 + 1] - flare_time[bep2])
                    if popt[0] == flare_time[bep2]:
                        t_peak_frac2 = 0
                    if popt[0] == flare_time[bep2 + 1]:
                        t_peak_frac2 = 1.
                t_peak_frac_opt_weird_fit.append(t_peak_frac2)

                t_start_weird.append(t_start)
                t_fast_decay_end_weird.append(t_fast_decay_end)

                eq_dur_true_weird_fit = np.trapz(y_synth_noscatter, x=x_synth)
                eq_dur_true_weird_fit *= (24 * 60 * 60)  # convert days to seconds

                eq_duration_true_weird_fit_noscatter.append(eq_dur_true_weird_fit)

                print('Saving The Flares That Produced A Weird Fit...')
                weird_fit_dict = {'Fraction between points of true peak time of flare': t_peak_weird_fit,
                                  'fwhm as a fraction of cadence': fwhm_over_cadence_weird,
                                  'percent difference from true fwhm': fwhm_weird_fit,
                                  'true equivalent duration (noscatter)': eq_duration_true_weird_fit_noscatter,
                                  'cadences (min)': cadence_weird_fit,
                                  'impulsive index': impulsive_index_weird_fit,
                                  'modified impulsive index': impulsive_index_weird_fit_modified,
                                  'true tpeak': tpeak_weird_true,
                                  'true fwhm': fwhm_weird_true,
                                  'true ampl': ampl_weird_true,
                                  'true flare start': t_start_weird,
                                  'fast-decay end': t_fast_decay_end_weird,
                                  'Fraction between points of best fit peak time': t_peak_frac_opt_weird_fit,
                                  }
                save_as_weird_fit = where_to_save
                df = pd.DataFrame(data=weird_fit_dict)
                df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as_weird_fit + '/Weird_Fit.csv',index=False)

                continue


        L_star = 1.2  # solar luminosity
        L_star *= 3.827e33  # convert to erg/s

        eq_dur = np.trapz(y_fit, x=x_fit)
        eq_dur *= 86400  # convert days to seconds
        flare_energy = L_star * eq_dur

        eqdur_fit.append(eq_dur)

        #eq_dur_true = np.trapz(y_true, x=x_fit)
        eq_dur_true = np.trapz(y_synth, x=x_synth)
        eq_dur_true *= (24 * 60 * 60)  # convert days to seconds
        flare_energy_true = L_star * eq_dur_true

        eq_duration_true.append(eq_dur_true)
        energy_true.append(flare_energy_true)

        # eq_dur_true = np.trapz(y_true, x=x_fit)
        eq_dur_true_noscatter = np.trapz(y_synth_noscatter, x=x_synth)
        eq_dur_true_noscatter *= (24 * 60 * 60)  # convert days to seconds
        flare_energy_true_noscatter = L_star * eq_dur_true_noscatter

        eq_duration_true_noscatter.append(eq_dur_true_noscatter)
        energy_true_noscatter.append(flare_energy_true_noscatter)

        eq_duration_opt.append((eq_dur - eq_dur_true) / eq_dur_true * 100)
        energy_opt.append((flare_energy - flare_energy_true) / flare_energy_true * 100)

        eq_duration_opt_noscatter.append((eq_dur - eq_dur_true_noscatter) / eq_dur_true_noscatter * 100)
        energy_opt_noscatter.append((flare_energy - flare_energy_true_noscatter) / flare_energy_true_noscatter * 100)

        # print(popt[0])
        # print(flare_properties["tpeak"])
        for bep2 in range(len(flare_time)-1):
            if (popt[0] > flare_time[bep2]) and (popt[0] < flare_time[bep2 + 1]):
                t_peak_frac2 = (popt[0] - flare_time[bep2]) / (flare_time[bep2 + 1] - flare_time[bep2])
            if popt[0] == flare_time[bep2]:
                t_peak_frac2 = 0
            if popt[0] == flare_time[bep2 + 1]:
                t_peak_frac2 = 1.
        t_peak_frac_opt.append(t_peak_frac2)

        t_peak_opt.append(((popt[0] - flare_properties["tpeak"])[0]*24*60)/set_lc_cadence)
        # fwhm_opt.append((popt[1] - flare_properties["fwhm"]) / flare_properties["fwhm"] * 100)
        ampl_opt.append(((np.abs(popt[2]) - np.abs(flare_properties["amplitude"])) / np.abs(flare_properties["amplitude"]))[0] * 100)

        t_peak_true.append(flare_properties["tpeak"])
        fwhm_true.append(flare_properties["fwhm"]*24.*60.)
        fwhm_true_over_cadence.append((flare_properties["fwhm"]*24.*60.) / set_lc_cadence)

        ampl_true.append(np.abs(flare_properties["amplitude"]))
        impulsive_index_true.append((np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"]*24.*60.))[0])

        t_fast_decay, fast_decay_start, fast_decay_end = get_t_fast_decay(x_fit, y_fit)
        where_above = np.where(y_fit > 0.001)[0]
        t_start = x_fit[where_above][0]
        t_rise = fast_decay_start - t_start
        symmetry_measure = t_rise/(t_fast_decay)

        best_fit_impulsive_index.append(popt[2]/(popt[1]*24.*60.))
        best_fit_impulsive_index_modified.append((popt[2] / (popt[1] * 24. * 60.)) / symmetry_measure)

        t_fast_decay, fast_decay_start, fast_decay_end = get_t_fast_decay(x_synth, y_synth_noscatter)
        where_above = np.where(y_synth_noscatter > 0.001)[0]
        t_start = x_synth[where_above][0]
        t_rise = fast_decay_start - t_start
        symmetry_measure = t_rise/(t_fast_decay)
        impulsive_index_true_modified.append(((np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"]*24.*60.))[0]) / symmetry_measure)

        fwhms.append(flare_properties["fwhm"]*24*60)
        good_cadences.append(set_lc_cadence)

        impulsive_index_over_cadence.append(((np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"]*24.*60.))[0]) / set_lc_cadence)
        impulsive_index_modified_over_cadence.append((((np.abs(flare_properties["amplitude"]) / (flare_properties["fwhm"]*24.*60.))[0]) / symmetry_measure) / set_lc_cadence)

        #print(np.min(impulsive_index_over_cadence),np.max(impulsive_index_over_cadence))
        #print(np.min(impulsive_index_over_cadence), np.max(impulsive_index_over_cadence)*(300/np.max(impulsive_index_true)))

        results = [t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,fwhm_true, impulsive_index_true_modified, good_cadences]
        results_labels = ['Difference From True\nPeak Time / Cadence', '% Difference from\nTrue FWHM',
                          '% Difference from\nTrue Amplitude', '% Difference from\nTrue Equivalent Duration',
                          'True FWHM\n/ Cadence','Modified Impulsive Index' + '\n' + r'$\frac{Amplitude/FWHM}{t_{rise}/t_{fast-decay}}$','Cadence (min)']

        # print(impulsive_index_true)
        # print(impulsive_index_true_modified)
        # print(impulsive_index_over_cadence)
        # print(impulsive_index_modified_over_cadence)

        # thres_x = [fast_decay_end,fast_decay_end]
        # start_x = [t_start,t_start]
        # peak_x = [flare_properties["tpeak"][0],flare_properties["tpeak"][0]]
        # thres_y = [np.min(y_synth_noscatter), np.max(y_synth_noscatter)]
        #
        # save_test1 = where_to_save + '/test_slope1'
        # quick_test_plot(any_x=[x_synth,thres_x,start_x,peak_x], any_y=[y_synth_noscatter,thres_y,thres_y,thres_y], label_x='time', label_y=['y_synth_noscatter','thres','start','peak'], y_axis_label='flux', x_axis_range=[0.45,0.60],y_axis_range=[np.min(y_synth),np.max(y_synth)], save_as=save_test1)
        #
        # import pdb; pdb.set_trace()


        t_start_time.append(t_start)
        t_fast_decay_end.append(fast_decay_end)

        all_synth_x.append(x_synth)
        all_synth_y.append(y_synth)
        all_synth_y_noscatter.append(y_synth_noscatter)
        all_downsampled_x.append(x_flare)
        all_downsampled_y.append(y_flare)
        all_downsampled_y_noscatter.append(y_noscatter)

        if np.mod(rep + 1, check_in_rep) == 0:
            print('Saving Results...')
            results_dict = {'diff from true peak time over cadence': t_peak_opt,
                           'percent diff from true fwhm': fwhm_opt,
                           'percent diff from true ampl': ampl_opt,
                           'calculated equivalent duration': eqdur_fit,
                           'true equivalent duration (noscatter)': eq_duration_true_noscatter,
                           'precent diff from true equivalent duration': eq_duration_opt,
                           'precent diff from true equivalent duration (noscatter)': eq_duration_opt_noscatter,
                           'cadences (min)': good_cadences,
                           'best fit impulsive index': best_fit_impulsive_index,
                           'best fit modified impulsive index': best_fit_impulsive_index_modified,
                           'impulsive index': impulsive_index_true,
                           'modified impulsive index': impulsive_index_true_modified,
                           'impulsive index normalized to cadence': impulsive_index_over_cadence,
                           'modified impulsive index normalized to cadence': impulsive_index_modified_over_cadence,
                           'true tpeak': t_peak_true,
                           'true fwhm': fwhm_true,
                           'true ampl': ampl_true,
                           'true flare start': t_start_time,
                           'fast-decay end': t_fast_decay_end,
                           'Fraction between points of best fit peak time': t_peak_frac_opt,
                           }
            save_as_results = where_to_save
            df = pd.DataFrame(data=results_dict)
            df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as_results + '/Results.csv', index=False)


        # ALL the CSV Stuff...
        # if template == 1:
        #     deqdur = 1.5
        #     dfwhm_opt = 1.0
        #     dampl_opt = 1.0
        #     dtpeakfrac_opt = 0.02
        #
        #     min_eq_duration_opt = -100
        #     max_eq_duration_opt = 100
        #     min_fwhm_opt = -100
        #     max_fwhm_opt = 100
        #     min_amp_opt = -100
        #     max_amp_opt = 100
        # if (template == 2) or (template == 3):
        #     deqdur = 1.5
        #     dfwhm_opt = 1.0
        #     dampl_opt = 1.0
        #     dtpeakfrac_opt = 0.02
        #
        #     min_eq_duration_opt = -100
        #     max_eq_duration_opt = 100
        #     min_fwhm_opt = -100
        #     max_fwhm_opt = 100
        #     min_amp_opt = -100
        #     max_amp_opt = 100
        # if np.mod(rep + 1, check_in_rep) == 0:
        #     save_as_bayes = where_to_save + '/Bayes'
        #
        #     dcadence = 0.20
        #     damplitude_true = 0.025
        #
        #     max_fwhm_true = np.max(fwhm_true)
        #     dfwhm_true = (max_fwhm_true - np.min(fwhm_true))/100
        #     max_fwhm_true_over_cadence = 30
        #     dfwhm_true_over_cadence = (max_fwhm_true_over_cadence - np.min(fwhm_true_over_cadence))/100 #0.25
        #
        #     max_impulsive_index = np.min([0.15,np.max(impulsive_index_true)])
        #     dimpulsive_index = (max_impulsive_index - np.min(impulsive_index_true))/100
        #     max_impulsive_index_over_cadence = np.min([0.05,np.max(impulsive_index_over_cadence)])
        #     dimpulsive_index_over_cadence = (max_impulsive_index_over_cadence - np.min(impulsive_index_over_cadence)) / 100.
        #
        #     if template == 2:
        #         max_impulsive_index_modified = np.min([0.15, np.max(impulsive_index_true_modified)])
        #         dimpulsive_index_modified = (max_impulsive_index_modified - np.min(impulsive_index_true_modified)) / 100
        #         max_impulsive_index_modified_over_cadence = np.min([0.005,np.max(impulsive_index_modified_over_cadence)])
        #         dimpulsive_index_modified_over_cadence = (max_impulsive_index_modified_over_cadence - np.min(impulsive_index_modified_over_cadence)) / 100.
        #     if template == 1:
        #         max_impulsive_index_modified = np.min([0.00025, np.max(impulsive_index_true_modified)])
        #         dimpulsive_index_modified = (max_impulsive_index_modified - np.min(impulsive_index_true_modified)) / 100
        #         max_impulsive_index_modified_over_cadence = np.min([0.00005,np.max(impulsive_index_modified_over_cadence)])
        #         dimpulsive_index_modified_over_cadence = (max_impulsive_index_modified_over_cadence - np.min(impulsive_index_modified_over_cadence)) / 100.
        #
        #
        #     # Energy given cadence
        #     bayes_dict1 = {'givenvar': [good_cadences],
        #                    'testvar': [eq_duration_opt],
        #                    'givenlabel': 'Cadence (min)',
        #                    'testlabel': '%Diff From True Equivalent Duration',
        #                    'givensavelabel': 'Cadence',
        #                    'testsavelabel': 'EqDur_Fit',
        #                    'mingiven': 0.,
        #                    'maxgiven': max_cadence,
        #                    'dgiven': dcadence,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(0, max_cadence + dcadence, dcadence)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict1, save_plot=False, save_as=save_as_bayes)
        #
        #     # Energy_noscatter given cadence
        #     bayes_dict1 = {'givenvar': [good_cadences],
        #                    'testvar': [eq_duration_opt_noscatter],
        #                    'givenlabel': 'Cadence (min)',
        #                    'testlabel': '%Diff From True Equivalent Duration Calculated With No Scatter',
        #                    'givensavelabel': 'Cadence',
        #                    'testsavelabel': 'EqDur_Fit_Noscatter',
        #                    'mingiven': 0.,
        #                    'maxgiven': max_cadence,
        #                    'dgiven': dcadence,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(0, max_cadence + dcadence, dcadence)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict1, save_plot=False, save_as=save_as_bayes)
        #
        #     # Amplitude Fit given cadence
        #     bayes_dict1 = {'givenvar': [good_cadences],
        #                    'testvar': [ampl_opt],
        #                    'givenlabel': 'Cadence (min)',
        #                    'testlabel': '%Diff From True Amplitude',
        #                    'givensavelabel': 'Cadence',
        #                    'testsavelabel': 'Ampl_Fit',
        #                    'mingiven': 0.,
        #                    'maxgiven': max_cadence,
        #                    'dgiven': dcadence,
        #                    'mintest': min_amp_opt,
        #                    'maxtest': max_amp_opt,
        #                    'dtest': dampl_opt,
        #                    'testgivendiffs': [np.arange(0, max_cadence + dcadence, dcadence)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict1, save_plot=False, save_as=save_as_bayes)
        #
        #     # FWHM Fit given cadence
        #     bayes_dict1 = {'givenvar': [good_cadences],
        #                    'testvar': [fwhm_opt],
        #                    'givenlabel': 'Cadence (min)',
        #                    'testlabel': '%Diff From True Amplitude',
        #                    'givensavelabel': 'Cadence',
        #                    'testsavelabel': 'FWHM_Fit',
        #                    'mingiven': 0.,
        #                    'maxgiven': max_cadence,
        #                    'dgiven': dcadence,
        #                    'mintest': min_fwhm_opt,
        #                    'maxtest': max_fwhm_opt,
        #                    'dtest': dfwhm_opt,
        #                    'testgivendiffs': [np.arange(0, max_cadence + dcadence, dcadence)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict1, save_plot=False, save_as=save_as_bayes)
        #
        #     # Fraction between points of best fit peak time
        #     bayes_dict1 = {'givenvar': [good_cadences],
        #                    'testvar': [t_peak_frac_opt],
        #                    'givenlabel': 'Cadence (min)',
        #                    'testlabel': 'Peak Time Occurrence Fraction Within Consecutive Points',
        #                    'givensavelabel': 'Cadence',
        #                    'testsavelabel': 'TPeak_Fraction_Fit',
        #                    'mingiven': 0.,
        #                    'maxgiven': max_cadence,
        #                    'dgiven': dcadence,
        #                    'mintest': 0,
        #                    'maxtest': 1.,
        #                    'dtest': dtpeakfrac_opt,
        #                    'testgivendiffs': [np.arange(0, max_cadence + dcadence, dcadence)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict1, save_plot=False, save_as=save_as_bayes)
        #
        #     # Energy given Impulsive Index
        #     bayes_dict2 = {'givenvar': [impulsive_index_true],
        #                    'testvar': [eq_duration_opt],
        #                    'givenlabel': 'Impulsive Index (Amplitude/FWHM)',
        #                    'testlabel': '%Diff From True Equivalent Duration',
        #                    'givensavelabel': 'Impulsive_Index',
        #                    'testsavelabel': 'EqDur',
        #                    'mingiven': np.min(impulsive_index_true),
        #                    'maxgiven': max_impulsive_index,
        #                    'dgiven': dimpulsive_index,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(np.min(impulsive_index_true), max_impulsive_index+dimpulsive_index, dimpulsive_index)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict2, save_plot=False, save_as=save_as_bayes)
        #
        #     # Energy given Modified Impulsive Index
        #     bayes_dict2 = {'givenvar': [impulsive_index_true_modified],
        #                    'testvar': [eq_duration_opt],
        #                    'givenlabel': 'Modified Impulsive Index   ' + r'$\frac{Amplitude/FWHM}{t_{rise}/t_{fast-decay}}$',
        #                    'testlabel': '%Diff From True Equivalent Duration',
        #                    'givensavelabel': 'Impulsive_Index_Modified',
        #                    'testsavelabel': 'EqDur',
        #                    'mingiven': np.min(impulsive_index_true_modified),
        #                    'maxgiven': max_impulsive_index_modified,
        #                    'dgiven': dimpulsive_index_modified,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(np.min(impulsive_index_true_modified), max_impulsive_index_modified+dimpulsive_index_modified, dimpulsive_index_modified)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict2, save_plot=False, save_as=save_as_bayes)
        #
        #     # Energy given impulsive index normalized to cadence
        #     bayes_dict1 = {'givenvar': [impulsive_index_over_cadence],
        #                    'testvar': [eq_duration_opt],
        #                    'givenlabel': r'Impulsive Index Normalized To Cadence (min$^{-2}$)',
        #                    'testlabel': '%Diff From True Equivalent Duration',
        #                    'givensavelabel': 'Impulsive_Index_over_Cadence',
        #                    'testsavelabel': 'EqDur_Fit',
        #                    'mingiven': np.min(impulsive_index_over_cadence),
        #                    'maxgiven': max_impulsive_index_over_cadence,
        #                    'dgiven': dimpulsive_index_over_cadence,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(np.min(impulsive_index_over_cadence), max_impulsive_index_over_cadence + dimpulsive_index_over_cadence, dimpulsive_index_over_cadence)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict1, save_plot=False, save_as=save_as_bayes)
        #
        #     # Energy given modified impulsive index normalized to cadence
        #     bayes_dict1 = {'givenvar': [impulsive_index_modified_over_cadence],
        #                    'testvar': [eq_duration_opt],
        #                    'givenlabel': r'Modified Impulsive Index Normalized To Cadence (min$^{-2}$)',
        #                    'testlabel': '%Diff From True Equivalent Duration',
        #                    'givensavelabel': 'Impulsive_Index_Modified_over_Cadence',
        #                    'testsavelabel': 'EqDur_Fit',
        #                    'mingiven': np.min(impulsive_index_modified_over_cadence),
        #                    'maxgiven': max_impulsive_index_modified_over_cadence,
        #                    'dgiven': dimpulsive_index_modified_over_cadence,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(np.min(impulsive_index_modified_over_cadence), max_impulsive_index_modified_over_cadence + dimpulsive_index_modified_over_cadence, dimpulsive_index_modified_over_cadence)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict1, save_plot=False, save_as=save_as_bayes)
        #
        #     # Energy given true amplitude of flare
        #     bayes_dict2 = {'givenvar': [ampl_true],
        #                    'testvar': [eq_duration_opt],
        #                    'givenlabel': 'True Flare Amplitude',
        #                    'testlabel': '%Diff From True Equivalent Duration',
        #                    'givensavelabel': 'True_Amplitude',
        #                    'testsavelabel': 'EqDur',
        #                    'mingiven': 0.05,
        #                    'maxgiven': 1.,
        #                    'dgiven': damplitude_true,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(0.1, 1 + damplitude_true, damplitude_true)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict2, save_plot=False, save_as=save_as_bayes)
        #
        #     # Energy given true FWHM of flare
        #     bayes_dict2 = {'givenvar': [fwhm_true],
        #                    'testvar': [eq_duration_opt],
        #                    'givenlabel': 'True Flare FWHM (min)',
        #                    'testlabel': '%Diff From True Equivalent Duration',
        #                    'givensavelabel': 'True_FWHM',
        #                    'testsavelabel': 'EqDur',
        #                    'mingiven': np.min(fwhm_true),
        #                    'maxgiven': max_fwhm_true,
        #                    'dgiven': dfwhm_true,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(0, max_fwhm_true + dfwhm_true, dfwhm_true)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict2, save_plot=False, save_as=save_as_bayes)
        #
        #     # Energy given true FWHM of flare normalized to cadence
        #     bayes_dict2 = {'givenvar': [fwhm_true_over_cadence],
        #                    'testvar': [eq_duration_opt_noscatter],
        #                    'givenlabel': 'True Flare FWHM As A Fraction Of Cadence',
        #                    'testlabel': '%Diff From True Equivalent Duration',
        #                    'givensavelabel': 'True_FWHM_over_cadence',
        #                    'testsavelabel': 'EqDur',
        #                    'mingiven': 0.,
        #                    'maxgiven': max_fwhm_true_over_cadence,
        #                    'dgiven': dfwhm_true_over_cadence,
        #                    'mintest': min_eq_duration_opt,
        #                    'maxtest': max_eq_duration_opt,
        #                    'dtest': deqdur,
        #                    'testgivendiffs': [np.arange(0, 30. + dfwhm_true_over_cadence, dfwhm_true_over_cadence)],
        #                    }
        #
        #     do_bayes2(bayes_dict=bayes_dict2, save_plot=False, save_as=save_as_bayes)





        if np.mod(rep + 1, check_in_rep) == 0:
            save_as_corner = where_to_save + '/Figures/cornerplot.pdf'
            plot_corner2(results, results_labels, flare_template=template, save_as = save_as_corner)


        if np.mod(rep + 1, check_in_rep) == 0:
            if len(fwhm_cant_fit) > 0:
                save_as_cant_fit = where_to_save + '/Figures/fit_stat_hist_cant_fit.pdf'
                plot_hist_cant_fit(t_peak_cant_fit, fwhm_cant_fit, ampl_cant_fit, impulsive_index_cant_fit,
                                   cadence_cant_fit, save_as=save_as_cant_fit)

            if len(fwhm_cant_find) > 0:
                save_as_cant_find = where_to_save + '/Figures/fit_stat_hist_cant_find.pdf'
                plot_hist_cant_find(t_peak_cant_find, fwhm_cant_find, ampl_cant_find, impulsive_index_cant_find,
                                    cadence_cant_find, save_as=save_as_cant_find)

            #print('Max FWHM: ' + str(np.max(fwhms)))


        if len(flare_time) > 0:
            if (np.mod(rep + 1, check_in_rep) == 0) or (rep == 0):
                save_as_test = where_to_save + '/xample_flare_' + str(rep + 1) + '.pdf'
                try:
                    plot_test_fit(x_flare, y_flare + 1, y_flare_err, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1, x_synth,
                                  y_synth_noscatter + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)
                except:
                    pass

        # if np.mod(rep + 1, check_in_rep) == 0:
        #     save_as_hist_2D_fwhm = 'All_Cadences/fit_stat_hist_2D_fwhm_sum.pdf'
        #     global_vmax = sort_property8(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true,
        #                                  fwhm_true, ampl_true, eq_duration_true, energy_true, impulsive_index_true,
        #                                  'sum', save_as_hist_2D_fwhm, stddev, set_lc_cadence, set_max_fwhm,
        #                                  hist_inclusion, bin_slice_factor, rep + 1, global_vmax,
        #                                  property_to_sort='fwhm')



        # if (np.mod(rep + 1, check_in_rep) == 0) or (np.mod(rep + 1, n_reps) == 0):
        #
        #     save_as_hist = 'All_Cadences/fit_stat_hist.pdf'
        #     plot_stat_hist4(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true,
        #                     hist_inclusion, bin_slice_factor, set_lc_cadence, save_as_hist)
        #
        #     if (np.mod(rep + 1, n_reps) == 0):
        #         save_as_hist_2D_fwhm = 'All_Cadences/fit_stat_hist_2D_fwhm_sum.pdf'
        #         global_vmax = sort_property8(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true,
        #                                      fwhm_true, ampl_true, eq_duration_true, energy_true, impulsive_index_true,
        #                                      'sum', save_as_hist_2D_fwhm, stddev, set_lc_cadence, set_max_fwhm,
        #                                      hist_inclusion, bin_slice_factor, rep + 1, global_vmax,
        #                                      property_to_sort='fwhm')
# def fit_statistics9(cadence, template = 1, downsample = True, set_max_fwhm = 120, hist_inclusion=20, bin_slice_factor=1., check_in_rep=100, n_reps = 100):
#
#     # initialize parameter arrays
#     t_peak_opt = []
#     fwhm_opt = []
#     ampl_opt = []
#     eq_duration_opt = []
#     energy_opt = []
#
#     t_peak_true = []
#     fwhm_true = []
#     ampl_true = []
#     eq_duration_true = []
#     energy_true = []
#     impulsive_index_true = []
#
#     t_peak_cant_fit = []
#     fwhm_cant_fit = []
#     ampl_cant_fit = []
#     impulsive_index_cant_fit = []
#     cadence_cant_fit = []
#
#     t_peak_cant_find = []
#     fwhm_cant_find = []
#     ampl_cant_find = []
#     impulsive_index_cant_find = []
#     cadence_cant_find = []
#
#     cadences = []
#     fwhms = []
#     good_cadences = []
#
#     global_vmax = 1.
#
#     for rep in range(n_reps):
#
#         set_lc_cadence = np.random.uniform(10./60.,30,1)[0]
#         print(str(rep+1) + '  |   cadence: ' + str(np.round(set_lc_cadence,2)) + ' min')
#
#         cadences.append(set_lc_cadence)
#
#         if (np.mod(rep + 1, 2500) == 0) or (rep == 0):
#             if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Cadences/'):
#                 os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Cadences/')
#
#         # if (np.mod(rep+1,250) == 0) or (rep == 0):
#         #     print('Generating Flare Statistics...\nCad. ' + str(np.round(set_lc_cadence, 1)) +'    Rep. ' + str(rep + 1))
#         #     print(' ')
#
#         # Davenport et al. (2014)
#         if template == 1:
#             x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic(cadence, max_fwhm=set_max_fwhm)
#         # Jackman et al. (2018) -  numpy.convolve method
#         if template == 2:
#             x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic_jflare1(cadence, max_fwhm=set_max_fwhm)
#         # Jackman et al. (2018) -  straight calculus method
#         if template == 3:
#             x_synth, y_synth, y_synth_noscatter, flare_properties = create_single_synthetic_jflare1_equation(cadence, max_fwhm=set_max_fwhm)
#
#
#
#         if downsample == True:
#
#             cadence_bench = (set_lc_cadence)*60  # to put in terms of seconds because finest sampling done with 1 sec cadence
#
#             where_start = np.int(np.floor(np.random.uniform(0, cadence_bench+1, 1)))
#
#             x_flare = x_synth[where_start::int(cadence_bench)]
#             y_flare = y_synth[where_start::int(cadence_bench)]
#             # y_noscatter_downsample = y_synth_noscatter[where_start::15]
#         if downsample == False:
#             x_flare = x_synth[0::1]
#             y_flare = y_synth[0::1]
#
#
#         # flare duration
#         flc = FlareLightCurve(time=x_flare, flux=y_flare, flux_err=np.zeros_like(y_flare)+1e-4,detrended_flux=y_flare,detrended_flux_err=np.zeros_like(y_flare)+1e-4)
#         try:
#             flc = flc.find_flares()
#         except:
#             flare_time = []
#             flare_flux = []
#             # continue
#         #flc.flares.to_csv('flares_' + targ + '.csv', index=False)
#         if len(flc.flares) > 0:
#             flare_time = flc.time[flc.flares['istart'][0]-1:flc.flares['istop'][0] + 1] #flc.time[f.istart:f.istop + 1]
#             flare_flux = flc.flux[flc.flares['istart'][0]-1:flc.flares['istop'][0] + 1]
#
#         else:
#             flare_time = []
#             flare_flux = []
#
#         #import pdb; pdb.set_trace()
#
#
#
#         if len(flare_time) > 0:
#             popt = dofit(flare_time, flare_flux, set_lc_cadence)
#
#             # Attemp guesses at flare fits. Try other guess variations if curve_fit fails
#             if len(popt) == 0:
#                 for bep in range(len(flare_time)):
#                     if (flare_properties["tpeak"] > flare_time[bep]) and (
#                             flare_properties["tpeak"] < flare_time[bep + 1]):
#                         t_peak_frac = (flare_properties["tpeak"] - flare_time[bep]) / (
#                                     flare_time[bep + 1] - flare_time[bep])
#                         break
#                     if flare_properties["tpeak"] == flare_time[bep]:
#                         t_peak_frac = [0]
#                         break
#                 # t_peak_cant_fit.append((flare_properties["tpeak"][w] - np.min(x_window))/(np.max(x_window) - np.min(x_window)))
#                 t_peak_cant_fit.append(t_peak_frac)
#                 fwhm_cant_fit.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
#                 # fwhm_cant_fit.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
#                 ampl_cant_fit.append(np.abs(flare_properties["amplitude"]))
#                 impulsive_index_cant_fit.append(np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
#                 cadence_cant_fit.append(set_lc_cadence)
#                 continue
#
#
#
#
#         if len(flare_time) > 0:
#             x_fit = np.linspace(np.min(flare_time), np.max(flare_time), 10000)
#             y_fit = aflare1(x_fit, *popt)
#             #y_true = aflare1(x_fit, flare_properties["tpeak"][0], flare_properties["fwhm"][0], flare_properties["amplitude"][0])
#             y_true = aflare1(x_fit, flare_properties["tpeak"], flare_properties["fwhm"],flare_properties["amplitude"])
#         else:
#             for bep in range(len(x_flare)):
#                 if (flare_properties["tpeak"] > x_flare[bep]) and (flare_properties["tpeak"] < x_flare[bep+1]):
#                     t_peak_frac = (flare_properties["tpeak"] - x_flare[bep])/(x_flare[bep+1] - x_flare[bep])
#                     break
#                 if flare_properties["tpeak"][0] == x_flare[bep]:
#                     t_peak_frac = [0]
#                     break
#
#             t_peak_cant_find.append(t_peak_frac)
#             fwhm_cant_find.append(flare_properties["fwhm"] / np.float(set_lc_cadence))
#             # fwhm_cant_find.append(flare_properties["fwhm"]) # / (np.float(set_lc_cadence) / 24. / 60.))
#             ampl_cant_find.append(flare_properties["amplitude"])
#             impulsive_index_cant_find.append(np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])
#             cadence_cant_find.append(set_lc_cadence)
#
#             if np.random.uniform(0,1,1) <= 0.10:
#
#                 if not os.path.exists('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Cadences/flares_cant_find/'):
#                     os.mkdir('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Cadences/flares_cant_find/')
#                 save_as_flare_cant_find = 'All_Cadences/flares_cant_find/' + str(rep+1) + '_' + str(np.round(set_lc_cadence,2)) + '.pdf'
#
#                 plot_cant_find(x_flare, y_flare, save_as_flare_cant_find)
#
#             continue
#
#         # Check for weird fit parameters
#         weird_pars = check_fitpars(popt, flare_properties)
#         if len(flare_time) > 0:
#
#             if weird_pars["tpeak"] != 0:
#                 save_as_test = 'All_Cadences/weird_flare_fit_parameters/tpeak_' + str(rep + 1) + '.pdf'
#                 plot_test_fit_check(x_flare, y_flare + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1,
#                                     x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true,
#                                     save_as_test)
#             if weird_pars["fwhm"] != 0:
#                 save_as_test = 'All_Cadences/weird_flare_fit_parameters/fwhm_' + str(rep + 1) + '.pdf'
#                 plot_test_fit_check(x_flare, y_flare + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1,
#                                     x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true,
#                                     save_as_test)
#             if weird_pars["amplitude"] != 0:
#                 save_as_test = 'All_Cadences/weird_flare_fit_parameters/amp_' + str(rep + 1) + '.pdf'
#                 plot_test_fit_check(x_flare, y_flare + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1,
#                                     x_synth, y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true,
#                                     save_as_test)
#         if (weird_pars["tpeak"] != 0) or (weird_pars["fwhm"] != 0) or (weird_pars["amplitude"] != 0):
#             continue
#
#
#         L_star = 1.2  # solar luminosity
#         L_star *= 3.827e33  # convert to erg/s
#
#         eq_dur = np.trapz(y_fit, x=x_fit)
#         eq_dur *= 86400  # convert days to seconds
#         flare_energy = L_star * eq_dur
#
#         #eq_dur_true = np.trapz(y_true, x=x_fit)
#         eq_dur_true = np.trapz(y_synth, x=x_synth)
#         eq_dur_true *= (24 * 60 * 60)  # convert days to seconds
#         flare_energy_true = L_star * eq_dur_true
#
#         eq_duration_true.append(eq_dur_true)
#         energy_true.append(flare_energy_true)
#
#         eq_duration_opt.append((eq_dur - eq_dur_true) / eq_dur_true * 100)
#         energy_opt.append((flare_energy - flare_energy_true) / flare_energy_true * 100)
#
#         # print(popt[0])
#         # print(flare_properties["tpeak"])
#
#
#         t_peak_opt.append(((popt[0] - flare_properties["tpeak"])[0]*24*60)/set_lc_cadence)
#         fwhm_opt.append((popt[1] - flare_properties["fwhm"]) / flare_properties["fwhm"] * 100)
#         ampl_opt.append(((np.abs(popt[2]) - np.abs(flare_properties["amplitude"])) / np.abs(flare_properties["amplitude"]))[0] * 100)
#
#         t_peak_true.append(flare_properties["tpeak"])
#         fwhm_true.append((flare_properties["fwhm"]*24*60)/set_lc_cadence)
#         ampl_true.append(np.abs(flare_properties["amplitude"]))
#         impulsive_index_true.append((np.abs(flare_properties["amplitude"]) / flare_properties["fwhm"])[0])
#
#         fwhms.append(flare_properties["fwhm"]*24*60)
#         good_cadences.append(set_lc_cadence)
#
#
#
#
#
#         results = [t_peak_opt,fwhm_opt,ampl_opt,eq_duration_opt,fwhm_true, impulsive_index_true, good_cadences]
#         results_labels = ['Difference From True\nPeak Time / Cadence', '% Difference from\nTrue FWHM',
#                           '% Difference from\nTrue Amplitude', '% Difference from\nTrue Equivalent Duration',
#                           'True FWHM\n/ Cadence','Impulsive Index\n(Amplitude/FWHM)','Cadence (min)']
#
#         if np.mod(rep + 1, check_in_rep) == 0:
#             save_as_corner = 'All_Cadences/cornerplot.pdf'
#             plot_corner2(results, results_labels, flare_template=template, save_as = save_as_corner)
#
#
#         if np.mod(rep + 1, check_in_rep) == 0:
#             if len(fwhm_cant_fit) > 0:
#
#                 save_as_cant_fit = 'All_Cadences/fit_stat_hist_cant_fit.pdf'
#                 plot_hist_cant_fit(t_peak_cant_fit, fwhm_cant_fit, ampl_cant_fit, impulsive_index_cant_fit,
#                                    cadence_cant_fit, save_as=save_as_cant_fit)
#
#             if len(fwhm_cant_find) > 0:
#
#                 save_as_cant_find = 'All_Cadences/fit_stat_hist_cant_find.pdf'
#                 plot_hist_cant_find(t_peak_cant_find, fwhm_cant_find, ampl_cant_find, impulsive_index_cant_find,
#                                     cadence_cant_find, save_as=save_as_cant_find)
#
#             print('Max FWHM: ' + str(np.max(fwhms)))
#
#
#         if len(flare_time) > 0:
#             if (np.mod(rep + 1, check_in_rep) == 0) or (rep == 0):
#                 save_as_test = 'All_Cadences/xample_flare_' + str(rep + 1) + '.pdf'
#                 try:
#                     plot_test_fit(x_flare, y_flare + 1, x_fit, y_fit + 1, y_true + 1, flare_time, flare_flux + 1, x_synth,
#                                   y_synth + 1, eq_dur, flare_energy, eq_dur_true, flare_energy_true, save_as_test)
#                 except:
#                     continue
#
#         # if np.mod(rep + 1, check_in_rep) == 0:
#         #     save_as_hist_2D_fwhm = 'All_Cadences/fit_stat_hist_2D_fwhm_sum.pdf'
#         #     global_vmax = sort_property8(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true,
#         #                                  fwhm_true, ampl_true, eq_duration_true, energy_true, impulsive_index_true,
#         #                                  'sum', save_as_hist_2D_fwhm, stddev, set_lc_cadence, set_max_fwhm,
#         #                                  hist_inclusion, bin_slice_factor, rep + 1, global_vmax,
#         #                                  property_to_sort='fwhm')
#
#
#
#         # if (np.mod(rep + 1, check_in_rep) == 0) or (np.mod(rep + 1, n_reps) == 0):
#         #
#         #     save_as_hist = 'All_Cadences/fit_stat_hist.pdf'
#         #     plot_stat_hist4(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, impulsive_index_true,
#         #                     hist_inclusion, bin_slice_factor, set_lc_cadence, save_as_hist)
#         #
#         #     if (np.mod(rep + 1, n_reps) == 0):
#         #         save_as_hist_2D_fwhm = 'All_Cadences/fit_stat_hist_2D_fwhm_sum.pdf'
#         #         global_vmax = sort_property8(t_peak_opt, fwhm_opt, ampl_opt, eq_duration_opt, energy_opt, t_peak_true,
#         #                                      fwhm_true, ampl_true, eq_duration_true, energy_true, impulsive_index_true,
#         #                                      'sum', save_as_hist_2D_fwhm, stddev, set_lc_cadence, set_max_fwhm,
#         #                                      hist_inclusion, bin_slice_factor, rep + 1, global_vmax,
#         #                                      property_to_sort='fwhm')


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


numreps = 1000000
check_in_rep = 50

delete_all_before_running = True

flare_template = 2
if flare_template == 1:
    save_here = 'All_Cadences_DavFittingDav0'
    max_fwhm = 60.
if flare_template == 2:
    save_here = 'All_Cadences_DavFittingJack_Throwaway'
    max_fwhm = 30.
if flare_template == 3:
    save_here = 'All_Cadences_DavFittingJack_Equation_Throwaway'

fit_statistics8(cadence, template=flare_template, downsample=True, set_max_fwhm = max_fwhm, hist_inclusion=20., bin_slice_factor=2, check_in_rep=check_in_rep, n_reps=numreps, where_to_save = save_here, do_clean_directory=delete_all_before_running)
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




#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import matplotlib as mpl

from IPython.display import Latex, Math

import numpy as np
from celerite.modeling import Model
from scipy.signal import savgol_filter

# import exoplanet as xo
# from exoplanet.gp import terms, GP
# from exoplanet.gp.terms import TermSum
# from exoplanet.utils import eval_in_model
# from exoplanet.sampling import PyMC3Sampler
# from exoplanet.utils import get_samples_from_trace
# import corner
# import pymc3 as pm
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

np.random.seed(42)
#np.random.seed()

from scipy.interpolate import splrep, splev

import lightkurve as lk

from scipy import optimize
import scipy.stats as st

import sys, glob, os, shutil
import colour # https://www.colour-science.org/installation-guide/

from ast import literal_eval
import matplotlib.colors as clr





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

def flatten_list(input_list):
    return [item for sublist in input_list for item in sublist]
def create_directories(mother_dir):
    path_to_analysis_figures = '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/'
    if not os.path.exists(path_to_analysis_figures + mother_dir + '/'):
        os.mkdir(path_to_analysis_figures + mother_dir + '/')
    if not os.path.exists(path_to_analysis_figures + mother_dir + '/Figures/'):
        os.mkdir(path_to_analysis_figures + mother_dir + '/Figures/')

    return path_to_analysis_figures + mother_dir + '/Figures/'

def read_file(path_to_csv, csv_filename):

    df = pd.read_csv(path_to_csv + csv_filename)

    colnames = df.columns

    return df,colnames
def convert_to_float(temp):
    temp1 = temp
    temp1 = temp1.replace('\n', '')
    temp2 = ''
    for fleeb in range(len(temp1)):
        if temp[fleeb] != ' ':
            temp2 += temp1[fleeb]
        if temp1[fleeb] == ' ':
            if (temp2[-1] == ',') or (temp2[-1] == '['):
                continue
            else:
                temp2 += ','
    fixed_dat = np.array(literal_eval(temp2))
    if len(fixed_dat) == 1:
        fixed_dat = fixed_dat[0]

    return fixed_dat
# def extract_columns(input_dict, given_col, test_col, cad_col):
#
#     print('Extracting Columns...')
#
#     if type(input_dict[given_col][0]) == 'str':
#         for foo in range(len(input_dict[given_col])):
#             input_dict[given_col][foo] = convert_to_float(input_dict[given_col][foo])
#     if type(input_dict[test_col][0]) == 'str':
#         for foo in range(len(input_dict[test_col])):
#             input_dict[test_col][foo] = convert_to_float(input_dict[test_col][foo])
#     if type(input_dict[cad_col][0]) == 'str':
#         for foo in range(len(input_dict[cad_col])):
#             input_dict[cad_col][foo] = convert_to_float(input_dict[cad_col][foo])
#
#     return input_dict[given_col], input_dict[test_col], input_dict[cad_col]
def extract_columns(input_dict, input_columns):

    for col in range(len(input_columns)):
        if type(input_dict[input_columns[col]][0]) == 'str':
            for foo in range(len(input_dict[input_columns[col]])):
                input_dict[input_columns[col]][foo] = convert_to_float(input_dict[input_columns[col]][foo])

    extracted_columns = []
    for excol in range(len(input_columns)):
        extracted_columns.append(input_dict[input_columns[excol]])

    return extracted_columns
def do_bayes3(bayes_dict):

    given_variable = np.array(bayes_dict['givenvar'])[0].flatten()
    test_variable = np.array(bayes_dict['testvar'])[0].flatten()
    given_label = bayes_dict['givenlabel']
    test_label = bayes_dict['testlabel']

    min_given = bayes_dict['mingiven']
    max_given = bayes_dict['maxgiven']
    d_given = bayes_dict['dgiven']
    min_test = bayes_dict['mintest']
    max_test = bayes_dict['maxtest']
    d_test = bayes_dict['dtest']




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

    #print('Saving Bayes Table To CSV...')
    table_dat1 = []
    table_dat2 = []
    for mlem in range(len(P_testvar_given_givenvar)):
        table_dat1.append(np.array(test_bins[:-1]))
        table_dat2.append(np.array(P_testvar_given_givenvar[mlem]))
        #print(P_testvar_given_givenvar[mlem][P_testvar_given_givenvar[mlem] > 0])

    d = {given_label : given_bins[:-1],
         test_label : table_dat1,
         'Likelihood Dist' : table_dat2,
         }
    l_dict = pd.DataFrame(data=d)
    # df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as + '_' + test_save_label + '_given_' + given_save_label + '.csv', index=False)

    # import pdb; pdb.set_trace()

    return l_dict
def do_bayes4(bayes_dict,interesting_cad,is_given_cad=False):

    given_variable_in = np.array(bayes_dict['givenvar'])[0].flatten()
    test_variable_in = np.array(bayes_dict['testvar'])[0].flatten()
    cadence_variable = np.array(bayes_dict['cadvar'])[0].flatten()
    given_label = bayes_dict['givenlabel']
    test_label = bayes_dict['testlabel']

    min_given = bayes_dict['mingiven']
    max_given = bayes_dict['maxgiven']
    d_given = bayes_dict['dgiven']
    min_test = bayes_dict['mintest']
    max_test = bayes_dict['maxtest']
    d_test = bayes_dict['dtest']

    if is_given_cad == False:
        where_cad = np.where((cadence_variable >= interesting_cad-0.05) & (cadence_variable <= interesting_cad+0.05))[0]

        given_variable = given_variable_in[where_cad]
        test_variable = test_variable_in[where_cad]
    if is_given_cad == True:
        given_variable = given_variable_in
        test_variable = test_variable_in


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

    #print('Saving Bayes Table To CSV...')
    table_dat1 = []
    table_dat2 = []
    for mlem in range(len(P_testvar_given_givenvar)):
        table_dat1.append(np.array(test_bins[:-1]))
        table_dat2.append(np.array(P_testvar_given_givenvar[mlem]))
        #print(P_testvar_given_givenvar[mlem][P_testvar_given_givenvar[mlem] > 0])

    d = {given_label : given_bins[:-1],
         test_label : table_dat1,
         'Likelihood Dist' : table_dat2,
         }
    l_dict = pd.DataFrame(data=d)
    # df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as + '_' + test_save_label + '_given_' + given_save_label + '.csv', index=False)

    # import pdb; pdb.set_trace()

    return l_dict

def plot_posterior(lh_dict,given_lims,test_lims,interesting_cad,sample_size,is_given_cad=False):
    cols = lh_dict.columns
    xlab = cols[1]
    ylab = cols[0]
    zlab = cols[2]

    font_size = 'large'

    y = np.array(lh_dict[ylab])
    x = np.array(lh_dict[xlab])
    z = np.zeros((len(x[0]),len(y)))

    # import pdb; pdb.set_trace()

    for row in range(len(lh_dict)):

        z_array = np.array(lh_dict[zlab][row])

        z[:, row] = np.array(z_array, dtype='float')

    #import pdb; pdb.set_trace()

    fig = plt.figure(1, figsize=(7, 5.5), facecolor="#ffffff")  # , dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
    ax.set_ylabel(ylab, fontsize=font_size, style='normal', family='sans-serif')
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    xmesh, ymesh = np.meshgrid(x[0], y)
    mycolormap = choose_cmap()
    cmap = plt.get_cmap(mycolormap, z.max() + 1)
    z = np.ma.masked_where(z == 0, z)
    cmap.set_bad(color='white')

    # import pdb; pdb.set_trace()

    pcolor = ax.pcolormesh(xmesh, ymesh, z.T, cmap = cmap, shading='nearest', rasterized=True)
    cb = fig.colorbar(pcolor) #, ax=ax) #, cmap=cmap)
    cb.ax.tick_params(labelsize=font_size)
    cb.ax.set_ylabel('Posterior Probability', fontsize=font_size, rotation=270)
    cb.ax.get_yaxis().labelpad = 15
    ax.plot([0,0],[0,np.max(y)],'-',lw=0.75,c='#000000')

    ax.set_xlim(test_lims)
    ax.set_ylim(given_lims)
    if is_given_cad == False:
        ax.set_title('Cadence: ' + str(np.round(interesting_cad,2)) + ' min\nSample Size: ' + str(sample_size),
                     fontsize=font_size)
    else:
        ax.set_title('Sample Size: ' + str(sample_size), fontsize=font_size)
    plt.tight_layout()
    # print('Saving 2D Posterior Likelihood Plot...')
    # plt.savefig(save_as, dpi=300)
    # plt.close()
    plt.show()
def plot_cumulative(lh_dict,given_lims,test_lims,interesting_cad,sample_size,is_given_cad=False):
    cols = lh_dict.columns
    xlab = cols[1]
    ylab = cols[0]
    zlab = cols[2]

    font_size = 'large'

    y = np.array(lh_dict[ylab])
    x = np.array(lh_dict[xlab])
    z = np.zeros((len(x[0]),len(y)))

    #import pdb; pdb.set_trace()

    for row in range(len(lh_dict)):

        z_array = np.array(lh_dict[zlab][row])

        temp_z = []
        for el in range(len(z_array)):
            temp_z.append(np.sum(z_array[0:el]))

        z[:,row] = np.array(temp_z,dtype='float')


    fig = plt.figure(1, figsize=(7, 5.5), facecolor="#ffffff")  # , dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
    ax.set_ylabel(ylab, fontsize=font_size, style='normal', family='sans-serif')
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    xmesh, ymesh = np.meshgrid(x[0], y)
    mycolormap = choose_cmap()
    cmap = plt.get_cmap(mycolormap, z.max() + 1)
    z = np.ma.masked_where(z == 0, z)
    cmap.set_bad(color='white')

    #import pdb; pdb.set_trace()

    pcolor = ax.pcolormesh(xmesh, ymesh, z.T, cmap = cmap, shading='nearest', rasterized=True)
    cb = fig.colorbar(pcolor) #, ax=ax) #, cmap=cmap)
    cb.ax.tick_params(labelsize=font_size)
    cb.ax.set_ylabel('Posterior Probability', fontsize=font_size, rotation=270)
    cb.ax.get_yaxis().labelpad = 15
    ax.plot([0,0],[0,np.max(y)],'-',lw=0.75,c='#000000')

    ax.set_xlim(test_lims)
    ax.set_ylim(given_lims)
    if is_given_cad == False:
        ax.set_title('Cadence: ' + str(np.round(interesting_cad,2)) + ' min\nSample Size: ' + str(sample_size),
                     fontsize=font_size)
    else:
        ax.set_title('Sample Size: ' + str(sample_size), fontsize=font_size)
    plt.tight_layout()
    plt.tight_layout()
    # print('Saving 2D Cumulative Likelihood Plot...')
    # plt.savefig(save_as, dpi=300)
    # plt.close()
    plt.show()

def plot_specific_cadences_posterior(lh_dict,interesting_cads,interesting_labs,sample_size,test_lims):

    cols = lh_dict.columns
    xlab = cols[1]
    ylab = cols[0]
    zlab = cols[2]

    y = np.array(lh_dict[ylab])
    x = np.array(lh_dict[xlab])
    z_col = np.array(lh_dict[zlab])

    font_size = 'medium'
    fig = plt.figure(1, figsize=(7, 5), facecolor="#ffffff") # , dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    mycolormap = choose_cmap()
    colors = mycolormap(range(len(interesting_cads)))
    colors = ['#b30047','#ff6600','#009900','#3366ff']
    patterns = ['//','\\','.','']
    alphas = [0.8,0.6,0.4,0.2]

    ymax = 0

    for element in range(len(interesting_cads)):

        temp_array = np.abs(y - interesting_cads[element])
        where_interesting = np.where(np.abs(temp_array == np.min(temp_array)))[0]

        if len(where_interesting) > 1:
            print('check where_interesting array')
            import pdb; pdb.set_trace()

        else:
            z_array = z_col[where_interesting[0]]

        #import pdb; pdb.set_trace()
        ax.bar(x[0], z_array, width=np.concatenate((np.diff(x[0]),[np.diff(x[0])[0]])),
               color=clr.rgb2hex(colors[element]), edgecolor='None', align='edge', lw=1,
               label=interesting_labs[element], alpha=0.1, rasterized=True) #ccffef
        ax.step(x[0], z_array, color=clr.rgb2hex(colors[element]), lw=1, alpha=0.8, where='post', rasterized=True)  # ccffef
        if np.max(z_array) >= ymax:
            ymax = np.max(z_array)
    ax.set_xlim(test_lims)
    ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
    ax.set_ylabel('Posterior Likelihood', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_title('Sample Size: ' + str(sample_size), fontsize=font_size)
    plt.tight_layout()
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    ax.tick_params(axis='x', length=0.5)
    ax.plot([0, 0], [0, 1.1*ymax], '-', lw=0.75, c='#000000')
    ax.set_ylim([0, 1.1 * ymax])
    plt.legend(loc='upper left')  # ,fontsize=font_size)
    plt.tight_layout()
    # print('Saving Interesting Cadences Plot...')
    # plt.savefig(save_as, dpi=300)
    # plt.close()
    plt.show()
def plot_specific_cadences_cumulative(lh_dict,interesting_cads,interesting_labs,sample_size,test_lims):

    cols = lh_dict.columns
    xlab = cols[1]
    ylab = cols[0]
    zlab = cols[2]

    y = np.array(lh_dict[ylab])
    x = np.array(lh_dict[xlab])
    z_col = np.array(lh_dict[zlab])

    font_size = 'medium'
    fig = plt.figure(1, figsize=(7, 5), facecolor="#ffffff") # , dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    mycolormap = choose_cmap()
    colors = mycolormap(range(len(interesting_cads)))
    colors = ['#b30047','#ff6600','#009900','#3366ff']

    for element in range(len(interesting_cads)):

        temp_array = np.abs(y - interesting_cads[element])
        where_interesting = np.where(np.abs(temp_array == np.min(temp_array)))[0]

        if len(where_interesting) > 1:
            print('check where_interesting array')
            import pdb; pdb.set_trace()

        else:
            z_array = z_col[where_interesting[0]]

        # calculate cumulative array of probability distribution
        temp_z = []
        for el in range(len(z_array)):
            temp_z.append(np.sum(z_array[0:el]))
        z_array = np.array(temp_z,dtype='float')

        #import pdb; pdb.set_trace()
        # ax.bar(x[0], z_array, width=np.concatenate((np.diff(x[0]),[np.diff(x[0])[0]])),
        #        color='None', edgecolor=clr.rgb2hex(colors[element]), align='edge', lw=1,
        #        label=interesting_labels[element], alpha=0.5, rasterized=True) #ccffef
        ax.step(x[0], z_array, color=clr.rgb2hex(colors[element]), lw=1,
                label=interesting_labs[element], alpha=1.0, where='post', rasterized=True)  # ccffef
        ax.plot([0, 0], [0, 1.05], '-', lw=0.75, c='#000000')
        ax.set_xlim(test_lims)
        ax.set_ylim([0, 1.05])
        ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
        ax.set_ylabel('Posterior Likelihood', fontsize=font_size, style='normal', family='sans-serif')
        ax.set_title('Sample Size: ' + str(sample_size), fontsize=font_size)
        ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
        ax.tick_params(axis='x', length=0.5)

        plt.legend(loc='upper left')  # ,fontsize=font_size)
        plt.tight_layout()
    # print('Saving Interesting Cadences Plot...')
    # plt.savefig(save_as, dpi=300)
    # plt.close()
    plt.show()


#----------------------------------------
#----------------------------------------
# search_directories = ['All_Cadences_DavFittingDav7','All_Cadences_DavFittingDav8']
# mother_directory = 'DavFittingDav'
search_directories = ['All_Cadences_DavFittingJack7','All_Cadences_DavFittingJack8']
search_directories = ['All_Cadences_DavFittingJack_Throwaway']
mother_directory = 'DavFittingJack'
#----------------------------------------
#----------------------------------------
where_to_save = create_directories(mother_dir=mother_directory)
csv_filename_results = 'Results.csv'
csv_filename_cantfind = 'Cant_Find.csv'
csv_filename_weirdfit = 'Weird_Fit.csv'
savename_results = csv_filename_results.split('.')[0]
savename_cantfind = csv_filename_cantfind.split('.')[0]
savename_weirdfit = csv_filename_weirdfit.split('.')[0]
#----------------------------------------
#----------------------------------------
given = 'cadences (min)'  # 'modified impulsive index'
test = 'percent diff from true fwhm' # 'precent diff from true equivalent duration (noscatter)'
cad = 'cadences (min)'

given_plotlabel = 'Cadence (min)'  # 'Modified Impulsive Index'
test_plotlabel = '%Diff From True FWHM'# '%Diff From True Equivalent Duration (calculated without scatter)'

given_savelabel = 'Cadence'  # 'Modified_Impulsive_Index'
test_savelabel = 'FWHM' # 'EqDur'
# ----------------------------------------
# ----------------------------------------
results_columns = [given,test,cad]
eqdur_columns = ['calculated equivalent duration','true equivalent duration (noscatter)']
cantfind_columns = ['true equivalent duration (noscatter)']
weirdfit_columns = ['true equivalent duration (noscatter)']
# ----------------------------------------
# ----------------------------------------
given_dat = []
test_dat = []
cad_dat = []
eq_dur_fit = []
eq_dur_true_noscatter = []
eq_dur_cantfind = []
eq_dur_weirdfit = []
for dir in search_directories:
    csv_directory = dir
    path_to_csv = '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + csv_directory +'/'
    #----------------------------------------
    #----------------------------------------
    df_results, colnames = read_file(path_to_csv,csv_filename_results)
    try:
        df_cantfind, colnames_cantfind = read_file(path_to_csv, csv_filename_cantfind)
    except:
        pass
    try:
        df_weirdfit, colnames_weirdfit = read_file(path_to_csv, csv_filename_weirdfit)
    except:
        pass
    #----------------------------------------
    #----------------------------------------
    given_dat_dir, test_dat_dir, cad_dat_dir = extract_columns(input_dict=df_results, input_columns=results_columns)
    given_dat.append(np.array(given_dat_dir))
    test_dat.append(np.array(test_dat_dir))
    cad_dat.append(np.array(cad_dat_dir))
    try:
        eq_dur_fit_dir, eq_dur_true_noscatter_dir = extract_columns(input_dict=df_results, input_columns=eqdur_columns)
    except:
        pass
    else:
        eq_dur_fit.append(np.array(eq_dur_fit_dir))
        eq_dur_true_noscatter.append(np.array(eq_dur_true_noscatter_dir))

    try:
        eq_dur_cantfind_dir = extract_columns(input_dict=df_cantfind, input_columns=cantfind_columns)
    except:
        pass
    else:
        eq_dur_cantfind.append(np.array(eq_dur_cantfind_dir))

    try:
        eq_dur_weirdfit_dir = extract_columns(input_dict=df_weirdfit, input_columns=weirdfit_columns)
    except:
        pass
    else:
        eq_dur_weirdfit.append(np.array(eq_dur_weirdfit_dir))
#----------------------------------------
#----------------------------------------
given_dat = np.array(flatten_list(input_list=given_dat))
test_dat = np.array(flatten_list(input_list=test_dat))
cad_dat = np.array(flatten_list(input_list=cad_dat))
eq_dur_fit = np.array(flatten_list(input_list=eq_dur_fit))
eq_dur_true_noscatter = np.array(flatten_list(input_list=eq_dur_true_noscatter))
eq_dur_cantfind = np.array(flatten_list(input_list=eq_dur_cantfind))
eq_dur_cantfind = eq_dur_cantfind.flatten()
eq_dur_weirdfit = np.array(flatten_list(input_list=eq_dur_weirdfit))
eq_dur_weirdfit = eq_dur_weirdfit.flatten()
#----------------------------------------
#----------------------------------------
min_given = np.min(given_dat)
max_given = np.max(given_dat)
min_test = np.min(test_dat)
max_test = np.max(test_dat)

d_given = (max_given - min_given)/100
d_test = 2. # (max_test - min_test)/100

given_plotlims = [0,30.] # [0.,0.5]
test_plotlims = [-100.,100.]
#----------------------------------------
#----------------------------------------
interesting_cadences = [10./60., 1., 2., 30.] # minutes
interesting_labels = ['10 sec', '1 min', '2 min', '30 min']
#----------------------------------------
#----------------------------------------
input_dict = {'givenvar': [given_dat],
              'testvar': [test_dat],
              'cadvar': [cad_dat],
              'givenlabel': given_plotlabel,
              'testlabel': test_plotlabel,
              'givensavelabel': given_savelabel,
              'testsavelabel': given_savelabel,
              'mingiven': min_given,
              'maxgiven': max_given,
              'dgiven': d_given,
              'mintest': min_test,
              'maxtest': max_test,
              'dtest': d_test,
               }
#----------------------------------------
#----------------------------------------
do_for_specified_cadences = False
is_the_given_var_cadence = True
#----------------------------------------
#----------------------------------------
#likelihood_dict = do_bayes3(bayes_dict=input_dict)
if do_for_specified_cadences == True:
    for this_cad in range(len(interesting_cadences)):
        print('showing cadence: ' + str(np.round(interesting_cadences[this_cad],2)) + ' min')
        likelihood_dict = do_bayes4(bayes_dict=input_dict,interesting_cad=interesting_cadences[this_cad],
                                    is_given_cad=is_the_given_var_cadence)

        plot_posterior(lh_dict=likelihood_dict, given_lims=given_plotlims, test_lims=test_plotlims,
                       interesting_cad=interesting_cadences[this_cad], sample_size=len(given_dat),
                       is_given_cad=is_the_given_var_cadence)
        # plot_cumulative(lh_dict=likelihood_dict, given_lims=given_plotlims, test_lims=test_plotlims,
        #                 interesting_cad=interesting_cadences[this_cad], sample_size=len(given_dat),
        #                 is_given_cad=is_the_given_var_cadence)
#----------------------------------------
#----------------------------------------
if do_for_specified_cadences == False:
    likelihood_dict = do_bayes4(bayes_dict=input_dict, interesting_cad=interesting_cadences,
                                is_given_cad=is_the_given_var_cadence)

    plot_posterior(lh_dict=likelihood_dict, given_lims=given_plotlims, test_lims=test_plotlims,
                   interesting_cad=interesting_cadences, sample_size=len(given_dat),
                   is_given_cad=is_the_given_var_cadence)
    # plot_cumulative(lh_dict=likelihood_dict, given_lims=given_plotlims, test_lims=test_plotlims,
    #                 interesting_cad=interesting_cadences[this_cad], sample_size=len(given_dat),
    #                 is_given_cad=is_the_given_var_cadence)
# ----------------------------------------
# ----------------------------------------
if given == 'cadences (min)':
    plot_specific_cadences_posterior(lh_dict=likelihood_dict,interesting_cads=interesting_cadences,
                                     interesting_labs=interesting_labels,sample_size=len(given_dat),
                                     test_lims=test_plotlims)
    plot_specific_cadences_cumulative(lh_dict=likelihood_dict,interesting_cads=interesting_cadences,
                                      interesting_labs=interesting_labels,sample_size=len(given_dat),
                                      test_lims=test_plotlims)
#----------------------------------------
#----------------------------------------
#----------------------------------------
#----------------------------------------
#----------------------------------------
#----------------------------------------
#----------------------------------------
#----------------------------------------
def model_func(t, A, K, C):
    return A * np.exp(-K * t) # + C
def linear_func(t, M, B):
    return M*t + B
def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = scipy.optimize.curve_fit(model_func, t, y, p0=(0.9,0.002, 0.), maxfev=1000)
    A, K, C = opt_parms
    return A, K, C
def fit_linear(t, y):
    opt_parms, parm_cov = scipy.optimize.curve_fit(linear_func, t, y) #, maxfev=1000)
    M, B = opt_parms
    return M, B
def plot_ffd(eq_dur_fit, eq_dur_true_noscatter, eq_dur_cantfind, eq_dur_weirdfit, search_directories, do_energy=False):


    d_eq_dur_fit = (np.max(eq_dur_fit) - np.min(eq_dur_fit))/50
    eqdur_grid = np.arange(np.min(eq_dur_fit), np.max(eq_dur_fit) + d_eq_dur_fit, d_eq_dur_fit)
    d_eq_dur_true_noscatter = (np.max(eq_dur_true_noscatter) - np.min(eq_dur_true_noscatter)) / 50
    eqdur_grid_noscatter = np.arange(np.min(eq_dur_true_noscatter), np.max(eq_dur_true_noscatter) + d_eq_dur_true_noscatter, d_eq_dur_true_noscatter)

    L_V830Tau = 3.827e33
    eqdur_grid2 = eqdur_grid*L_V830Tau
    eqdur_grid_noscatter2 = eqdur_grid_noscatter*L_V830Tau
    eq_dur_fit2 = eq_dur_fit*L_V830Tau
    eq_dur_true_noscatter2 = eq_dur_true_noscatter*L_V830Tau

    #import pdb; pdb.set_trace()

    if len(eq_dur_cantfind) > 0:
        eq_dur_cantfind2 = np.array(eq_dur_cantfind) * L_V830Tau
        if len(eq_dur_cantfind2) <= len(search_directories):
            eq_dur_cantfind2 = np.concatenate((eq_dur_cantfind2))
    else:
        eq_dur_cantfind2 = []
    if len(eq_dur_weirdfit) > 0:
        eq_dur_weirdfit2 = np.array(eq_dur_weirdfit) * L_V830Tau
        if len(eq_dur_weirdfit2) <= len(search_directories):
            eq_dur_weirdfit2 = np.concatenate((eq_dur_weirdfit2))
    else:
        eq_dur_weirdfit2 = []

    #import pdb; pdb.set_trace()
    cumulative_eqdur = []
    cumulative_eqdur_noscatter = []

    if do_energy == True:
        for grid_element in range(len(eqdur_grid2)):
            where_above = np.where(eq_dur_fit2 >= eqdur_grid2[grid_element])[0]
            where_above_cantfind = np.where(eq_dur_cantfind2 >= eqdur_grid2[grid_element])[0]
            where_above_weirdfit = np.where(eq_dur_weirdfit2 >= eqdur_grid2[grid_element])[0]
            cumulative_eqdur.append(np.float(len(where_above))/(np.float(len(eq_dur_fit2)) + len(where_above_cantfind) + len(where_above_weirdfit)))
        for grid_element_noscatter in range(len(eqdur_grid_noscatter2)):
            where_above_noscatter = np.where(eq_dur_true_noscatter2 >= eqdur_grid_noscatter2[grid_element_noscatter])[0]
            where_above_noscatter_cantfind = np.where(eq_dur_cantfind2 >= eqdur_grid_noscatter2[grid_element_noscatter])[0]
            where_above_noscatter_weirdfit = np.where(eq_dur_weirdfit2 >= eqdur_grid_noscatter2[grid_element_noscatter])[0]
            cumulative_eqdur_noscatter.append(np.float(len(where_above_noscatter)) / (np.float(len(eq_dur_true_noscatter2)) + len(where_above_noscatter_cantfind) + len(where_above_noscatter_weirdfit)))

        A, K, C = fit_exp_nonlinear(eqdur_grid2, cumulative_eqdur)
        fit_cumulative = model_func(eqdur_grid2, A, K, C)
        print(np.log(A) * K)

    if do_energy == False:
        for grid_element in range(len(eqdur_grid)):
            where_above = np.where(eq_dur_fit >= eqdur_grid[grid_element])[0]
            where_above_cantfind = np.where(eq_dur_cantfind >= eqdur_grid[grid_element])[0]
            where_above_weirdfit = np.where(eq_dur_weirdfit >= eqdur_grid[grid_element])[0]
            cumulative_eqdur.append(np.float(len(where_above))/(np.float(len(eq_dur_fit)) + len(where_above_cantfind) + len(where_above_weirdfit)))
        for grid_element_noscatter in range(len(eqdur_grid_noscatter)):
            where_above_noscatter = np.where(eq_dur_true_noscatter >= eqdur_grid_noscatter[grid_element_noscatter])[0]
            where_above_noscatter_cantfind = np.where(eq_dur_cantfind >= eqdur_grid_noscatter[grid_element_noscatter])[0]
            where_above_noscatter_weirdfit = np.where(eq_dur_weirdfit >= eqdur_grid_noscatter[grid_element_noscatter])[0]
            cumulative_eqdur_noscatter.append(np.float(len(where_above_noscatter)) / (np.float(len(eq_dur_true_noscatter)) + len(where_above_noscatter_cantfind) + len(where_above_noscatter_weirdfit)))

        A, K, C = fit_exp_nonlinear(eqdur_grid, cumulative_eqdur)
        fit_cumulative = model_func(eqdur_grid, A, K, C)
        print(np.log(A)*K)

    # eqdur_grid2 = np.log10(eqdur_grid2)
    # eqdur_grid_noscatter2 = np.log10(eqdur_grid_noscatter2)
    # fit_cumulative = np.log10(fit_cumulative)
    # cumulative_eqdur = np.log10(cumulative_eqdur)
    # cumulative_eqdur_noscatter = np.log10(cumulative_eqdur_noscatter)

    # where_noinf = np.where(eqdur_grid2 > 0)[0]
    # eqdur_grid2 = eqdur_grid2[where_noinf]
    # cumulative_eqdur = np.array(cumulative_eqdur)[where_noinf]
    # fit_cumulative = fit_cumulative[where_noinf]

    font_size = 'medium'
    fig = plt.figure(1, figsize=(7, 5), facecolor="#ffffff") # , dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    #import pdb; pdb.set_trace()

    # M, B = fit_linear(eqdur_grid2, cumulative_eqdur)
    # fit_cumulative_linear = linear_func(eqdur_grid2, M, B)

    # import pdb; pdb.set_trace()

    if do_energy == True:
        # ax.plot(np.log10(eqdur_grid2), cumulative_eqdur, color='#ff0066', lw=2, label='determined with flare fit')
        # ax.plot(np.log10(eqdur_grid_noscatter2), cumulative_eqdur_noscatter, color='#33cc00', lw=2, label='true')
        ax.plot(eqdur_grid2, cumulative_eqdur, color='#ff0066', lw=2, label='determined with flare fit')
        ax.plot(eqdur_grid_noscatter2, cumulative_eqdur_noscatter, color='#33cc00', lw=2, label='true')
        # ax.plot(eqdur_grid2, fit_cumulative, color='blue', lw=2,
        #         label='Fitted Function:\n$y = %0.2f e^{%0.6f x} + %0.2f$' % (A, K, C))
        #ax.set_xlim(np.min(np.log10(eqdur_grid2)), np.max(np.log10(eqdur_grid2)))
        ax.set_xlim(np.min(eqdur_grid2), np.max(eqdur_grid2))
    if do_energy == False:
        ax.plot(eqdur_grid, cumulative_eqdur, color='#ff0066', lw=2, label='determined with flare fit')
        ax.plot(eqdur_grid_noscatter, cumulative_eqdur_noscatter, color='#33cc00', lw=2, label='true')
        # ax.scatter(eqdur_grid, cumulative_eqdur, color='#b30086', s=np.pi*1**2, label='determined with flare fit')
        # ax.scatter(eqdur_grid_noscatter, cumulative_eqdur_noscatter, color='#ff6600', s=np.pi*1**2, label='true')
        # ax.step(eqdur_grid, cumulative_eqdur, color='#b30086', lw=1, alpha=1.0, where='post', rasterized=True,
        #         label='determined with flare fit')  # ccffef
        # ax.step(eqdur_grid_noscatter, cumulative_eqdur_noscatter, color='#ff6600', lw=1, alpha=1.0, where='post',
        #         rasterized=True,
        #         label='true')  # ccffef
        ax.plot(eqdur_grid, fit_cumulative, color='blue', lw=2,
                label='Fitted Function:\n$y = %0.2f e^{%0.6f x} + %0.2f$' % (A, K, C))
        ax.set_xlim(np.min(eqdur_grid),np.max(eqdur_grid))
    # ax.set_ylim([0, 1.1*np.max(cumulative_eqdur)])
    ax.set_xlabel('Equivalent Duration', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_ylabel('Occurrence', fontsize=font_size, style='normal', family='sans-serif')
    # ax.set_title('Eq Dur:  ' + str(int(test_diff)) + ' – ' + str(int(test_diff+1)) + '% Diff From True Value', fontsize='large', style='normal', family='sans-serif')
    # ax.set_title(save_as.split('/')[-1], fontsize=font_size, style='normal', family='sans-serif')
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    ax.tick_params(axis='x', length=0.5)

    plt.legend(loc='upper right')  # ,fontsize=font_size)
    plt.tight_layout()
    # print('Saving Interesting Cadences Plot...')
    # plt.savefig(save_as, dpi=300)
    # plt.close()
    plt.show()

print(eq_dur_cantfind, eq_dur_weirdfit)

plot_ffd(eq_dur_fit, eq_dur_true_noscatter, eq_dur_cantfind, eq_dur_weirdfit, search_directories, do_energy=False)















#
# def make_cmap(colors, position=None, bit=False):
#     '''
#     make_cmap takes a list of tuples which contain RGB values. The RGB
#     values may either be in 8-bit [0 to 255] (in which bit must be set to
#     True when called) or arithmetic [0 to 1] (default). make_cmap returns
#     a cmap with equally spaced colors.
#     Arrange your tuples so that the first color is the lowest value for the
#     colorbar and the last is the highest.
#     position contains values from 0 to 1 to dictate the location of each color.
#     '''
#     import matplotlib as mpl
#     import numpy as np
#     bit_rgb = np.linspace(0,1,256)
#     if position == None:
#         position = np.linspace(0,1,len(colors))
#     else:
#         if len(position) != len(colors):
#             sys.exit("position length must be the same as colors")
#         elif position[0] != 0 or position[-1] != 1:
#             sys.exit("position must start with 0 and end with 1")
#     if bit:
#         for i in range(len(colors)):
#             colors[i] = (bit_rgb[colors[i][0]],
#                          bit_rgb[colors[i][1]],
#                          bit_rgb[colors[i][2]])
#     cdict = {'red':[], 'green':[], 'blue':[]}
#     for pos, color in zip(position, colors):
#         cdict['red'].append((pos, color[0], color[0]))
#         cdict['green'].append((pos, color[1], color[1]))
#         cdict['blue'].append((pos, color[2], color[2]))
#
#     cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
#     return cmap
# def choose_cmap():
#     # colors1 = [(102/255, 0/255, 204/255), (255/255, 128/255, 0/255), (0/255, 153/255, 153/255)]
#     # colors2 = [(204/255, 0/255, 102/255), (51/255, 51/255, 204/255), (153/255, 204/255, 0/255)]
#     # colors3 = [(128/255, 0/255, 64/255),(51/255, 51/255, 204/255),(0/255, 255/255, 153/255)]
#     # colors4 = [(255/255, 255/255, 255/255),(0/255, 255/255, 204/255),(0/255, 153/255, 204/255),(0/255, 153/255, 255/255),(102/255, 0/255, 204/255)]
#     # colors5 = [(255/255, 255/255, 255/255),(153/255, 255/255, 153/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255)]
#     # colors6 = [(255/255, 255/255, 255/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255),(0/255, 0/255, 77/255)]
#     colors7 = [(255 / 255, 255 / 255, 255 / 255), (255 / 255, 204 / 255, 0 / 255), (255 / 255, 0 / 255, 102 / 255), (134 / 255, 0 / 255, 179 / 255), (0 / 255, 0 / 255, 77 / 255)]
#     # colors8 = [(255 / 255, 255 / 255, 255 / 255), (255 / 255, 0 / 255, 102 / 255), (153 / 255, 0 / 255, 204 / 255), (0 / 255, 0 / 255, 77 / 255)]
#     # colors9 = [(255/255, 255/255, 255/255),(255/255, 204/255, 0/255),(255/255, 0/255, 102/255),(115/255, 0/255, 123/255)]
#     colors10 = [(0 / 255, 255 / 255, 204 / 255), (255 / 255, 204 / 255, 0 / 255), (255 / 255, 0 / 255, 102 / 255), (134 / 255, 0 / 255, 179 / 255), (0 / 255, 0 / 255, 77 / 255)]
#     colors11 = [(0 / 255, 102 / 255, 153 / 255), (255 / 255, 204 / 255, 0 / 255), (255 / 255, 0 / 255, 102 / 255), (134 / 255, 0 / 255, 179 / 255), (0 / 255, 0 / 255, 77 / 255)]
#
#
#     # position = [0, 0.5, 1]
#     # position2 = [0, 0.25, 0.5, 0.75, 1]
#     position2_2 = [0, 0.25, 0.5, 0.75, 1]
#     # position3 = [0, 1./3., 2./3., 1]
#     mycolormap = make_cmap(colors11, position=position2_2)
#
#     return mycolormap
#
# def condense_filedata(path_to_csv, directories, data_type):
#
#     for dir in range(len(directories)):
#         file = glob.glob(path_to_csv + directories[dir] + data_type + '.csv')
#         csv_filename = file.split('/')[-1]
#         savename = file.split('/')[-1].split('.')[0]  # + '.pdf'
#
#         if savename == 'Results':
#             continue
#
#         print('Opening ' + savename + '...')
#
#         df, xlab, ylab, zlab = read_file(csv_filename)
#         plot_posterior(df, xlab, ylab, zlab, save_as=path_to_csv + '/Figures/' + savename)
#
#         if ylab == 'Cadence (min)':
#             savename_interesting = savename + '_interesting_ones'  # .pdf'
#             plot_specific(df, xlab, ylab, zlab, interesting_cadences, interesting_labels,
#                           save_as=path_to_csv + '/Figures/' + savename_interesting)
#
#
# def read_file(csv_filename):
#
#     df = pd.read_csv(path_to_csv + csv_filename)
#
#     colnames = df.columns
#
#     return df,colnames
# def convert_to_float(temp):
#     temp1 = temp
#     temp1 = temp1.replace('\n', '')
#     temp2 = ''
#     for fleeb in range(len(temp1)):
#         if temp[fleeb] != ' ':
#             temp2 += temp1[fleeb]
#         if temp1[fleeb] == ' ':
#             if (temp2[-1] == ',') or (temp2[-1] == '['):
#                 continue
#             else:
#                 temp2 += ','
#     fixed_dat = np.array(literal_eval(temp2))
#     if len(fixed_dat) == 1:
#         fixed_dat = fixed_dat[0]
#
#     return fixed_dat
# # def extract_columns(input_dict, given_col, test_col, cad_col):
# #
# #     print('Extracting Columns...')
# #
# #     if type(input_dict[given_col][0]) == 'str':
# #         for foo in range(len(input_dict[given_col])):
# #             input_dict[given_col][foo] = convert_to_float(input_dict[given_col][foo])
# #     if type(input_dict[test_col][0]) == 'str':
# #         for foo in range(len(input_dict[test_col])):
# #             input_dict[test_col][foo] = convert_to_float(input_dict[test_col][foo])
# #     if type(input_dict[cad_col][0]) == 'str':
# #         for foo in range(len(input_dict[cad_col])):
# #             input_dict[cad_col][foo] = convert_to_float(input_dict[cad_col][foo])
# #
# #     return input_dict[given_col], input_dict[test_col], input_dict[cad_col]
# def extract_columns(input_dict, input_columns):
#
#     print('Extracting Columns...')
#
#     for col in range(len(input_columns)):
#         if type(input_dict[input_columns[col]][0]) == 'str':
#             for foo in range(len(input_dict[input_columns[col]])):
#                 input_dict[input_columns[col]][foo] = convert_to_float(input_dict[input_columns[col]][foo])
#
#     extracted_columns = []
#     for excol in range(len(input_columns)):
#         extracted_columns.append(input_dict[input_columns[excol]])
#
#     return extracted_columns
# def do_bayes3(bayes_dict):
#
#     given_variable = np.array(bayes_dict['givenvar'])[0].flatten()
#     test_variable = np.array(bayes_dict['testvar'])[0].flatten()
#     given_label = bayes_dict['givenlabel']
#     test_label = bayes_dict['testlabel']
#
#     min_given = bayes_dict['mingiven']
#     max_given = bayes_dict['maxgiven']
#     d_given = bayes_dict['dgiven']
#     min_test = bayes_dict['mintest']
#     max_test = bayes_dict['maxtest']
#     d_test = bayes_dict['dtest']
#
#
#
#
#     #import pdb; pdb.set_trace()
#
#     # ------------- given variable fraction ------------- #
#
#     # d_given = 1. # minutes
#     #cadence_bins = np.arange(min_cadence,max_cadence+d_cadence,d_cadence)
#     given_bins = np.arange(min_given, max_given + d_given, d_given)
#
#     P_givens = []
#
#     for i_cad in range(len(given_bins)-1):
#
#         where_given = np.where((given_variable > given_bins[i_cad]) & (given_variable <= given_bins[i_cad+1]))[0]
#
#         num_given = np.float(len(where_given))
#         tot_given = np.float(len(given_variable))
#
#         P_given = num_given/tot_given
#         P_givens.append(P_given)
#
#
#     # ------------- test variable fraction ------------- #
#
#     # d_test = 1  # 10 seconds to minutes
#     test_bins = np.arange(min_test, max_test + d_test, d_test)
#
#     P_givenvar_given_testvar = []
#
#     P_tests = []
#
#     for i_test in range(len(test_bins) - 1):
#         where_test = np.where((test_variable > test_bins[i_test]) & (test_variable <= test_bins[i_test + 1]))[0]
#
#         num_test = np.float(len(where_test))
#         tot_test = np.float(len(test_variable))
#
#         P_test = num_test / tot_test
#         P_tests.append(P_test)
#
#
#         # ------------- cad given eqdur ------------- #
#
#         givens_per_test = []
#
#         for i_cad2 in range(len(given_bins)-1):
#
#             test_givens = np.array(given_variable)[where_test]
#             where_test2 = np.where((test_givens >= given_bins[i_cad2]) & (test_givens < given_bins[i_cad2+1]))[0]
#
#             if num_test == 0:
#                 givens_per_test.append(0)
#             else:
#                 givens_per_test.append(np.float(len(where_test2))/num_test)
#
#         P_givenvar_given_testvar.append(np.array(givens_per_test))
#
#
#     # ------------- eq_dur given cad ------------- #
#
#     P_testvar_given_givenvar = []
#
#     for i_cad3 in range(len(given_bins) - 1):
#
#         tests_per_given = []
#
#         for i_test2 in range(len(test_bins) - 1):
#
#             if P_givens[i_cad3] == 0:
#                 P_Bayes = 0
#             else:
#                 P_Bayes = (P_givenvar_given_testvar[i_test2][i_cad3] * P_tests[i_test2]) / P_givens[i_cad3]
#                 # if P_Bayes > 0:
#                 #     print(P_Bayes)
#
#             tests_per_given.append(P_Bayes)
#
#         # if np.sum(tests_per_given) > 0:
#         #     P_testvar_given_givenvar.append(np.array(tests_per_given)/np.sum(tests_per_given))
#         # else:
#         #     P_testvar_given_givenvar.append(np.array(tests_per_given))
#
#         P_testvar_given_givenvar.append(np.array(tests_per_given))
#
#
#
#     # import pdb; pdb.set_trace()
#
#     #print('Saving Bayes Table To CSV...')
#     table_dat1 = []
#     table_dat2 = []
#     for mlem in range(len(P_testvar_given_givenvar)):
#         table_dat1.append(np.array(test_bins[:-1]))
#         table_dat2.append(np.array(P_testvar_given_givenvar[mlem]))
#         #print(P_testvar_given_givenvar[mlem][P_testvar_given_givenvar[mlem] > 0])
#
#     d = {given_label : given_bins[:-1],
#          test_label : table_dat1,
#          'Likelihood Dist' : table_dat2,
#          }
#     l_dict = pd.DataFrame(data=d)
#     # df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as + '_' + test_save_label + '_given_' + given_save_label + '.csv', index=False)
#
#     # import pdb; pdb.set_trace()
#
#     return l_dict
# def do_bayes4(bayes_dict,interesting_cad,is_given_cad=False):
#
#     given_variable_in = np.array(bayes_dict['givenvar'])[0].flatten()
#     test_variable_in = np.array(bayes_dict['testvar'])[0].flatten()
#     cadence_variable = np.array(bayes_dict['cadvar'])[0].flatten()
#     given_label = bayes_dict['givenlabel']
#     test_label = bayes_dict['testlabel']
#
#     min_given = bayes_dict['mingiven']
#     max_given = bayes_dict['maxgiven']
#     d_given = bayes_dict['dgiven']
#     min_test = bayes_dict['mintest']
#     max_test = bayes_dict['maxtest']
#     d_test = bayes_dict['dtest']
#
#     if is_given_cad == False:
#         where_cad = np.where((cadence_variable >= interesting_cad-0.05) & (cadence_variable <= interesting_cad+0.05))[0]
#
#         given_variable = given_variable_in[where_cad]
#         test_variable = test_variable_in[where_cad]
#     if is_given_cad == True:
#         given_variable = given_variable_in
#         test_variable = test_variable_in
#
#
#     #import pdb; pdb.set_trace()
#
#     # ------------- given variable fraction ------------- #
#
#     # d_given = 1. # minutes
#     #cadence_bins = np.arange(min_cadence,max_cadence+d_cadence,d_cadence)
#     given_bins = np.arange(min_given, max_given + d_given, d_given)
#
#     P_givens = []
#
#     for i_cad in range(len(given_bins)-1):
#
#         where_given = np.where((given_variable > given_bins[i_cad]) & (given_variable <= given_bins[i_cad+1]))[0]
#
#         num_given = np.float(len(where_given))
#         tot_given = np.float(len(given_variable))
#
#         P_given = num_given/tot_given
#         P_givens.append(P_given)
#
#
#     # ------------- test variable fraction ------------- #
#
#     # d_test = 1  # 10 seconds to minutes
#     test_bins = np.arange(min_test, max_test + d_test, d_test)
#
#     P_givenvar_given_testvar = []
#
#     P_tests = []
#
#     for i_test in range(len(test_bins) - 1):
#         where_test = np.where((test_variable > test_bins[i_test]) & (test_variable <= test_bins[i_test + 1]))[0]
#
#         num_test = np.float(len(where_test))
#         tot_test = np.float(len(test_variable))
#
#         P_test = num_test / tot_test
#         P_tests.append(P_test)
#
#
#         # ------------- cad given eqdur ------------- #
#
#         givens_per_test = []
#
#         for i_cad2 in range(len(given_bins)-1):
#
#             test_givens = np.array(given_variable)[where_test]
#             where_test2 = np.where((test_givens >= given_bins[i_cad2]) & (test_givens < given_bins[i_cad2+1]))[0]
#
#             if num_test == 0:
#                 givens_per_test.append(0)
#             else:
#                 givens_per_test.append(np.float(len(where_test2))/num_test)
#
#         P_givenvar_given_testvar.append(np.array(givens_per_test))
#
#
#     # ------------- eq_dur given cad ------------- #
#
#     P_testvar_given_givenvar = []
#
#     for i_cad3 in range(len(given_bins) - 1):
#
#         tests_per_given = []
#
#         for i_test2 in range(len(test_bins) - 1):
#
#             if P_givens[i_cad3] == 0:
#                 P_Bayes = 0
#             else:
#                 P_Bayes = (P_givenvar_given_testvar[i_test2][i_cad3] * P_tests[i_test2]) / P_givens[i_cad3]
#                 # if P_Bayes > 0:
#                 #     print(P_Bayes)
#
#             tests_per_given.append(P_Bayes)
#
#         # if np.sum(tests_per_given) > 0:
#         #     P_testvar_given_givenvar.append(np.array(tests_per_given)/np.sum(tests_per_given))
#         # else:
#         #     P_testvar_given_givenvar.append(np.array(tests_per_given))
#
#         P_testvar_given_givenvar.append(np.array(tests_per_given))
#
#
#
#     # import pdb; pdb.set_trace()
#
#     #print('Saving Bayes Table To CSV...')
#     table_dat1 = []
#     table_dat2 = []
#     for mlem in range(len(P_testvar_given_givenvar)):
#         table_dat1.append(np.array(test_bins[:-1]))
#         table_dat2.append(np.array(P_testvar_given_givenvar[mlem]))
#         #print(P_testvar_given_givenvar[mlem][P_testvar_given_givenvar[mlem] > 0])
#
#     d = {given_label : given_bins[:-1],
#          test_label : table_dat1,
#          'Likelihood Dist' : table_dat2,
#          }
#     l_dict = pd.DataFrame(data=d)
#     # df.to_csv('/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_as + '_' + test_save_label + '_given_' + given_save_label + '.csv', index=False)
#
#     # import pdb; pdb.set_trace()
#
#     return l_dict
#
# def plot_posterior(lh_dict,given_lims,test_lims,interesting_cad,is_given_cad=False):
#     cols = lh_dict.columns
#     xlab = cols[1]
#     ylab = cols[0]
#     zlab = cols[2]
#
#     font_size = 'large'
#
#     y = np.array(lh_dict[ylab])
#     x = np.array(lh_dict[xlab])
#     z = np.zeros((len(x[0]),len(y)))
#
#     # import pdb; pdb.set_trace()
#
#     for row in range(len(lh_dict)):
#
#         z_array = np.array(lh_dict[zlab][row])
#
#         z[:, row] = np.array(z_array, dtype='float')
#
#     #import pdb; pdb.set_trace()
#
#     fig = plt.figure(1, figsize=(7, 5.5), facecolor="#ffffff")  # , dpi=300)
#     ax = fig.add_subplot(111)
#     ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
#     ax.set_ylabel(ylab, fontsize=font_size, style='normal', family='sans-serif')
#     ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
#
#     xmesh, ymesh = np.meshgrid(x[0], y)
#     mycolormap = choose_cmap()
#     cmap = plt.get_cmap(mycolormap, z.max() + 1)
#     z = np.ma.masked_where(z == 0, z)
#     cmap.set_bad(color='white')
#
#     # import pdb; pdb.set_trace()
#
#     pcolor = ax.pcolormesh(xmesh, ymesh, z.T, cmap = cmap, shading='nearest', rasterized=True)
#     cb = fig.colorbar(pcolor) #, ax=ax) #, cmap=cmap)
#     cb.ax.tick_params(labelsize=font_size)
#     cb.ax.set_ylabel('Posterior Probability', fontsize=font_size, rotation=270)
#     cb.ax.get_yaxis().labelpad = 15
#     ax.plot([0,0],[0,np.max(y)],'-',lw=0.75,c='#000000')
#
#     ax.set_xlim(test_lims)
#     ax.set_ylim(given_lims)
#     if is_given_cad == False:
#         ax.set_title('Cadence: ' + str(np.round(interesting_cad,2)) + ' min', fontsize=font_size)
#     plt.tight_layout()
#     # print('Saving 2D Posterior Likelihood Plot...')
#     # plt.savefig(save_as, dpi=300)
#     # plt.close()
#     plt.show()
# def plot_cumulative(lh_dict,given_lims,test_lims,interesting_cad,is_given_cad=False):
#     cols = lh_dict.columns
#     xlab = cols[1]
#     ylab = cols[0]
#     zlab = cols[2]
#
#     font_size = 'large'
#
#     y = np.array(lh_dict[ylab])
#     x = np.array(lh_dict[xlab])
#     z = np.zeros((len(x[0]),len(y)))
#
#     #import pdb; pdb.set_trace()
#
#     for row in range(len(lh_dict)):
#
#         z_array = np.array(lh_dict[zlab][row])
#
#         temp_z = []
#         for el in range(len(z_array)):
#             temp_z.append(np.sum(z_array[0:el]))
#
#         z[:,row] = np.array(temp_z,dtype='float')
#
#
#     fig = plt.figure(1, figsize=(7, 5.5), facecolor="#ffffff")  # , dpi=300)
#     ax = fig.add_subplot(111)
#     ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
#     ax.set_ylabel(ylab, fontsize=font_size, style='normal', family='sans-serif')
#     ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
#
#     xmesh, ymesh = np.meshgrid(x[0], y)
#     mycolormap = choose_cmap()
#     cmap = plt.get_cmap(mycolormap, z.max() + 1)
#     z = np.ma.masked_where(z == 0, z)
#     cmap.set_bad(color='white')
#
#     #import pdb; pdb.set_trace()
#
#     pcolor = ax.pcolormesh(xmesh, ymesh, z.T, cmap = cmap, shading='nearest', rasterized=True)
#     cb = fig.colorbar(pcolor) #, ax=ax) #, cmap=cmap)
#     cb.ax.tick_params(labelsize=font_size)
#     cb.ax.set_ylabel('Posterior Probability', fontsize=font_size, rotation=270)
#     cb.ax.get_yaxis().labelpad = 15
#     ax.plot([0,0],[0,np.max(y)],'-',lw=0.75,c='#000000')
#
#     ax.set_xlim(test_lims)
#     ax.set_ylim(given_lims)
#     if is_given_cad == False:
#         ax.set_title('Cadence: ' + str(np.round(interesting_cad,2)) + ' min', fontsize=font_size)
#     plt.tight_layout()
#     # print('Saving 2D Cumulative Likelihood Plot...')
#     # plt.savefig(save_as, dpi=300)
#     # plt.close()
#     plt.show()
#
# def plot_specific_cadences_posterior(lh_dict,interesting_cads,interesting_labs,test_lims):
#
#     cols = lh_dict.columns
#     xlab = cols[1]
#     ylab = cols[0]
#     zlab = cols[2]
#
#     y = np.array(lh_dict[ylab])
#     x = np.array(lh_dict[xlab])
#     z_col = np.array(lh_dict[zlab])
#
#     font_size = 'medium'
#     fig = plt.figure(1, figsize=(7, 5), facecolor="#ffffff") # , dpi=300)
#     ax = fig.add_subplot(1, 1, 1)
#     mycolormap = choose_cmap()
#     colors = mycolormap(range(len(interesting_cads)))
#     colors = ['#b30047','#ff6600','#009900','#3366ff']
#     patterns = ['//','\\','.','']
#     alphas = [0.8,0.6,0.4,0.2]
#
#     ymax = 0
#
#     for element in range(len(interesting_cads)):
#
#         temp_array = np.abs(y - interesting_cads[element])
#         where_interesting = np.where(np.abs(temp_array == np.min(temp_array)))[0]
#
#         if len(where_interesting) > 1:
#             print('check where_interesting array')
#             import pdb; pdb.set_trace()
#
#         else:
#             z_array = z_col[where_interesting[0]]
#
#         #import pdb; pdb.set_trace()
#         ax.bar(x[0], z_array, width=np.concatenate((np.diff(x[0]),[np.diff(x[0])[0]])),
#                color=clr.rgb2hex(colors[element]), edgecolor='None', align='edge', lw=1,
#                label=interesting_labs[element], alpha=0.1, rasterized=True) #ccffef
#         ax.step(x[0], z_array, color=clr.rgb2hex(colors[element]), lw=1, alpha=0.8, where='post', rasterized=True)  # ccffef
#         if np.max(z_array) >= ymax:
#             ymax = np.max(z_array)
#     ax.set_xlim(test_lims)
#     ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
#     ax.set_ylabel('Posterior Likelihood', fontsize=font_size, style='normal', family='sans-serif')
#     ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
#     ax.tick_params(axis='x', length=0.5)
#     ax.plot([0, 0], [0, 1.1*ymax], '-', lw=0.75, c='#000000')
#     ax.set_ylim([0, 1.1 * ymax])
#     plt.legend(loc='upper left')  # ,fontsize=font_size)
#     plt.tight_layout()
#     # print('Saving Interesting Cadences Plot...')
#     # plt.savefig(save_as, dpi=300)
#     # plt.close()
#     plt.show()
# def plot_specific_cadences_cumulative(lh_dict,interesting_cads,interesting_labs,test_lims):
#
#     cols = lh_dict.columns
#     xlab = cols[1]
#     ylab = cols[0]
#     zlab = cols[2]
#
#     y = np.array(lh_dict[ylab])
#     x = np.array(lh_dict[xlab])
#     z_col = np.array(lh_dict[zlab])
#
#     font_size = 'medium'
#     fig = plt.figure(1, figsize=(7, 5), facecolor="#ffffff") # , dpi=300)
#     ax = fig.add_subplot(1, 1, 1)
#     mycolormap = choose_cmap()
#     colors = mycolormap(range(len(interesting_cads)))
#     colors = ['#b30047','#ff6600','#009900','#3366ff']
#
#     for element in range(len(interesting_cads)):
#
#         temp_array = np.abs(y - interesting_cads[element])
#         where_interesting = np.where(np.abs(temp_array == np.min(temp_array)))[0]
#
#         if len(where_interesting) > 1:
#             print('check where_interesting array')
#             import pdb; pdb.set_trace()
#
#         else:
#             z_array = z_col[where_interesting[0]]
#
#         # calculate cumulative array of probability distribution
#         temp_z = []
#         for el in range(len(z_array)):
#             temp_z.append(np.sum(z_array[0:el]))
#         z_array = np.array(temp_z,dtype='float')
#
#         #import pdb; pdb.set_trace()
#         # ax.bar(x[0], z_array, width=np.concatenate((np.diff(x[0]),[np.diff(x[0])[0]])),
#         #        color='None', edgecolor=clr.rgb2hex(colors[element]), align='edge', lw=1,
#         #        label=interesting_labels[element], alpha=0.5, rasterized=True) #ccffef
#         ax.step(x[0], z_array, color=clr.rgb2hex(colors[element]), lw=1,
#                 label=interesting_labs[element], alpha=1.0, where='post', rasterized=True)  # ccffef
#         ax.plot([0, 0], [0, 1.05], '-', lw=0.75, c='#000000')
#         ax.set_xlim(test_lims)
#         ax.set_ylim([0, 1.05])
#         ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
#         ax.set_ylabel('Posterior Likelihood', fontsize=font_size, style='normal', family='sans-serif')
#         # ax.set_title('Eq Dur:  ' + str(int(test_diff)) + ' – ' + str(int(test_diff+1)) + '% Diff From True Value', fontsize='large', style='normal', family='sans-serif')
#         # ax.set_title(save_as.split('/')[-1], fontsize=font_size, style='normal', family='sans-serif')
#         ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
#         ax.tick_params(axis='x', length=0.5)
#
#         plt.legend(loc='upper left')  # ,fontsize=font_size)
#         plt.tight_layout()
#     # print('Saving Interesting Cadences Plot...')
#     # plt.savefig(save_as, dpi=300)
#     # plt.close()
#     plt.show()
#
#
# search_directories = ['All_Cadences_DavFittingJack6','All_Cadences_DavFittingJack7']
# #----------------------------------------
# #----------------------------------------
# # save_directory = 'All_Cadences_DavFittingJack5'
# save_directory = 'All_Cadences_DavFittingJack7'
# path_to_csv = '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/' + save_directory +'/'
# #----------------------------------------
# #----------------------------------------
# if not os.path.exists(path_to_csv + '/Figures/'):
#     os.mkdir(path_to_csv + '/Figures/')
# #----------------------------------------
# #----------------------------------------
# csv_filename = 'Results.csv'
# savename = csv_filename.split('.')[0]
# print('Opening ' + savename + '...')
# #----------------------------------------
# #----------------------------------------
# df, colnames = read_file(csv_filename)
# #----------------------------------------
# #----------------------------------------
# given = 'cadences (min)' # 'modified impulsive index'
# test = 'precent diff from true equivalent duration (noscatter)'
# cad = 'cadences (min)'
#
# given_plotlabel = 'Cadence (min)' # 'Modified Impulsive Index'
# test_plotlabel = '%Diff From True Equivalent Duration (calculated without scatter)'
#
# given_savelabel = 'Cadence' # 'Modified_Impulsive_Index'
# test_savelabel = 'EqDur'
# #----------------------------------------
# #----------------------------------------
# in_columns = [given,test,cad]
# given_dat, test_dat, cad_dat = extract_columns(input_dict=df, input_columns=in_columns)
#
# min_given = np.min(given_dat)
# max_given = np.max(given_dat)
# min_test = np.min(test_dat)
# max_test = np.max(test_dat)
#
# d_given = (max_given - min_given)/100
# d_test = 2. # (max_test - min_test)/100
#
# given_plotlims = [0,30.] # [0.,0.5]
# test_plotlims = [-100.,100.]
# #----------------------------------------
# #----------------------------------------
# interesting_cadences = [10./60., 1., 2., 30.] # minutes
# interesting_labels = ['10 sec', '1 min', '2 min', '30 min']
# #----------------------------------------
# #----------------------------------------
# input_dict = {'givenvar': [given_dat],
#               'testvar': [test_dat],
#               'cadvar': [cad_dat],
#               'givenlabel': given_plotlabel,
#               'testlabel': test_plotlabel,
#               'givensavelabel': given_savelabel,
#               'testsavelabel': given_savelabel,
#               'mingiven': min_given,
#               'maxgiven': max_given,
#               'dgiven': d_given,
#               'mintest': min_test,
#               'maxtest': max_test,
#               'dtest': d_test,
#                }
# #----------------------------------------
# #----------------------------------------
# do_for_specified_cadences = False
# is_the_given_var_cadence = True
# #----------------------------------------
# #----------------------------------------
# #likelihood_dict = do_bayes3(bayes_dict=input_dict)
# if do_for_specified_cadences == True:
#     for this_cad in range(len(interesting_cadences)):
#         print('showing cadence: ' + str(np.round(interesting_cadences[this_cad],2)) + ' min')
#         likelihood_dict = do_bayes4(bayes_dict=input_dict,interesting_cad=interesting_cadences[this_cad],
#                                     is_given_cad=is_the_given_var_cadence)
#
#         plot_posterior(lh_dict=likelihood_dict, given_lims=given_plotlims, test_lims=test_plotlims,
#                        interesting_cad=interesting_cadences[this_cad], is_given_cad=is_the_given_var_cadence)
#         # plot_cumulative(lh_dict=likelihood_dict, given_lims=given_plotlims, test_lims=test_plotlims,
#         #                 interesting_cad=interesting_cadences[this_cad], is_given_cad=is_the_given_var_cadence)
# #----------------------------------------
# #----------------------------------------
# if do_for_specified_cadences == False:
#     likelihood_dict = do_bayes4(bayes_dict=input_dict, interesting_cad=interesting_cadences,
#                                 is_given_cad=is_the_given_var_cadence)
#
#     plot_posterior(lh_dict=likelihood_dict, given_lims=given_plotlims, test_lims=test_plotlims,
#                    interesting_cad=interesting_cadences, is_given_cad=is_the_given_var_cadence)
#     # plot_cumulative(lh_dict=likelihood_dict, given_lims=given_plotlims, test_lims=test_plotlims,
#     #                 interesting_cad=interesting_cadences[this_cad], is_given_cad=is_the_given_var_cadence)
# # ----------------------------------------
# # ----------------------------------------
# if given == 'cadences (min)':
#     plot_specific_cadences_posterior(lh_dict=likelihood_dict,interesting_cads=interesting_cadences,
#                                      interesting_labs=interesting_labels,test_lims=test_plotlims)
#     plot_specific_cadences_cumulative(lh_dict=likelihood_dict,interesting_cads=interesting_cadences,
#                                       interesting_labs=interesting_labels,test_lims=test_plotlims)
# #----------------------------------------
# #----------------------------------------
# #----------------------------------------
# #----------------------------------------
# #----------------------------------------
# #----------------------------------------
# #----------------------------------------
# #----------------------------------------
# csv_filename_cantfind = 'Cant_Find.csv'
# savename_cantfind = csv_filename_cantfind.split('.')[0]
# print('Opening ' + savename_cantfind + '...')
# #----------------------------------------
# #----------------------------------------
# columns = ['calculated equivalent duration','true equivalent duration (noscatter)']
# eq_dur_fit, eq_dur_true_noscatter = extract_columns2(input_dict=df, input_columns=columns)
#
# df_cantfind, colnames = read_file(csv_filename_cantfind)
# columns_cantfind = ['true equivalent duration (noscatter)']
# eq_dur_cantfind = extract_columns2(input_dict=df_cantfind, input_columns=columns_cantfind)
#
# def model_func(t, A, K, C):
#     return A * np.exp(-K * t) # + C
# def linear_func(t, M, B):
#     return M*t + B
# def fit_exp_nonlinear(t, y):
#     opt_parms, parm_cov = scipy.optimize.curve_fit(model_func, t, y, p0=(0.9,0.02, 0.)) #, maxfev=1000)
#     A, K, C = opt_parms
#     return A, K, C
# def fit_linear(t, y):
#     opt_parms, parm_cov = scipy.optimize.curve_fit(linear_func, t, y) #, maxfev=1000)
#     M, B = opt_parms
#     return M, B
# def plot_ffd(eq_dur_fit, eq_dur_true_noscatter, eq_dur_cantfind):
#
#
#     d_eq_dur_fit = (np.max(eq_dur_fit) - np.min(eq_dur_fit))/100
#     eqdur_grid = np.arange(np.min(eq_dur_fit), np.max(eq_dur_fit) + d_eq_dur_fit, d_eq_dur_fit)
#     d_eq_dur_true_noscatter = (np.max(eq_dur_true_noscatter) - np.min(eq_dur_true_noscatter)) / 100
#     eqdur_grid_noscatter = np.arange(np.min(eq_dur_true_noscatter), np.max(eq_dur_true_noscatter) + d_eq_dur_true_noscatter, d_eq_dur_true_noscatter)
#
#     L_V830Tau = 3.827e33
#     eqdur_grid2 = eqdur_grid*L_V830Tau
#     eqdur_grid_noscatter2 = eqdur_grid_noscatter*L_V830Tau
#     eq_dur_fit2 = eq_dur_fit*L_V830Tau
#     eq_dur_true_noscatter2 = eq_dur_true_noscatter * L_V830Tau
#     eq_dur_cantfind2 = np.array(eq_dur_cantfind) * L_V830Tau
#
#     cumulative_eqdur = []
#     cumulative_eqdur_noscatter = []
#     for grid_element in range(len(eqdur_grid2)):
#         where_above = np.where(eq_dur_fit2 >= eqdur_grid2[grid_element])[0]
#         where_above_cantfind = np.where(eq_dur_cantfind2 >= eqdur_grid2[grid_element])[0]
#         cumulative_eqdur.append(np.float(len(where_above))/(np.float(len(eq_dur_fit2)) + len(where_above_cantfind)))
#         # cumulative_eqdur.append(np.float(len(where_above)) / np.float(len(eq_dur_fit)))
#     for grid_element_noscatter in range(len(eqdur_grid_noscatter2)):
#         where_above_noscatter = np.where(eq_dur_true_noscatter2 >= eqdur_grid_noscatter2[grid_element_noscatter])[0]
#         where_above_noscatter_cantfind = np.where(eq_dur_cantfind2 >= eqdur_grid_noscatter2[grid_element_noscatter])[0]
#         cumulative_eqdur_noscatter.append(np.float(len(where_above_noscatter)) / (np.float(len(eq_dur_true_noscatter2)) + len(where_above_noscatter_cantfind)))
#         # cumulative_eqdur_noscatter.append(np.float(len(where_above_noscatter)) / np.float(len(eq_dur_true_noscatter)))
#
#
#     A, K, C = fit_exp_nonlinear(eqdur_grid, cumulative_eqdur)
#     fit_cumulative = model_func(eqdur_grid, A, K, C)
#
#     print(np.log(A)*K)
#
#     # eqdur_grid2 = np.log10(eqdur_grid2)
#     # eqdur_grid_noscatter2 = np.log10(eqdur_grid_noscatter2)
#     # fit_cumulative = np.log10(fit_cumulative)
#     # cumulative_eqdur = np.log10(cumulative_eqdur)
#     # cumulative_eqdur_noscatter = np.log10(cumulative_eqdur_noscatter)
#
#     # where_noinf = np.where(eqdur_grid2 > 0)[0]
#     # eqdur_grid2 = eqdur_grid2[where_noinf]
#     # cumulative_eqdur = np.array(cumulative_eqdur)[where_noinf]
#     # fit_cumulative = fit_cumulative[where_noinf]
#
#     font_size = 'medium'
#     fig = plt.figure(1, figsize=(7, 5), facecolor="#ffffff") # , dpi=300)
#     ax = fig.add_subplot(1, 1, 1)
#
#     #import pdb; pdb.set_trace()
#
#     # M, B = fit_linear(eqdur_grid2, cumulative_eqdur)
#     # fit_cumulative_linear = linear_func(eqdur_grid2, M, B)
#
#     # import pdb; pdb.set_trace()
#
#     ax.step(eqdur_grid, cumulative_eqdur, color='#990033', lw=1, alpha=1.0, where='post', rasterized=True,
#             label='determined with flare fit')  # ccffef
#     ax.step(eqdur_grid_noscatter, cumulative_eqdur_noscatter, color='#206020', lw=1, alpha=1.0, where='post', rasterized=True,
#             label='true')  # ccffef
#     ax.plot(eqdur_grid, fit_cumulative, color='blue', label='Fitted Function:\n$y = %0.2f e^{%0.6f x} + %0.2f$' % (A, K, C))
#     # ax.plot(eqdur_grid2, fit_cumulative_linear, color='red', label='Fitted Function:\n$y = %0.2f x + %0.2f$' % (M, B))
#     ax.set_xlim(np.min(eqdur_grid),np.max(eqdur_grid))
#     # ax.set_ylim([0, 1.1*np.max(cumulative_eqdur)])
#     ax.set_xlabel('Equivalent Duration', fontsize=font_size, style='normal', family='sans-serif')
#     ax.set_ylabel('Occurrence', fontsize=font_size, style='normal', family='sans-serif')
#     # ax.set_title('Eq Dur:  ' + str(int(test_diff)) + ' – ' + str(int(test_diff+1)) + '% Diff From True Value', fontsize='large', style='normal', family='sans-serif')
#     # ax.set_title(save_as.split('/')[-1], fontsize=font_size, style='normal', family='sans-serif')
#     ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
#     ax.tick_params(axis='x', length=0.5)
#
#     plt.legend(loc='upper right')  # ,fontsize=font_size)
#     plt.tight_layout()
#     # print('Saving Interesting Cadences Plot...')
#     # plt.savefig(save_as, dpi=300)
#     # plt.close()
#     plt.show()
#
# plot_ffd(eq_dur_fit, eq_dur_true_noscatter, eq_dur_cantfind)
#
#


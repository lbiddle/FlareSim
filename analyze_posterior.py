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

np.random.seed(42)
#np.random.seed()

from scipy.interpolate import splrep, splev

import lightkurve as lk

from scipy import optimize
import scipy.stats as st

import sys, glob, os, shutil
import colour # https://www.colour-science.org/installation-guide/

import ast


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


    # position = [0, 0.5, 1]
    # position2 = [0, 0.25, 0.5, 0.75, 1]
    position2_2 = [0, 0.25, 0.5, 0.75, 1]
    # position3 = [0, 1./3., 2./3., 1]
    mycolormap = make_cmap(colors10, position=position2_2)

    return mycolormap




def read_file(csv_filename):

    df = pd.read_csv(path_to_csv + csv_filename) #, quotechar='"', sep=',', converters={1:ast.literal_eval})
    #
    # df['%Diff From True Equivalent Duration'].apply(lambda i: ast.literal_eval(i))
    # df['Likelihood Dist'].apply(lambda i: ast.literal_eval(i))

    #colnames = ['Cadence (min)', '%Diff From True Equivalent Duration', 'Likelihood Dist']
    colnames = df.columns
    xlab = colnames[1]
    ylab = colnames[0]
    zlab = colnames[2]

    #import pdb; pdb.set_trace()

    return df,xlab,ylab,zlab

def plot_posterior(df,xlab,ylab,zlab,save_as):
    font_size = 'large'

    y = np.array(df[ylab]) #np.zeros_like(y) + df['Cadence (min)'][row]
    x = df[xlab][0]
    x = x.replace('  ', ',')
    x = x.replace('\n','')
    x = x.replace('[', '')
    x = x.replace(']', '')

    # x = x.replace('  ', '')
    x = x.split(',')

    temp_x = []

    for boop in range(len(x)):
        try:
            x_float = np.float(x[boop])
        except:
            continue
        else:
            temp_x.append(x_float)

    #import pdb; pdb.set_trace()
    #x = np.array(x[:-1]).astype(np.float)
    x = np.array(temp_x)

    z = np.zeros((len(x),len(y)))

    for row in range(len(df)):
        # y = np.array(df['%Diff From True Equivalent Duration'][row])
        # z = np.array(df['Likelihood Dist'][row])
        # x = np.zeros_like(y) + df['Cadence (min)'][row]

        temp_z = []

        z_array = df[zlab][row]
        z_array = z_array.replace('\n', '')
        z_array = z_array.replace('[', '')
        z_array = z_array.replace(']', '')
        z_array = z_array.split(' ')
        for bep in range(len(z_array[:-1])):
            try:
                z_float = np.float(z_array[bep])
            except:
                #import pdb; pdb.set_trace()
                continue
            else:
                temp_z.append(z_float)

        z_array = np.array(temp_z)

        try:
            z[:,row] = z_array#[:-1]
        except:
            # import pdb; pdb.set_trace()
            continue

    # deltaX = np.diff(x)[0]
    # deltaY = np.diff(y)[0]
    # #deltaZ = np.diff(Z)[0]
    # xmin = xlim_min - 1 * deltaX
    # xmax = xlim_max + 1 * deltaX
    # ymin = ylim_min - 1 * deltaY
    # ymax = ylim_max + 1 * deltaY
    # # zmin = 0
    # # zmax = 1.
    # xx = np.arange(xmin, xmax + deltaX, deltaX)
    # yy = np.arange(ymin, ymax + deltaY, deltaY)
    # ax.scatter(xplot, yplot, s=np.pi * 1.5 ** 2, c=zplot, cmap='rainbow', vmin=np.min(zplot), vmax=np.max(zplot),
    #            alpha=1.0, rasterized=True)

    fig = plt.figure(1, figsize=(7, 5.5), facecolor="#ffffff")  # , dpi=300)
    ax = fig.add_subplot(111)
    # ax1.set_xlim([np.min(flare_id_time) - 0.005, np.max(flare_id_time) + 0.01])
    ax.set_xlabel(xlab, fontsize=font_size, style='normal', family='sans-serif')
    ax.set_ylabel(ylab, fontsize=font_size, style='normal', family='sans-serif')
    # ax1.set_title('Equivalent Duration = ' + str(np.round(eq_dur, 2)) + ' sec' + '\nTrue Equivalent Duration = ' + str(
    #     np.round(eq_dur_true, 2)) + ' sec', pad=10, fontsize=font_size, style='normal', family='sans-serif')
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    xmesh, ymesh = np.meshgrid(x, y)
    # clevels_regular = ax.contour(xmesh, ymesh, H.T, lw=0.5, colors='green')  # , cmap='winter')

    mycolormap = choose_cmap()
    # bounds = np.arange(0, H.max() + 1, 1)
    cmap = plt.get_cmap(mycolormap, z.max() + 1)
    z = np.ma.masked_where(z == 0, z)
    cmap.set_bad(color='white')
    pcolor = ax.pcolormesh(xmesh, ymesh, z.T, cmap = cmap, vmin = z.min(), rasterized=True)
    cb = fig.colorbar(pcolor) #, ax=ax) #, cmap=cmap)
    cb.ax.tick_params(labelsize=font_size)
    cb.ax.set_ylabel('Posterior Probability', fontsize=font_size, rotation=270)
    cb.ax.get_yaxis().labelpad = 15

    ax.plot([0,0],[np.min(y),np.max(y)],'-',lw=0.75,c='#000000')
    # import pdb; pdb.set_trace()

    ax.set_ylim([np.min(y), np.max(y)])
    # ax1.legend(fontsize=font_size, loc='upper right')
    plt.tight_layout()
    print('Saving Test Fig...')
    plt.savefig(path_to_csv + save_as, dpi=300, rasterized=True)
    plt.close()




path_to_csv = '/Users/lbiddle/PycharmProjects/Flares/TESS/Flares/K2_Project/K2_Figures/test_flare_fits/All_Cadences/'

for file in glob.glob(path_to_csv + '*.csv'):
    csv_filename = file.split('/')[-1]
    savename = file.split('/')[-1].split('.')[0]

    print('Opening ' + savename + '...')

    df,xlab,ylab,zlab = read_file(csv_filename)
    plot_posterior(df,xlab,ylab,zlab,save_as = savename)







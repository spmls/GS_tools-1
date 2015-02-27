# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:06:03 2014

@author: blunghino
"""
import matplotlib
from matplotlib import pyplot as plt

from .gsfile import GSFile


def compare_bulk_dists(dgs_bins, dgs_dist, sl_bins, sl_dist, pc_bins, *pc_dists, 
                      bin_units='phi', bin_range=(-2,5), figsize=(13,10)):
    """
    create a figure plotting bulk grain size distributions 
    from 3 different analysis methods
    
    distribution units should be fraction of a whole (distributions sum to 1)
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.title('Distributions')
    for pc_dist in pc_dists:
        ax.plot(pc_bins, pc_dist, 'r--', label='Point Count', lw=2)
    ax.plot(dgs_bins, dgs_dist, 'b-', label='Digital Grain Size', lw=2)
    ax.plot(sl_bins, sl_dist, 'g-.', label='Sed Lab', lw=2)
    ax.legend(loc=1)
    if bin_range:
        plt.xlim(left=bin_range[0], right=bin_range[1])
    if bin_units == 'phi':
        plt.xlabel(r'Size ($\mathsf{\phi}$)')
    else:
        plt.xlabel('Size (%s)' % bin_units)
    plt.ylabel('Fraction')
    return fig


def add_new_dists(fig, new, bin_units='phi'):
    """
    add distributions from GSFile 'new' to the first axis of figure 'fig'
    """
    c = ['DarkOrange', 'DarkRed', 'DimGray', 'm']
    ax = fig.get_axes()[0]
    hands, labs = ax.get_legend_handles_labels()
    if bin_units == 'phi':
        bins = new.bins_phi
    elif bin_units == 'mm':
        bins = 2**-new.bins_phi
    else:
        return fig
    for ii, dist in enumerate(new.dists.T):
        p = ax.plot(bins, dist, '--', color=c[ii], lw=1.5)[0]
        hands.append(p)
        labs.append(new.sample_id[ii])
    ax.legend(hands, labs)
    return fig
    

def compare_bulk_means(dgs_mean, sl_mean, *pc_means, figsize=(10,10), 
                      y_units='phi', y_range=(5,-2)):
    """
    pc_means is a list of tuples:
        [(mean1_ax1, mean1_ax2, [id]), (mean2_ax1, mean2_ax2, [id])]
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.title('Means')
    means = [dgs_mean, sl_mean] + list(np.asarray(pc_means).flatten())
    colors = 'bgkr'
    tick_labels = ['', 'Digital Grain Size', 'Sed Lab']
    for jj, p in enumerate(pc_means):
        try:
            tick_labels += p[2]
        except KeyError:
            tick_labels += 'Point Count %i' % jj+1
    labels = ['Digital Grain Size', 'Sed Lab', 'Point Count Major Axis', 
               'Point Count Minor Axis']
    handles = []
    for ii, mn in enumerate(means):
        if ii < 2:
            ctr = ii + 1
        else:
            ctr = ii + 1 + ii%2
        p, = plt.plot(ctr, mn, 'o', c=colors[ii], label=labels[ii])
        if ii < 4:
            handles.append(p)
    ax.legend(handles, labels, loc=1, numpoints=1)
    ax.set_xlim(0, len(tick_labels))
    ax.xaxis.set_major_locator(matplotlib.ticker.LinearLocator(len(tick_labels)+1))
    ax.set_xticklabels(tick_labels, rotation=45)
    if y_units == 'phi':
        plt.ylabel(r'Size ($\mathsf{\phi}$)')
        ax.invert_yaxis()
    else:
        plt.xlabel('Size (%s)' % y_units)
    if y_range:
        ax.set_ylim(bottom=y_range[0], top=y_range[1])
    plt.tight_layout()
    return fig

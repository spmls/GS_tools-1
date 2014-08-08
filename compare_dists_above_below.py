import os
import datetime as dt

import matplotlib
from matplotlib import gridspec, pyplot as plt
import numpy as np

from GS_tools.gsfile import GSFile


matplotlib.rcParams['font.size'] = 16


def compare_dists_above_below(gsf, tsunami_only=True, min_layer=None,
                              figsize=(18,11), phi_min_max=None,
                              depth_min_max=None, save_fig=False):
    """
    make a set of figures comparing a grain size distribution with
    the distributions directly above and below it in the core/trench

    tsunami_only: if True and min_layer is None, uses GSFile.layer to
        exclude non-tsunami layers (GSFile.layer >= 1)

    min_layer: specify a minimum cutoff value of GSFile.layer to include
        overrides tsunami_only

    figsize: set the figsize for plt.figure call

    phi_min_max: can pass a tuple (min_phi, max_phi) used to set the x limits for
        the distribution plots (min_phi is the largest grain size to include)

    depth_min_max: tuple (min_depth, max_depth) used to specify a range of depth
        values to plot

    save_fig: if True, saves each fig as a png
    """
    figs = []
    ## set layer filter value
    if min_layer is None:
        if tsunami_only:
            min_layer = 1
        else:
            min_layer = -1
    ## filter dists so that only layer values >= min_layer are plotted
    f1 = gsf.layer >= min_layer
    dists = gsf.dist_normed(normed_to=100)[:, f1]
    mid_depth = gsf.mid_depth[f1]
    means = gsf.dist_means()[f1]
    f2 = np.isfinite(mid_depth)
    dists = dists[:, f2]
    mid_depth = mid_depth[f2]
    means = means[f2]
    if gsf.bins_phi is None:
        bins = gsf.bins_phi_mid
    else:
        bins = gsf.bins_phi
    if bins is None:
        print('GS_tools: {}: No grain size bin data'.format(gsf.id))
#    sample_id = np.asarray(
#        [gsf.sample_id[ii] for ii, L in enumerate(gsf.layer) if L >= min_layer]
#    )
    sample_id = gsf.sample_id[f1]
    sample_id = sample_id[f2]
    y2 = np.zeros_like(bins)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ms = 7.5
    for ii, d in enumerate(mid_depth):
        fig = plt.figure(figsize=figsize)
        fig.suptitle('%s: %s' % (gsf.id, sample_id[ii]))
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax1.plot(means, mid_depth, 'k.')
        if ii != 0:
            ax0.plot(bins, dists[:,ii-1], 'b')
            ax0.fill_between(bins, dists[:,ii-1], y2, color='b', alpha=.5)
            ax1.plot(means[ii-1], mid_depth[ii-1], 'o', c='b', ms=ms)
            ax1.annotate('{:.2f}'.format(means[ii-1]),
                         (means[ii-1], mid_depth[ii-1]), (4, 0),
                         textcoords='offset points', fontsize=10)
        if ii != len(mid_depth) - 1:
            ax0.plot(bins, dists[:,ii+1], 'r')
            ax0.fill_between(bins, dists[:,ii+1], y2, color='r', alpha=.5)
            ax1.plot(means[ii+1], mid_depth[ii+1], 'o', c='r', ms=ms)
            ax1.annotate('{:.2f}'.format(means[ii+1]),
                         (means[ii+1], mid_depth[ii+1]), (4, 0),
                         textcoords='offset points', fontsize=10)
        ax0.plot(bins, dists[:,ii], 'k', zorder=10)
        ax0.fill_between(bins, dists[:,ii], y2, color='DimGray', zorder=9,
                         alpha=.9)
        ax1.plot(means[ii], d, 'o', c='DimGray', ms=ms)
        ax1.annotate('{:.2f}'.format(means[ii]),
                     (means[ii], mid_depth[ii]), (4, 0),
                     textcoords='offset points', fontsize=10)
        ax1.invert_yaxis()
        ax1.xaxis.set_major_locator(matplotlib.ticker.LinearLocator(3))
        ax1.set_xlabel(r'Mean Grain Size ($\mathsf{\phi}$)')
        ax1.set_ylabel('Depth (%s)' % gsf.depth_units)
        ax0.set_xlabel(r'Grain Size ($\mathsf{\phi}$)')
        ax0.set_ylabel('Percent')
        ax0.set_ylim(bottom=0)
        if phi_min_max:
            ax0.set_xlim(left=phi_min_max[0], right=phi_min_max[1])
        figs.append(fig)
    if save_fig:
        figsaver(figs, sample_id, save_fig=save_fig, dir_title=gsf.id,
                 dir_path=os.path.join(gsf.project_directory, 'Figures'))
    return figs


def figsaver(figs, fig_titles, save_fig='png', dir_path=None, dir_title='',
             overwrite=False, dpi=300, transparent=False):
    """
    save figure "fig"

    if save_fig is a string it must specify a file format supported by
    matplotlib. eg: save_fig='jpg'
    this will initiate autosave which will create a new directory in the cwd
    and save the figure automatically using the string fig_title

    if overwrite is True the function will allow old directories with the same name
    to be overwritten
    """
    if dir_path is None:
        dir_path = os.getcwd()
    if save_fig not in ('png', 'eps', 'tif', 'jpg', 'svg', 'pdf'):
        save_fig = 'png'
    if not dir_title:
        dir_title = dt.datetime.strftime(dt.datetime.today(), '%Y-%m-%d')
    save_dir = os.path.join(dir_path, 'Figures_%s' % dir_title)
    if os.path.exists(save_dir) and not overwrite:
        ii = 0
        save_dir += '__'
        while os.path.exists(save_dir):
            if ii < 10:
                ii += 1
                save_dir = save_dir[:-1] + str(ii)
            else:
                print('')
                print('compare_dists_above_below.figsaver:')
                print('Could not save, specify different dir_title')
                return
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for ii, fig in enumerate(figs):
        plt.figure(fig.number)
        fname = '%s.%s' % (fig_titles[ii].replace(" ", "_"), save_fig)
        full_path = os.path.join(save_dir, fname)
        plt.savefig(full_path, dpi=dpi, transparent=transparent)


if __name__ == "__main__":
    gsf_name = 'GS_Chile_Coliumo_Trench7.csv'
    pd_pc = r'C:\Users\blunghino\Field Sites\Tsunami_Deposit_Database\TsuDepData\Uniform_GS_Data\\'
    pd_mac = '/Users/blunghino/Field_Sites/Tsunami_Deposit_Database/TsuDepData/Uniform_GS_Data/'
    gsf = GSFile(gsf_name, project_directory=pd_pc)
    figs = compare_dists_above_below(gsf, phi_min_max=(0,4))
    plt.show()

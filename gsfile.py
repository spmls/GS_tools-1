# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:05:07 2014

Created while working for the US Geological Survey

@author: brent lunghino
"""
import os
import csv
import warnings

import numpy as np
from matplotlib import cm, pyplot as plt
from scipy.stats import nanmean


layer_type_lookup = {
    -1: 'Post-tsunami',
    0: 'Pre-tsunami',
    1: 'Suspension graded',
    2: 'Inverse graded',
    3: 'Normal graded',
    4: 'Massive',
    5: 'Not classified',
    6: 'Not suspension graded',
    7: 'Mud',
    8: 'Mud cap',
    9: 'Unknown',
}


class BaseGSFile:
    """
    base class to store and manipulate grain size data

    when initialized, reads in data from a uniform format csv file stored in
    csv_directory

    layers can be classified according to layer_type_lookup
    """
    ## allows layer classifications to be specified by subclassing
    layer_type_lookup = {
        -1: 'Post-tsunami',
        0: 'Pre-tsunami',
        1: 'Suspension graded',
        2: 'Inverse graded',
        3: 'Normal graded',
        4: 'Massive',
        5: 'Not classified',
        6: 'Not suspension graded',
        7: 'Mud',
        8: 'Mud cap',
        9: 'Unknown',
    }
    ## allows a directory to be specified by subclassing
    project_directory = ''
    ## allows mm/pixel ratio to be specified by subclassing
    mm_pix = None

    def __init__(
            self,
            csv_file_location,
            project_directory='',
            mm_pix=None,
            layer_type_lookup=None,
            metadata_rows=17,
            col_header_rows=6,
            numeric_fields=('Min Depth', 'Max Depth', 'Layer Type', 'Layer'),
    ):
        """
        csv_file_location is the name of the csv file in project_directory,
        (also can set file path by overriding get_csv_file_path method)

        metadata_rows is the number of rows in the csv file before the start
        of the grain size distribution data

        col_header_rows is the number of rows in the csv file that contain
        metadata specific to each sample in the csv file.
        These rows must come between the trench scale metadata and the
        grain size distribution

        mm_pix is the mm per pixel conversion factor
        """
        ## allows settings to be overridden when the object is initiated
        if project_directory:
            self.project_directory = project_directory
        elif not self.project_directory:
            self.project_directory = os.getcwd()
        if mm_pix:
            self.mm_pix = mm_pix
        if layer_type_lookup:
            self.layer_type_lookup = layer_type_lookup
        ## get the full file path for the csv file
        self.csv_file_path = self.get_csv_file_path(csv_file_location)
        self.gsfileuniform = os.path.split(self.csv_file_path)[1]
        ## keep track of sequences
        self.sequence_attrs = []
        ## parse csv file contents by row
        try:
            with open(self.csv_file_path, 'r') as csvfile:
                rdr = csv.reader(csvfile, dialect='excel', strict=True,
                                 skipinitialspace=True)
                ## sequence of lists where each list is a row from the csv file
                lines = [line for line in rdr]
        except:
            with open(self.csv_file_path, 'rU') as csvfile:
                rdr = csv.reader(csvfile, dialect=csv.excel, strict=True,
                                 skipinitialspace=True)
                lines = [line for line in rdr]
        for ii, m in enumerate(lines[:metadata_rows]):
            if m[0]:
                ## values for numeric fields are converted to numpy arrays
                if m[0] in numeric_fields:
                    att = np.asarray([x if x != '' else "nan" for x in m[1:]],
                                     dtype=np.float64)
                ## values for non numeric fields are kept as strings
                else:
                    att = np.asarray(m[1:])
                ## attribute name
                name = m[0].replace(' ', '_').lower()
                ## first group of meta data store a single value associated
                ## with the entire file
                if ii < metadata_rows - col_header_rows - 1:
                    setattr(self, name, att[0])
                ## second group of meta data rows stores a sequence of values
                ## with one value for each grain size sample (col_header_rows)
                elif len(att) > 0:
                    setattr(self, name, att)
                    self.sequence_attrs.append(name)
                ## when no data exists in a col_header_row setattr to None
                else:
                    setattr(self, name, None)
        self.mid_depth = (self.min_depth+self.max_depth) / 2.
        ## get indices to sort with
        ind = np.argsort(self.mid_depth)
        self.mid_depth = self.mid_depth[ind]
        ## get strings values for layer type codes
        self.layer_type_strings = np.asarray(
            [self.layer_type_lookup[x] for x in self.layer_type])[ind]
        ## get data into numpy array
        temp = np.asarray(lines[metadata_rows:], dtype=np.float64)
        self.bins = temp[:, 0]
        self.bins_phi = self._convert_bins_to_phi()
        self.bins_phi_mid = self._convert_bins_to_phi_mid()
        self.dists = temp[:, 1:]
        self.dists = self.dists[:, ind]
        for seq in self.sequence_attrs:
            sorted = getattr(self, seq)[ind]
            setattr(self, seq, sorted)

    def get_csv_file_path(self, csv_file_location):
        """return the full path to the csv file"""
        return os.path.join(self.project_directory, csv_file_location)

    def _convert_bins_to_phi(self):
        """
        internal method to convert bins to phi units
        """
        if self.bin_units == 'phi':
            return self.bins
        ## mid point between phi bin edges (used for statistics)
        elif self.bin_units == 'phi mid':
            return None
        ## settling velocity
        elif self.bin_units == 'psi':
            return None
        elif self.bin_units == 'mm':
            return -np.log2(self.bins)
        elif self.bin_units == 'pixels' and self.mm_pix is not None:
            return -np.log2(self.bins * self.mm_pix)
        else:
            return None

    def _convert_bins_to_phi_mid(self):
        """
        internal method to convert bins to phi midpoints
        """
        if self.bin_units == 'phi mid':
            return self.bins
        elif self.bins_phi is not None:
            mpt1 = np.asarray(self.bins_phi[0] + 0.5 * (self.bins_phi[0] - self.bins_phi[1]))
            bins_phi_mid = self.bins_phi[1:] + 0.5 * (self.bins_phi[:-1] - self.bins_phi[1:])
            return np.hstack((mpt1, bins_phi_mid))
        else:
            return None

    def __str__(self):
        return self.id


class GSFile(BaseGSFile):
    """
    subclass of BaseGSFile 
    adds methods to calculate statistics and make plots

    Class to store and manipulate grain size data

    When initialized, reads in data from a uniform format csv file stored in
    csv_directory

    Layers can be classified according to layer_type_lookup

    Stats calculated using formulations in sedstats.m by Bruce Jaffe 10/2/03
    formulas originally from sedsize version 3.3 documentation (7/12/89)
    """
    def dist_normed(self, normed_to=1, sensitivity=.05):
        """
        calculate the normalized distribution
        """
        sums = self.dists.sum(axis=0)
        mean = sums.mean()
        if abs(sums - mean).any() > sensitivity * mean:
            warnings.warn(
                '%s - distributions have inconsistent sums when normalizing'
                % self.__str__()
            )
        normed_dists = np.ones_like(self.dists)
        for ii, dist in enumerate(self.dists.T):
            normed_dists[:,ii] = normed_to * dist / dist.sum()
        return normed_dists

    def dist_means(self, min_size=None):
        """
        calculate the mean of each distribution
        """
        means = np.zeros_like(self.mid_depth)
        if min_size:
            filtr = self.bins_phi_mid < min_size
            dists = self.dists[filtr, :]
            bins_phi_mid = self.bins_phi_mid[filtr]
        else:
            dists = self.dists
            bins_phi_mid = self.bins_phi_mid
        for ii, dist in enumerate(dists.T):
            means[ii] = np.sum(dist * bins_phi_mid) / dist.sum()
        return means

    def dist_devs(self):
        """
        returns deviation from the mean and the mean
        """
        means = self.dist_means()
        devs = np.zeros_like(self.dists)
        for ii, m in enumerate(means):
            devs[:, ii] = self.bins_phi_mid - m
        return devs, means

    def dist_stds(self):
        """
        calculate the standard deviation of each distribution
        """
        devs = self.dist_devs()[0]
        variances = np.zeros_like(self.mid_depth)
        for ii, dist in enumerate(self.dists.T):
            variances[ii] = np.sum(dist * (devs[:, ii] ** 2)) / dist.sum()
        return np.sqrt(variances)

    def dist_moments(self):
        """
        calculate 1st through 4th moments for each distribution
        """
        devs, m1 = self.dist_devs()
        m2 = np.zeros_like(self.mid_depth)
        m3 = np.zeros_like(self.mid_depth)
        m4 = np.zeros_like(self.mid_depth)
        for ii, dist in enumerate(self.dists.T):
            dist_sum = dist.sum()
            m2[ii] = np.sum(dist * devs[:, ii] ** 2) / dist_sum
            std = np.sqrt(m2[ii])
            m3[ii] = np.sum(dist * (devs[:, ii] / std) ** 3) / dist_sum
            m4[ii] = np.sum(dist * (devs[:, ii] / std) ** 4) / dist_sum
        return m1, m2, m3, m4

    def bulk_dist(self, depth_range=None):
        """
        calculate bulk distribution for all samples of tsunami
        sediments in trench
        """
        ## check if depth data exists for all grain size distributions
        if not np.isnan(self.min_depth).any() and len(self.sample_id) > 1:
            dists = self.dists[:, self.layer > 0]
            diffs = [x - self.min_depth[ii] for ii, x in enumerate(self.max_depth)]
            length = sum(diffs)
            ## weight distributions by depth range
            for ii in range(dists.shape[1]):
                dists[:, ii] = dists[:, ii] * diffs[ii] / length
        else:
            dists = self.dists[:, self.layer > 0]
        bulk_dist = nanmean(dists, axis=1)
        bulk_dist = 100. * bulk_dist / bulk_dist.sum()
        return bulk_dist

    def bulk_mean(self, gs_min_max=None):
        """
        calculate bulk mean of all samples of tsunami sediments in trench

        gs_min_max is a sequence of length 2 specifying the minimum grain size
        and maximum grain size to include in the calculations (in phi)
        """
        if self.bins_phi_mid is None:
            return np.nan
        else:
            dist = self.bulk_dist()
            if gs_min_max is not None:
                f1 = self.bins_phi_mid <= gs_min_max[0]
                f2 = self.bins_phi_mid >= gs_min_max[1]
                filtr = f1 * f2
                dist = dist[filtr]
                bins = self.bins_phi_mid[filtr]
            else:
                bins = self.bins_phi_mid
            return np.sum(dist * bins) / dist.sum()

    def bulk_std(self, gs_min_max=None):
        """
        calculate bulk standard deviation of all samples of tsunami sediments
        in trench
        """
        if self.bins_phi_mid is None:
            return np.nan
        else:
            dist = self.bulk_dist()
            mean = self.bulk_mean(gs_min_max=gs_min_max)
            if gs_min_max is not None:
                f1 = self.bins_phi_mid <= gs_min_max[0]
                f2 = self.bins_phi_mid >= gs_min_max[1]
                filtr = f1 * f2
                dist = dist[filtr]
                bins = self.bins_phi_mid[filtr]
            else:
                bins = self.bins_phi_mid
            dev = bins - mean
            variance = np.sum(dist * (dev ** 2)) / dist.sum()
            return np.sqrt(variance)
            
    def n_layers_in_layer_type(self, layer_type=1):
        """
        calculate the number of layers with a given classification
        """
        return len(set(self.layer[self.layer_type == layer_type]))
        
    def thickness_of_layers_in_layer_type(self, layer_type=1):
        """
        calculate the thicknesses of all layers of a given layer type
        """
        out = []
        ## filter to get only layers of `layer_type`
        f1 = self.layer_type == layer_type
        for x in sorted(set(self.layer[f1])):
            ## for each layer, calculate the thickness 
            ## by subtracting the minimum and maximum depth
            f2 = self.layer == x
            f3 = f1 * f2
            mn = min(self.min_depth[f3])
            mx = max(self.max_depth[f3])
            out.append(mx-mn)
        if not out:
            return None
        else:
            return np.asarray(out)

    def _get_depth_bin_edges(self, min_layer=-1):
        """
        internal method to deal with uneven depth spacing when plotting pcolor

        eg if min_depth = [0, 1, 2.5], max_depth = [1, 2, 3.5]
        returns [0, 1, 2.25, 3.5]

        min_layer designates the lower boundary of layers to use from the
        layer attribute field
        """
        min_depth = self.min_depth[self.layer >= min_layer]
        max_depth = self.max_depth[self.layer >= min_layer]
        ## all sample edges match
        if np.array_equal(min_depth[1:], max_depth[:-1]):
            return np.hstack((min_depth, max_depth[-1]))
        ## some sample edges do not match
        else:
            depths = np.zeros(min_depth.size + 1)
            depths[0] = min_depth[0]
            depths[-1] = max_depth[-1]
            for ii, (n, x) in enumerate(zip(min_depth[1:],
                                            max_depth[:-1])):
                ii += 1
                if n == x:
                    depths[ii] = n
                else:
                    depths[ii] = n + (x-n) / 2
            return depths

    def fig_dists_depth(self, figsize=(8, 10), phi_min=-2, phi_max=4,
                        pcolor=True, tsunami_only=True, min_layer=None,
                        unicode_label=False, show_sg=False):
        """
        create a matplotlib figure plotting grain size distribution with depth

        phi_min is the minimum phi value (maximum grain size)
        phi_max is the maximum phi value (minimum grain size)
        """
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        plt.title('Grain-size distributions at %s' % self.id)
        ## set layer filter value
        if min_layer is None:
            if tsunami_only:
                min_layer = 1
            else:
                min_layer = -1
        ## filter dists so that only layer values >= min_layer are plotted
        dists = self.dists[:, self.layer >= min_layer]
        max_depth = self.max_depth[self.layer >= min_layer]
        min_depth = self.min_depth[self.layer >= min_layer]
        layer_type = self.layer_type[self.layer >= min_layer]
        ## check that depth data exists
        if np.isnan(max_depth).all():
            plt.text(.5, .5, 'No depth values associated with grain-size data',
                     ha='center')
            return fig
        elif np.isnan(max_depth).any():
            pcolor = False
        ## create pcolor
        if pcolor:
            depths = self._get_depth_bin_edges(min_layer=min_layer)
            plt.pcolormesh(self.bins_phi_mid, depths, dists.T)
            color = 'w'
            cbar = plt.colorbar(orientation='vertical', fraction=.075, pad=.1,
                                aspect=30, shrink=.75)
            cbar.set_label(self.distribution_units)
        else:
            color = 'k'
        ## set phi bins
        if self.bins_phi is not None:
            bins = self.bins_phi
        elif self.bins_phi_mid is not None:
            bins = self.bins_phi_mid
        else:
            plt.text(.5, .5,
                     'Grain size bins must convert to phi for this figure',
                     ha='center')
            return fig
        ## set up line color to correspond to suspension grading
        color = list(color) * len(layer_type)
        if show_sg:
            for ii, L in enumerate(layer_type):
                if L != 1:
                    if pcolor:
                        color[ii] = 'k'
                    else:
                        color[ii] = 'b'
        ## plot a line for each distribution
        for ii, d in enumerate(max_depth):
            ## normalize to the max, and scale to plot within the depth range
            normed = dists[:, ii] * (min_depth[ii] - d) * .95 / dists[:, ii].max()
            plt.plot(bins, d + normed, color[ii], lw=2.25)
        ax.invert_yaxis()
        ax.set_xlim((phi_min, phi_max))
        ax.set_ylim(bottom=np.nanmax(max_depth))
        if unicode_label:
            plt.xlabel('Size (\u03D5)')
        else:
            plt.xlabel(r'Size ($\mathsf{\phi}$)')
        plt.ylabel('Depth (%s)' % self.depth_units)
        return fig


    def fig_dists_stacked(self, figsize=(16, 12), phi_min=-2, phi_max=4,
                          tsunami_only=True, min_layer=None,
                          unicode_label=False):
        """
        plot grain size distributions on one axis
        """
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        plt.title('Grain size distributions at %s' % self.id)
        ## set layer filter value
        if min_layer is None:
            if tsunami_only:
                min_layer = 1
            else:
                min_layer = -1
        ## filter dists so that only layer values >= min_layer are plotted
        dists = self.dists[:, self.layer >= min_layer]
        n_dists = dists.shape[1]
        labels = [self.sample_id[ii] for ii, L in enumerate(self.layer) if L >= min_layer]
        ## set up custom cmap
        cmap = cm.get_cmap('spectral')
        c = [cmap(1. * ((ii + 1) / (n_dists + 1))) for ii in range(n_dists)]
        ## set phi bins
        if self.bins_phi is not None:
            bins = self.bins_phi
        elif self.bins_phi_mid is not None:
            bins = self.bins_phi_mid
        else:
            plt.text(.5, .5,
                     'Grain size bins must convert to phi for this figure',
                     ha='center')
            return fig
        ## plot each distribution
        for ii, d in enumerate(dists.T):
            plt.plot(bins, d, c=c[ii], label=labels[ii], lw=1.5)
        plt.legend(loc=0)
        ax.set_xlim((phi_min, phi_max))
        plt.ylabel(self.distribution_units)
        if unicode_label:
            plt.xlabel('Size (\u03D5)')
        else:
            plt.xlabel(r'Size ($\mathsf{\phi}$)')
        return fig

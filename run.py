#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Look for streams and substructure in the current GES data """

from __future__ import absolute_import, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import os
from collections import Counter
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pyfits
from astroML.plotting import hist as blocks_hist
from sklearn.cluster import MeanShift


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG)


def mark_duplicates(data):

    duplicate_cnames = [cname \
        for (cname, count) in Counter(data["CNAME"]).iteritems() if count > 1]

    # For each duplicate CNAME, find all row indices, then find the best entry
    duplicate_row_indices = []
    num_duplicate_cnames = len(duplicate_cnames)
    for i, cname in enumerate(duplicate_cnames):
        indices = (data["CNAME"] == cname)

        # Which has the lowest velocity error?
        best_row_index = np.where(indices)[0][np.argmin(data[indices]["E_VEL"])]
        logging.debug("({0}/{1}) {2} has {3} entries. Taking index {4} with VEL"
            " = {5:.1f} +/- {6:.1f} km/s as the best row.".format(i,
                num_duplicate_cnames, cname, indices.sum(), best_row_index,
                data[best_row_index]["VEL"], data[best_row_index]["E_VEL"]))

        duplicate_row_indices.extend(
            set(np.where(indices)[0]).difference([best_row_index]))

    # Create a mask
    is_duplicate = np.zeros(len(data), dtype=bool)
    for index in duplicate_row_indices:
        is_duplicate[index] = True

    return is_duplicate


def assign_field_numbers(data, **kwargs):
    """
    Assign field numbers to each star
    """

    X = np.vstack([data["RA"], data["DEC"]]).T

    plot = kwargs.pop("plot", False)
    ms_kwargs = {
        "bandwidth": 0.5,
        "bin_seeding": True
    }
    ms_kwargs.update(kwargs)
    ms = MeanShift(**ms_kwargs).fit(X)

    labels = ms.labels_
    field_centers = ms.cluster_centers_

    unique_labels = set(labels)
    num_clusters = len(unique_labels)
    logging.debug("Found {} unique labels".format(num_clusters))

    if plot:
        fig, ax = plt.subplots(figsize=(16,16))
        colors = cycle("bgrcmy")

        scat = ax.scatter(X[:, 0], X[:, 1], c=labels)
    
        for k, col in zip(range(num_clusters), colors):
            members = (labels == k)
            field_center = field_centers[k]

            ax.scatter([field_center[0]], [field_center[1]], facecolors='none',
                s=30, lw=2, edgecolor="k", zorder=100)

        cbar = plt.colorbar(scat)
        cbar.set_label("Field Number")
        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")

        return (labels, fig)

    return labels







if __name__ == "__main__":

    iDR3_filenames = ("GES_iDR2iDR3_WG11_Recommended.fits",
        "GES_iDR2iDR3_WG10_Recommended.fits")
    iDR4_filename = "GES_iDR4.fits"

    data = pyfits.open(iDR4_filename)[1].data
    # UVES, GIRAFFE
    idr3_data = [pyfits.open(filename)[1].data for filename in iDR3_filenames]

    matched_subset_filename = "GES_MW_iDR4+iDR3_matched.fits"

    if os.path.exists(matched_subset_filename):
        subset = pyfits.open(matched_subset_filename)[1].data

    else:
        # Look for duplicate CNAMES and take the row with the lowest E_VEL
        # Note: this part takes time so we 
        duplicate_rows_filename = "duplicates.pickle"
        if os.path.exists(duplicate_rows_filename):
            with open(duplicate_rows_filename, "rb") as fp:
                is_duplicate = pickle.load(fp)
        else:
            is_duplicate = mark_duplicates(data)
            with open(duplicate_rows_filename, "wb") as fp:
                pickle.dump(duplicate, fp, -1)

        # Is the position information sensible?
        has_sensible_position = data["RA"] > 0

        # Let's just use milky way stars from now on
        is_mw_field_star = np.array([r["GES_FLD"].startswith("GES_MW_") \
            for r in data])

        # Assign field numbers to each star
        subset_filter = has_sensible_position * is_mw_field_star * ~is_duplicate

        subset = data[subset_filter]

        subset_field_numbers, fig = assign_field_numbers(subset, plot=True)
        fig.savefig("fields.pdf")

        # Find all the stars we can in iDR3. Preference UVES over GIRAFFE
        get_recommended_keys = ("TEFF", "LOGG", "FEH", "ALPHA_FE")
        idr3_recommended = []
        for i, row in enumerate(subset):
            recommended_row_data = [np.nan] * len(get_recommended_keys)

            file_index = [0, 1][row["FILENAME"].startswith("g")]
            matches = (row["CNAME"] == idr3_data[file_index]["CNAME"])
            if matches.sum() > 0:

                if file_index > 0:
                    use_keys = ["TEFF", "LOGG", "MH", "ALPHA_FE"]

                else:
                    use_keys = list(get_recommended_keys)

                match = np.where(matches)[0][0]
                recommended_row_data = [idr3_data[file_index][match][k] \
                    for k in use_keys]

            logging.debug("({0}/{1}) iDR3 matched data for {2} is {3}".format(i,
                len(subset), row["CNAME"], recommended_row_data))
            idr3_recommended.append(recommended_row_data)


        # Join all the data together.
        columns_to_append = ["REMARK"]
        idr3_recommended = np.array(idr3_recommended)
        for i, key in enumerate(get_recommended_keys):
            if key in subset.columns.names:
                logging.debug("Overwriting column {0} with iDR3 data".format(key))
                subset[key] = idr3_recommended[:, i]

        # Put the field names in the REMARK column
        logging.debug("Writing field names to REMARK column")
        subset["REMARK"] = np.array(subset_field_numbers, dtype='|S250')

        # Save the file
        hdu = pyfits.BinTableHDU(subset)
        hdulist = pyfits.HDUList([pyfits.PrimaryHDU(), hdu])
        hdulist.writeto(matched_subset_filename)

    # Filter out high velocity stars
    is_high_velocity = np.abs(subset["VEL"]) > 500. # km/s

    # At each field pointing show:
    # 1) the velocity histogram
    # 2) the RA/DEC coloured by velocity
    # 3) metallicity vs alpha coloured by velocity

    plt.close("all")

    field_numbers = np.unique(subset["REMARK"])
    for field in field_numbers:
        members = subset["REMARK"] == field

        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.25, bottom=0.15, left=0.05, right=0.95)

        # Use only finite data, and stars that are not hyper velocity ones
        sensible_members = ~is_high_velocity * members \
            * np.isfinite(subset["VEL"])

        velocities = subset["VEL"][sensible_members]

        ax_vel, ax_vel_fancy, ax_pos, ax_feh = axes
        bin_size = 10 # km/s
        bins = np.arange(velocities.min(), velocities.max() + bin_size, bin_size)
        ax_vel.hist(subset["VEL"][sensible_members],
            bins=bins, facecolor="#666666")

        try:
            blocks_hist(velocities, bins="blocks",
                facecolor="#cccccc", edgecolor="#cccccc", ax=ax_vel_fancy)
        except:
            None

        try:
            blocks_hist(velocities, bins="knuth",
                facecolor="none", edgecolor="k", ax=ax_vel_fancy)
        except:
            None

        #vel_limits = (-500, 500)
        #ax_vel.set_xlim(*vel_limits)
        #ax_vel_fancy.set_xlim(*vel_limits)

        ax_vel.set_ylabel("N")
        ax_vel_fancy.set_ylabel("N")
        ax_vel.set_xlabel("$v$ (km s$^{-1}$)")
        ax_vel_fancy.set_xlabel("$v$ (km s$^{-1}$)")

        sensible_positions = members

        positions = (subset["RA"][sensible_positions],
            subset["DEC"][sensible_positions])
        scat = ax_pos.scatter(positions[0], positions[1],
            c=subset["VEL"][sensible_positions])

        cbar = plt.colorbar(scat)
        cbar.set_label("$v$ (km s$^{-1}$")

        ax_pos.set_xlabel("RA")
        ax_pos.set_ylabel("DEC")

        # Metallicities need to be MH *or* FEH, depending on the WG
        # ...or not?
        metallicities = subset["FEH"][sensible_members]

        # Alphas
        alphas = subset["ALPHA_FE"][sensible_members]
        ax_feh.scatter(metallicities, alphas, c=velocities)

        ax_feh.set_xlabel("[Fe/H]")
        ax_feh.set_ylabel("[$\\alpha$/Fe]")

        fig.savefig("field-{0}.png".format(field))

        plt.close()

# 168, 236
    raise a


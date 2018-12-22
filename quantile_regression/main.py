# coding: utf-8

from processing import (
    conditional_quantiles, quantile_regression, delta,
    importance_weights, density_weights, histogram
)

from figures import table_1, figure_1, figure_2


def main():
    year = "00"

    if year in [80, 90]:
        print("Processing data for the 19{} census ...".format(year))
    elif year == "00":
        print("Processing data for the 2000 census ...")

    # Obtains nonparametric estimates of the conditional quantiles of log
    # earnings given schooling. These estimates are just the sample quantiles
    # of log earnings for each level of schooling.
    conditional_quantiles(year)

    # Quantile regressions and Chamberlain’s minimum distance estimates
    # fitting conditional quantiles to schooling across cells. The outcomes of
    # this programs are the csv data files census80g.csv, which contains
    # estimates of the conditional quantiles, and the quantile regression
    # and Chamberlain fitted values for each level of schooling;
    # and census80qr.csv, which contains the quantile regression residuals
    # and specification errors for each individual.
    quantile_regression(year)

    # Auxiliary program for estimating the importance weights.
    # This program obtains, for each level of schooling, the grid of values
    # for log earnings where the density is estimated. Using the csv file
    # census80qr.csv, the outcome of this program is the file census80delta.csv
    # that contains the grid of values for each level of schooling.
    delta(year)

    # Derives the importance weights by estimating kernel densities in the grid
    # of points obtained from delta() and weighted-averaging these densities.
    # The results, together with the quantile regression weights
    # (importance weights × histogram of schooling), are saved in the csv file
    # census80gimp.csv.
    importance_weights(year)

    # Obtains the density weights by estimating kernel densities at the
    # nonparametric estimates of the conditional quantiles. This program uses
    # the csv file census80qr.csv created by quantile_regression() and saves
    # the results to census80gd.csv.
    density_weights(year)

    # Saves individual levels of schooling in the csv file census80i.csv to
    # generate the histogram.
    histogram(year)
    print("Done !")

    # Reproduces Figure N°1 of the paper
    print("Reproducing Figure N°1 ...")
    figure_1()
    print("Done !")

    # Reproduces Table N°1 of the paper
    print("Reproducing Table N°1 (as csv files)...")
    table_1()
    print("Done !")

    # Reproduces Figure N°2 of the paper
    print("Reproducing Figure N°2 ...")
    figure_2()
    print("Done !")


if __name__ == '__main__':
    main()

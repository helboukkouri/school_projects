from __future__ import division

import pandas as pd
from generators import (
    CQGenerator, QRGenerator, DeltaGenerator, IWGenerator, DWGenerator
)


import warnings
warnings.filterwarnings('ignore')


def conditional_quantiles(year):
    census_data = pd.read_stata("Data/census{}.dta".format(year))

    cq_generator = CQGenerator(census_data)
    result = cq_generator.process()

    result.to_csv("Data/census{}cq.csv".format(year))


def quantile_regression(year):
    census_data = pd.read_stata("Data/census{}.dta".format(year))
    census_cq = pd.read_csv("Data/census{}cq.csv".format(year))

    qr_generator = QRGenerator(census_data, census_cq)
    result = qr_generator.process()

    result[0].to_csv("Data/census{}qr.csv".format(year), index=False)
    result[1].to_csv("Data/census{}g.csv".format(year), index=False)


def delta(year):
    census_qr = pd.read_csv("Data/census{}qr.csv".format(year))

    delta_generator = DeltaGenerator(census_qr)
    result = delta_generator.process()

    result.to_csv("Data/census{}delta.csv".format(year), index=False)


def importance_weights(year):
    census_qr = pd.read_csv("Data/census{}qr.csv".format(year))
    census_delta = pd.read_csv("Data/census{}delta.csv".format(year))
    census_g = pd.read_csv("Data/census{}g.csv".format(year))

    imp_generator = IWGenerator(census_qr, census_delta, census_g)
    result = imp_generator.process()

    result.to_csv("Data/census{}gimp.csv".format(year), index=False)


def density_weights(year):
    census_qr = pd.read_csv("Data/census{}qr.csv".format(year))

    density_weights = DWGenerator(census_qr)
    result = density_weights.process()

    result.to_csv("Data/census{}gd.csv".format(year), index=False)


def histogram(year):
    census_data = pd.read_stata("Data/census{}.dta".format(year))
    census_i = census_data[["educ"]].astype(int)
    census_i.to_csv("Data/census{}i.csv".format(year), index=False)

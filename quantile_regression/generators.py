from __future__ import division

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import KernelDensity



QUANTILES_LIST = [10, 25, 50, 75, 90]


def silverman_factor(x):
    '''
    Computes the Silverman factor.
    :param x: A pandas.DataFrame column
    :return: the ideal bandwidth according to the Silverman rule of thumb
    '''
    iqr = x.quantile(.75) - x.quantile(.25)
    m = min(x.std()**2, iqr / 1.349)
    n = len(x)
    return (.9 * m) / n**(.2)


class CQGenerator(object):
    def __init__(self, data):
        self.data = data

    def process(self):
        census_data = self.data

        for i in [10, 25, 50, 75, 90]:
            census_data["cqlogwk_q%i" % i] = census_data.logwk

        columns = [
            'educ', 'cqlogwk_q10', 'cqlogwk_q25',
            'cqlogwk_q50', 'cqlogwk_q75', 'cqlogwk_q90'
        ]

        census_data_cq = census_data[columns]

        result = census_data_cq.groupby('educ').agg({
            'cqlogwk_q10': (lambda x: x.quantile(.1)),
            u'cqlogwk_q25': (lambda x: x.quantile(.25)),
            u'cqlogwk_q50': (lambda x: x.quantile(.5)),
            u'cqlogwk_q75': (lambda x: x.quantile(.75)),
            u'cqlogwk_q90': (lambda x: x.quantile(.9))
        })

        return result


class QRGenerator(object):
    def __init__(self, data, data_cq):
        self.data = data
        self.data_cq = data_cq

    @staticmethod
    def fit(model, q):
        res = model.fit(q=q)
        return np.int(q * 100), res.params['Intercept'], res.params['educ']

    @staticmethod
    def predict(intercept, slope, x):
        f = np.vectorize(lambda a, b, x: a + b * x)
        y = f(intercept, slope, x)
        return y

    def process(self):
        # data_qr creation
        data_qr = pd.merge(self.data_cq, self.data, how="inner", on="educ")
        data_qr["logwk_weighted"] = data_qr["logwk"] * data_qr["perwt"]

        QR = smf.quantreg('logwk_weighted ~ educ', data_qr)

        models = [self.fit(QR, q / 100.) for q in QUANTILES_LIST]
        models = pd.DataFrame(models, columns=['tau', 'a', 'b'])

        for tau in QUANTILES_LIST:
            a = models.loc[models.tau == tau, "a"]
            b = models.loc[models.tau == tau, "b"]
            x = data_qr["educ"]

            data_qr["qrlogwk_q" + str(tau)] = self.predict(a, b, x)

        ols = smf.ols('logwk_weighted ~ educ', data_qr).fit()
        intercept = ols.params['Intercept']
        slope = ols.params['educ']
        x = data_qr["educ"]
        data_qr["qrlogwk_ols"] = self.predict(intercept, slope, x)
        data_qr["cqlogwk_ols"] = data_qr["logwk"]

        for tau in QUANTILES_LIST:
            data_qr["delta_q" + str(tau)] = data_qr["qrlogwk_q" + str(tau)]
            data_qr["delta_q" + str(tau)] += - data_qr["cqlogwk_q" + str(tau)]

            data_qr["epsilon_q" + str(tau)] = data_qr["logwk"]
            data_qr["epsilon_q" + str(tau)] += - data_qr["cqlogwk_q" + str(tau)]

        # data_g creation
        collapse_mean = []
        for name in ['delta', 'qrlogwk', 'cqlogwk']:
            collapse_mean += [
                col for col in data_qr.columns if col.startswith(name)
            ]

        collapse_dico = {}
        for column in collapse_mean:
            collapse_dico[column] = 'mean'
        collapse_dico['perwt'] = 'sum'

        data_g = data_qr.groupby("educ").agg(collapse_dico)
        data_g.reset_index(inplace=True)
        data_g['educ'] = data_g['educ'].apply(lambda x: np.int(x))

        sperwt = sum(data_g['perwt'])
        data_g['preduc'] = data_g['perwt'] / sperwt

        for tau in QUANTILES_LIST:
            data_g['weighted_cqlogwk_q' + str(tau)] = data_g['cqlogwk_q' + str(tau)]
            # data_g['weighted_cqlogwk_q' + str(tau)] *= data_g['preduc']

            instructions = 'weighted_cqlogwk_q' + str(tau) + ' ~ educ'
            ols_tau = smf.ols(instructions, data_g).fit()
            a = ols_tau.params['Intercept']
            b = ols_tau.params['educ']
            x = data_g["educ"]

            data_g["cqrlogwk_q" + str(tau)] = self.predict(a, b, x)

        keep_columns = []
        for name in ['delta', 'preduc', 'educ',
                     'cqlogwk', 'qrlogwk', 'cqrlogwk']:

            keep_columns += [
                col for col in data_g.columns if col.startswith(name)
            ]

        data_g = data_g[keep_columns]
        data_g.sort_values('educ', inplace=True)

        result = [data_qr, data_g]
        return(result)


class DeltaGenerator(object):
    def __init__(self, qr):
        self.qr = qr

    def process(self):
        df = self.qr[["educ"] + ["delta_q%i" % i for i in QUANTILES_LIST]]
        for i in [10, 25, 50, 75, 90]:
            df["delta_q%i" % i] = df["delta_q%i" % i] * self.qr['perwt']

        df_educ = df.groupby("educ").mean()

        dict_values = []
        for tau in QUANTILES_LIST:
            for index in df_educ.index:
                dict_values.append(
                    ("delta%i_q%i" % (index, tau),
                     df_educ.loc[index]["delta_q%i" % tau])
                )
        dft = pd.DataFrame(dict_values).set_index(0).T
        result = pd.concat([dft] * 101).reset_index(drop='True')
        result["u"] = range(1, 102)
        return result


class IWGenerator(object):
    def __init__(self, qr, delta, g):
        self.qr = qr
        self.delta = delta
        self.g = g
        self.g.index = g["educ"]

    def process(self):
        data = self.delta
        data_qr = self.qr
        groups = data_qr.groupby("educ")

        for tau in [10, 25, 50, 75, 90]:
            for educ in groups.groups:
                educ = int(educ)
                estimation_points = groups.get_group(educ)
                estimation_points = estimation_points["epsilon_q{}".format(tau)]
                kde = KernelDensity(
                    kernel='gaussian',
                    bandwidth=silverman_factor(estimation_points)
                )

                kde.fit(estimation_points.values.reshape(-1, 1))

                s = "{}_q{}".format(educ, tau)
                data["epsilon" + s] = data["delta" + s] * (data["u"] - 1) / 100.

                data["wdensity" + s] = 1 - (data["u"] - 1) / 100.
                data["wdensity" + s] *= np.exp(
                    kde.score_samples(data["epsilon" + s].values.reshape(-1, 1))
                )

        means = data.mean(axis=0)
        for tau in [10, 25, 50, 75, 90]:
            data_qr["wdensity_q{}".format(tau)] = 0

            for educ in groups.groups:
                s = "{}_q{}".format(int(educ), tau)
                idx = groups.get_group(educ).index

                data_qr.loc[
                    idx, "wdensity_q{}".format(tau)
                ] = means["wdensity" + s]

        weights = groups.mean()[[c for c in data_qr if "density" in c]]
        weights.columns = ["impweight_" + c.split('_')[1] for c in weights]

        weights_rescaled = (weights / weights.sum(axis=0)).copy()
        weights_rescaled.columns = ["wqr5_" + c.split('_')[1] for c in weights]

        data = weights.join(weights_rescaled)
        for q in QUANTILES_LIST:
            col = data["wqr5_q" + str(q)] * self.g["preduc"]
            col = col / col.sum(axis=0)
            data["awqr5_q" + str(q)] = col

        data["educ"] = data.index.astype(int)

        return data


class DWGenerator(object):

    def __init__(self, qr):
        self.qr = qr

    def process(self):
        density_estimates = pd.DataFrame([])
        educ_groups = self.qr.groupby("educ")
        for tau in [10, 25, 50, 75, 90]:
            silverman_fact = silverman_factor(self.qr["epsilon_q%i" % tau])
            kde = KernelDensity(bandwidth=silverman_fact)
            values_tau = np.zeros(16)
            i = 0
            for index in educ_groups.groups:
                group_educ = educ_groups.get_group(index)
                values_tau[i] = self.estimate_density(
                    group_educ["epsilon_q%i" % tau], kde
                )

            density_estimates["dweight_q%i" % tau] = values_tau
            density_estimates["dwqr2_q%i" % tau] = self.normalize(
                density_estimates["dweight_q%i" % tau])
            i += 1
        density_estimates["educ"] = range(5, 21)
        return density_estimates

    @staticmethod
    def estimate_density(x, kde_instance):
        kde_instance.fit(
            x.values.reshape(-1, 1)
        )
        return np.exp(kde_instance.score_samples(0)) / 2.

    @staticmethod
    def normalize(x):
        return x / sum(x)


class TableGenerator(object):

    def __init__(self, census):
        self.census = census

    def fit_model(self, q, modele):
        res = modele.fit(q=q)
        return [np.int(q * 100),
                res.params['Intercept'],
                res.params['educ'],
                res.params['black'],
                res.params['exper'],
                res.params['exper2']]

    def process(self, census=None):
        if census is not None:
            self.census = census
        census80t2 = self.census.copy()

        census80t2["highschool"] = (census80t2["educ"] == 12).apply(
            lambda x: np.int(x)
        )
        census80t2["college"] = (census80t2["educ"] == 16).apply(
            lambda x: np.int(x)
        )

        QR = smf.quantreg('logwk ~ educ + black + exper + exper2', self.census)

        models = [self.fit_model(x / 100, QR) for x in QUANTILES_LIST]
        models = np.array(models)

        def get_y(a, b, x): return a + np.dot(x, b)

        for tau in QUANTILES_LIST:
            census80t2["q" + str(tau)] = census80t2["logwk"]
            census80t2["aq" + str(tau)] = get_y(
                models[models[:, 0] == tau, 1],
                models[models[:, 0] == tau, 2:6].T,
                census80t2[["educ", "black", "exper", "exper2"]]
            )

        census80t2.sort_values(["educ", "black", "exper"], inplace=True)

        collapse_mean = [
            col for col in census80t2.columns if col.startswith("aq")
        ]
        collapse_mean += ["highschool", "college", "exper2"]

        collapse_dico = {}
        for col in collapse_mean:
            collapse_dico[col] = lambda x: np.mean(x)

        collapse_dico["q10"] = lambda x: np.percentile(x, q=10)
        collapse_dico["q25"] = lambda x: np.percentile(x, q=25)
        collapse_dico["q50"] = lambda x: np.percentile(x, q=50)
        collapse_dico["q75"] = lambda x: np.percentile(x, q=75)
        collapse_dico["q90"] = lambda x: np.percentile(x, q=90)

        collapse_dico["perwt"] = "sum"

        census80t2g = census80t2.groupby(["educ", "black", "exper"]).agg(collapse_dico)
        census80t2g.reset_index(inplace=True)

        # Second part
        census80t2g["True"] = 1
        list_tab = []
        for cond in [
            census80t2g['educ'] >= 0,
            census80t2g['educ'] == 12,
            census80t2g['educ'] == 16
        ]:
            census80t2g_filtered = census80t2g.loc[cond, :]
            mean_list = [
                col for col in census80t2g_filtered.columns
                if col.startswith("aq") or col.startswith("q")
            ]
            mean_list += ["highschool", "college"]

            perwt = np.sum(census80t2g_filtered["perwt"])
            census80t2g_filtered['perwt'] = census80t2g_filtered["perwt"] / perwt

            census80t2g_agg = census80t2g_filtered[mean_list].aggregate(
                lambda x: np.average(x, weights=census80t2g_filtered['perwt'])
            )
            census80t2g_agg['perwt'] = np.int(perwt)

            for prefix in ["", "a"]:
                census80t2g_agg[prefix + 'd9010'] = census80t2g_agg[prefix + 'q90']
                census80t2g_agg[prefix + 'd9010'] -= census80t2g_agg[prefix + 'q10']

                census80t2g_agg[prefix + 'd7525'] = census80t2g_agg[prefix + 'q75']
                census80t2g_agg[prefix + 'd7525'] -= census80t2g_agg[prefix + 'q25']

                census80t2g_agg[prefix + 'd9050'] = census80t2g_agg[prefix + 'q90']
                census80t2g_agg[prefix + 'd9050'] -= census80t2g_agg[prefix + 'q50']

                census80t2g_agg[prefix + 'd5010'] = census80t2g_agg[prefix + 'q50']
                census80t2g_agg[prefix + 'd5010'] -= census80t2g_agg[prefix + 'q10']

            list_tab += [census80t2g_agg]

        return(census80t2g, list_tab)

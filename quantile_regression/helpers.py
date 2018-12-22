from __future__ import division

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf


def add_columns(df):
    df['weight * educ'] = df["weight"] * df["educ"]
    df["weight * exper"] = df["weight"] * df["exper"]
    df["weight * black"] = df["weight"] * df["black"]
    df["weight * exper2"] = df["weight"] * df["exper"] ** 2
    return df


def sigma(df, n, tau, res):
    df['sub'] = (tau - (res <= 0))
    df["weight"] = df["perwt"] * df['sub']
    add_columns(df)
    x = df[['weight', 'weight * educ', 'weight * exper',
            'weight * exper2', 'weight * black']]
    return np.dot(x.T.values, x.values) / float(n)


def sigma2(df, n, tau):
    df["weight"] = df["perwt"] * np.sqrt(tau * (1 - tau))
    add_columns(df)
    x = df[["weight"]]
    return np.dot(x.T.values, x.values) / float(n)


def sigma0(df, n, tau, res):
    df["weight"] = df["perwt"] * np.sqrt(tau * (1 - tau))
    add_columns(df)
    x = ["weight", "weight * educ", "weight * exper",
         "weight * exper2", "weight * black"]
    return np.dot(df[x].T.values, df[x].values) / float(n)


def jacobian(df, n, tau, res, alpha):
    hn = ((norm.ppf(1. - alpha / 2.) ** (2. / 3.))
          * ((1.5 * norm.pdf(norm.ppf(tau)) ** 2.)
             / (2. * norm.ppf(tau)**2 + 1.)) ** (1. / 3) * (n**(-1. / 3)))

    df["sub"] = (abs(res) <= hn)
    df["weight"] = np.sqrt(df["perwt"]) * df["sub"]
    add_columns(df)
    x = ["weight", "weight * educ", "weight * exper",
         "weight * exper2", "weight * black"]
    return(np.dot(df[x].T.values, df[x].values) / (2 * hn * float(n)))


def jacobian2(df, n, tau, res, alpha):
    hn = ((norm.ppf(1. - alpha / 2.) ** (2. / 3.))
          * ((1.5 * norm.pdf(norm.ppf(tau)) ** 2.)
             / (2. * norm.ppf(tau)**2 + 1.))**(1. / 3) * (n**(-1. / 3)))

    df["sub"] = (abs(res) <= hn)
    df["weight"] = np.sqrt(df["perwt"]) * df["sub"]
    x = ["weight"]
    return(np.dot(df[x].T.values, df[x].values) / (2 * hn * float(n)))


def subsamplek(formula, V, tau, coeffs, data, n, b, B, R):
    k = np.zeros(B)
    RVR = (np.float(np.dot(np.dot(R.T, V), R) / b))**(-1 / 2)
    probs = np.array(data['perwt']) / np.sum(np.array(data['perwt']))
    for s in range(B):
        sing = 0
        while sing == 0:
            sample = np.random.choice(np.arange(0, n), size=int(b),
                                      replace=True, p=probs)
            sdata = data.iloc[sample, :]

            x = sdata[["educ", "exper", "exper2", "black", "perwt"]]
            x = x.as_matrix()
            sing = np.linalg.det(np.dot(x.T, x))
        # Didn't use weights here
        sqr_model = smf.quantreg(formula, sdata)
        sqr = sqr_model.fit(q=tau)
        k[s] = np.abs(np.dot(np.dot(RVR, R.T), coeffs - np.array(sqr.params)))
    return(k)


def table_rq_res(formula, taus, data, alpha, R,  n, sigma, jacobian):
    m = len(taus)
    tab = pd.DataFrame([], index=[0])
    setab = pd.DataFrame([], index=[0])
    for i in range(m):
        fit_model = smf.quantreg(formula, data)
        fit = fit_model.fit(q=taus[i])
        coeff = np.dot(R.T, np.array(fit.params))
        tab[str(i)] = coeff
        sigmatau = sigma(data, n, taus[i], fit.resid)
        jacobtau = jacobian(data, n, taus[i], fit.resid, alpha)
        solved_jacobtau = np.linalg.inv(jacobtau)
        V = np.dot(np.dot(solved_jacobtau, sigmatau), solved_jacobtau) / n
        secoeff = np.float(np.dot(np.dot(R.T, V), R))**.5
        setab[str(i)] = secoeff
    tab = tab.transpose()
    setab = setab.transpose()
    return(tab, setab)

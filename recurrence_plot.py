""" Plot frequency-magnitude and recurrence plot from rainfall timeseries """

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime

from plot_utils import initialize_plot_settings
matplotlib.style.use('ggplot')
initialize_plot_settings()


def main():
    files = os.listdir('data')
    files = [f for f in files if 'ghcn_' in f and '.csv' in f]
    for f in files:
        site = f.split('_')[1].split('.')[0]
        print('Processing {}...'.format(site))
        df = pd.read_csv('data/' + f)
        df.precip *= 10 # provided in 0.1 mm units by GHCN
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        #df = df[df.dist == df.dist.min()]
        
        mean = np.mean(df.precip.values)
        med = np.percentile(df.precip.values, 50)
        p90 = np.percentile(df.precip.values, 90)

        print('Mean: {:.0f}'.format(mean))
        print('P50: {:.0f}'.format(med))
        print('P90: {:.0f}'.format(p90))
        print('Peak to median: {:.2f}'.format(p90 / med))
        print('Peak to mean: {:.2f}'.format(p90 / mean))
        
        #df = df[df.precip > 0]

        #plot_monthly_max(df)
        #plt.savefig('fig/monthly_' + site + '.png', dpi=150, bbox_inches='tight')
        #plot_frequency(df)
        #plt.savefig('fig/freq_' + site + '.png', dpi=150, bbox_inches='tight')

        plot_monthly_mean(df)
        plt.savefig('fig/monthlymean_' + site + '.png', dpi=150, bbox_inches='tight')
        #plot_monthly_max(df)
        #plt.savefig('fig/monthlymax_' + site + '_nearest_nils.png', dpi=150, bbox_inches='tight')
        plot_log_cdf(df)
        plt.savefig('fig/cdf_' + site + '.png', dpi=150, bbox_inches='tight')
        #plot_log_probability(df)
        #plt.savefig('fig/pdf_monthlymean_' + site + '_nearest.png', dpi=150, bbox_inches='tight')
        #plot_frequency(df)
        #plt.savefig('fig/freq_' + site + '_nearest.png', dpi=150, bbox_inches='tight')
        #plt.close('all')
        
#    fig, ax = plt.subplots(1,1,)
#    for f in files:
#        site = f.split('_')[1].split('.')[0]
#        print('Processing {}...'.format(site))
#        df = pd.read_csv('data/' + f)
#        df['date'] = pd.to_datetime(df['date'])
#        df = df.set_index('date')
#        df = df[df.dist == df.dist.min()]
#        df = df[df.precip > 0]
#        plot_cdf_sqrt(df, site)
#
#    ax.set_ylabel('Cumulative probability')
#    ax.set_xlabel('Rainfall$^{1/2}$ [mm$^0.5$]')
#    ax.legend()
#    plt.savefig('fig/cdf_nearest.png', dpi=150, bbox_inches='tight')
#    plt.close('all')


def calculate_recurrence(df):
    pass


def plot_cdf_sqrt(df, label):
    ax.hist(np.sqrt(df.precip), bins=100, normed=True, cumulative=True, histtype='step', label=label, lw=1, alpha=0.9)


def plot_monthly_max(df):
    g = df.groupby(df.index.month).max()
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    g.plot.bar(y='precip', facecolor=colors[0], edgecolor='k', legend=None)

    ax = plt.gca()
    months = ['January', '', 'March', '', 'May', '', 'July', '', 'September', '', 'November', '']
    ax.set_xticklabels(months, rotation=45)
    ax.set_xlabel('Month')
    ax.set_ylabel('Rainfall [mm]')
    
    text = summary_text(df)
    ax.text(0.05, 1.05, text, transform=ax.transAxes)

    plt.show(block=False)


def plot_monthly_mean(df):
    g = df.groupby(df.index.month).mean()
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    g.plot.bar(y='precip', facecolor=colors[1], edgecolor='k', legend=None)

    ax = plt.gca()
    months = ['January', '', 'March', '', 'May', '', 'July', '', 'September', '', 'November', '']
    ax.set_xticklabels(months, rotation=45)
    ax.set_xlabel('Month')
    ax.set_ylabel('Rainfall [mm]')
    
    text = summary_text(df)
    ax.text(0.05, 1.05, text, transform=ax.transAxes)

    plt.show(block=False)


def plot_frequency(df):
    data = df.precip.values
    n = len(data)
    data.sort()
    data = data[::-1]
    rank = np.arange(n) + 1
    prob = 100 * rank / (n + 1)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(prob, data, 'ks')

    x = np.log10(prob)
    x = sm.add_constant(x)
    model = sm.OLS(data, x)
    fitted = model.fit()

    x_pred = np.linspace(x.min(), x.max(), num=1000)
    x_pred = sm.add_constant(x_pred)
    y_pred = fitted.predict(x_pred)

    B0, B1 = fitted.params
    pval = fitted.pvalues[1]
    r2 = fitted.rsquared

    if pval < 0.005:
        label = '{:.2f} $\log$ P + {:.2f}\n($p$ < 0.005, $R^2$ = {:.2f})'.format(B1, B0, r2)
    else:
        label = '{:.2f} $\log$ P + {:.2f}\n($p$ = {:.3f}, $R^2$ = {:.2f})'.format(B1, B0, pval, r2)

    label += '\nQ$_{50}$' + ' = {:.0f} mm'.format(fitted.predict([1., np.log10(50)])[0])
    label += '\nQ$_{1}$' + ' = {:.0f} mm'.format(fitted.predict([1., np.log10(1)])[0])

    ax.plot(10 ** x_pred[:, 1], y_pred, '-', lw=1)

    ax.set(xscale='log')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Rainfall [mm]')

    text = summary_text(df)
    text += '\n\n' + label
    
    ax.text(0.6, 0.8, text, transform=ax.transAxes)


def plot_histogram(df):
    fig, ax = plt.subplots(1,1)
    ax.hist(df.precip, bins=100, normed=True)
    ax.set_ylabel('Normalized frequency')
    ax.set_xlabel('Rainfall [mm]')


def plot_histogram_sqrt(df):
    fig, ax = plt.subplots(1,1)
    ax.hist(np.sqrt(df.precip), bins=100, normed=True)
    ax.set_ylabel('Normalized frequency')
    ax.set_xlabel('Rainfall$^{1/2}$ [mm$^{0.5}$]')


def plot_min_median_max(df):
    pass


def plot_log_cdf(df):
    mean = np.mean(df.precip.values)
    med = np.percentile(df.precip.values, 50)
    p90 = np.percentile(df.precip.values, 90)

    print('Mean: {:.0f}'.format(mean))
    print('P50: {:.0f}'.format(med))
    print('P90: {:.0f}'.format(p90))
    print('Peak to median: {:.2f}'.format(p90 / med))
    print('Peak to mean: {:.2f}'.format(p90 / mean))

    #data = np.log10(df.precip.values)
    data = df.precip.values
    
    fig, ax = plt.subplots(1,1)
    bins = np.logspace(0, 5)
    ax.hist(data, bins=bins, normed=True, histtype='stepfilled', cumulative=True)

    ax.set(xscale='log')
    ax.set_ylabel('Cumulative frequency')
    ax.set_xlabel('Rainfall [mm]')

    text = summary_text(df)
    ax.text(0.05, 0.8, text, transform=ax.transAxes)


def plot_log_probability(df):
    data = np.log10(df.precip.values)
    
    fig, ax = plt.subplots(1,1)
    bins = np.logspace(0, 5)
    ax.hist(data, bins=bins, normed=True, histtype='stepfilled')

    ax.set(xscale='log')
    ax.set_ylabel('Normalized frequency')
    ax.set_xlabel('Rainfall [mm]')

    text = summary_text(df)
    ax.text(0.6, 0.8, text, transform=ax.transAxes)


def plot_probability_sqrt(df):
    data = np.sqrt(df.precip.values)
    n = len(data)
    data.sort()
    data = data[::-1]
    rank = np.arange(n) + 1
    prob = 100 * rank / (n + 1)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(data, prob, 'ks')

    ax.set_ylabel('Probability [%]')
    ax.set_xlabel('Rainfall$^{1/2}$ [mm$^0.5$]')

    text = '{} - {}\n$n$ = {}'.format(df.date.min()[0:4], df.date.max()[0:4], len(df))
    ax.text(0.6, 0.8, text, transform=ax.transAxes)


def plot_recurrence(df):
    pass


def summary_text(df):
    text = '{} - {}\n$n$ = {}'.format(df.index.year.min(), df.index.year.max(), len(df))

    n_stations = len(np.unique(df.station.values))
    if n_stations > 1:
        text += '\n{} stations'.format(n_stations)
    else:
        text += '\nGHCN Station: ' + df.station.iloc[0] # + '\nDistance: {:.2f} deg.'.format(df.dist.iloc[0])

    return text



if __name__ == "__main__":
    main()

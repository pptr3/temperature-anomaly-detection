#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq

# Configuration
anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'
figsize=(9, 3)
autoclose = True

def load_series(file_name, data_folder):
    # Load the input data
    data_path = f'{data_folder}/data/{file_name}'
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    # Load the labels
    label_path = f'{data_folder}/labels/combined_labels.json'
    with open(label_path) as fp:
        labels = pd.Series(json.load(fp)[file_name])
    labels = pd.to_datetime(labels)
    # Load the windows
    window_path = f'{data_folder}/labels/combined_windows.json'
    window_cols = ['begin', 'end']
    with open(window_path) as fp:
        windows = pd.DataFrame(columns=window_cols,
                data=json.load(fp)[file_name])
    windows['begin'] = pd.to_datetime(windows['begin'])
    windows['end'] = pd.to_datetime(windows['end'])
    # Return data
    return data, labels, windows


def plot_series(data, labels=None,
                    windows=None,
                    predictions=None,
                    highlights=None,
                    val_start=None,
                    test_start=None,
                    figsize=figsize,
                    show_sampling_points=False,
                    show_markers=False,
                    filled_version=None):
    # Open a new figure
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    # Plot data
    if not show_markers:
        plt.plot(data.index, data.values, zorder=0)
    else:
        plt.plot(data.index, data.values, zorder=0,
                marker='.', markersize=3)
    if filled_version is not None:
        filled = filled_version.copy()
        filled[~data['value'].isnull()] = np.nan
        plt.scatter(filled.index, filled,
                marker='.', c='tab:orange', s=3);
    if show_sampling_points:
        vmin = data.min()
        lvl = np.full(len(data.index), vmin)
        plt.scatter(data.index, lvl, marker='.',
                c='tab:red', s=3)
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color=anomaly_color, zorder=2, s=3)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color=anomaly_color, alpha=0.3, zorder=1)
    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3,
                    s=3)
    plt.tight_layout()


def plot_autocorrelation(data, max_lag=100, figsize=figsize):
    # Open a new figure
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    # Autocorrelation plot
    pd.plotting.autocorrelation_plot(data['value'])
    # Customized x limits
    plt.xlim(0, max_lag)
    # Rotated x ticks
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_histogram(data, bins=10, vmin=None, vmax=None, figsize=figsize):
    # Build a new figure
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist(data, density=True, bins=bins)
    # Update limits
    lims = plt.xlim()
    if vmin is not None:
        lims = (vmin, lims[1])
    if vmax is not None:
        lims = (lims[0], vmax)
    plt.xlim(lims)
    plt.tight_layout()


def plot_histogram2d(xdata, ydata, bins=10, figsize=figsize):
    # Build a new figure
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist2d(xdata, ydata, density=True, bins=bins)
    plt.tight_layout()


def plot_density_estimator_1D(estimator, xr, figsize=figsize):
    # Build a new figure
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    # Plot the estimated density
    xvals = xr.reshape((-1, 1))
    dvals = np.exp(estimator.score_samples(xvals))
    plt.plot(xvals, dvals)
    plt.tight_layout()


def plot_density_estimator_2D(estimator, xr, yr, figsize=figsize):
    # Plot the estimated density
    nx = len(xr)
    ny = len(yr)
    xc = np.repeat(xr, ny)
    yc = np.tile(yr, nx)
    data = np.vstack((xc, yc)).T
    dvals = np.exp(estimator.score_samples(data))
    dvals = dvals.reshape((nx, ny))
    # Build a new figure
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    plt.pcolor(dvals)
    plt.tight_layout()
    # plt.xticks(np.arange(0, len(xr)), xr)
    # plt.yticks(np.arange(0, len(xr)), yr)


def get_pred(signal, thr):
    return pd.Series(signal.index[signal >= thr])


def get_metrics(pred, labels, windows):
    tp = [] # True positives
    fp = [] # False positives
    fn = [] # False negatives
    advance = [] # Time advance, for true positives
    # Loop over all windows
    used_pred = set()
    for idx, w in windows.iterrows():
        # Search for the earliest prediction
        pmin = None
        for p in pred:
            if p >= w['begin'] and p < w['end']:
                used_pred.add(p)
                if pmin is None or p < pmin:
                    pmin = p
        # Compute true pos. (incl. advance) and false neg.
        l = labels[idx]
        if pmin is None:
            fn.append(l)
        else:
            tp.append(l)
            advance.append(l-pmin)
    # Compute false positives
    for p in pred:
        if p not in used_pred:
            fp.append(p)
    # Return all metrics as pandas series
    return pd.Series(tp), \
            pd.Series(fp), \
            pd.Series(fn), \
            pd.Series(advance)


class ADSimpleCostModel:

    def __init__(self, c_alrm, c_missed, c_late):
        self.c_alrm = c_alrm
        self.c_missed = c_missed
        self.c_late = c_late

    def cost(self, signal, labels, windows, thr):
        # Obtain predictions
        pred = get_pred(signal, thr)
        # Obtain metrics
        tp, fp, fn, adv = get_metrics(pred, labels, windows)
        # Compute the cost
        adv_det = [a for a in adv if a.total_seconds() <= 0]
        cost = self.c_alrm * len(fp) + \
           self.c_missed * len(fn) + \
           self.c_late * (len(adv_det))
        return cost


def opt_thr(signal, labels, windows, cmodel, thr_range):
    costs = [cmodel.cost(signal, labels, windows, thr)
            for thr in thr_range]
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return thr_range[best_idx], costs[best_idx]


def sliding_window_1D(data, wlen):
    assert(len(data.columns) == 1)
    # Get shifted columns
    m = len(data)
    lc = [data.iloc[i:m-wlen+i+1].values for i in range(0, wlen)]
    # Stack
    wdata = np.hstack(lc)
    # Wrap
    wdata = pd.DataFrame(index=data.index[wlen-1:],
            data=wdata, columns=range(wlen))
    return wdata


def plot_prediction_scatter(target, pred, figsize=figsize):
    plt.figure(figsize=figsize)
    plt.scatter(target, pred, marker='x', alpha=.3, s=3)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([0, plt.xlim()[1]], [0, plt.ylim()[1]], ':', c='black')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(linestyle=':')
    plt.tight_layout()


def apply_differencing(data, lags):
    deltas = []
    data_d = data.copy()
    for d in lags:
        delta = data_d.iloc[:-d]
        data_d = data_d.iloc[d:] - delta.values
        deltas.append(delta)
    return data_d, deltas


def deapply_differencing(pred, deltas, lags, extra_wlen=0):
    dsum = 0
    pred_dd = pred.copy()
    for i, d in reversed(list(enumerate(lags))):
        delta = deltas[i].values.reshape((-1,))
        pred_dd = pred_dd + delta[extra_wlen+dsum:]
        dsum +=  d
    return pred_dd


def separate_normal_behavior(data, labels, windows,
        start=None, end=None):
    # Apply the start/end mask to the windows
    windows_sep = windows.copy()
    if start is not None:
        windows_sep = windows_sep[windows_sep['begin'] >= start]
    if end is not None:
        windows_sep = windows_sep[windows_sep['end'] < end]
    # Apply the start/end mask to the labels
    labels_sep = labels.copy()
    if start is not None:
        labels_sep = labels_sep[labels >= start]
    if end is not None:
        labels_sep = labels_sep[labels < end]
    # Apply start/end mask to the data
    mask = np.full(len(data), True)
    if start is not None:
        mask = mask & (data.index >= start)
    if end is not None:
        mask = mask & (data.index < end)
    # Remove data within windows
    for _, w in windows_sep.iterrows():
        m2 = (data.index < w['begin']) | \
             (data.index >= w['end'])
        mask = mask & m2
    return data[mask], labels_sep, windows_sep


def binning_avg(data, labels, binsize):
    rit = data.rolling(window=binsize)
    datab = rit.mean()[::binsize].dropna()
    # Map labels to the closest (early) index
    labelsb = []
    for lbl in labels:
        maplbl = datab.index[datab.index <= lbl][-1]
        labelsb.append(maplbl)
    labelsb = pd.Series(labelsb)
    return datab, labelsb


def densify(data, freq):
    # Build a dense index
    dx = pd.date_range(data.index[0], data.index[-1], freq=freq)
    # Build a dataframe with a dense index
    dd = pd.DataFrame(index=dx, columns=data.columns)
    # Build a realigned index for the original data
    rx = data.index.round(freq=freq)
    # Assign the realigned index
    rd = data.set_index(rx)
    # Remove duplicates (keep first occurrence)
    rd = rd[~rd.index.duplicated()]
    # Assign all columns
    for col in data.columns:
        dd[col] = rd[col]
    return dd

def plot_gp(target=None, pred=None, std=None, samples=None,
        target_samples=None, figsize=figsize):
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(target.index, target, c='black', label='target')
    if pred is not None:
        plt.plot(pred.index, pred, c='tab:blue',
                label='predictions')
    if std is not None:
        plt.fill_between(pred.index, pred-1.96*std, pred+1.96*std,
                alpha=.3, fc='tab:blue', ec='None',
                label='95% C.I.')
    # Add scatter plots
    if samples is not None:
        try:
            x = samples.index
            y = samples.values
        except AttributeError:
            x = samples[0]
            y = samples[1]
        plt.scatter(x, y, color='tab:orange',
              label='samples', s=3)
    if target_samples is not None:
        try:
            x = target_samples.index
            y = target_samples.values
        except AttributeError:
            x = target_samples[0]
            y = target_samples[1]
        plt.scatter(x, y,
                color='black', label='target', s=3)
    plt.legend()
    plt.tight_layout()


def plot_fft_abs(data):
    n = len(data)
    # Compute the FFT
    y = fft(data)
    # Obtain the corresponding frequencies
    f = fftfreq(n)
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(f[1:n//2], np.abs(y[1:n//2]))
    plt.tight_layout()
    # Return results
    return f[1:n//2], np.abs(y[1:n//2]).ravel()

def plot_bars(data, figsize=figsize, generate_x=False):
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    if generate_x:
        x = 0.5 + np.arange(len(data))
    else:
        x = data.index
    plt.bar(x, data, width=0.7)
    plt.xticks(x[::10], data.index[::10])
    plt.tight_layout()

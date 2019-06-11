import os
import re
from datetime import datetime

import numpy as np
from scipy.io import wavfile
import pandas as pd


class EEG:
    """
    A class for loading eeg data saved as a wavefile with some metadata on teh side, handles both 'newer' and 'older' style formats
    Resamples data to a target_samplerate
    """

    def __init__(self, wf, tf, target_samplerate=666.6, tone_names=None, subtract_baseline=True):
        if tone_names is None:
            tone_names = {'1': 'standard', '2': 'oddball'}

        # Read in wave data
        samplerate, data = wavfile.read(wf)
        # print(samplerate, data.shape)

        # Extract date
        yearstr = os.path.splitext(os.path.split(wf)[1])[0]
        dt = datetime.strptime(yearstr[len('BYB_Recording_'):len('BYB_Recording_') + 19], '%Y-%m-%d_%H.%M.%S')  # Reference -  http://strftime.org/
        self.datetime = dt
        # Determines how to read in the data based on the yea the data was collected
        if dt > datetime(year=2016, month=1, day=1):
            # New format
            toneonsets = pd.read_csv(tf, sep=',\t')
            wavedata = data.copy()
        else:
            # Old format
            # TODO read in eye positions
            eyes = pd.read_csv(tf, sep=',\t')
            wavedata = data[:, 0].copy()

            # Extract tone onsets from all channels in dataset
            all_onsets = []
            for ch_num in range(1, data.shape[1]):
                baseline = np.median(data[:, ch_num])

                thresholded = (data[:, ch_num] - baseline) > 1
                thresholded_idx = np.where(thresholded)[0]
                dif = np.diff(thresholded_idx)

                # Determines where the tone onset occurs based on signal deviation from baseline above 1000
                onsets = thresholded_idx[1:][dif > 1000]
                onsets = np.hstack((thresholded_idx[0], onsets)) / samplerate

                # Appends all tone onset timepoints to the same list
                all_onsets.append(pd.Series(onsets, index=[str(ch_num), ] * len(onsets)))

            # now tones are extracted, convert to pandas series
            toneonsets = pd.concat(all_onsets)
            toneonsets.sort_values(inplace=True)  # Reorder by time

        # Now old and new data are in the same format
        toneonsets = toneonsets.rename(tone_names)
        self.toneonsets = toneonsets
        self.samplerate = target_samplerate

        if subtract_baseline:
            wavedata -= np.median(wavedata).astype(wavedata.dtype)

        # resample to a fixed sample rate
        # have wavedata at samplerate, and want it at target_samplerate
        sample_times = np.arange(len(wavedata)) / samplerate
        new_sample_times = np.linspace(0, sample_times.max(), sample_times.max() * target_samplerate)
        wavedata_resamp = np.interp(new_sample_times, sample_times, wavedata)

        self.sample_times = new_sample_times
        self.wavedata = wavedata_resamp

    @property
    def duration(self):
        """Duration in seconds"""
        return self.sample_times.max() - self.sample_times.min()

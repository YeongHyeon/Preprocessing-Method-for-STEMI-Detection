import os, inspect, glob, random, scipy, peakutils, argparse
from scipy import signal
from wfdb import processing

import numpy as np

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

class Preprocess(object):

    def __init__(self, rawpath, setname="BP", fs=500, nfft=4096, rmfreqs=[60, 120, 180, 240], outdim=600):

        self.rawpath = rawpath
        self.setname = setname
        self.fs = fs
        self.nfft = nfft
        self.rmfreqs = rmfreqs
        self.outdim = outdim

    def make(self):

        self.makedir(path="dataset_%s" %(self.setname))

        subclasses = glob.glob(os.path.join(self.rawpath, "*"))
        subclasses.sort()
        for sidx, subclass in enumerate(subclasses):
            subname = subclass.split("/")[-1]
            print("\n%s" %(subname))
            self.makedir(path=os.path.join("dataset_%s" %(self.setname), subname))

            npys = glob.glob(os.path.join(subclass, "*.npy"))
            npys.sort()
            for nidx, npy in enumerate(npys):
                npyname = npy.split("/")[-1].replace(".npy", "")
                print(npyname)
                data = np.load(npy)
                origin = data.copy()
                data = data[:, 250:5250]

                maxlen = data.shape[1]

                Y, x_freq, x_freq_val = self.fast_fourier_transform(sig=data[0], fs=self.fs, nfft=self.nfft)

                x_notch = np.zeros_like(data)
                x_high = np.zeros_like(data)
                x_total = np.zeros_like(data)
                for didx, dat in enumerate(data):
                    if("N" in self.setname):
                        x_total[didx] = self.notchfilter(sig=dat, fs=self.fs, freqs=self.rmfreqs, Q=1)
                    elif("H" in self.setname):
                        x_total[didx] = self.highpassfilter(data=dat, cutoff=1, fs=self.fs)
                    elif("B" in self.setname):
                        x_notch[didx] = self.notchfilter(sig=dat, fs=self.fs, freqs=self.rmfreqs, Q=1)
                        x_total[didx] = self.highpassfilter(data=x_notch[didx], cutoff=1, fs=self.fs)
                    else:
                        x_total[didx] = dat

                if(("N" in self.setname) and not("NP" in self.setname) or
                    ("H" in self.setname) and not("HP" in self.setname) or
                    ("B" in self.setname) and not("BP" in self.setname) or
                    ("R" in self.setname) and not("RP" in self.setname)):
                    np.save(os.path.join("dataset_%s" %(self.setname), subname, "%s" %(npyname)), x_total)
                else:
                    """Start point of peak voting process"""
                    x_vote = np.zeros((maxlen))

                    x_total_flip = x_total * (-1)

                    peak_indices = []
                    for cidx in range(12):
                        indices = self.peak_selection(signal=x_total[cidx], threshold=0.8)
                        indices_flip = self.peak_selection(signal=x_total_flip[cidx], threshold=0.8)
                        peak_indices.append(indices)
                        peak_indices.append(indices_flip)
                        for idx in indices:
                            x_vote[idx-10:idx+10] += 1
                        for idx in indices_flip:
                            x_vote[idx-10:idx+10] += 1
                    x_vote[:250] /= 10
                    x_vote[250:] /= 10
                    indices = self.peak_selection(signal=x_vote, threshold=0.5)
                    indices_filtered, interval = self.peak_filtering(indices=indices, maxlen=maxlen)
                    """End point of peak voting process"""

                    """Start point of slicing process"""
                    for i, pidx in enumerate(indices_filtered):

                        term = int(interval / 2)
                        sp, ep = pidx - term, pidx + term
                        if(sp < 0): sp = 0
                        if(ep >= x_total.shape[1]): ep = (x_total.shape[1]-1)
                        if(abs(sp-ep) < interval*0.9): continue

                        rows = np.zeros((0, abs(sp-ep)))
                        for idx in range(12):
                            row = x_total[idx][sp:ep].reshape((1, abs(sp-ep)))
                            rows = np.append(rows, row, axis=0)
                        rows = rows.T

                        rows = self.linear_interpolation(data=rows, outdim=self.outdim)
                        rows = self.range_regularization(data=rows)
                        np.save(os.path.join("dataset_%s" %(self.setname), subname, "%s_%d" %(npyname, i)), rows)
                    """End point of slicing process"""

    def makedir(self, path):
        try: os.mkdir(path)
        except: pass

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpassfilter(self, data, cutoff, fs, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def notchfilter(self, sig, fs=500, freqs=[60], Q=1):

        for f0 in freqs:
            w0 = f0/(fs/2)  # Normalized Frequency
            # Design notch filter
            b, a = signal.iirnotch(w0, Q)

            nfft = 4096
            cp = int(nfft/2)

            sig = scipy.signal.lfilter(b=b, a=a, x=sig)

        return sig

    def fast_fourier_transform(self, sig, fs=500, nfft=4096):

        sig_ft = np.fft.fft(sig, n=nfft)

        fftn = int(nfft/2)
        fstep = (fs/2)/fftn

        x_freq, x_freq_val = [], []
        cnt = 0
        for i in range(fftn):
            if(int(i*fstep) == int(50*cnt)):
                cnt += 1
                x_freq.append(i)
                x_freq_val.append(int(i*fstep))

        return abs(sig_ft), x_freq, x_freq_val

    def magnitude2dB(self, mag):

        mag[0] = 0 # Remove DC term
        db = (np.log(mag+1e-9) / np.log(np.ones_like(mag)*10)) * 10

        return db

    def peak_selection(self, signal, fs=500, threshold=0.2):

        while(True):
            indices = peakutils.indexes(signal, thres=threshold, min_dist=100)

            if(len(indices) >= int((signal.shape[0]/fs) - 1)): break
            else: threshold *= 0.95

        return indices

    def peak_filtering(self, indices, fs=500, maxlen=5500):

        interval = 0
        limit = 1.0
        while(True):
            for i, idx in enumerate(indices):
                if(i != 0):
                    interval_tmp = abs(indices[i] - indices[i-1])
                    if((interval_tmp > interval) and (interval_tmp < (fs*limit))): interval = interval_tmp
            if(interval != 0): break
            else: limit += 0.1

        indices = list(indices)

        indices.reverse()
        i = 0
        while(True):
            try:
                # print(i, indices[i], indices[i+1], indices[i] - indices[i+1], abs(indices[i] - indices[i+1]))
                if(abs(indices[i] - indices[i+1]) < (interval * 0.7)):
                    try:
                        if(abs(indices[i] - indices[i+1]) < abs(indices[i+1] - indices[i+2])): indices.pop(i)
                        else: indices.pop(i+1)
                    except: indices.pop(i)
                else: i += 1
            except: break

        indices.reverse()
        i = 0
        while(True):
            try:
                if(indices[i] - interval < 0): indices.pop(i)
                elif(indices[i] + interval > maxlen): indices.pop(i)
                else: i += 1
            except: break

        return indices, interval

    def range_regularization(self, data):
        if(np.min(data) < 0): data += abs(np.min(data))
        else: data -= abs(np.min(data))
        data /= np.max(data)

        return data

    def linear_interpolation(self, data, outdim):

        inter_unit = outdim / data.shape[0]

        outdata = np.zeros((outdim, data.shape[1]))

        for sigidx in range(data.shape[0]):
            x1 = int((sigidx-1)*inter_unit)
            x2 = int((sigidx+1)*inter_unit)
            if(sigidx == data.shape[0]-1): x2 = outdim - 1
            for chdix in range(data.shape[1]):
                outdata[x2][chdix] = data[sigidx][chdix]
                if(sigidx != 0):
                    diff = (data[sigidx][chdix] - data[sigidx-1][chdix]) / (x2 - x1)
                    for inter in range(x2-x1):
                        if(inter == 0): continue
                        else: outdata[x1+inter][chdix] = data[sigidx-1][chdix] + (inter * diff)

        return outdata

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--rawpath', type=str, default="SNUBH_ECG", help='Path of rawdata.')
    parser.add_argument('--set', type=str, default="BP", help='Kind of dataset.')
    parser.add_argument('--fs', type=int, default=500, help='Sampling rate of raw data.')
    parser.add_argument('--nfft', type=int, default=4096, help='FFT point for Fourier transform.')
    parser.add_argument('--rmfreq', type=int, default=60, help='Frequency for Notch filter.')
    parser.add_argument('--outdim', type=int, default=600, help='Dimension of output.')

    FLAGS, unparsed = parser.parse_known_args()

    rmfreqs = []
    for i in range(4):
        rmfreqs.append(FLAGS.rmfreq * (i+1))

    process = Preprocess(rawpath=FLAGS.rawpath, setname=FLAGS.set.upper(), fs=FLAGS.fs, nfft=FLAGS.nfft, rmfreqs=rmfreqs, outdim=FLAGS.outdim)
    process.make()

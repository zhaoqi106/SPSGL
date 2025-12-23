import glob
import os
import re
import pandas as pd
from joblib import dump, load
import numpy as np
from scipy import signal
def get_mean_std(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    result = np.stack((mean, std), axis=1)
    return result

def choose_nperseg(num_time: int, cap: int = 256, overlap_ratio: float = 0.5):
    m = max(1, min(cap, int(num_time)))
    nperseg = 1 if m == 1 else 2 ** int(np.floor(np.log2(m)))
    noverlap = int(nperseg * overlap_ratio)
    noverlap = min(noverlap, nperseg - 1)
    return nperseg, noverlap

def get_per_freq_features(time_series_array, fs=1.0 / 0.72,
                                     bands=[(0.01, 0.04), (0.04, 0.07), (0.07, 0.1), (0.01, 0.1)],
                                     cap=128, overlap_ratio=0.5):
    num_roi, num_time = time_series_array.shape
    nperseg_val, noverlap_val = choose_nperseg(num_time, cap=cap, overlap_ratio=overlap_ratio)
    num_feature = len(bands)
    features = np.zeros((num_roi, num_feature))
    freqs, psd = signal.welch(time_series_array, fs=fs, axis=1,
                              nperseg=nperseg_val, noverlap=noverlap_val,
                              detrend='linear') # 添加了 detrend 和 noverlap

    for b, (low, high) in enumerate(bands):
        idx = (freqs >= low) & (freqs < high)
        if idx.sum() > 1:
            band_power = np.trapz(psd[:, idx], freqs[idx], axis=1)
        elif idx.sum() == 1:
            if len(freqs) > 1:
                df = freqs[1] - freqs[0]
            elif len(freqs) == 1:
                df = freqs[0]
            else:
                df = 0
            band_power = psd[:, idx].flatten() * df
        else:
            band_power = np.zeros(num_roi)
        features[:, b] = band_power

    return features

if __name__ == '__main__':
    fc_folder = ""
    series_time_folder = ""
    labels_dict = {}
    save_path = ""
    pattern = re.compile(r"^([^_]+)_")
    time_series_mean_std_list = []
    time_series_list = []
    id_list = []
    npy_list = []
    score_list = []
    for filename in os.listdir(fc_folder):
        if filename.endswith(".npy"):
            match = pattern.match(filename)
            if match:
                file_id = match.group(1)
                file_path = os.path.join(fc_folder, filename)
                series_file_path = os.path.join(series_time_folder, filename.replace("fc", "ts"))
                data = np.load(file_path)
                time_series_data = np.load(series_file_path)
                # time_series_mean_std = get_mean_std(time_series_data)
                time_series_mean_std = get_per_freq_features(time_series_data)
                time_series_list.append(time_series_data)
                time_series_mean_std_list.append(time_series_mean_std)
                id_list.append(file_id)
                npy_list.append(data)
                scores = labels_dict.get(file_id)
                score_list.append(scores[0])

    fc_array = np.stack(npy_list)
    time_series_array = np.stack(time_series_list)
    time_series_mean_std_array = np.stack(time_series_mean_std_list)
    score_array = np.array(score_list)
    id_array = np.array(id_list)
    data_dict = {
        "fc": fc_array,
        "time_series": time_series_array,
        "time_series_freq": time_series_mean_std_array,
        "score": score_array,
        # "group": id_array
    }
    np.save(save_path, data_dict)
    dump(data_dict, save_path)



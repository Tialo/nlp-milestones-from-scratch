import os

import requests


def download_data(rewrite=False, save_path="train_data.txt"):
    if os.path.isfile(save_path) and not rewrite:
        return print("data is already downloaded")

    base_url = "https://raw.githubusercontent.com/mashashma/WMT2022-data/main/en-ru/en-ru.1m_{}.tsv"

    # there are 10 files in this repo
    n_files = 10
    with open(save_path, "w", encoding="utf-8") as f:
        for file_idx in range(1, n_files + 1):
            r = requests.get(base_url.format(file_idx))
            f.write(r.text)
            print(f"{file_idx}/{n_files} saved")


if __name__ == '__main__':
    download_data()

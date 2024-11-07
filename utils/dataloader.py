import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from utils.dataloader_utils import interpolate_table, random_flip
from tqdm import tqdm
from sys import exit

class Data(Dataset):
    def __init__(self, obj_dir, augment=False, scale=False, mean_std=None, min_max=None, tanh=False, preload=False, img_size=(81, 41)):
        self.scale = scale
        self.augment = augment
        self.mean_std = mean_std
        self.min_max = min_max
        self.tanh = tanh
        self.preload = preload
        self.paths = []
        self.data_cache = {}
        self.conditions = []
        self.img_size = img_size

        if scale:
            if self.mean_std:
                self.mean_stress, self.std_stress, self.mean_height, self.std_height, self.mean_punch, self.std_punch = self.mean_std

            elif self.min_max:
                self.min_stress, self.max_stress, self.min_height, self.max_height, self.min_punch, self.max_punch = self.min_max

            else:
                print()
                exit(0)

        self.paths = self._collect_paths(obj_dir)

        if self.preload:
            self._preload_data()

    def _collect_paths(self, obj_dir):
        def extract_number(filename):
            number = "".join(filter(str.isdigit, filename))
            return int(number) if number else 0

        paths = []
        geometrie_arten = sorted(os.listdir(obj_dir), key=extract_number)

        for geometrie in geometrie_arten:
            geometrie_path = os.path.join(obj_dir, geometrie)
            matrizen_arten = os.listdir(geometrie_path)
            for matrizen_art in matrizen_arten:
                matrizen_path = os.path.join(geometrie_path, matrizen_art)
                werkzeug_arten = os.listdir(matrizen_path)
                for werkzeug_art in werkzeug_arten:
                    excel_path = os.path.join(matrizen_path, werkzeug_art)
                    excel_arten = os.listdir(excel_path)
                    for excel_art in excel_arten:
                        if "stempel" in excel_art.lower():
                            continue
                        tool_path = os.path.join(excel_path, excel_art)
                        paths.append(tool_path)
                        self.conditions.append(float(excel_art.replace("Matrize_", "").replace(".xlsx", "")))

        return paths

    def _preload_data(self):
        for path in tqdm(self.paths, desc="Preloading data"):
            stempel_path = "Stempel".join(path.rsplit("Matrize", 1))
            matrize = interpolate_table(path, self.img_size[0], self.img_size[1], "segment_pressure_value (MPa)")
            stempel = interpolate_table(stempel_path, self.img_size[0], self.img_size[1], "segment_pressure_value (MPa)")
            geo = interpolate_table(path, self.img_size[0], self.img_size[1],  "z (mm)")
            self.data_cache[path] = (matrize, geo, stempel)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        if self.preload:
            matrize, geo, stempel = self.data_cache[path]
        else:
            stempel_path = "Stempel".join(path.rsplit("Matrize", 1))
            matrize = interpolate_table(path, self.img_size[0], self.img_size[1], "segment_pressure_value (MPa)")
            stempel = interpolate_table(stempel_path, self.img_size[0], self.img_size[1],"segment_pressure_value (MPa)")
            geo = interpolate_table(path, self.img_size[0], self.img_size[1], "z (mm)")

        if self.scale:
            matrize, geo, stempel = self._scale_data(matrize, geo, stempel)

        if self.augment:
            matrize, geo, stempel = self._augment_data(matrize, geo, stempel)

        return (
            torch.tensor(matrize.copy(), dtype=torch.float32),
            torch.tensor(geo.copy(), dtype=torch.float32),
            torch.tensor(stempel.copy(), dtype=torch.float32),
            torch.tensor(self.conditions[idx], dtype=torch.float32)
        )

    def _scale_data(self, matrize, geo, stempel):
        if self.mean_std:
            matrize = (matrize - self.mean_stress) / self.std_stress
            # geo = geo - 1
            geo = (geo - self.mean_height) / self.std_height
            stempel = (stempel - self.mean_punch) / self.std_punch
        elif self.min_max and not self.tanh:
            matrize = (matrize - self.min_stress) / (self.max_stress - self.min_stress)
            # geo = geo - 1
            geo = (geo - self.min_height) / (self.max_height - self.min_height)
            stempel = (stempel - self.min_punch) / (self.max_punch - self.min_punch)
        elif self.tanh:
            matrize = 2 * ((matrize - self.min_stress) / (self.max_stress - self.min_stress)) - 1
            geo = 2 * ((geo - self.min_height) / (self.max_height - self.min_height)) - 1
            stempel = 2 * ((stempel - self.min_punch) / (self.max_punch - self.min_punch)) - 1
        return matrize, geo, stempel

    def _augment_data(self, matrize, geo, stempel):
        n_rotation = random.choice([0, 2])
        matrize = np.rot90(matrize, k=n_rotation)
        geo = np.rot90(geo, k=n_rotation)
        stempel = np.rot90(stempel, k=n_rotation)
        return random_flip([matrize, geo, stempel])


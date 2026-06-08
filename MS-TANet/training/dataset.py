"""
Dataset construction for MS-TANet.

Each sample = (one athlete, one monthly test event) ⇒ supervised target.

  X        : [L=90, D=18] daily load features ending at test date
  s_static : [D_s_static=6]
  s_dyn    : [D_s_dyn=4]
  y        : scalar — standardised performance score (0-100, within sport)

The split is per-athlete chronological (Section 3.7.3):
   first 70 % of athlete's monthly tests → train
   middle 15 %                            → validation
   last 15 %                              → test
A 5-fold blocked TS-CV is constructed by sliding the boundary in 10 % increments.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Order of the 18 daily features (D = 18)
DAILY_FEATURES = [
    "Training_duration_min", "Total_distance_km",
    "Mean_HR_bpm", "Peak_HR_bpm", "TRIMP",
    "HRV_RMSSD_morning_ms", "Stroke_count_total",
    "Stroke_rate_per_min", "PlayerLoad_au",
    "Session_RPE_0_10", "sRPE_load_AU",
    "Prior_night_sleep_h", "Subjective_wellness_1_10",
    # Phase one-hot (5 dims)
    "Phase_Base", "Phase_Intensive", "Phase_Tapering",
    "Phase_Competition", "Phase_Recovery",
]

# Tier 1: strictly static
STATIC_FEATURES = [
    "Sport_code",                      # 0/1/2 — swimming/distance/rowing
    "Sex_code",                        # 0 female, 1 male
    "Age_at_season_start_years",
    "Height_cm",
    "Dominant_side_code",              # 0 right, 1 left
    "Years_of_structured_training",
]

# Tier 2: quasi-dynamic (refreshed monthly)
DYN_FEATURES = [
    "Body_mass_kg", "Body_fat_percent",
    "VO2max_ml_kg_min", "Resting_HR_bpm",
]

SPORT_CODE = {"Swimming": 0, "Distance Running": 1, "Rowing": 2}
SIDE_CODE = {"R": 0, "L": 1}


# ----------------------------------------------------------------------
def load_raw_data(data_dir: str = "data"):
    """Read the three CSV tables that ship with the project."""
    cohort = pd.read_csv(f"{data_dir}/athlete_cohort.csv")
    daily = pd.read_csv(f"{data_dir}/daily_training_records.csv",
                        parse_dates=["Date"])
    monthly = pd.read_csv(f"{data_dir}/monthly_test_results.csv",
                          parse_dates=["Test_date"])
    return cohort, daily, monthly


def _phase_one_hot(phase: str):
    return [
        int(phase == "Base"),
        int(phase == "Intensive"),
        int(phase == "Tapering"),
        int(phase == "Competition"),
        int(phase == "Recovery"),
    ]


def _expand_daily_with_phase_dummies(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    for col in ["Phase_Base", "Phase_Intensive", "Phase_Tapering",
                "Phase_Competition", "Phase_Recovery"]:
        daily[col] = 0
    phase_map = {"Base": "Phase_Base", "Intensive": "Phase_Intensive",
                 "Tapering": "Phase_Tapering", "Competition": "Phase_Competition",
                 "Recovery": "Phase_Recovery"}
    for phase, col in phase_map.items():
        daily.loc[daily["Training_phase"] == phase, col] = 1
    return daily


# ----------------------------------------------------------------------
class AthleteSampleIndex:
    """
    Build the index of (athlete, test_date, X_window) supervised samples plus
    per-athlete chronological standardisation parameters.
    """

    def __init__(self, data_dir: str = "data", look_back: int = 90):
        self.data_dir = data_dir
        self.look_back = look_back
        cohort, daily, monthly = load_raw_data(data_dir)

        # Encode static attributes
        cohort["Sport_code"] = cohort["Sport"].map(SPORT_CODE)
        cohort["Sex_code"] = (cohort["Sex"] == "Male").astype(int)
        cohort["Dominant_side_code"] = cohort["Dominant_side"].map(SIDE_CODE).fillna(0).astype(int)

        # Expand training phase to dummies
        daily = _expand_daily_with_phase_dummies(daily)

        self.cohort = cohort.set_index("Athlete_ID")
        self.daily = daily.sort_values(["Athlete_ID", "Date"]).reset_index(drop=True)
        self.monthly = monthly.sort_values(["Athlete_ID", "Test_date"]).reset_index(drop=True)

        # Build standardisation parameters from train-only portion of each athlete
        self._build_standardization()
        self.samples: List[Dict] = self._build_samples()

    # ------------------------------------------------------------------
    def _build_standardization(self,
                                train_pct: float = 0.70):
        self.daily_means: Dict[str, np.ndarray] = {}
        self.daily_stds: Dict[str, np.ndarray] = {}
        for aid, grp in self.daily.groupby("Athlete_ID"):
            n = len(grp)
            n_train = int(n * train_pct)
            train_part = grp.iloc[:n_train]
            self.daily_means[aid] = train_part[DAILY_FEATURES].mean(numeric_only=True).values
            self.daily_stds[aid] = train_part[DAILY_FEATURES].std(numeric_only=True).replace(0, 1.0).values

    # ------------------------------------------------------------------
    def _build_samples(self) -> List[Dict]:
        samples = []
        daily_grouped = {aid: grp.reset_index(drop=True)
                         for aid, grp in self.daily.groupby("Athlete_ID")}

        for _, test_row in self.monthly.iterrows():
            aid = test_row["Athlete_ID"]
            test_date = pd.Timestamp(test_row["Test_date"])
            grp = daily_grouped.get(aid)
            if grp is None or len(grp) == 0:
                continue
            # Take last L training days strictly before or on test date
            window = grp[grp["Date"] <= test_date].tail(self.look_back)
            if len(window) < 30:
                continue
            # Pad with zeros if window < L
            if len(window) < self.look_back:
                pad_len = self.look_back - len(window)
                # Pad rows of zeros — phase is set as Base by default
                pad = pd.DataFrame(
                    {c: [0.0] * pad_len for c in DAILY_FEATURES})
                X_raw = pd.concat([pad, window[DAILY_FEATURES]], axis=0).values
            else:
                X_raw = window[DAILY_FEATURES].values
            # Standardise per athlete
            means = self.daily_means[aid]
            stds = self.daily_stds[aid]
            X = (X_raw - means) / (stds + 1e-6)

            ath_meta = self.cohort.loc[aid]
            s_static = np.array([
                ath_meta["Sport_code"], ath_meta["Sex_code"],
                ath_meta["Age_at_season_start_years"],
                ath_meta["Height_cm"], ath_meta["Dominant_side_code"],
                ath_meta["Years_of_structured_training"],
            ], dtype=np.float32)
            s_dyn = np.array([
                test_row["Body_mass_kg"], test_row["Body_fat_percent"],
                test_row["VO2max_ml_kg_min"], test_row["Resting_HR_bpm"],
            ], dtype=np.float32)
            y = float(test_row["Sport_performance_standardized_0_100"])

            samples.append({
                "athlete_id": aid,
                "sport": ath_meta["Sport"],
                "test_month_index": int(test_row["Test_month_index"]),
                "test_date": test_date,
                "X": X.astype(np.float32),
                "s_static": s_static,
                "s_dyn": s_dyn,
                "y": y,
            })
        return samples


# ----------------------------------------------------------------------
class AthleteDataset(Dataset):
    """Wrap a subset of samples."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["X"]),
            torch.from_numpy(s["s_static"]),
            torch.from_numpy(s["s_dyn"]),
            torch.tensor(s["y"], dtype=torch.float32),
            idx,
        )


# ----------------------------------------------------------------------
def chronological_split(samples: List[Dict],
                        train_pct: float = 0.70,
                        val_pct: float = 0.15,
                        test_pct: float = 0.15) -> Tuple[List, List, List]:
    """Per-athlete chronological split — train then val then test in time."""
    by_ath = {}
    for s in samples:
        by_ath.setdefault(s["athlete_id"], []).append(s)
    train, val, test = [], [], []
    for aid, lst in by_ath.items():
        lst = sorted(lst, key=lambda x: x["test_date"])
        n = len(lst)
        n_train = max(1, int(n * train_pct))
        n_val = max(1, int(n * val_pct))
        train.extend(lst[:n_train])
        val.extend(lst[n_train:n_train + n_val])
        test.extend(lst[n_train + n_val:])
    return train, val, test


def blocked_ts_cv_folds(samples, n_folds: int = 5,
                        train_pct_start: float = 0.50,
                        increment: float = 0.10,
                        val_pct: float = 0.15):
    """Sliding 5-fold blocked TS-CV — Section 3.7.3."""
    folds = []
    by_ath = {}
    for s in samples:
        by_ath.setdefault(s["athlete_id"], []).append(s)
    for k in range(n_folds):
        tr_pct = train_pct_start + k * increment
        train, val, test = [], [], []
        for aid, lst in by_ath.items():
            lst = sorted(lst, key=lambda x: x["test_date"])
            n = len(lst)
            n_train = max(1, int(n * tr_pct))
            n_val = max(1, int(n * val_pct))
            train.extend(lst[:n_train])
            val.extend(lst[n_train:n_train + n_val])
            test.extend(lst[n_train + n_val:])
        folds.append((train, val, test))
    return folds


def athlete_wise_cv(samples, n_folds: int = 5, seed: int = 42):
    """Random partition of athletes within each sport into n_folds."""
    rng = np.random.RandomState(seed)
    by_sport_ath = {}
    for s in samples:
        by_sport_ath.setdefault(s["sport"], set()).add(s["athlete_id"])
    folds = []
    for k in range(n_folds):
        train, test = [], []
        for sport, athletes in by_sport_ath.items():
            athletes = sorted(athletes)
            rng.shuffle(athletes)
            chunks = np.array_split(athletes, n_folds)
            test_athletes = set(chunks[k])
            for s in samples:
                if s["sport"] != sport:
                    continue
                if s["athlete_id"] in test_athletes:
                    test.append(s)
                else:
                    train.append(s)
        folds.append((train, train[:1], test))   # LOAO/LOSO: no separate val split
    return folds


def leave_one_athlete_out(samples):
    by_ath = {}
    for s in samples:
        by_ath.setdefault(s["athlete_id"], []).append(s)
    athletes = sorted(by_ath.keys())
    folds = []
    for left_out in athletes:
        train = [s for s in samples if s["athlete_id"] != left_out]
        test = by_ath[left_out]
        folds.append((train, train[:1], test))
    return folds


def leave_one_sport_out(samples):
    sports = sorted({s["sport"] for s in samples})
    folds = []
    for left_out in sports:
        train = [s for s in samples if s["sport"] != left_out]
        test = [s for s in samples if s["sport"] == left_out]
        folds.append((train, train[:1], test))
    return folds

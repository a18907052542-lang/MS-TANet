# Longitudinal Athlete Training-Load and Performance Dataset

This directory contains the three CSV tables that comprise the self-built
longitudinal dataset used in MS-TANet evaluation (Section 3.7.1 of the paper).

## Cohort

* **47 athletes** across three endurance sports
  * Swimming        — 18 athletes (10 male, 8 female), 14–20 yr
  * Distance Running — 17 athletes (11 male, 6 female), 15–21 yr
  * Rowing           — 12 athletes (8 male, 4 female), 16–22 yr
* All athletes recruited from a provincial sports school
* Observation period: 18 consecutive months (April 2024 – October 2025)

## Tables

| File                              | Rows    | Description                              |
| --------------------------------- | ------- | ---------------------------------------- |
| `athlete_cohort.csv`              | 47      | Static + baseline quasi-dynamic attributes |
| `daily_training_records.csv`      | 15,481  | Day-level training load (18 features)    |
| `monthly_test_results.csv`        | 846     | Monthly body-composition + VO₂max + sport-specific test |

## Daily features (D = 18)

Aggregated from continuous-monitoring devices to one record per athlete-day:

| Source                            | Variables                                                       |
| --------------------------------- | --------------------------------------------------------------- |
| Polar Vantage V2 (1 Hz watch)     | Training duration, total distance, mean / peak HR, TRIMP        |
| Garmin HRM-Pro chest belt         | Morning HRV-RMSSD                                               |
| 100 Hz accelerometer (summary)    | Stroke count, stroke rate, PlayerLoad                           |
| Tablet-entered sRPE               | Session RPE (Borg CR-10), sRPE-load                             |
| Self-report                       | Prior-night sleep, subjective wellness                          |
| Training phase one-hot (5 dims)   | Base, Intensive, Tapering, Competition, Recovery                |

## Monthly tests

* Body mass (kg)
* Body-fat percentage by bioelectrical impedance
* VO₂max — 20 m shuttle run (running/rowing) or treadmill GXT (swimming)
* Sport-specific performance:
  * Swimming — 100 m freestyle time (s)
  * Distance Running — 5 km time-trial (s)
  * Rowing — 2 km ergometer time (s)
* Standardised within sport-event to 0–100 (higher = better)

## Standardisation

Per-athlete chronological standardisation (Section 3.7.2): the first 70 % of
each athlete's records define the standardisation mean and standard deviation;
all subsequent records (the validation and test partitions) are normalised
using these *training-set* statistics to prevent leakage.

## Ethics

Ethics approval CSU-EC-2024-XXX from the Institutional Review Board of
Central South University.  Written informed consent obtained from every
athlete (and from a parent / guardian for those under 18).  All records
were de-identified prior to analysis.

## Licence

This dataset is released under the **CC BY-NC 4.0** licence for academic and
non-commercial research use.  Re-distribution must preserve the licence and
acknowledge the original collection effort.

# data/

Git does not track the files in this folder, except this README. EEG recordings are clinical data.
Do not commit them, and do not put them in an issue, a log or a report.

## Source

| Item | Value |
|---|---|
| Name | MDD Patients and Healthy Controls EEG Data (Mumtaz et al.) |
| URL | <https://figshare.com/articles/dataset/EEG_Data_New/4244171> |
| License and terms | Read the terms on the figshare page before you use the data. Cite the original papers |
| Content | 19-channel resting-state EEG (10-20 montage) of patients with major depressive disorder, with treatment-outcome labels for a subset |

The prototype used per-subject `.mat` files that were made from this dataset. Check the outcome
labels and the license yourself. eeg-rx does not download data.

## Expected layout (`.mat` loader)

```
data/SSRI/
├── SSRI_R_1.mat     # responder 1
├── ...
├── SSRI_NR_1.mat    # non-responder 1
└── ...
```

- Each file contains one variable `EEG` with shape (samples, 19) or (19, samples), in microvolts.
- Channel order: Fp1, Fp2, F7, F3, Fz, F4, F8, C3, Cz, C4, P3, Pz, P4, T3, T4, T5, T6, O1, O2.
- Sampling rate: 256 Hz (`EEG_RX_SFREQ`).
- The number of files and the length of each recording can be any value. Nothing is hard-coded.

## EDF files (optional)

`eeg_rx.io.load_edf_manifest` reads a CSV with the columns `subject_id,label,path` and EDF files
through MNE (`pip install -e ".[mne]"`). `label` is 1 for a responder and 0 for a non-responder.

## No download

`eeg-rx synth --out data/synthetic` writes synthetic `.mat` files in the same layout, and the
`--synthetic` flag makes the same data in memory. The tests use only synthetic EEG.

"""eeg-rx: a subject-wise, leakage-safe EEG study of SSRI treatment response. Research use only."""

__version__ = "0.1.0"

# 10-20 montage order of the 19-channel recordings.
CHANNELS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3", "Cz", "C4",
            "P3", "Pz", "P4", "T3", "T4", "T5", "T6", "O1", "O2")

DISCLAIMER = ("RESEARCH USE ONLY. This output is not a diagnosis and not a treatment recommendation. "
              "A clinician must make every treatment decision.")

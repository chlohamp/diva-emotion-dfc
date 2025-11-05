import pandas as pd
import random
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define the data parameters
onset_values = np.arange(3, 24, 1.5)  # From 3 to 24 in 1.5 increments
duration = 1.5  # All durations are 1.5

# Define word lists
nouns = [
    "bicycle",
    "leaf",
    "cat",
    "house",
    "tree",
    "book",
    "flower",
    "car",
    "dog",
    "bird",
    "chair",
    "table",
    "computer",
    "phone",
    "mountain",
    "river",
    "ocean",
    "cloud",
]

verbs = [
    "run",
    "jump",
    "ride",
    "walk",
    "swim",
    "fly",
    "climb",
    "dance",
    "sing",
    "read",
    "write",
    "drive",
    "sleep",
    "eat",
    "play",
    "work",
    "study",
    "laugh",
]

# Define character list for on-screen appearances
characters = ["Mike Wheeler", "Eleven", "Nancy"]


def create_rater_data(rater_id):
    n_trials = len(onset_values)

    # Create the dataframe
    data = {
        "onset": onset_values,
        "duration": [duration] * n_trials,
        "nouns": random.choices(nouns, k=n_trials),
        "verbs": random.choices(verbs, k=n_trials),
        "valence": np.random.randint(1, 8, n_trials),  # 1-7
        "arousal": np.random.randint(1, 8, n_trials),  # 1-7
        "characters": random.choices(
            characters, k=n_trials
        ),  # Random character selection
    }

    return pd.DataFrame(data)


# Generate data for both raters
print("Generating data for Rater A1...")
rater_a1_data = create_rater_data("A1")

# Reset random seed to ensure different values for second rater
random.seed(123)
np.random.seed(123)

print("Generating data for Rater A2...")
rater_a2_data = create_rater_data("A2")

# Define file names
filename_a1 = "dset/derivatives/simulated/ses-01_task-strangerthings_acq-A1_run-1_events.tsv"
filename_a2 = "dset/derivatives/simulated/ses-01_task-strangerthings_acq-A2_run-1_events.tsv"

# Create output directory
Path("dset/derivatives/simulated").mkdir(parents=True, exist_ok=True)

# Save to TSV files
print(f"Saving {filename_a1}...")
rater_a1_data.to_csv(filename_a1, sep="\t", index=False)

print(f"Saving {filename_a2}...")
rater_a2_data.to_csv(filename_a2, sep="\t", index=False)

print("\nFiles generated successfully!")
print(f"- {filename_a1}")
print(f"- {filename_a2}")

# Display preview of the data
print(f"\nPreview of {filename_a1}:")
print(rater_a1_data.head())

print(f"\nPreview of {filename_a2}:")
print(rater_a2_data.head())

# Display summary statistics
print("\nSummary for Rater A1:")
print(f"Number of trials: {len(rater_a1_data)}")
a1_onset_min = rater_a1_data["onset"].min()
a1_onset_max = rater_a1_data["onset"].max()
print(f"Onset range: {a1_onset_min} to {a1_onset_max}")
a1_val_min = rater_a1_data["valence"].min()
a1_val_max = rater_a1_data["valence"].max()
print(f"Valence range: {a1_val_min} to {a1_val_max}")
a1_aro_min = rater_a1_data["arousal"].min()
a1_aro_max = rater_a1_data["arousal"].max()
print(f"Arousal range: {a1_aro_min} to {a1_aro_max}")

print("\nSummary for Rater A2:")
print(f"Number of trials: {len(rater_a2_data)}")
a2_onset_min = rater_a2_data["onset"].min()
a2_onset_max = rater_a2_data["onset"].max()
print(f"Onset range: {a2_onset_min} to {a2_onset_max}")
a2_val_min = rater_a2_data["valence"].min()
a2_val_max = rater_a2_data["valence"].max()
print(f"Valence range: {a2_val_min} to {a2_val_max}")
a2_aro_min = rater_a2_data["arousal"].min()
a2_aro_max = rater_a2_data["arousal"].max()
print(f"Arousal range: {a2_aro_min} to {a2_aro_max}")

# Generate participants-characters data
print("\nGenerating participants-characters data...")

# Reset random seed for participants data
random.seed(456)
np.random.seed(456)

participants_characters_data = {
    "participant_id": ["sub-Blossom", "sub-Bubbles", "sub-Buttercup"],
    "ses_id": ["ses-01", "ses-01", "ses-01"],
    "dusty_valence": np.random.randint(1, 8, 3),  # 1-7
    "dusty_arousal": np.random.randint(1, 8, 3),  # 1-7
    "nancy_valence": np.random.randint(1, 8, 3),  # 1-7
    "nancy_arousal": np.random.randint(1, 8, 3),  # 1-7
    "steve_valence": np.random.randint(1, 8, 3),  # 1-7
    "steve_arousal": np.random.randint(1, 8, 3),  # 1-7
}

participants_characters_df = pd.DataFrame(participants_characters_data)

# Save participants-characters file
participants_characters_filename = "derivatives/simulated/participants-characters.tsv"
print(f"Saving {participants_characters_filename}...")
participants_characters_df.to_csv(
    participants_characters_filename, sep="\t", index=False
)

print("\nParticipants-characters file generated:")
print(f"{participants_characters_filename}")
print("\nParticipants-characters data:")
print(participants_characters_df)

# Generate participants demographics data
print("\nGenerating participants demographics data...")

# Reset random seed for different demographic data
random.seed(789)
np.random.seed(789)

# Define the participants with random ages and sex
participants_demo_data = {
    "participant_id": [
        "sub-Blossom",
        "sub-Bubbles",
        "sub-Buttercup",
    ],
    "age": np.random.randint(18, 65, 3).tolist(),  # Random ages 18-64
    "sex": ["F", "F", "F"],
}

participants_demo_df = pd.DataFrame(participants_demo_data)

# Save participants demographics file
participants_demo_filename = "dset/derivatives/simulated/participants.tsv"
print(f"Saving {participants_demo_filename}...")
participants_demo_df.to_csv(participants_demo_filename, sep="\t", index=False)

print("\nParticipants demographics file generated:")
print(f"{participants_demo_filename}")
print("\nParticipants demographics data:")
print(participants_demo_df)

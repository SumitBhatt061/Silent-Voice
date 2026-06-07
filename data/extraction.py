import os
import shutil
import re

DATASET_ROOT = r"C:/Users/prhtt/American-Sign-Language-Dataset"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "filtered_dataset")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# your vocabulary set (normalized)
VOCAB = {
    "afternoon","angry","boss","bye","camera","clean","coffee","college","come",
    "computer","cook","door","down","friend","give","go","good","great","happy",
    "hate","he","hello","help","hungry","know","left","love","maybe","me","mine",
    "morning","move","no","now","phone","play","please","room","sad","scared",
    "see","sit","sleep","sorry","stand","sure","take","tea","think","thirsty",
    "tired","today","tomorrow","up","vegetable","wait","walk","water","when",
    "where","which","who","why","work","yes","yesterday","you"
}

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def extract_label(filename):
    name = os.path.splitext(filename)[0].lower()

    # split into tokens using -, _, space, numbers
    tokens = re.split(r"[-_\s]+", name)

    # keep only alphabetic tokens
    tokens = [t for t in tokens if t.isalpha()]

    # match against vocabulary
    for t in tokens:
        if t in VOCAB:
            return t.upper()

    # fallback: try partial match (safer for messy names)
    for t in tokens:
        for word in VOCAB:
            if word in t:
                return word.upper()

    return "UNKNOWN"


for part in os.listdir(DATASET_ROOT):
    part_path = os.path.join(DATASET_ROOT, part)

    if not os.path.isdir(part_path):
        continue

    print(f"Processing {part}...")

    for file in os.listdir(part_path):
        if not file.lower().endswith(VIDEO_EXTS):
            continue

        src_path = os.path.join(part_path, file)

        label = extract_label(file)

        dest_folder = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(dest_folder, exist_ok=True)

        dst_path = os.path.join(dest_folder, file)

        shutil.copy2(src_path, dst_path)   # change to move if needed
        # shutil.move(src_path, dst_path)

print("Done!")
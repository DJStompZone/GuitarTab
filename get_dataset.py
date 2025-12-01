import os
import shutil
import guitarpro
from tqdm import tqdm

def build_dataset(root_A, dataset_root):
    """
    Search for all GP files under root_A that have exactly
    1 track and 6 strings. Copy them to dataset_root while
    preserving the original folder structure.
    """

    # First: collect all GP file paths (for a progress bar)
    print("Scanning all GP files...")
    all_gp_files = []

    for current_dir, _, files in os.walk(root_A):
        for fname in files:
            if fname.lower().endswith((".gp3", ".gp4")):
                all_gp_files.append(os.path.join(current_dir, fname))

    valid_files = []

    # Second: parse GP files with progress bar
    print("Checking GP files for (1-track & 6-strings)...")
    for file_path in tqdm(all_gp_files, desc="Parsing GP files"):
        try:
            tab = guitarpro.parse(file_path)
            if len(tab.tracks) == 1 and len(tab.tracks[0].strings) == 6:
                valid_files.append(file_path)
        except Exception:
            # skip errors quietly or print if needed
            pass

    # Third: copy with progress bar
    print("\nCopying valid files...")
    for src_path in tqdm(valid_files, desc="Copying"):
        rel_path = os.path.relpath(src_path, root_A)
        dst_path = os.path.join(dataset_root, rel_path)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

    print(f"\nTotal valid GP files copied: {len(valid_files)}")


# Example usage
ROOT_A = "DadaGP-v1.1"
DATASET_ROOT = "Dataset_new"   # your dataset folder
build_dataset(ROOT_A, DATASET_ROOT)

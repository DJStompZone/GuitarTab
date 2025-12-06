# pyright: reportCallInDefaultInitializer=false
import os
import shutil
import json
import guitarpro
from tqdm import tqdm
import typer

app = typer.Typer()


@app.command()
def build_dataset(
    dadagp_root: str = typer.Argument(
        "DadaGP-v1.1", help="Root directory to search for GP files"
    ),
    max_gp_files: int = typer.Option(
        None,
        help="Maximum number of valid GP files to find (None for unlimited)",
    ),
    output_json_path: str = typer.Option(
        "selected_files.json", help="Path to save the selected files"
    ),
):
    """
    Search for all GP files under root_a that have exactly
    1 track and 6 strings. Copy them to dataset_root while
    preserving the original folder structure.
    """

    # First: collect all GP file paths (for a progress bar)
    print("Scanning all GP files...")
    all_gp_files = []

    for current_dir, _, files in os.walk(dadagp_root):
        for fname in files:
            if fname.lower().endswith((".gp3", ".gp4")):
                all_gp_files.append(os.path.join(current_dir, fname))

    valid_files = []

    # Second: parse GP files with progress bar
    print(f"Checking GP files for (1-track & 6-strings)...")
    if max_gp_files:
        print(f"Will stop after finding {max_gp_files} valid files.")
        # Progress bar based on valid files to find
        pbar = tqdm(total=max_gp_files, desc="Finding valid files")
        processed = 0
        for file_path in all_gp_files:
            processed += 1
            try:
                tab = guitarpro.parse(file_path)
                if len(tab.tracks) == 1 and len(tab.tracks[0].strings) == 6:
                    valid_files.append(file_path)
                    pbar.update(1)
                    # Stop if we've reached the max
                    if len(valid_files) >= max_gp_files:
                        pbar.set_description(
                            f"Found {len(valid_files)}/{max_gp_files} (processed {processed}/{len(all_gp_files)})"
                        )
                        break
            except Exception:
                # skip errors quietly or print if needed
                pass
            # Update description after every file (valid or not)
            pbar.set_description(
                f"Found {len(valid_files)}/{max_gp_files} (processed {processed}/{len(all_gp_files)})"
            )
        pbar.close()
    else:
        # Original behavior: show all files being parsed
        for file_path in tqdm(all_gp_files, desc="Parsing GP files"):
            try:
                tab = guitarpro.parse(file_path)
                if len(tab.tracks) == 1 and len(tab.tracks[0].strings) == 6:
                    valid_files.append(file_path)
            except Exception:
                # skip errors quietly or print if needed
                pass

    print(f"\nTotal valid GP files copied: {len(valid_files)}")

    # Fourth: save file list to JSON
    with open(output_json_path, "w") as f:
        json.dump(valid_files, f, indent=2)
    print(f"File list saved to: {output_json_path}")


if __name__ == "__main__":
    app()

# Personale
#
# Scarica uno o più dataset VPR da Google Drive e li estrae nella cartella data/.
# Il file zip viene eliminato automaticamente dopo l'estrazione.
#
# Utilizzo:
#   python download_datasets_personal.py tokyo_xs
#   python download_datasets_personal.py tokyo_xs sf_xs
#   python download_datasets_personal.py --help
#
# Dataset disponibili: tokyo_xs | sf_xs | gsv_xs | svox

URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=share_link",
    "sf_xs": "https://drive.google.com/file/d/1tQqEyt3go3vMh4fj_LZrRcahoTbzzH-y/view?usp=share_link",
    "gsv_xs": "https://drive.google.com/file/d/1q7usSe9_5xV5zTfN-1In4DlmF5ReyU_A/view?usp=share_link",
    "svox": "https://drive.google.com/file/d/16iuk8voW65GaywNUQlWAbDt6HZzAJ_t9/view?usp=drive_link"
}
import os
import gdown
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("datasets", nargs="+", choices=list(URLS.keys()), metavar="DATASET",
                    help=f"Dataset da scaricare. Scegli tra: {', '.join(URLS.keys())}")
args = parser.parse_args()
os.makedirs("data", exist_ok=True)
for dataset_name, url in URLS.items():
    if dataset_name not in args.datasets:
        continue
    print(f"Downloading {dataset_name}")
    zip_filepath = f"data/{dataset_name}.zip"
    gdown.download(url, zip_filepath, fuzzy=True)
    shutil.unpack_archive(zip_filepath, extract_dir="data")
    os.remove(zip_filepath)

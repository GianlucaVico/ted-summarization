import os
import sys
sys.path.append("../")
sys.path.append("../../") # to import tools
import requests
from bs4 import BeautifulSoup
from tools import progress_bar, make_name, file_exists, name_to_url
import json
from datetime import datetime
import re
import subprocess
import time
import glob
import pandas as pd
import numpy as np
import argparse
import math

script_query = {"data-spec":"q"}
json_extract = re.compile("\{.*\}")
transcript_service = "https://ted2srt.org/api/talks/{}/transcripts/download/{}?lang=en" # video id, format
list_video = "https://www.ted.com/talks?page={}&sort=newest&language=en"

base_url = "https://www.ted.com"
fmt = "srt"

def load_video_file(file):
    """Read list of video url"""
    urls = []
    with open(file, "r") as f:
        urls = f.readlines()
    return urls

def load_video_data(name):
    d = {}
    if file_exists(name):
        with open(name, "rt") as f:
            d = json.load(f)
    return d

def save_audio(id_, data_dict, talk_id, folder, freq=16000, mono=True, err="missing.fail"):
    """
    Args:
        id_: talk id
        data_dict: data_urls.json file
        talks_id: dataframe with talk ids. The index is the id
        folder: where to save the audio
    Returns:
        None
    """
    with open(err, "a+") as missing:
        url = talk_id.loc[id_]["ted"]
        v_url = data_dict[make_name(url)][0][2].split("?")[0]
        fmt = v_url.strip()[-3:] # get the format
        if "youtube" in v_url:
            pass
        elif fmt not in ["mp4"]:
            print("Unknown format:", fmt)
        else:
            name = make_name(url)+".wav"
            if not file_exists(f"{folder}/{name}"):
                print("ID:", id_)
                try:
                    r = requests.get(v_url, timeout=60)
                    if r.ok:
                        print("Converting")
                        # run ffmpeg, input from pipe, map only audio, on gpu 0, do not overwrite existing files
                        cmd = ["ffmpeg", "-i", "pipe:", "-map", "a", "-ar", "16000", "-ac", "1", "-n", "-v", "error", f"{folder}/{name}"]
                        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=None) # output will be ignored
                        proc.communicate(r.content)
                        time.sleep(5)
                    else:
                        print("Request not ok on:", id_)
                        missing.write(f"{id_}\n")
                except requests.Timeout:
                    print("Timeout:", id_)
                    missing.write(f"{id_}\n")
                except requests.ConnectionError: # e.g. invalid URL
                    print("Connection error on:", id_)
                    missing.write(f"{id_}\n")
                except requests.exceptions.MissingSchema:
                    print("Missing schema:", id_)
                    missing.write(f"{id_}\n")


def get_ids(index, total, talk_id):
    start = int((index - 1) * math.ceil(len(talk_id) / total))
    end = int((index) * math.ceil(len(talk_id) / total))

    return list(talk_id.index)[start: end]

def get_missing_id(folder):
    files = glod.glob(f"{folder}/missing_*.fail")
    print(f"Found {len(files)} files")
    ids = []
    for i in files:
        with open(i) as f:
            ids.extend([int(l) for l in f.readlines()])
    return ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pegasus')

    parser.add_argument("-id", "--id", type=str, required=True, help="Talk id csv file")
    parser.add_argument("-d", "--data", type=str, required=True, help="Data json file")
    parser.add_argument("-f", "--folder", type=str, required=True, help="Output folder")
    parser.add_argument("--info", action="store_true", help="Print info about the files")
    parser.add_argument("--missing", action="store_true", help="Redownload faild talks")

    args = parser.parse_args()

    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
    total = int(os.environ.get("TOTAL_JOBS", 1))

    talk_id = pd.read_csv(args.id).set_index("id")
    data = load_video_data(args.data)

    if args.info:
        print("IDs:", len(talk_id))
        ids = len(data)

        print("ID assigned:", ids/total)
        # from 1 to total (included)
        print("Range:", [(index-1) * (ids/total), index*(ids/total)])
        ids = get_ids(index, total, data)
        print("From:", ids[0])
        print("To:", ids[-1])
    else:
        print("Starting...")
        if args.missing:
            ids = get_missing_id(args.folder)
        else:
            ids = get_ids(index, total, talk_id)
        print("IDS:", len(ids))
        for id_ in ids:
            save_audio(id_, data, talk_id, args.folder, err=f"{args.folder}/missing_{index}.fail")
        print("Done....")

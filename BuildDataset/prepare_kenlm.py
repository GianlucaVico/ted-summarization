import sys
sys.path.append("../")
import joblib
import nltk
import argparse
import glob
import tools
import text_tools

pipeline = tools.get_pipeline(
    text_tools.preprocess_numbers,
    text_tools.preprocess_clean,
    text_tools.preprocess_no_punctuation,
    lambda x: text_tools.preprocess_case_normalization(x), # lower case: for compatibility with wav2letter
    text_tools.preprocess_spaces,
)

def flatten(x):
    return [j for i in x for j in i]

def no_none(x):
    return [i for i in x if i is not None]

def get_transcripts(folder):
    parts = glob.glob(f"{folder}/*.pkl")
    transcripts = [None for i in parts]
    for i, p in enumerate(parts):
        print(f"Loading part {i+1}/{len(parts)}")
        _, _, transcripts[i] = joblib.load(p)
        # flatten
    return flatten(no_none(transcripts))

def do_work(data, name, pipeline):
    with open(f"{name}_words.txt", "w") as words:
        with open(f"{name}_chars.txt", "w") as chars:
            data = pipeline(get_transcripts(data))
            for i in data: # each fragment
                words.write(i + "\n")
                chars.write(" ".join(i.replace(" ", "|")) + "\n")


if __name__ == "__main__":
    # run: python3 prepare_kenlm.py --saved-test $HPCWORK/test_audio --saved-train $HPCWORK/train_audio -o $HPCWORK/kenlm_data
    parser = argparse.ArgumentParser(description="Prepare the trannscript for training a KenLM model")
    parser.add_argument(
        "--saved-test",
        type=str,
        help="Path to the folder with the transcripts and audio from ted",
    )

    parser.add_argument(
        "--saved-train",
        type=str,
        help="Path to the folder with the transcripts and audio from ted",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Path to the folder for the output files",
    )

    args = parser.parse_args()

    print("Processing test data...")
    do_work(args.saved_test, f"{args.output_dir}/test", pipeline)

    print("Processing train data...")
    do_work(args.saved_train, f"{args.output_dir}/train", pipeline)

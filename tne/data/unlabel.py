import argparse
from copy import deepcopy
from tqdm import tqdm

from tne.data.split_ood import to_file
from tne.data.split_ood import read_data


def filter_labels(doc):
    filtered_doc = {}
    for k, v in doc.items():
        if k != 'np_relations':
            filtered_doc[k] = deepcopy(v)
    return filtered_doc


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_file", type=str, help="input file", default="data/processed/test.jsonl")
    parse.add_argument("--out_file", type=str, help="output file", default="data/processed/test_unlabaled.jsonl")

    args = parse.parse_args()

    docs = read_data(args.in_file)

    filtered_docs = [filter_labels(doc) for doc in tqdm(docs)]

    to_file(filtered_docs, args.out_file)


if __name__ == '__main__':
    main()

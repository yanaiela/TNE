"""
This script converts the original data format to a different one,
 that aligns with huggingface datasets library.
Specifically, it converts the nps entry, from a dictionary that maps between
 an np's id to its values, into a list of dictionaries with the relevant information.
"""

import argparse
from copy import deepcopy
from tqdm import tqdm

from tne.data.split_ood import to_file
from tne.data.split_ood import read_data


def convert(doc):
    converted_doc = deepcopy(doc)
    new_nps = []
    for val in doc['nps'].values():
        new_nps.append(val)
    converted_doc['nps'] = new_nps

    return converted_doc


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_file", type=str, help="input file", default="data/test.jsonl")
    parse.add_argument("--out_file", type=str, help="output file", default="data/test-v1.1.jsonl")

    args = parse.parse_args()

    docs = read_data(args.in_file)

    converted_docs = [convert(doc) for doc in tqdm(docs)]

    to_file(converted_docs, args.out_file)


if __name__ == '__main__':
    main()

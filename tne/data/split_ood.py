import argparse
import json

from tqdm import tqdm


def read_data(in_f):
    with open(in_f, 'r') as f:
        data = f.readlines()
    data = [json.loads(x) for x in data]
    return data


def to_file(documents, out_f):
    with open(out_f, 'w') as f:
        for doc in documents:
            json.dump(doc, f)
            f.write('\n')


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_file", type=str, help="ood input file", default="data/processed/ood.jsonl")

    args = parse.parse_args()

    docs = read_data(args.in_file)

    sources = {
        'books': [],
        'imdb': [],
        'reddit_YouShouldKnow': [],
        'reddit_LifeProTips': [],
        'reddit_AskHistorians': [],
        'reddit_atheism': [],
        'reddit_askedscience': [],
        'reddit_depressed': [],
        'reddit': [],
    }

    for doc in tqdm(docs):
        source = doc['metadata']['source']
        if 'imdb' in source:
            sources['imdb'].append(doc)
        elif 'txt_gut' in source:
            sources['books'].append(doc)
        elif 'reddit_depressed' in source:
            sources['reddit_depressed'].append(doc)
        elif 'reddit_atheism' in source:
            sources['reddit_atheism'].append(doc)
        elif 'reddit_asked_science' in source:
            sources['reddit_askedscience'].append(doc)
        elif 'reddit_YouShouldKnow' in source:
            sources['reddit_YouShouldKnow'].append(doc)
        elif 'reddit_AskHistorians' in source:
            sources['reddit_AskHistorians'].append(doc)
        elif 'reddit_LifeProTips' in source:
            sources['reddit_LifeProTips'].append(doc)
        else:
            print('unknown source', source)
        if 'reddit' in source:
            sources['reddit'].append(doc)

    for k, v in sources.items():
        to_file(v, f'data/processed/ood_{k}.jsonl')


if __name__ == '__main__':
    main()

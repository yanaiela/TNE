import ast
import collections
from collections import defaultdict
import subprocess
import argparse


def get_decisions(annotation_doc):
    """
    Get a list of labels based on the ordered entities
    """
    entities = ast.literal_eval(annotation_doc['entities'])
    entities_labels = {}
    for entity in entities:
        source = entity['source']
        for member in entity['members']:
            entities_labels[member] = source

    ordered_dic = collections.OrderedDict(sorted(entities_labels.items()))
    return list(ordered_dic.values())


def convert2conll(entities, doc_id):
    data_by_rows = []
    i = 0
    for entity_id, entity in enumerate(entities):
        # skipping singletons (and therefore, also the other labels types: time/date and idiomatic)
        if len(entity['members']) == 1:
            continue
        for member in entity['members']:
            data_by_rows.append('({})'.format(str(entity['id'])))
    data_by_rows = [f'#begin document {doc_id + 1}'] + data_by_rows + ['#end document']
    return data_by_rows


def conll2file(fname, conll_data):
    with open(fname, 'w') as f:
        for row in conll_data:
            f.write('\t'.join([str(row)]) + '\n')


def run_scorer(annotation_file1, annotation_file2):
    result = subprocess.run(['python','/home/nlp/lazary/workspace/github/coval/scorer.py',
                        annotation_file1,
                        annotation_file2], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    return float(output.split('\n')[1].split()[-1]), float(output.split('\n')[-1].split()[-1])


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-in", "--input_file", type=str, help="input file name of the annotations")
    parse.add_argument("-out", "--output_dir", type=str, help="output dir")
    args = parse.parse_args()

    out_d = args.output_dir

    with open(args.input_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()


    tasks = defaultdict(list)
    all_turkers = set()
    for line in lines:
        task_json = ast.literal_eval(line.strip())
        tasks[task_json['internal_hit_id']].append(task_json)
        all_turkers.add(task_json['annotator'])

    annotation1, annotation2 = [], []
    for ind, task in enumerate(tasks.values()):
        # for ind, annotation in task:
        assert len(task) <= 2
        entities = ast.literal_eval(task[0]['entities'])
        conll_format = convert2conll(entities, ind)
        annotation1.extend(conll_format)

        entities = ast.literal_eval(task[1]['entities'])
        conll_format = convert2conll(entities, ind)
        annotation2.extend(conll_format)

    conll2file('{}/task_verification_ann{}.conll'.format(out_d, 1), annotation1)
    conll2file('{}/task_verification_ann{}.conll'.format(out_d, 2), annotation2)


    muc, conll = run_scorer(
        f'{out_d}/task_verification_ann1.conll',
        f'{out_d}/task_verification_ann2.conll',
    )

    print(f'muc: {muc}, conll f1: {conll}')


if __name__ == '__main__':
    main()

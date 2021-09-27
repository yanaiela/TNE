# coding: utf8

from collections import defaultdict
import spacy
from spacy import displacy

from flask import Flask, request
from flask_cors import CORS, cross_origin

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from tne.modeling.models.tne_coupled import TNECoupledModel
from tne.modeling.dataset_readers.tne_reader import TNEReader
from tne.modeling.predictors.tne_predictor import TNEPredictor

app = Flask(__name__)
CORS(app)
nlp = spacy.load("en_core_web_sm", exclude=['ner'])


class TNE(object):
    """
    """

    def __init__(self):
        """Initialise the pipeline component.

        nlp (Language): The shared nlp object. Used to initialise the matcher
            with the shared `Vocab`, and create `Doc` match patterns.
        RETURNS (callable): A spaCy pipeline component.
        """

        archive_model = load_archive('models/coupled_spanbert_large/model.tar.gz')
        self.nte_predictor = Predictor.from_archive(archive_model, 'tne_predictor')

    def __call__(self, text):
        """Apply the pipeline component to a `Doc` object.

        doc (Doc): The `Doc` returned by the previous pipeline component.
        RETURNS (Doc): The modified `Doc` object.
        """
        input_json = {'text': text}
        out = self.nte_predictor.predict_json(input_json)
        return out


tne_model = TNE()


def add_annotation(text):
    output = tne_model(text)
    all_links = output['full_links']

    doc = nlp(text)

    np_chunks = {}
    for ind, chunk in enumerate(doc.noun_chunks):
        np_chunks[chunk.start] = chunk

    annotations = defaultdict(list)
    for link in all_links:
        annotations[(np_chunks[link['from_first']].start_char, np_chunks[link['from_first']].end_char)].append(
            f"{link['preposition']} {np_chunks[link['to_first']].text}")

    annotation_links = []
    for (s, e), to_text in annotations.items():
        annotation_links.append({
            'start': s,
            'end': e,
            'label': '|'.join(to_text),
        })

    ex = [{"text": text,
           "ents": annotation_links,
           "title": None}]
    html = displacy.render(ex, style="ent", manual=True)

    return html


@app.route('/tne/', methods=['POST'])
@cross_origin()
def serve():
    text = request.form.get('text')

    if text.strip() == '' or text is None:
        return ''

    try:
        html = add_annotation(text)

    except Exception as e:
        print(e)
        html = 'some error occurred while trying to find the TNE'

    return html

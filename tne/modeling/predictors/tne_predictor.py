from allennlp.common.util import JsonDict
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register("tne_predictor")
class TNEPredictor(Predictor):
    """
    TODO - fix documentation
    Predictor for the [`BridgingModel`](../models/coreference_resolution/coref.md) model.
    Registered as a `Predictor` with name "bridging_resolution".
    """

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)

        # We have to use spacy to tokenize our document here, because we need
        # to also know sentence boundaries to propose valid mentions.
        self._spacy = get_spacy_model(language, pos_tags=True, parse=True, ner=False)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"document": "string of document text"}`
        """
        document = json_dict['text']
        spacy_document = self._spacy(document)
        tokens = [token.text for token in spacy_document]
        entities = {}
        for ind, chunk in enumerate(spacy_document.noun_chunks):
            start = chunk.start
            end = chunk.end - 1
            entities[ind] = {
                'first_token': start,
                'last_token': end
            }

        instance = self._dataset_reader.text_to_instance(document, tokens, entities)
        return instance

    # @overrides
    # def dump_line(self, outputs: JsonDict) -> str:
    #     return 'my output\n'

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    ListField,
    TextField,
    SpanField,
    MetadataField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers import Token
from overrides import overrides

from tne.modeling.fields.multilabel_span_field import MultiLabelSpanField

logger = logging.getLogger(__name__)


@DatasetReader.register("tne_reader")
class TNEReader(DatasetReader):
    """

    Returns a `Dataset` where the `Instances` have four fields : `text`, a `TextField`
    containing the full document text, `spans`, a `ListField[SpanField]` of inclusive start and
    end indices for span candidates, and `metadata`, a `MetadataField` that stores the instance's
    original text. For data with gold cluster labels, we also include the original `clusters`
    (a list of list of index pairs) and a `SequenceLabelField` of cluster ids for every span
    candidate.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = `None`)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    """

    def __init__(
            self,
            prepositions: List[str],
            token_indexers: Dict[str, TokenIndexer] = None,
            wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
            development: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._development = development

        # assuming the enumerate function retain the order of the list and thus `no-relation` will get the 0 label
        self._prep_dict = {k: v for v, k in enumerate(prepositions)}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        if self._development:
            lines = lines[:100]

        documents = []
        for line in lines:
            doc = json.loads(line)
            documents.append(doc)

        for doc in documents:
            text = doc['text']
            tokens = doc['tokens']
            tokenized_entities = doc['nps']
            links = doc.get('np_relations', None)

            yield self.text_to_instance(text, tokens, tokenized_entities, links)

    @overrides
    def text_to_instance(
            self,  # type: ignore
            text: str,
            tokens: List[str],
            tokenized_entities: Dict,
            links: Optional[List[Dict]] = None,
    ) -> Instance:
        return self.make_tne_instance(
            text,
            tokens,
            tokenized_entities,
            self._token_indexers,
            links,
            self._wordpiece_modeling_tokenizer,
        )

    def make_tne_instance(self,
                          text: str,
                          tokens: List[str],
                          tokenized_entities: Dict,
                          token_indexers: Dict[str, TokenIndexer],
                          links: Optional[List[Dict]] = None,
                          wordpiece_modeling_tokenizer: PretrainedTransformerTokenizer = None,
                          ):
        if self._wordpiece_modeling_tokenizer is not None:
            flat_sentences_tokens, offsets = wordpiece_modeling_tokenizer.intra_word_tokenize(
                tokens
            )
        else:
            flat_sentences_tokens = [Token(word) for word in tokens]

        text_field = TextField(flat_sentences_tokens, token_indexers)

        entity_spans = {}
        for entity_id, row in tokenized_entities.items():
            if len(row) == 1: continue
            entity_spans[entity_id] = (row['first_token'], row['last_token'])

        spans: List[Field] = []

        for entity_id, (span_start, span_end) in entity_spans.items():
            spans.append(SpanField(span_start, span_end, text_field))

        span_field = ListField(spans)

        metadata: Dict[str, Any] = {"original_text": text,
                                    "tokenized_entities": tokenized_entities,
                                    "tokens": tokens}

        fields: Dict[str, Field] = {
            "text": text_field,
            "spans": span_field,
        }
        if links is not None:
            links_dic = defaultdict(list)
            prepositions_dic = defaultdict(dict)
            extended_prep_dic = defaultdict(dict)
            for row in links:
                links_dic[row['anchor']].append(row['complement'])
                prepositions_dic[row['anchor']][row['complement']] = self._prep_dict[row['preposition']]

                if row['anchor'] not in extended_prep_dic or row['complement'] not in extended_prep_dic[row['anchor']]:
                    extended_prep_dic[row['anchor']][row['complement']] = [self._prep_dict[row['preposition']]]
                else:
                    extended_prep_dic[row['anchor']][row['complement']].append([self._prep_dict[row['preposition']]])

            # Building a vector in the size of N^2, for all possible links
            # This includes the links on the diagonal (between the same NP), which will be filtered
            # in the model
            link_labels = []
            preposition_labels = []
            extended_prep_labels = []
            for ind1, span_id1 in enumerate(entity_spans.keys()):
                for ind2, span_id2 in enumerate(entity_spans.keys()):
                    if span_id1 in links_dic and span_id2 in links_dic[span_id1]:
                        link_labels.append(1)
                        preposition_labels.append(prepositions_dic[span_id1][span_id2])
                        extended_prep_labels.append(extended_prep_dic[span_id1][span_id2])
                    else:
                        link_labels.append(0)
                        preposition_labels.append(self._prep_dict['no-relation'])
                        extended_prep_labels.append([self._prep_dict['no-relation']])

                    # overriding the exist_link label on the diagonal (link between the same entity)
                    if ind1 == ind2:
                        link_labels[-1] = -1

            # if not self._joint_modeling:
            fields["link_labels"] = MultiLabelSpanField(link_labels, skip_indexing=True, num_labels=3,
                                                        label_namespace='link_labels')
            fields["preposition_labels"] = MultiLabelSpanField(preposition_labels, skip_indexing=True,
                                                               num_labels=len(self._prep_dict),
                                                               label_namespace='preposition_labels')

            metadata['extended_prepositions'] = extended_prep_labels
        metadata_field = MetadataField(metadata)
        fields['metadata'] = metadata_field

        return Instance(fields)

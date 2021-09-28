# Text-based NP Enrichment (TNE)

<img align="left" src="media/tne.svg" height="100"></img>
TNE is an NLU task, which focus on relations between noun phrases (NPs) that can be
mediated via prepositions. 
The dataset contains 5,497 documents, annotated exhaustively with all possible
links between the NPs in each document.

For more details check out our paper ["Text-based NP Enrichment"](https://arxiv.org/abs/2109.12085),
and [website](https://yanaiela.github.io/TNE).  


* Key Links
	* **TNE Dataset**: [Download](https://github.com/yanaiela/TNE#Download)
	* **Paper**: ["Text-based NP Enrichment"](https://arxiv.org/abs/2109.12085)
	* **Models Code**: [https://github.com/yanaiela/TNE/tree/main/tne/modeling](https://github.com/yanaiela/TNE/tree/main/tne/modeling)
	* **Leaderboard**:  
		* TNE:  [Leaderboard](https://leaderboard.allenai.org/tne/)
		* TNE-OOD:  [Leaderboard](https://leaderboard.allenai.org/tne-ood/)
		* Evaluator Code: [https://github.com/allenai/tne-evaluator](https://github.com/allenai/tne-evaluator)
	* **Website**: [https://yanaiela.github.io/TNE](https://yanaiela.github.io/TNE)


## Data

### Download
* [Train](https://github.com/yanaiela/TNE/raw/main/data/train.jsonl.gz)
* [Dev](https://github.com/yanaiela/TNE/raw/main/data/dev.jsonl.gz)
* [Test (unlabeled)](https://github.com/yanaiela/TNE/raw/main/data/test_unlabeled.jsonl.gz)
* [Test-OOD (unlabeled)](https://github.com/yanaiela/TNE/raw/main/data/ood_unlabeled.jsonl.gz)

### Data Format

The dataset is spread across four files, for the four different splits: train, dev, test and ood.
Each file is in a jsonl format, containing a dictionary of a single document.
A document consists of:

* `id`: a unique identifier of a document, beginning with `r` and followed by a number
* `text`: the text of the document. The title and subtitles (if exists) are separated with two new lines. The paragraphs
are separated by a single new line.
* `tokens`: a list of string, containing the tokenized tokens
* `nps`: a dictionary, where each key is an np identifier, which contains another dictionary:
  * `text`: the text of the np
  * `start_index`: an integer indicating the starting index in the text
  * `end_index`: an integer indicating the ending index in the text
  * `start_token`: an integer indicating the first token of the np out of the tokenized tokens
  * `end_token`: an integer indicating the last token of the np out of the tokenized tokens
  * `id`: the same as the key
* `np_relations`: these are the relation labels of the document. It is a list of dictionaries, where each
dictionary contains:
  * `anchor`: the id of the anchor np
  * `complement`: the id of the complement np
  * `preposition`: the preposition that links between the anchor and the complement
  * `complement_coref_cluster_id`: the coreference id, which the complement is part of.
* `coref`: the coreference labels. It contains a list of dictionaries, where each dictionary contains:
  * `id`: the id of the coreference cluster
  * `members`: the ids of the nps members of such cluster
  * `np_type`: the type of cluster. It can be either 
    * `standard`: regular coreference cluster
    * `time/date/measurement`: a time / date / measurement np. These will be singletons.
    * `idiomatic`: an idiomatic expression
* `metadata`: metadata of the document. It contains the following:
  * `annotators`: a dictionary with anonymized annotators id
    * `coref_worker`: the coreference worker id
    * `consolidator_worker`: the consolidator worker id
    * `np-relations_worker`: the np relations worker id
  * `url`: the url where the document was taken from (not always existing)
  * `source`: the original file name where the document was taken from

## Models

We train the models using [allennlp](https://allennlp.org/)

To run the coupled-large model, run:

```bash
allennlp train tne/modeling/configs/coupled_large.jsonnet \
         --include-package tne \
         -s models/coupled_spanbert_large
```

### Trained Model
We release the best model we achieved: `coupled-large` and it can be downloaded [here](https://storage.googleapis.com/ai2i/datasets/tne/coupled_spanbert_large.tar.gz).
If there's interest in other models from the paper, please let me know via email or open an issue, 
and I will upload them as well.


## Citation

```markdown
@article{tne,
Author = {Yanai Elazar and Victoria Basmov and Yoav Goldberg and Reut Tsarfaty},
Title = {Text-based NP Enrichment},
Year = {2021},
Eprint = {arXiv:2109.12085},
}
```


## Changelog
- `27/09/2021` TNE was released: paper + dataset + exploration + demo


## Q&A

#### Q: But what about huggingface (dataset, hub, implementation)
I found it easier to use the allennlp framework, but I might consider using hf infrastructure 
as well in the future. Feel free to upload the dataset there, or suggest an 
implementation using hf codebase.

#### Q: If I find a bug?
It happens! Please open an [issue](https://github.com/yanaiela/TNE/issues) and I'll do my best
to address it.

#### Q: What about additional trained models?
I uploaded the best model we trained from the paper. 
If there's interest, I can upload the others as well. Open an 
[issue](https://github.com/yanaiela/TNE/issues) or email me.


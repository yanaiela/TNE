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

### Load from Huggingface's Datasets Library
```
from datasets import load_dataset

dataset = load_dataset("tne")
```

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
* `nps`: a list of dictionaries, where each key is an np identifier, which contains another dictionary (note that this field was changed in v1.1 from a dictionary where each key is the np id to the dictionary, to a list of dictionaries, to match the huggingface datasets library):
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


## Getting Started

Install dependencies
```shell
conda create -n tne python==3.7 anaconda
conda activate tne

pip install -r requirements.txt
```

## Models

We train the models using [allennlp](https://allennlp.org/)

To run the coupled-large model, run:

```bash
allennlp train tne/modeling/configs/coupled_large.jsonnet \
         --include-package tne \
         -s models/coupled_spanbert_large
```

After training a model (or using the trained one), you can get the predictions file using:
```
allennlp predict models/coupled_spanbert_large/model.tar.gz data/test.jsonl --output-file coupled_large_predictions.jsonl --include-package bridging --use-dataset-reader --predictor tne_predictor
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

## Submitting to the Leaderboard
To submit your model's prediction to the leaderboard, you need to create an answer file.
You can find details on the submission pocess [here](https://leaderboard.allenai.org/tne/submissions/get-started), and follow the evaluation code and tests [here](https://github.com/allenai/tne-evaluator)


## Changelog
- `03/05/2021` The TNE dataset is on huggingface's [datasets library](https://huggingface.co/datasets/tne)
- `12/04/2021` Released v1.1 of the dataset. Changed the `nps` field from a dictionary of diciontaries, to a list of dictionaries, to match with huggingface's `datasets` library.
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

#### Why are there no labels in the released test-sets files?
We decided to keep the labels hidden, to avoid overfitting on this dataset. 
However, once you have a good model, you can upload your predictions to the [leaderboard](https://leaderboard.allenai.org/tne/) (and the [ood leaderboard](https://leaderboard.allenai.org/tne-ood/)), and find out your score!


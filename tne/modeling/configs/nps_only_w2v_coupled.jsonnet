// Configuration for a tne model
//   + SpanBERT-base

local transformer_model = "SpanBERT/spanbert-large-cased";
local max_length = 512;
local max_span_width = 30;
local emb_dim = 300;
local transformer_dim = 1024;  # uniquely determined by transformer_model
local span_embedding_dim = 300;
local span_pair_embedding_dim = 3 * span_embedding_dim;
local preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                          'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                          'inside', 'outside', 'into', 'around'];
local num_labels = std.length(preposition_list);

{
  "dataset_reader": {
    "type": "tne_reader",
    "token_indexers": {
            "tokens": { "type": "single_id" },
    },
    "development": false,
    "prepositions": preposition_list,
  },
  "validation_dataset_reader": {
    "type": "tne_reader",
    "token_indexers": {
            "tokens": { "type": "single_id" },
    },
    "prepositions": preposition_list,
  },
  "train_data_path": 'data/train.jsonl',
  "validation_data_path": 'data/dev.jsonl',
  "test_data_path": 'data/test_unlabeled.jsonl',
  "evaluate_on_test": false,
  "model": {
    "type": "nps_only_coupled",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/word2vec/GoogleNews-vectors-negative300.txt.gz",
            "embedding_dim": emb_dim,
            "trainable": false
        },
      }
    },
    "anchor_feedforward": {
        "input_dim": span_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 500,
        "activations": "relu",
        "dropout": 0.3
    },
    "complement_feedforward": {
        "input_dim": span_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 500,
        "activations": "relu",
        "dropout": 0.3
    },
    "preposition_predictor": {
        "input_dim": 500 + 500,
        "num_layers": 2,
        "hidden_dims": [100, num_labels],
        "activations": ["relu", "linear"],
        "dropout": 0.3
    },

    "prepositions": preposition_list,
    "freeze": false,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we a span field.
      "sorting_keys": ["text"],
      "batch_size": 1
    }
  },
  "trainer": {
    "num_epochs": 40,
    "patience" : 10,
    "cuda_device" : 0,
    "validation_metric": "+overall_micro_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-5,
    },
    "callbacks": [
        {
            "type": "wandb",
            "summary_interval": 1,
            "batch_size_interval": 1,
            "should_log_learning_rate": false,
            "should_log_parameter_statistics": false,
            "project": "tne",
            "name": "nps_only_w2v"
        },
    ],
  }
}

// Configuration for a tne model
//   + SpanBERT-base

local transformer_model = "SpanBERT/spanbert-base-cased";
local max_length = 512;
local max_span_width = 30;

local transformer_dim = 768;  # uniquely determined by transformer_model
local span_embedding_dim = 2 * transformer_dim;
local span_pair_embedding_dim = 3 * span_embedding_dim;
local preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                          'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                          'inside', 'outside', 'into', 'around'];
local num_labels = std.length(preposition_list);

{
  "dataset_reader": {
    "type": "tne_reader",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length,
      },
    },
    "development": false,
    "prepositions": preposition_list,
  },
  "validation_dataset_reader": {
    "type": "tne_reader",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length,
      },
    },
    "prepositions": preposition_list,
  },
  "train_data_path": 'data/train.jsonl',
  "validation_data_path": 'data/dev.jsonl',
  "test_data_path": 'data/test_unlabeled.jsonl',
  "evaluate_on_test": false,
  "model": {
    "type": "tne_decoupled_model",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length,
        }
      }
    },
    "context_layer": {
        "type": "pass_through",
        "input_dim": transformer_dim
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
    "link_predictor": {
        "input_dim": 500 + 500,
        "num_layers": 2,
        "hidden_dims": [100, 2],
        "activations": ["relu", "linear"],
        "dropout": 0.3
    },
    "initializer": {
      "regexes": [
        [".*_span_updating_gated_sum.*weight", {"type": "xavier_normal"}],
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer.*weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
      ]
    },
    "prepositions": preposition_list,
    "freeze": true,
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
    "validation_metric": "+overall_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-5,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
    "callbacks": [
        {
            "type": "wandb",
            "summary_interval": 1,
            "batch_size_interval": 1,
            "should_log_learning_rate": false,
            "should_log_parameter_statistics": false,
            "project": "tne",
            "name": "decoupled_probe_spanbert-base"
        },
    ],
  }
}

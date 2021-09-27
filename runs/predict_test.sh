for model in 'probe_decoupled_spanbert_base' 'probe_decoupled_spanbert_large' 'probe_coupled_spanbert_base' 'probe_coupled_spanbert_large' 'decoupled_spanbert_base' 'decoupled_spanbert_large' 'coupled_spanbert_base' 'coupled_spanbert_large'
do
  mkdir -p models/${model}/results
  allennlp evaluate models/${model}/model.tar.gz \
           data/processed/test.jsonl \
           --include-package tne \
           --cuda-device 0 \
           --output-file models/${model}/results/test_results.json
done

for ood in 'ood_books' 'ood_imdb' 'ood_reddit_askedscience' 'ood_reddit_atheism' 'ood_reddit_LifeProTips' 'ood_reddit_AskHistorians' 'ood_reddit_depressed' 'ood_reddit_YouShouldKnow' 'ood_reddit' 'ood'
do
  allennlp evaluate models/coupled_spanbert_large/model.tar.gz \
           data/processed/${ood}.jsonl \
           --include-package tne \
           --cuda-device 0 \
           --output-file models/coupled_spanbert_large/results/${ood}_results.json
done


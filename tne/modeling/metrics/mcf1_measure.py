from typing import Optional

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import dist_reduce_sum


@Metric.register("mc_f1")
class MCF1Measure(Metric):
    """
    An adaptation of the allennlp.training.metrics.fbeta_measure.FBetaMeasure class
    to be used with multi-class, but allowing to ignore one of the labels as part
    of the calculation.

    Compute precision, recall, F-measure and support for each class.

    The precision is the ratio `tp / (tp + fp)` where `tp` is the number of
    true positives and `fp` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio `tp / (tp + fn)` where `tp` is the number of
    true positives and `fn` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    If we have precision and recall, the F-beta score is simply:
    `F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)`

    The F-beta score weights recall more than precision by a factor of
    `beta`. `beta == 1.0` means recall and precision are equally important.

    The support is the number of occurrences of each class in `y_true`.

    # Parameters

    beta : `float`, optional (default = `1.0`)
        The strength of recall versus precision in the F-score.

    """

    def __init__(self, beta: float = 1.0) -> None:
        if beta <= 0:
            raise ConfigurationError("`beta` should be >0 in the F-beta score.")
        self._beta = beta

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: int = -1
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum: int = -1
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: int = -1
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: int = -1

    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum == -1:
            self._true_positive_sum = 0
            self._true_sum = 0
            self._pred_sum = 0
            self._total_sum = 0

        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = predictions.sum(dim=-1) != 0

        links = predictions != 0

        true_positives = (gold_labels == predictions) & links & mask & pred_mask

        true_positive_sum = torch.sum(true_positives).item()

        preds = (predictions != 0) & mask & pred_mask
        pred_sum = preds.sum().item()

        gold_labels = (gold_labels != 0) & mask
        true_sum = gold_labels.sum().item()

        self._true_positive_sum += dist_reduce_sum(true_positive_sum)
        self._pred_sum += dist_reduce_sum(pred_sum)
        self._true_sum += dist_reduce_sum(true_sum)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        precisions : `List[float]`
        recalls : `List[float]`
        f1-measures : `List[float]`

        !!! Note
            If `self.average` is not `None`, you will get `float` instead of `List[float]`.
        """
        if self._true_positive_sum == -1:
            raise RuntimeError("You never call this metric before.")

        else:
            tp_sum = self._true_positive_sum
            pred_sum = self._pred_sum
            true_sum = self._true_sum

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        if pred_sum == 0:
            precision = 0
        else:
            precision = tp_sum / pred_sum
        if true_sum == 0:
            recall = 0
        else:
            recall = tp_sum / true_sum
        if precision == 0 or recall == 0:
            fscore = 0
        else:
            fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)

        if reset:
            self.reset()

        return {"precision": precision, "recall": recall, "fscore": fscore}

    @overrides
    def reset(self) -> None:
        self._true_positive_sum = -1
        self._pred_sum = -1
        self._true_sum = -1

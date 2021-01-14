# coding=utf-8
# Copyright 2021 The Uncertainty Metrics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Oracle Collaborative Accuracy measures for probabilistic predictions.

Oracle Collaborative Accuracy measures the usefulness of model uncertainty
scores in facilitating human-computer collaboration (e.g., between a neural
model and an "oracle" human moderator in moderating online toxic comments).

The idea is that given a large amount of testing examples, the model will first
generate predictions for all examples, and then send a certain percentage of
examples that it is not confident about to the human moderators, whom we assume
can label those examples correctly.

The goal of this metric is to understand, under capacity constraints on the
human moderator (e.g., the model is only allowed to send 0.1% of total examples
to humans), how well the model can collaborate with the human to achieve the
highest overall accuracy. In this way, the metric attempts to quantify the
behavior of the full model-moderator system rather than of the model alone.

A model that collaborates with a human oracle well should not be accurate, but
also capable of quantifying its uncertainty well (i.e., its uncertainty should
be calibrated such that uncertainty ≅ model accuracy).
"""
import tensorflow as tf
from uncertainty_metrics.tensorflow import calibration


def _bin_probabilities(num_bins, index, dtype):
  """Computing corresponding probabilities.

  Args:
    num_bins: Number of bins to maintain over the interval [0, 1], and the bins
      are uniformly spaced.
    index: Which index to return.
    dtype: Data type

  Returns:
    bin_probabilities: Tensor, the corresponding probabilities.
  """
  return tf.cast(tf.linspace(0.0 + 1.0 / num_bins, 1.0, num_bins)[index], dtype)


class OracleCollaborativeAccuracy(calibration.ExpectedCalibrationError):
  """Oracle Collaborative Accuracy."""

  def __init__(self,
               fraction=0.01,
               num_bins=100,
               binary_threshold=0.5,
               name=None,
               dtype=None):
    """Constructs an expected collaborative accuracy metric.

    The class probabilities are computed using the argmax by default, but a
    custom threshold can be used in the binary case. This binary threshold is
    applied to the second (taken to be the positive) class.

    Args:
      fraction: the fraction of total examples to send to moderators.
      num_bins: Number of bins to maintain over the interval [0, 1].
      binary_threshold: Threshold to use in the binary case.
      name: Name of this metric.
      dtype: Data type.
    """
    super(OracleCollaborativeAccuracy, self).__init__(
        num_bins=num_bins, name=name, dtype=dtype)
    self.fraction = fraction
    self.binary_threshold = binary_threshold

  def _compute_pred_labels(self, probs):
    """Computes predicted labels, using binary_threshold in the binary case.

    Args:
      probs: Tensor of shape [..., k] of normalized probabilities associated
        with each of k classes.

    Returns:
      Predicted class labels.
    """
    return tf.cond(
        tf.shape(probs)[-1] == 2,
        lambda: tf.cast(probs[:, 1] > self.binary_threshold, tf.int64),
        lambda: tf.math.argmax(probs, axis=-1))

  def _compute_pred_probs(self, probs):
    """Computes predicted probabilities associated with the predicted labels."""
    pred_labels = self._compute_pred_labels(probs)
    indices = tf.stack(
        [tf.range(tf.shape(probs)[0], dtype=tf.int64), pred_labels], axis=1)
    return tf.gather_nd(probs, indices)

  def update_state(self,
                   labels,
                   probabilities,
                   custom_binning_score=None,
                   **kwargs):
    if self.binary_threshold != 0.5 and not custom_binning_score:
      # Bin by distance from threshold, i.e. send to the oracle in that order.
      custom_binning_score = tf.abs(probabilities - self.binary_threshold)

    super().update_state(
        labels, probabilities, custom_binning_score, kwargs=kwargs)

  def result(self):
    """Returns the expected calibration error."""
    num_total_examples = tf.reduce_sum(self.counts)
    num_oracle_examples = tf.cast(
        tf.floor(num_total_examples * self.fraction), self.dtype)

    non_empty_bin_mask = self.counts != 0
    counts = tf.boolean_mask(self.counts, non_empty_bin_mask)
    correct_sums = tf.boolean_mask(self.correct_sums, non_empty_bin_mask)
    cum_counts = tf.cumsum(counts)

    # Identify the final bin the oracle sees examples from, and the remaining
    # number of predictions it can make on that bin.
    final_oracle_bin = tf.cast(
        tf.argmax(cum_counts > num_oracle_examples), tf.int32)
    oracle_predictions_used = tf.cast(
        tf.cond(final_oracle_bin > 0, lambda: cum_counts[final_oracle_bin - 1],
                lambda: 0.), self.dtype)
    remaining_oracle_predictions = num_oracle_examples - oracle_predictions_used

    expected_correct_final_bin = (
        correct_sums[final_oracle_bin] / counts[final_oracle_bin] *
        (counts[final_oracle_bin] - remaining_oracle_predictions))
    expected_correct_after_final_bin = tf.reduce_sum(
        correct_sums[final_oracle_bin + 1:])

    expected_correct = (
        num_oracle_examples + expected_correct_final_bin +
        expected_correct_after_final_bin)
    return expected_correct / num_total_examples

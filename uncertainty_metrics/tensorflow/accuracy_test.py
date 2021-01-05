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

"""Tests for Oracle-Model Collaborative Accuracy."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import uncertainty_metrics as um


class OracleCollaborativeAccuracyTest(parameterized.TestCase, tf.test.TestCase):

  def testOracleCollaborativeAccuracy(self):
    num_bins = 10
    fraction = 0.4
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = np.array([0, 0, 0, 0, 0, 2, 3, 1, 2, 0])
    bin_correct_sums = np.array([0, 0, 0, 0, 0, 1, 2, 0, 2, 0])
    bin_prob_sums = np.array(
        [0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68, 0.71, 0.81 + 0.85, 0])
    # `(3 - 1)` refers to the rest examples in this bin
    # (minus the examples sent to the moderators), while `2/3` is
    # the accuracy in this bin.
    bin_collab_correct_sums = np.array(
        [0, 0, 0, 0, 0, 2, 1 * 1.0 + (3 - 1) * (2 / 3), 0, 2, 0])

    correct_acc = np.sum(bin_collab_correct_sums) / np.sum(bin_counts)

    metric = um.OracleCollaborativeAccuracy(
        fraction, num_bins, name='collab_acc', dtype=tf.float64)

    acc1 = metric(labels, pred_probs)
    self.assertAllClose(acc1, correct_acc)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    actual_bin_bin_collab_correct_sums = tf.convert_to_tensor(
        metric.collab_correct_sums)

    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)
    self.assertAllClose(bin_collab_correct_sums,
                        actual_bin_bin_collab_correct_sums)

  def testOracleCollaborativeAccuracyThresholdNearHalfSameResults(self):
    """Test that thresholds 0.5 and just above 0.5 give the same results."""
    num_bins = 13
    fraction = 0.267
    pred_probs = np.array([
        0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85, 0.49, 0.12, 0.37, 0.73,
        0.95, 0.14
    ])
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 0.])

    metric1 = um.OracleCollaborativeAccuracy(
        fraction, num_bins, name='collab_acc', dtype=tf.float64)
    acc1 = metric1(labels, pred_probs)

    metric2 = um.OracleCollaborativeAccuracy(
        fraction,
        num_bins,
        binary_threshold=0.5001,
        name='collab_acc2',
        dtype=tf.float64)

    metric2.update_state(labels, pred_probs)
    acc2 = metric2.result()

    self.assertAlmostEqual(acc1, acc2)

  def testOracleCollaborativeAccuracyBinaryThreshold(self):
    num_bins = 10
    fraction = 0.3
    binary_threshold = 0.655
    pred_probs = np.array([0.51, 0.45, 0.39, 0.76, 0.68, 0.29, 0.81, 0.85])
    # threshold_pred_probs: [0.49, 0.55, 0.61, 0.76, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [0, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 1., 1., 0., 1., 1., 1.])
    # Bins distances from 0.65 into (0, 0.1), [0.1, 0.2), ...,  [0.9, 1).
    bin_counts = np.array([1, 4, 2, 1, 0, 0, 0, 0, 0, 0])
    bin_correct_sums = np.array([0, 4, 1, 0, 0, 0, 0, 0, 0, 0])
    bin_prob_sums = np.array(
        [0.68, 0.49 + 0.76 + 0.81 + 0.85, 0.55 + 0.61, 0.71, 0, 0, 0, 0, 0, 0])
    # `(3 - 1)` refers to the rest examples in this bin
    # (minus the examples sent to the moderators), while `2/3` is
    # the accuracy in this bin.
    bin_collab_correct_sums = np.array([1, 4, 1, 0, 0, 0, 0, 0, 0, 0])

    correct_acc = np.sum(bin_collab_correct_sums) / np.sum(bin_counts)

    metric = um.OracleCollaborativeAccuracy(
        fraction,
        num_bins,
        binary_threshold=binary_threshold,
        name='collab_acc',
        dtype=tf.float64)

    acc1 = metric(labels, pred_probs)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    actual_bin_collab_correct_sums = tf.convert_to_tensor(
        metric.collab_correct_sums)

    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)
    self.assertAllClose(bin_collab_correct_sums, actual_bin_collab_correct_sums)

    self.assertAllClose(acc1, correct_acc)


if __name__ == '__main__':
  tf.test.main()

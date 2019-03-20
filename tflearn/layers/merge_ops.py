# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

@tf.custom_gradient
def getpow(y, w):
    out = tf.pow(y, w)
    def grad(dy):
        return w * out / y
    return out, grad


def merge(tensors_list, mode, axis=1, name="Merge"):
    """ Merge.

    Merge a list of `Tensor` into a single one. A merging 'mode' must be
    specified, check below for the different options.

    Input:
        List of Tensors.

    Output:
        Merged Tensors.

    Arguments:
        tensors_list: A list of `Tensor`, A list of tensors to merge.
        mode: `str`. Merging mode, it supports:
            ```
            'concat': concatenate outputs along specified axis
            'elemwise_sum': outputs element-wise sum
            'elemwise_mul': outputs element-wise mul
            'sum': outputs element-wise sum along specified axis
            'mean': outputs element-wise average along specified axis
            'prod': outputs element-wise multiplication along specified axis
            'max': outputs max elements along specified axis
            'min': outputs min elements along specified axis
            'and': `logical and` btw outputs elements along specified axis
            'or': `logical or` btw outputs elements along specified axis
            ```
        axis: `int`. Represents the axis to use for merging mode.
            In most cases: 0 for concat and 1 for other modes.
        name: A name for this layer (optional). Default: 'Merge'.

    """

    assert len(tensors_list) > 1, "Merge required 2 or more tensors."

    with tf.name_scope(name) as scope:
        tensors = [l for l in tensors_list]
        if mode == 'concat':
            inference = tf.concat(tensors, axis)
        elif mode == 'elemwise_sum':
            inference = tensors[0]
            for i in range(1, len(tensors)):
                inference = tf.add(inference, tensors[i])
        elif mode == 'elemwise_mul':
            inference = tensors[0]
            for i in range(1, len(tensors)):
                inference = tf.multiply(inference, tensors[i])
        elif mode == 'sum':
            inference = tf.reduce_sum(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'mean':
            inference = tf.reduce_mean(tf.concat(tensors, axis),
                                       reduction_indices=axis)
        elif mode == 'prod':
            inference = tf.reduce_prod(tf.concat(tensors, axis),
                                       reduction_indices=axis)
        elif mode == 'max':
            inference = tf.reduce_max(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'min':
            inference = tf.reduce_min(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'and':
            inference = tf.reduce_all(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'or':
            inference = tf.reduce_any(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'weighted':
            # Model outputs predictions in form [y0, y1] = [y0, 1 - y0]
            # Want to weight each pair of predictions with weight w
            # To make the loss linear wrt weight, the correct prediction
            # Needs to be raised to the power of w
            # If truth is [1, 0]
            # Desired result is [y0 ** w, 1 - y0 ** w]
            # If truth is [0, 1]
            # Desired result is [1 - (1 - y0) ** w, (1 - y0) ** w]
            # Note, if w = 1, truth does not matter

            y0 = tensors[0][:, :1]
            y1 = tensors[0][:, 1:]
            weights = tensors[1][:, :1]
            y0truth = tensors[1][:, 1:]
            y1truth = tf.subtract(tf.ones(tf.shape(y0truth)), y0truth)

            # If any value is 0, the gradient is NaN
            y0 = tf.clip_by_value(y0, 1e-10, 1 - 1e-10)
            y1 = tf.clip_by_value(y1, 1e-10, 1 - 1e-10)

            # Raise both predictions to the power w
            weighted_y0 = tf.pow(y0, weights)
            weighted_y1 = tf.pow(y1, weights)

            # Only keep the prediction which is true
            # Set false prediction to 0
            weighted_y0 = tf.multiply(weighted_y0, y0truth)
            weighted_y1 = tf.multiply(weighted_y1, y1truth)

            # 1 - weighted_prediction for each
            anti_loss_y0 = tf.subtract(tf.ones(tf.shape(weighted_y0)), weighted_y0)
            anti_loss_y1 = tf.subtract(tf.ones(tf.shape(weighted_y1)), weighted_y1)

            # Only keep (1 - weighted_value) for prediction which is true
            # Set false prediction to 0
            anti_loss_y0 = tf.multiply(anti_loss_y0, y0truth)
            anti_loss_y1 = tf.multiply(anti_loss_y1, y1truth)

            # Add weighted_prediction for one prediction, and 1 - weighted_prediction for the other
            inference = tf.add(weighted_y0, anti_loss_y1)
            anti_inference = tf.add(weighted_y1, anti_loss_y0)

            inference = tf.concat([inference, anti_inference], 1)

        else:
            raise Exception("Unknown merge mode", str(mode))

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def merge_outputs(tensor_list, name="MergeOutputs"):
    """ Merge Outputs.

    A layer that concatenate all outputs of a network into a single tensor.

    Input:
        List of Tensors [_shape_].

    Output:
        Concatenated Tensors [nb_tensors, _shape_].

    Arguments:
        tensor_list: list of `Tensor`. The network outputs.
        name: `str`. A name for this layer (optional).

    Returns:
        A `Tensor`.

    """
    with tf.name_scope(name) as scope:
        x = tf.concat(tensor_list, 1)

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, x)

    return x

import tensorflow as tf
import tree

from . import indexing_ops


def retrace(lambda_,
            qs,
            targnet_qs,
            actions,
            rewards,
            pcontinues,
            target_policy_probs,
            behaviour_policy_probs,
            stop_targnet_gradients=True,
            name=None):
    all_args = [
        lambda_, qs, targnet_qs, actions, rewards, pcontinues,
        target_policy_probs, behaviour_policy_probs
    ]
    # Mainly to simplify testing:
    (lambda_, qs, targnet_qs, actions, rewards, pcontinues, target_policy_probs,
     behaviour_policy_probs) = (
         tf.convert_to_tensor(arg) for arg in all_args)

    # Deduce the shapes of the arguments we'll create for retrace_core.
    qs_shape = tf.shape(qs)
    timesteps = qs_shape[0]  # Batch size is qs_shape[1].

    # Deduce the time indices for the arguments we'll create for retrace_core.
    timestep_indices_tm1 = tf.range(0, timesteps - 1)
    timestep_indices_t = tf.range(1, timesteps)

    # Construct arguments for retrace_core and call.
    q_tm1 = tf.gather(qs, timestep_indices_tm1)
    a_tm1 = tf.gather(actions, timestep_indices_tm1)

    r_t = tf.gather(rewards, timestep_indices_tm1)
    pcont_t = tf.gather(pcontinues, timestep_indices_tm1)

    target_policy_t = tf.gather(target_policy_probs, timestep_indices_t)
    behaviour_policy_t = tf.gather(behaviour_policy_probs, timestep_indices_t)
    targnet_q_t = tf.gather(targnet_qs, timestep_indices_t)
    a_t = tf.gather(actions, timestep_indices_t)

    core = retrace_core(lambda_, q_tm1, a_tm1, r_t, pcont_t, target_policy_t,
                        behaviour_policy_t, targnet_q_t, a_t,
                        stop_targnet_gradients)

    return {'loss': core['loss'], 'target': core['target']}
    # return base_ops.LossOutput(core.loss, None)


@tf.function(experimental_relax_shapes=True)
def _reverse_seq(sequence, sequence_lengths=None):
    """Reverse sequence along 0th dimension.

    Args:
      sequence: Tensor of shape [T, B, ...].
      sequence_lengths: (optional) tensor of shape [B]. If `None`, only reverse
        along dim 0.

    Returns:
      Tensor of same shape as sequence with dim 0 reversed up to sequence_lengths.
    """
    if sequence_lengths is None:
        return tf.reverse(sequence, [0])

    sequence_lengths = tf.convert_to_tensor(sequence_lengths)
    with tf.control_dependencies(tf.debugging.assert_equal(
            tf.shape(sequence)[1], tf.shape(sequence_lengths)[0])):
        return tf.reverse_sequence(
            sequence, sequence_lengths, seq_axis=0, batch_axis=1)


@tf.function(experimental_relax_shapes=True)
def cumulative_discounted_sum(sequence, decay, initial_value, reverse=False,
                              sequence_lengths=None, back_prop=True,
                              name="cumulative_discounted_sum"):
    """Evaluates a cumulative discounted sum along dimension 0.

      ```python
      if reverse = False:
        result[1] = sequence[1] + decay[1] * initial_value
        result[k] = sequence[k] + decay[k] * result[k - 1]
      if reverse = True:
        result[last] = sequence[last] + decay[last] * initial_value
        result[k] = sequence[k] + decay[k] * result[k + 1]
      ```

    Respective dimensions T, B and ... have to be the same for all input tensors.
    T: temporal dimension of the sequence; B: batch dimension of the sequence.

      if sequence_lengths is set then x1 and x2 below are equivalent:
      ```python
      x1 = zero_pad_to_length(
        scan_discounted_sum(
            sequence[:length], decays[:length], **kwargs), length=T)
      x2 = scan_discounted_sum(sequence, decays,
                               sequence_lengths=[length], **kwargs)
      ```

    Args:
      sequence: Tensor of shape `[T, B, ...]` containing values to be summed.
      decay: Tensor of shape `[T, B, ...]` containing decays/discounts.
      initial_value: Tensor of shape `[B, ...]` containing initial value.
      reverse: Whether to process the sum in a reverse order.
      sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
        (reversed and then) summed.
      back_prop: Whether to backpropagate.
      name: Sets the name_scope for this op.

    Returns:
      Cumulative sum with discount. Same shape and type as `sequence`.
    """
    if sequence_lengths is not None:
        raise NotImplementedError
        # Zero out sequence and decay beyond sequence_lengths.
        with tf.control_dependencies(
                [tf.assert_equal(sequence.shape[0], decay.shape[0])]):
            mask = tf.sequence_mask(sequence_lengths, maxlen=sequence.shape[0],
                                    dtype=sequence.dtype)
            mask = tf.transpose(mask)

        # Adding trailing dimensions to mask to allow for broadcasting.
        sequence_broadcast_shape = tf.concat((
            tf.shape(mask), tf.ones(tf.rank(sequence) - tf.rank(mask))))
        sequence *= tf.reshape(mask, sequence_broadcast_shape)
        decay_broadcast_shape = tf.concat((
            tf.shape(mask), tf.ones(tf.rank(decay) - tf.rank(mask))))
        decay *= tf.reshape(mask, decay_broadcast_shape)

    sequences = (sequence, decay)

    if reverse:
        sequences = tree.map_structure(
            lambda s: _reverse_seq(s, sequence_lengths), sequences)

    summed = tf.scan(
        lambda a, x: x[0] + x[1] * a,
        sequences,
        initializer=initial_value,
        parallel_iterations=1)

    if not back_prop:
        summed = tf.stop_gradient(summed)
    if reverse:
        summed = _reverse_seq(summed, sequence_lengths)
    return summed


@tf.function(experimental_relax_shapes=True)
def off_policy_corrected_multistep_target(rewards,
                                          continuation_probs,
                                          traces,
                                          expected_Q_values,
                                          Q_values,
                                          back_prop=False,
                                          name=None):
    """Evaluates targets for various off-policy value correction based algorithms.
    `target_policy_t` is the policy that this function aims to evaluate. New
    action-value estimates (target values `T`) must be expressible in this
    recurrent form:
    ```none
    T(x_{t-1}, a_{t-1}) = rewards + γ * [
        𝔼_π Q(x_t, .)
        - traces * Q(x_t, a_t)
        + traces * T(x_t, a_t) ]
    ```
    `T(x_t, a_t)` is an estimate of expected discounted future returns based
    on the current Q value estimates `Q(x_t, a_t)` and rewards `rewards`. The
    evaluated target values can be used as supervised targets for learning the Q
    function itself or as returns for various policy gradient algorithms.
    `Q==T` if convergence is reached. As the formula is recurrent, it will
    evaluate multistep returns for non-zero importance weights `c_t`.
    In the usual moving and target network setup `Q_values` should be calculated by
    the target network while the `target_policy_t` may be evaluated by either of
    the networks. If `target_policy_t` is evaluated by the current moving network
    the algorithm implemented will have a similar flavour as double DQN.
    Depending on the choice of traces, the algorithm can implement:
    ```none
    Importance Sampling             traces = π(x_t, a_t) / μ(x_t, a_t),
    Harutyunyan's et al. Q(lambda)  traces = λ,
    Precup's et al. Tree-Backup     traces = π(x_t, a_t),
    Munos' et al. Retrace           traces = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).
    ```
    Please refer to page 3 for more details:
    https://arxiv.org/pdf/1606.02647v1.pdf
    Args:
      rewards: 2-D tensor holding rewards received during the transition
        that corresponds to each major index.
        Shape is `[T, B]`.
      continuation_probs: 2-D tensor holding continuation probability values
        received during the transition that corresponds to each major index.
        Shape is `[T, B]`.
      target_policy_t:  3-D tensor holding per-action policy probabilities for
        the states encountered just AFTER the transitions that correspond to
        each major index, according to the target policy (i.e. the policy we
        wish to learn). These usually derive from the learning net.
        Shape is `[T, B, num_actions]`.
      traces: 2-D tensor holding importance weights; see discussion above.
        Shape is `[T, B]`.
      Q_values: 3-D tensor holding per-action Q-values for the states
        encountered just AFTER taking the transitions that correspond to each
        major index. Shape is `[T, B, num_actions]`.
      back_prop: whether to backpropagate gradients through time.
      name: name of the op.
    Returns:
      Tensor of shape `[T, B, num_actions]` containing Q values.
    """
    # Formula (4) in https://arxiv.org/pdf/1606.02647v1.pdf can be expressed
    # in a recursive form where T is a new target value:
    # T(x_{t-1}, a_{t-1}) = rewards + γ * [
    #     𝔼_π Q(x_t, .)
    #     - traces * Q(x_t, a_t)
    #     + traces * T(x_t, a_t) ]
    # This recurrent form allows us to express Retrace by using
    # `cumulative_discounted_sum`.
    # Define:
    #   T_tm1             = T(x_{t-1}, a_{t-1})
    #   T_t               = T(x_t, a_t)
    #   expected_Q_values = 𝔼_π Q(x_t,.)
    #   Q_values          = Q(x_t, a_t)
    # Hence:
    #   T_tm1   = rewards + γ * (
    #               expected_Q_values - traces * Q_values
    #             ) + γ * traces * T_t
    # Define:
    #   current = rewards + γ * (expected_Q_values - traces * Q_values)
    # Thus:
    #   T_tm1 = cumulative_discounted_sum(current, γ * traces, reverse=True)
    current = rewards + continuation_probs * (
        expected_Q_values - traces * Q_values)
    initial_value = Q_values[-1]
    return cumulative_discounted_sum(
        current,
        continuation_probs * traces,
        initial_value,
        reverse=True,
        back_prop=back_prop)


@tf.function(experimental_relax_shapes=True)
def _retrace_weights(target_policy_probs, behavior_policy_probs):
    """Evaluates importance weights for the Retrace algorithm.
    Args:
      target_policy_probs: taken action probabilities according to target
        policy. Shape is `[T, B]`.
      behavior_policy_probs: taken action probabilities according to behaviour
        policy. Shape is `[T, B]`.
    Returns:
      Tensor of shape `[T, B]` containing importance weights.
    """
    # tf.minimum seems to handle potential NaNs when
    # behavior_policy_probs[i] = 0
    # TODO(hartikainen): This might have the same nan-propagation issue as we
    # had in vod.py!
    return tf.minimum(1.0, target_policy_probs / behavior_policy_probs)


@tf.function(experimental_relax_shapes=True)
def retrace_core(lambda_,
                 Q_values_t_0,
                 actions_t_0,
                 rewards_t_0,
                 continuation_probs,
                 target_policy_probs_all_actions,
                 behaviour_policy_probs,
                 target_Q_values,
                 actions_t_1,
                 stop_target_gradients=True,
                 name=None):
    target_policy_probs = indexing_ops.batched_index(
        target_policy_probs_all_actions, actions_t_1)
    # Evaluate importance weights.
    traces = _retrace_weights(
        target_policy_probs,
        behaviour_policy_probs) * lambda_
    # Targets are evaluated by using only Q values from the target network.
    # This provides fixed regression targets until the next target network
    # update.
    expected_Q_values = tf.reduce_sum(
        target_policy_probs_all_actions * target_Q_values, axis=2)
    chosen_Q_values = indexing_ops.batched_index(target_Q_values, actions_t_1)
    target = off_policy_corrected_multistep_target(
        rewards_t_0,
        continuation_probs,
        traces,
        expected_Q_values,
        chosen_Q_values,
        not stop_target_gradients)

    if stop_target_gradients:
        target = tf.stop_gradient(target)
    # Regress Q values of the learning network towards the targets evaluated
    # by using the target network.
    Q_actions_t_0 = indexing_ops.batched_index(Q_values_t_0, actions_t_0)
    delta = target - Q_actions_t_0
    loss = 0.5 * tf.square(delta)

    return {
        'loss': loss,
        'retrace_weights': traces,
        'target': target,
    }

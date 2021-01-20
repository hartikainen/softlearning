import numpy as np
import tensorflow as tf

from . import retrace_ops


class RetraceOpsTest(tf.test.TestCase):
    """Tests for `Retrace` ops."""

    def setUp(self):
        """Defines example data for, and an expected result of, Retrace operations.

        The example data comprises a minibatch of two sequences of four
        consecutive timesteps, allowing the data to be interpreted by Retrace
        as three successive transitions.
        """
        super(RetraceOpsTest, self).setUp()

        # Example input data:

        self.lambda_ = 0.9
        self.qs = np.array([
            [[2.2, 3.2, 4.2],
             [5.2, 6.2, 7.2]],

            [[7.2, 6.2, 5.2],
             [4.2, 3.2, 2.2]],

            [[3.2, 5.2, 7.2],
             [4.2, 6.2, 9.2]],

            [[np.nan] * 3,
             [np.nan] * 3]], dtype=np.float32)
        self.targnet_qs = np.array([
            [[np.nan] * 3,
             [np.nan] * 3],

            [[7., 6., 5.],
             [4., 3., 2.]],

            [[3., 5., 7.],
             [4., 6., 9.]],

            [[2., 8., 4.],
             [9., 1., 8.]]], dtype=np.float32)
        self.actions = np.array([
            [2,
             0], [1,
                  2], [0,
                       1], [9999,
                            9999]])
        self.rewards = np.array([
            [1.9,
             2.9], [3.9,
                    4.9], [5.9,
                           6.9], [np.nan,  # nan marks entries we should never use.
                                  np.nan]], dtype=np.float32)
        self.pcontinues = np.array([
            [0.8,
             0.9], [0.7,
                    0.8], [0.6,
                           0.5], [np.nan,
                                  np.nan]], dtype=np.float32)
        self.target_policy_probs = np.array([
            [[np.nan] * 3,
             [np.nan] * 3],

            [[0.41, 0.28, 0.31],
             [0.19, 0.77, 0.04]],

            [[0.22, 0.44, 0.34],
             [0.14, 0.25, 0.61]],

            [[0.16, 0.72, 0.12],
             [0.33, 0.30, 0.37]]], dtype=np.float32)
        self.behaviour_policy_probs = np.array([
            [np.nan,
             np.nan], [0.85,
                       0.86], [0.87,
                               0.88], [0.89,
                                       0.84]], dtype=np.float32)

        ### Expected results of Retrace as applied to the above:

        # NOTE: To keep the test code compact, we don't use the example data when
        # manually computing our expected results, but instead duplicate their
        # values explictly in those calculations. Some patterns in the values can
        # help you track who's who: for example, note that target network Q values
        # are integers, whilst learning network Q values all end in 0.2.

        # In a purely theoretical setting, we would compute the quantity we call
        # the "trace" using this recurrence relation:
        #
        #     ΔQ_tm1 = δ_tm1  +  λγπ(a_t | s_t)/μ(a_t | s_t) ⋅ ΔQ_t
        #      δ_tm1 = r_t + γ𝔼_π[Q(s_t, .)] - Q(s_tm1, a_tm1)
        #
        # In a target network setting, you might rewrite ΔQ_t as ΔQ'_t, indicating
        # that this value is the next-timestep trace as computed when all
        # Q(s_tm1, a_tm1) terms (in δ_t, δ_t+1, ...) come from the target network,
        # not the learning network.
        #
        # To generate our collection of expected outputs, we'll first compute
        # "ΔQ'_tm1" (the "target network trace") at all timesteps.
        #
        # We start at the end of the sequence and work backward, like the
        # implementation does.
        targ_trace = np.zeros((3, 2))
        targ_trace[2, 0] = (5.9 + 0.6*(0.16*2 + 0.72*8 + 0.12*4) - 3)  # δ_tm1[2,0]
        targ_trace[2, 1] = (6.9 + 0.5*(0.33*9 + 0.30*1 + 0.37*8) - 6)  # δ_tm1[2,1]

        targ_trace[1, 0] = (3.9 + 0.7*(0.22*3 + 0.44*5 + 0.34*7) - 6 +  # δ_tm1[1,0]
                            0.9*0.7*0.22/0.87 * targ_trace[2, 0])
        targ_trace[1, 1] = (4.9 + 0.8*(0.14*4 + 0.25*6 + 0.61*9) - 2 +  # δ_tm1[1,1]
                            0.9*0.8*0.25/0.88 * targ_trace[2, 1])

        targ_trace[0, 0] = (1.9 + 0.8*(0.41*7 + 0.28*6 + 0.31*5) - 4 +  # δ_tm1[0,0]
                            0.9*0.8*0.28/0.85 * targ_trace[1, 0])
        targ_trace[0, 1] = (2.9 + 0.9*(0.19*4 + 0.77*3 + 0.04*2) - 5 +  # δ_tm1[0,1]
                            0.9*0.9*0.04/0.86 * targ_trace[1, 1])

        # We can evaluate target Q values by adding targ_trace to single step
        # returns.
        target_q = np.zeros((3, 2))
        target_q[2, 0] = (5.9 + 0.6*(0.16*2 + 0.72*8 + 0.12*4))
        target_q[2, 1] = (6.9 + 0.5*(0.33*9 + 0.30*1 + 0.37*8))

        target_q[1, 0] = (3.9 + 0.7*(0.22*3 + 0.44*5 + 0.34*7) +
                          0.9*0.7*0.22/0.87 * targ_trace[2, 0])
        target_q[1, 1] = (4.9 + 0.8*(0.14*4 + 0.25*6 + 0.61*9) +
                          0.9*0.8*0.25/0.88 * targ_trace[2, 1])

        target_q[0, 0] = (1.9 + 0.8*(0.41*7 + 0.28*6 + 0.31*5) +
                          0.9*0.8*0.28/0.85 * targ_trace[1, 0])
        target_q[0, 1] = (2.9 + 0.9*(0.19*4 + 0.77*3 + 0.04*2) +
                          0.9*0.9*0.04/0.86 * targ_trace[1, 1])

        # Now we can compute the "official" trace (ΔQ_tm1), which involves the
        # learning network. The only difference from the "target network trace"
        # calculations is the Q(s_tm1, a_tm1) terms we use:
        trace = np.zeros((3, 2))  #     ↓ Q(s_tm1, a_tm1)
        trace[2, 0] = target_q[2, 0] - 3.2  # δ_tm1[2,0]
        trace[2, 1] = target_q[2, 1] - 6.2  # δ_tm1[2,1]

        trace[1, 0] = target_q[1, 0] - 6.2   # δ_tm1[1,0]
        trace[1, 1] = target_q[1, 1] - 2.2   # δ_tm1[1,1]

        trace[0, 0] = target_q[0, 0] - 4.2   # δ_tm1[0,0]
        trace[0, 1] = target_q[0, 1] - 5.2   # δ_tm1[0,0]

        self.expected_result = 0.5 * np.square(trace)
        self.target_q = target_q

    def _get_retrace_core(self):
        """Constructs a tf subgraph from `retrace_core` op.

        A retrace core namedtuple is built from a two-sequence, three-timestep
        input minibatch.

        Returns:
          Tuple of size 3 containing non-differentiable inputs, differentiable
          inputs and retrace_core namedtuple.
        """
        # Here we essentially replicate the preprocessing that `retrace` does
        # as it constructs the inputs to `retrace_core`.
        # These ops must be Tensors so that we can use them in the
        # `testNoOtherGradients` unit test. TensorFlow can only compute gradients
        # with respect to other parts of the graph
        lambda_ = tf.constant(self.lambda_)
        q_tm1 = tf.constant(self.qs[:3])
        a_tm1 = tf.constant(self.actions[:3])
        r_t = tf.constant(self.rewards[:3])
        pcont_t = tf.constant(self.pcontinues[:3])
        target_policy_t = tf.constant(self.target_policy_probs[1:4])
        behaviour_policy_t = tf.constant(self.behaviour_policy_probs[1:4])
        targnet_q_t = tf.constant(self.targnet_qs[1:4])
        a_t = tf.constant(self.actions[1:4])
        static_args = [lambda_, a_tm1, r_t, pcont_t, target_policy_t,
                       behaviour_policy_t, targnet_q_t, a_t]
        diff_args = [q_tm1]
        with tf.GradientTape() as tape:
            tape.watch(diff_args)
            core_result = retrace_ops.retrace_core(
                lambda_, q_tm1, a_tm1, r_t, pcont_t,
                target_policy_t, behaviour_policy_t,
                targnet_q_t, a_t)
        static_gradients, diff_gradients = tape.gradient(
            [core_result['loss']], [static_args, diff_args])
        result = (
            static_args,
            diff_args,
            core_result,
            static_gradients,
            diff_gradients,
        )
        return result

    def testRetraceCoreTargetQThreeTimeSteps(self):
        """Tests whether retrace_core evaluates correct targets for regression."""
        _, _, retrace, _, _ = self._get_retrace_core()
        self.assertAllClose(retrace['target'], self.target_q)

    def testRetraceCoreLossThreeTimeSteps(self):
        """Tests whether retrace_core evaluates correct losses."""
        _, _, retrace, _, _ = self._get_retrace_core()
        self.assertAllClose(retrace['loss'], self.expected_result)

    def testNoOtherGradients(self):
        """Tests no gradient propagates through things other than q_tm1."""
        static_args, _, retrace, static_gradients, _ = self._get_retrace_core()
        self.assertEqual(static_gradients, [None] * len(static_gradients))

    def testMovingNetworkGradientIsEvaluated(self):
        """Tests that gradients are evaluated w.r.t. q_tm1."""
        _, diff_args, retrace, _, diff_gradients = self._get_retrace_core()
        for gradient in diff_gradients:
            self.assertNotEqual(gradient, None)


if __name__ == '__main__':
    tf.test.main()

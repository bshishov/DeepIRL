import tensorflow as tf

from deepirl.models.base import ModelBase
from deepirl.models.policy import PolicyValueHead


def log_sigmoid(a):
    """Equivalent to tf.log(tf.sigmoid(a))"""
    return -tf.nn.softplus(-a)


def logit_bernoulli_entropy(logits):
    """ Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
    ent = (1.-tf.nn.sigmoid(logits))*logits - log_sigmoid(logits)
    return ent


def conv_embedding(inputs: tf.Tensor,
                   residual_blocks: int = 2,
                   outputs: int = 256,
                   output_activation=tf.nn.tanh,
                   conv_filters: int = 32,
                   reuse: bool = False,
                   scope_name: str = 'ConvEmbedding') -> tf.Tensor:
    with tf.variable_scope(scope_name, reuse=reuse):
        with tf.variable_scope('ConvBlock'):
            conv1 = tf.layers.conv2d(inputs, conv_filters, [3, 3], padding='same', name='Conv1')
            batch_norm1 = tf.layers.batch_normalization(conv1, name='BatchNorm1')
            relu1 = tf.nn.relu(batch_norm1, name='ReLu1')

        x = relu1
        for i in range(residual_blocks):
            with tf.variable_scope('Residual{0}'.format(i)):
                conv1 = tf.layers.conv2d(x, conv_filters, [3, 3], name='Conv1', padding='same')
                batch_norm1 = tf.layers.batch_normalization(conv1, name='BatchNorm1')
                relu1 = tf.nn.relu(batch_norm1, name='Relu1')
                conv2 = tf.layers.conv2d(relu1, conv_filters, [3, 3], name='Conv2', padding='same')
                batch_norm2 = tf.layers.batch_normalization(conv2, name='BatchNorm2')
                relu2 = tf.nn.relu(batch_norm2 + x, name='Relu2')
                x = relu2
        conv_out = tf.layers.flatten(x, name='Flatten')
        out = tf.layers.dense(conv_out, outputs, activation=output_activation, name='DenseEmbedding')
        return out


def discriminator_head(observation_embedding: tf.Tensor,
                       actions: tf.Tensor,
                       hidden_size: int = 64,
                       scope_name: str ='DiscriminatorHead',
                       reuse: bool = False):
    with tf.variable_scope(scope_name, reuse=reuse):
        inputs = tf.concat((observation_embedding, actions), axis=1, name='Inputs')
        dense_1 = tf.layers.dense(inputs, hidden_size, name='Dense1', activation=tf.nn.tanh)
        dense_2 = tf.layers.dense(dense_1, hidden_size, name='Dense2', activation=tf.nn.tanh)
        logits = tf.layers.dense(dense_2, 1, name='Logits', activation=None)
        logits = tf.reshape(logits, [-1])
        return logits


def discriminator(observations: tf.Tensor,
                  actions: tf.Tensor,
                  scope_name: str = 'Discriminator',
                  reuse: bool = False):
    with tf.variable_scope(scope_name, reuse=reuse):
        conv_embed = conv_embedding(observations, reuse=reuse)
        logits = discriminator_head(conv_embed, actions, reuse=reuse)
        return logits


def policy_value(observations: tf.Tensor,
                 actions_dim: int,
                 scope_name: str = 'PolicyValue',
                 reuse: bool = False):
    with tf.variable_scope(scope_name, reuse=reuse):
        conv_embed = conv_embedding(observations, reuse=reuse)

        with tf.variable_scope('PolicyHead'):
            x = conv_embed
            x = tf.layers.dense(x, 64, name='fc1', activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, name='fc2', activation=tf.nn.tanh)
            x = tf.layers.dense(x, actions_dim, name='Mean', activation=None)
            policy_mean = x

            # Std just from 1 variable
            log_std = tf.Variable(tf.zeros([actions_dim]) - 2.0, dtype=tf.float32, trainable=True, name='LogStd')
            policy_std = tf.nn.softplus(log_std) + 1e-5

        with tf.variable_scope('ValueHead'):
            x = conv_embed
            x = tf.layers.dense(x, 64, name='fc1', activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, name='fc2', activation=tf.nn.tanh)
            x = tf.layers.dense(x, 1, name='fc3', activation=None)
            value = tf.reshape(x, [-1])

        with tf.variable_scope('Distribution'):
            # Build a MultivariateNormal distribution with Mu from nn and std as trainable variable
            dist = tf.contrib.distributions.MultivariateNormalDiag(policy_mean, policy_std)

        return dist, value


class GailPpoCnn(ModelBase):
    SCOPE_DISCRIMINATOR = 'Discriminator'
    SCOPE_POLICY_VALUE = 'PolicyValue'
    SCOPE_BEHAVIORAL_CLONE = 'BehavioralClone'

    def __init__(self,
                 state_shape: tuple,
                 action_dim: int,
                 entropy_coeff: float = 0.001,
                 clip_epsilon: float = 1e-2):
        super().__init__()
        with tf.variable_scope('Inputs'):
            with tf.variable_scope(self.SCOPE_DISCRIMINATOR):
                self.d_exp_observations = tf.placeholder(dtype=tf.float32, shape=(None, ) + state_shape,
                                                         name='ExpertObservations')
                self.d_exp_actions = tf.placeholder(dtype=tf.float32, shape=(None, action_dim),
                                                    name='ExpertActions')

                self.d_gen_observations = tf.placeholder(dtype=tf.float32, shape=(None,) + state_shape,
                                                         name='GeneratorObservations')
                self.d_gen_actions = tf.placeholder(dtype=tf.float32, shape=(None, action_dim),
                                                    name='GeneratorActions')

            with tf.variable_scope(self.SCOPE_POLICY_VALUE):
                self.ppo_observations = self.d_gen_observations
                self.ppo_actions = self.d_gen_actions
                self.ppo_target_values = tf.placeholder(dtype=tf.float32, shape=(None, ), name='TargetValues')
                self.ppo_advantages = tf.placeholder(dtype=tf.float32, shape=(None,), name='Advantages')
                self.ppo_old_logprob = tf.placeholder(dtype=tf.float32, shape=(None,), name='NegLogProb')

            with tf.variable_scope(self.SCOPE_BEHAVIORAL_CLONE):
                self.bc_observations = self.d_gen_observations
                self.bc_actions = self.d_gen_actions

        exp_logits = discriminator(self.d_exp_observations, self.d_exp_actions,
                                   scope_name=self.SCOPE_DISCRIMINATOR, reuse=False)
        gen_logits = discriminator(self.d_gen_observations, self.d_gen_actions,
                                   scope_name=self.SCOPE_DISCRIMINATOR, reuse=True)

        self.policy_dist, value = policy_value(self.ppo_observations, action_dim, scope_name=self.SCOPE_POLICY_VALUE)

        with tf.variable_scope('Reward'):
            self.reward = -tf.log(1 - tf.nn.sigmoid(gen_logits) + 1e-8)

        with tf.variable_scope('Losses'):
            with tf.variable_scope(self.SCOPE_DISCRIMINATOR):
                """ DISCRIMINATOR LOSS """
                generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_logits,
                                                                         labels=tf.zeros_like(gen_logits))
                generator_loss = tf.reduce_mean(generator_loss, name='GeneratorLoss')

                expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=exp_logits,
                                                                      labels=tf.ones_like(exp_logits))
                expert_loss = tf.reduce_mean(expert_loss, name='ExpertLoss')

                logits = tf.concat([gen_logits, exp_logits], 0)
                entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
                entropy_loss = -entropy_coeff * entropy

                self.d_loss = generator_loss + expert_loss + entropy_loss

            with tf.variable_scope(self.SCOPE_BEHAVIORAL_CLONE):
                """ BEHAVIORAL CLONING LOSS """
                bc_entropy_loss = - entropy_coeff * tf.reduce_mean(self.policy_dist.entropy())
                bc_prob_loss = - tf.reduce_mean(self.policy_dist.log_prob(self.bc_actions))
                bc_mode_loss_deterministic = tf.losses.mean_squared_error(self.bc_actions, self.policy_dist.mode())
                self.bc_loss = bc_mode_loss_deterministic

            with tf.variable_scope(self.SCOPE_POLICY_VALUE):
                """ PROXIMAL POLICY OPTIMIZATION LOSS """
                # Entropy
                entropy = tf.reduce_mean(self.policy_dist.entropy())
                loss_entropy = (-entropy_coeff) * entropy

                # L^VF - Value function loss
                # loss_value = tf.reduce_mean(tf.squared_difference(self.policy.value, self.target_values))
                loss_value = tf.losses.huber_loss(self.ppo_target_values, value)

                # L^CLIP - Clipped loss (surrogate objectives)
                dist_ratio = tf.exp(self.policy_dist.log_prob(self.ppo_actions) - self.ppo_old_logprob)

                surr_1 = dist_ratio * self.ppo_advantages
                surr_2 = tf.clip_by_value(dist_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * self.ppo_advantages
                loss_clip = -tf.reduce_mean(tf.minimum(surr_1, surr_2))
                self.ppo_loss = loss_clip + loss_value + loss_entropy

        with tf.variable_scope('Optimization'):
            bc_opt = tf.train.AdamOptimizer()
            self.bc_opt_op = bc_opt.minimize(self.bc_loss)

            ppo_opt = tf.train.AdamOptimizer()
            self.ppo_opt_op = ppo_opt.minimize(self.ppo_loss)

            d_opt = tf.train.AdamOptimizer()
            self.d_opt_op = d_opt.minimize(self.d_loss)

    def train_bc(self, sess: tf.Session, observations, actions):
        loss, _ = sess.run([self.bc_loss, self.bc_opt_op], feed_dict={
            self.bc_observations: observations,
            self.bc_actions: actions
        })
        return loss

    def train_ppo(self, sess: tf.Session, observations, advantages, values, old_log_prob):
        loss, _ = sess.run([self.ppo_loss, self.ppo_opt_op], feed_dict={
            self.ppo_observations: observations,
            self.ppo_advantages: advantages,
            self.ppo_target_values: values,
            self.ppo_old_logprob: old_log_prob
        })
        return loss

    def train_discriminator(self, sess: tf.Session,
                            observations_exp, actions_exp,
                            observations_gen, actions_gen):
        loss, _ = sess.run([self.d_loss, self.ppo_opt_op], feed_dict={
            self.d_exp_observations: observations_exp,
            self.d_exp_actions: actions_exp,
            self.d_gen_observations: observations_gen,
            self.d_gen_actions: actions_gen,
        })
        return loss

    def predict_actions(self, sess: tf.Session, observations):
        actions, log_prob = sess.run(self.policy_dist.sample(), feed_dict={
            self.ppo_observations: observations,
        })
        return actions


def main():
    model = GailPpoCnn((32, 32, 5), 2)


if __name__ == '__main__':
    main()

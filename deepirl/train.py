import tensorflow as tf
import argparse

from deepirl.utils.replay import StateActionReplay
from deepirl.utils.config import instantiate
from deepirl.models.dqn import Model


IRL_MEMORY_SIZE = 10000
IRL_BATCH_SIZE = 256
IRL_STOP_LOSS = 2.0


def train_irl(sess: tf.Session, model: Model, replay: StateActionReplay, epochs: int):
    for epoch in range(epochs):
        if len(replay) > IRL_BATCH_SIZE:
            states, actions = replay.sample(IRL_BATCH_SIZE)
            loss = model.train_irl(sess, states, actions)
            print('IRL: Epoch: {0}/{1} Loss: {2:.3f}'.format(epoch, epochs, loss))
            if loss < IRL_STOP_LOSS:
                print('Loss is already too small, stopping IRL training')
                return


def main(arguments):
    env = instantiate(arguments.env)

    # Load expert trajectories
    expert_replay = StateActionReplay(100000, env.state_shape)
    print('Loading trajectories from {0}'.format(arguments.expert_replay))
    expert_replay.load(arguments.expert_replay)

    log_path = arguments.model_path + '/logs'

    with tf.device(arguments.device):
        model = instantiate(arguments.model, input_shape=env.state_shape, outputs=env.num_actions)
        strategy = instantiate(arguments.rl, env, model)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model.load_if_exists(sess, arguments.model_path)

            writer = tf.summary.FileWriter(log_path, graph=sess.graph)
            model.set_writer(writer)

            for i in range(arguments.train):
                if arguments.irl_epochs > 0:
                    train_irl(sess, model, expert_replay, epochs=arguments.irl_epochs)
                    model.save(sess, arguments.model_path)
                    writer.flush()

                if arguments.rl_epochs > 0:
                    strategy.run(sess, num_episodes=arguments.rl_epochs, verbose=True)
                    model.save(sess, arguments.model_path)
                    writer.flush()

            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument("--env", type=str, help="Path to environment configuration JSON file")
    parser.add_argument("--model", type=str, help="Path to model configuration JSON file")
    parser.add_argument("--rl", type=str, help="Path to RL strategy JSON file")

    # Expert data
    parser.add_argument("--expert_replay", type=str, help="Path to expert replay")

    # Meta parameters
    parser.add_argument("--train", type=int, help="Run training both IRL and RL parts N times", default=100)
    parser.add_argument("--irl_epochs", type=int, help="IRL training epochs", default=1000)
    parser.add_argument("--rl_epochs", type=int, help="RL training epochs", default=1000)
    parser.add_argument("--model_path", type=str, help="Path to save model to", default='/netscratch/shishov/eye_irl')
    parser.add_argument("--device", type=str, help="Device to use", default='/device:GPU:0')

    main(parser.parse_args())

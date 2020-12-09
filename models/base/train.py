''' Train operations'''
import os, sys, yaml

import tensorflow as tf
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
from utils.general import get_logger, Progbar, export_plot

# todo: add schedulers
def train(model, stim_train, stim_test, nsteps):
    """
    """

    t = 0  # time control of nb of steps
    loss_train = []
    loss_test = []

    prog = Progbar(target=nsteps)

    # interact with environment
    while t < nsteps:
        t += 1

        # perform a training step
        train_data = utils_train.tensorize_trial(stim_train.generate_trial(), dtype=model.dtype)
        loss_struct, grad = model.update_step(train_data)
        loss_train += [loss_struct['loss']]

        if model.trainable_variables is None:
            print('No trainable variables. Done training! ')
            break

        # evaluate loss on test set
        test_data = utils_train.tensorize_trial(stim_test.generate_trial(), dtype=model.dtype)
        test_lossStruct, test_outputs = model.evaluate(test_data)
        loss_test += [test_lossStruct['loss']]

        # todo: output behavior summary
        # test_outputs['dec_output']
        # test_outputs['est_output']

        # ground_truth, estim_mean, raw_error, beh_perf = utils_analysis.behavior_summary(test_data, test_logits,
        #                                                                                 stim_test)
        # errors_circular = np.arctan2(np.sum(np.sin(2 * raw_error)), np.sum(np.cos(2 * raw_error))) / 2
        # errors          = np.arctan2(np.sin(2 * raw_error), np.cos(2 * raw_error)) / 2
        # # note that this error is not circular mean.

        # # add losses to tensorboard
        # with self.writer.as_default():
        #     tf.summary.scalar('Test average error', np.mean(np.abs(errors)), step=t)
        #     for (name, val) in loss_struct.items():
        #         tf.summary.scalar('train ' + name, val, step=t)
        #     for (name, val) in test_lossStruct.items():
        #         tf.summary.scalar('test ' + name, val, step=t)

        # logging stuff
        prog.update(t, exact=[("Train Loss", loss_struct['loss']),
                              ("Test Loss", test_lossStruct['loss']),
                              ("Grads", grad)])

        if (t % model.hp['saving_freq']) == 1:
            template = '\n Epoch {}, Train Loss: {}, Test Loss: {}'
            model.logger.info(template.format(t + 1,
                                              loss_struct['loss'],
                                              test_lossStruct['loss']))

            save_model(model,t, stim_test)  # Save Model

    # last words
    model.logger.info("- Training done.")
    save_model(model,'final', stim_test)  # save and output behavior
    export_plot(loss_train, loss_test, 'loss', model.hp['output_path'] + os.path.sep + 'losses')

    # output graph on tensorflow using trace (need to enable tf.function
    # tf.summary.trace_on(graph=True, profiler=False)
    # test_data       = utils_train.tensorize_trial(stim_test.generate_trial(), dtype = self.dtype)
    # test_input      = test_data['input_tuning']
    # test_Y          = self.__call__(test_input)
    # with self.writer.as_default():
    #     tf.summary.trace_export(name="loss",step=t)


def save_model(model, t, stim_test):
    filename = model.hp['model_output'] + os.path.sep + 'iter' + str(t)
    tf.saved_model.save(model, filename)
    # self.plot_summary(stim_test, filename=filename)

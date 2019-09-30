"""
This module implements the graph convolutional neural network.

Most of the code is based on https://github.com/mdeff/cnn_graph/.
"""
from __future__ import division

import os
import time
import collections
import shutil
from builtins import range

import numpy as np
from scipy import sparse
import sklearn
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.losses.python.metric_learning import triplet_semihard_loss
from tensorflow.nn import l2_normalize

from . import utils


# Python 2 compatibility.
if hasattr(time, 'process_time'):
    process_time = time.process_time
    perf_counter = time.perf_counter
else:
    import warnings
    warnings.warn('The CPU time is not working with Python 2.')
    def process_time():
        return np.nan
    def perf_counter():
        return np.nan

# def show_all_variables():
#     import tensorflow as tf
#     import tensorflow.contrib.slim as slim
#     model_vars = tf.trainable_variables()
#     slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# This class is necessary for the dataset
class LoadableGenerator(object):
    def __init__(self):
        self.curr = None
        self.it = None

    def iter(self):
        return self.__iter__()
    def __iter__(self):
        self.update()
        while self.curr:
            yield self.curr
            self.update()        # Isn't it better with this? Or i just don't understand this part of the code
    def load(self, it):
        self.it = it
    def update(self):
        if self.it:
            try:
                self.curr = next(self.it)
            except StopIteration:
                self.curr = None


                
class base_model(object):
    """Common methods for all models."""

    def __init__(self):
        self.regularizers = []
        self.regularizers_size = []

    # High-level interface which runs the constructed computational graph.
    
    def get_descriptor(self, data, sess=None, cache=True):
        sess = self._get_session(sess)
        if cache:
            size = data.N
        else:
            size = data.shape[0]
        descriptors = np.empty((size, self.op_descriptor.shape[-2], self.op_descriptor.shape[-1]))
        if cache:
            if cache is 'TF':
                dataset = data.get_tf_dataset(self.batch_size)
                data_iter = dataset.make_one_shot_iterator()
            else:
                data_iter = data.iter(self.batch_size)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            if cache:
                if cache is 'TF':
                    batch_data, _ = data_iter.get_next()
                else:
                    batch_data, _ = next(data_iter)
            else:
                batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
                tmp_data = data[begin:end, :, :]
                if type(tmp_data) is not np.ndarray:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                batch_data[:end-begin] = tmp_data
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()
            feed_dict = {self.ph_data: batch_data, self.ph_training: False}    
            batch_des = sess.run(self.op_descriptor, feed_dict)
            
            descriptors[begin:end] = batch_des[:end-begin]
            
        return descriptors

    def predict(self, data, labels=None, sess=None, cache=False):
        loss = 0
        if cache:
            size = data.N
        else:
            size = data.shape[0]
        if self.dense:
            M0 = self.L[0].shape[0]
            predictions = np.empty((size, M0))
            label = np.empty((size, M0))
        else:
            predictions = np.empty(size)
            label = np.empty(size)
        sess = self._get_session(sess)
        if cache:
            if cache is 'TF':
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess2 = tf.Session(config=config)
                dataset = data.get_tf_dataset(self.batch_size)
                data_iter = dataset.make_one_shot_iterator().get_next()
            else:
                data_iter = data.iter(self.batch_size)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            if cache:
                if cache is 'TF':
                    batch_data, batch_labels = sess2.run(data_iter)
                    label[begin:end] = np.asarray(batch_labels)[:end-begin]
                else:
                    batch_data, batch_labels = next(data_iter)
                if type(batch_data) is not np.ndarray:
                    batch_data = batch_data.toarray()  # convert sparse matrices
                feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_training: False}
            else:
                batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
                tmp_data = data[begin:end, :, :]
                if type(tmp_data) is not np.ndarray:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                batch_data[:end-begin] = tmp_data
                feed_dict = {self.ph_data: batch_data, self.ph_training: False}

            # Compute loss if labels are given.
            if labels is not None:
                if self.dense:
                    batch_labels = np.zeros((self.batch_size, M0))
                else:
                    batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            elif cache and batch_labels is not None:
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
               
            predictions[begin:end] = batch_pred[:end-begin]

        if labels is not None or (cache and batch_labels is not None):
            if cache is 'TF':
                del sess2
                return predictions, label, loss * self.batch_size / size
            else:
                return predictions, loss * self.batch_size / size
        else:
            return predictions
        
    def probs(self, data, nb_class, labels=None, sess=None, cache=False):
        loss = 0
        if cache:
            size = data.N
        else:
            size = data.shape[0]
        if self.dense:
            M0 = self.L[0].shape[0]
            while True:
                try:
                    probabilities = np.empty((size, M0, nb_class))
                    label = np.empty((size, M0))
                    break
                except Exception as e:
                    print(e)
                    size = size//2
                    if size<1:
                        raise e
        else:
            probabilities = np.empty((size, nb_class))
            label = np.empty(size)
        sess = self._get_session(sess)
        if cache:
            if cache is 'TF':
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess2 = tf.Session(config=config)
                dataset = data.get_tf_dataset(self.batch_size)
                data_iter = dataset.make_one_shot_iterator().get_next()
            else:
                data_iter = data.iter(self.batch_size)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            if cache:
                if cache is 'TF':
                    batch_data, batch_labels = sess2.run(data_iter)
                    label[begin:end] = np.asarray(batch_labels)[:end-begin]
                else:
                    batch_data, batch_labels = next(data_iter)
                if type(batch_data) is not np.ndarray:
                    batch_data = batch_data.toarray()  # convert sparse matrices
                feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_training: False}
            else:
                batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
                tmp_data = data[begin:end, :, :]
                if type(tmp_data) is not np.ndarray:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                batch_data[:end-begin] = tmp_data
                feed_dict = {self.ph_data: batch_data, self.ph_training: False}

            # Compute loss if labels are given.
            if labels is not None:
                if self.dense:
                    batch_labels = np.zeros((self.batch_size, M0))
                else:
                    batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_prob, batch_loss = sess.run([self.op_probabilities, self.op_loss], feed_dict)
                loss += batch_loss
            elif cache and batch_labels is not None:
                batch_prob, batch_loss = sess.run([self.op_probabilities, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_prob = sess.run(self.op_probabilities, feed_dict)

            probabilities[begin:end] = batch_prob[:end-begin]

        if labels is not None or (cache and batch_labels is not None):
            if cache is 'TF':
                del sess2
                return probabilities, label, loss * self.batch_size / size
            else:
                return probabilities, loss * self.batch_size / size
        else:
            return probabilities

    def evaluate_TF(self, data, labels=None, sess=None, cache='TF'):
        loss = 0
        accu = np.zeros(3)
        aps = np.zeros(3)
        sess = self._get_session(sess)
        def accuracy(pred_cls, true_cls, nclass=3):
            accu = []
            tot_int = 0
            tot_cl = 0
            for i in range(3):
                intersect = np.sum(((pred_cls == i) * (true_cls == i)))
                thiscls = np.sum(true_cls == i)
                accu.append(intersect / thiscls * 100)
#                 tot_int += intersect
#                 tot_cl += thiscls
            return np.array(accu)#, np.mean(accu) #tot_int/tot_cl * 100
        if cache:
            size = data.N
        else:
            size = data.shape[0]

        if cache:
            if cache is 'TF':
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess2 = tf.Session(config=config)
                dataset = data.get_tf_dataset(self.batch_size)
                data_iter = dataset.make_one_shot_iterator().get_next()
            else:
                data_iter = data.iter(self.batch_size)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            if cache:
                if cache is 'TF':
                    batch_data, batch_labels = sess2.run(data_iter)
                else:
                    batch_data, batch_labels = next(data_iter)
                if type(batch_data) is not np.ndarray:
                    batch_data = batch_data.toarray()  # convert sparse matrices
                feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_training: False}
            else:
                batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
                tmp_data = data[begin:end, :, :]
                if type(tmp_data) is not np.ndarray:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                batch_data[:end-begin] = tmp_data
                feed_dict = {self.ph_data: batch_data, self.ph_training: False}

            if labels is not None:
                feed_dict[self.ph_labels] = batch_labels
                batch_prob, batch_loss = sess.run([self.op_probabilities, self.op_loss], feed_dict)
                loss += batch_loss
            elif cache and batch_labels is not None:
                batch_prob, batch_loss = sess.run([self.op_probabilities, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_prob = sess.run(self.op_probabilities, feed_dict)
            batch_pred = np.argmax(batch_prob, axis=-1)

            batch_labels = batch_labels.flatten()
            true = sklearn.preprocessing.label_binarize(batch_labels, classes=[0, 1, 2])
            batch_prob = batch_prob.reshape(-1, 3)
            AP = sklearn.metrics.average_precision_score(true, batch_prob, None)
            batch_pred = batch_pred.flatten()
#             ncorrects = sum(predictions == labels)
            class_acc = accuracy(batch_pred, batch_labels)
            accu += class_acc
            aps += AP

        if labels is not None or (cache and batch_labels is not None):
            if cache is 'TF':
                del sess2
                return aps/size, accu/size, loss * self.batch_size / size
            else:
                return aps/size, accus/size, loss * self.batch_size / size
        else:
            return aps/size, accu/size


    def evaluate(self, data, labels, sess=None, cache=False):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_cpu, t_wall = process_time(), time.time()
        if cache is 'TF':
            if self.dense:
                beg = perf_counter()
                probabilities, labels, loss = self.probs(data, 3, labels, sess, cache=cache)
                print("probs time: ", perf_counter()-beg)
                predictions = np.argmax(probabilities, axis=-1)
            else:
                predictions, labels, loss = self.predict(data, labels, sess, cache=cache)
        elif self.dense:
            probabilities, loss = self.probs(data, 3, labels, sess, cache=cache)
            predictions = np.argmax(probabilities, axis=-1)
            if cache:
                labels = data.get_labels()
        else:
            predictions, loss = self.predict(data, labels, sess, cache=cache)
            if cache:
                labels = data.get_labels()
        if hasattr(self, 'val_mask'):
            predictions = predictions * data[..., -2]
            labels = labels * data[..., -1]
        if self.regression:
            exp_var = sklearn.metrics.explained_variance_score(labels, predictions)
            r2 = sklearn.metrics.r2_score(labels, predictions)
            mae = sklearn.metrics.mean_absolute_error(labels, predictions)
            string = 'explained variance: {:.4f}, r2: {:.4f}, loss (MSE): {:.3e}, loss (MAE): {:.3e}'.format(
                exp_var, r2, loss, mae)
            accuracy = exp_var
            f1 = r2
            # labels, predictions = sklearn.utils.check_array(labels, predictions)
            mre = np.mean(np.abs((labels - predictions) /np.clip(labels, 1, None))) * 100
            metrics = mae, mre
        elif self.dense:
            def accuracy(pred_cls, true_cls, nclass=3):
                accu = []
                tot_int = 0
                tot_cl = 0
                for i in range(3):
                    intersect = np.sum(((pred_cls == i) * (true_cls == i)))
                    thiscls = np.sum(true_cls == i)
                    accu.append(intersect / thiscls * 100)
#                     tot_int += intersect
#                     tot_cl += thiscls
                return np.array(accu), np.mean(accu) #tot_int/tot_cl * 100
            labels = labels.flatten()
            true = sklearn.preprocessing.label_binarize(labels, classes=[0, 1, 2])
            probabilities = probabilities.reshape(-1, 3)
            AP = sklearn.metrics.average_precision_score(true, probabilities, None)
            print("AP time: ", perf_counter()-beg)
#             AP = [0, 0, 0]
#             predictions = predictions.flatten()
#             ncorrects = sum(predictions == labels)
#             class_acc, accuracy = accuracy(predictions, labels)
            ncorrects = 0
            print("acc time: ", perf_counter()-beg)
            class_acc, accuracy = [0,0,0],0
#             f1 = sklearn.metrics.f1_score(labels, predictions, average=None)
            print("f1 time: ", perf_counter()-beg)
            f1 = [0, 0, 0]
            string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (TC): {:.2f}, f1 (AR): {:.2f}, loss: {:.2e}'.format(
                    accuracy, ncorrects, len(labels), 100*f1[1], 100*f1[2], loss)
            metrics = AP, class_acc
        else:
            metrics = None
            labels = labels.flatten()
            predictions = predictions.flatten()
            ncorrects = sum(predictions == labels)
            accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
            f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
            string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                    accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
            string += '\nCPU time: {:.0f}s, wall time: {:.0f}s'.format(process_time()-t_cpu, time.time()-t_wall)
        return string, accuracy, f1, loss, metrics

    def fit(self, train_dataset, val_dataset, use_tf_dataset=False, restore=True, verbose=True, cache=False):
        
        # Load the dataset
#         if use_tf_dataset:
#             self.loadable_generator.load(train_dataset.iter(self.batch_size))

        total_acc = 0
        acc = 0
        mre = 0
        
        if not verbose:
            tf.logging.set_verbosity(tf.logging.WARN)
        else:
            tf.logging.set_verbosity(tf.logging.DEBUG)
        
        t_cpu, t_wall = process_time(), time.time()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=self.graph, config=config)
        if restore:
            try:
                filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
                self.op_saver.restore(sess, filename)
                start_step = int(filename.split('-')[-1])+1
                print("training from last checkpoint")
            except ValueError:
                start_step=1
                shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
                shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
                os.makedirs(self._get_path('checkpoints'))
                restore = False
                print("training from scratch")
        else:
            print("training from scratch")
            start_step=1
            shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
            shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
            os.makedirs(self._get_path('checkpoints'))
        if self.debug:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        path = os.path.join(self._get_path('checkpoints'), 'model')
        
        # Initialization
        if not restore:
            sess.run(self.op_metrics_init)
            sess.run(self.op_init)
#         print("after init")
        # Training.
        accuracies_validation = []
        losses_validation = []
        losses_training = []
        #num_steps = int(self.num_epochs * np.ceil(train_dataset.N / self.batch_size))
        num_steps = int(self.num_epochs * train_dataset.N / self.batch_size)
        if not use_tf_dataset:
            train_iter = train_dataset.iter(self.batch_size)
        else:
            pass
            #sess.run(self.tf_data_iterator.initializer)

        if not cache:
            val_data, val_labels = val_dataset.get_all_data()
            if len(val_data.shape) is 2:
                val_data = np.expand_dims(val_data, axis=2)
                
#         print("begin loop, eval freq = ", self.eval_frequency)
        times = []
        for step in range(start_step, num_steps+1):
            t_begin = perf_counter()
            if not use_tf_dataset:
                batch_data, batch_labels = next(train_iter)
                if type(batch_data) is not np.ndarray:
                    batch_data = batch_data.toarray()  # convert sparse matrices
                if len(batch_data.shape) is 2:
                    batch_data = np.expand_dims(batch_data, axis=2)
                feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_training: True}
            else:
                feed_dict = {self.ph_training: True}

#             learning_rate, loss = sess.run([self.op_train, self.op_loss], feed_dict)
            evaluate = (step % self.eval_frequency == 0) or (step == num_steps)
            if evaluate and self.profile:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None
            
#             if step%10==0:
#                 print('{}/{}'.format(step, num_steps))
            
            t_begin_load = perf_counter()
            
            if self.regression:
                batch_acc = np.nan
                learning_rate, loss, batch_mre = sess.run([self.op_train, self.op_loss, self.tf_mre_update], 
                                                          feed_dict, run_options, run_metadata)
                if step%(train_dataset.N//self.batch_size)==0:
                    mre = sess.run(self.tf_mre)
                    sess.run(self.op_metrics_init)
                pass
            else:
                learning_rate, loss = sess.run([self.op_train, self.op_loss], # , self.tf_accuracy_update], 
                                                          feed_dict, run_options, run_metadata)
                if step%(train_dataset.N//self.batch_size)==0:
                    pass
#                     acc = sess.run(self.tf_accuracy)
#                     sess.run(self.op_metrics_init)
                
            t_end = perf_counter()
            times.append(t_end-t_begin)
            # Periodical evaluation of the model.
            if evaluate:
                # Change evaluation in case of augmentation, maybe? In order to get a more accurate response of the model (in evaluation, same as Cohen) But need change of datastructure
                epoch = step * self.batch_size / train_dataset.N
                if verbose:
                    print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                    if self.regression:
                        print('  learning_rate = {:.2e}, training mean relative error = {:.2e},' \
                              ' training loss = {:.2e}'.format(learning_rate, batch_mre, loss))
                    else:
                        print('  learning_rate = {:.2e}, training mAP = {:.2f}, training loss = {:.2e}'.format(learning_rate, 
                                                                                                                    acc, loss))
                losses_training.append(loss)
                if self.regression:
                    string, exp_var, r2, loss, (mae, mre_val) = self.evaluate(val_data, val_labels, sess, cache=cache)
                else:
                    if cache:
#                         pass
#                         string, accuracy, f1, loss, metrics = "", 0, [0, 0, 0], 0., ([0,0,0],[0,0,0])
                        string, accuracy, f1, loss, metrics = self.evaluate(val_dataset, None, sess, cache=cache)
                    else:
                        string, accuracy, f1, loss, metrics = self.evaluate(val_data, val_labels, sess)
                    if self.dense:
                        AP, class_acc = metrics
                    accuracies_validation.append(accuracy)
                losses_validation.append(loss)
                if verbose:
                    print('  validation {}'.format(string))
                    print('  CPU time: {:.0f}s, wall time: {:.0f}s, perf_time_load: {:.3f}s, perf_time: {:.3f}s'.format(process_time()-t_cpu, time.time()-t_wall, t_end-t_begin_load, times[-1]))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                if self.regression:
                    summary.value.add(tag='validation/exp_variance', simple_value=exp_var)
                    summary.value.add(tag='validation/r2', simple_value=r2)
                    summary.value.add(tag='validation/loss', simple_value=loss)
                    summary.value.add(tag='validation/mae', simple_value=mae)
                    summary.value.add(tag='validation/mre', simple_value=mre_val)
                    summary.value.add(tag='training/mre', simple_value=mre)
                else:
                    if self.dense:
#                         summary.value.add(tag='training/epoch_map', simple_value=acc)
                        summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                        summary.value.add(tag='validation/accuracy_BG', simple_value=class_acc[0])
                        summary.value.add(tag='validation/accuracy_TC', simple_value=class_acc[1])
                        summary.value.add(tag='validation/accuracy_AR', simple_value=class_acc[2])
                        summary.value.add(tag='validation/f1_BG', simple_value=f1[0])
                        summary.value.add(tag='validation/f1_TC', simple_value=f1[1])
                        summary.value.add(tag='validation/f1_AR', simple_value=f1[2])
                        summary.value.add(tag='validation/AP_BG', simple_value=AP[0])
                        summary.value.add(tag='validation/AP_TC', simple_value=AP[1])
                        summary.value.add(tag='validation/AP_AR', simple_value=AP[2])
                    else:
                        summary.value.add(tag='training/epoch_accuracy', simple_value=acc)
                        summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                        summary.value.add(tag='validation/f1', simple_value=f1)
                    summary.value.add(tag='validation/loss', simple_value=loss)
#                     summary.value.add(tag='training/batch_accuracy', simple_value=batch_acc)
                writer.add_summary(summary, step)
                if self.profile:
                    writer.add_run_metadata(run_metadata, 'step{}'.format(step))

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)
                # save if best in ckpt form

        if verbose and not self.regression:
            print('validation accuracy: best = {:.2f}, mean = {:.2f}'.format(max(accuracies_validation), np.mean(accuracies_validation[-10:])))
        writer.close()
        sess.close()
        if verbose:
            print('time per batch: mean = {:.5f}, var = {:.5f}'.format(np.mean(times), np.var(times))) 
        t_step = (time.time() - t_wall) / num_steps
        return accuracies_validation, losses_validation, losses_training, t_step, np.mean(times)

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val
    
    def get_nbr_var(self):
        #print(self.graph.get_collection('trainable_variables'))
        total_parameters = 0
        for variable in self.graph.get_collection('trainable_variables'):
            # print(variable.name)
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            #print(shape)
            #print(len(shape))
            variable_parameters = 1
            for dim in shape:
                #print(dim)
                variable_parameters *= dim.value
            print(variable.name, variable_parameters)
            total_parameters += variable_parameters
        #print(total_parameters)
        return total_parameters

    # Methods to construct the computational graph.

    def build_graph(self, M_0, nfeature=1, tf_dataset=None, regression=False, dtype=tf.float32):
        """Build the computational graph of the model."""

        gstart = time.time()
        self.loadable_generator = LoadableGenerator()
        try:
            if not self.restore:
                raise ValueError
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.startstep = int(filename.split('-')[-1])
        except:
            self.startstep = 0

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Make the dataset
            self.tf_train_dataset = tf.data.Dataset.from_generator(self.loadable_generator.iter, 
                                                                     output_types=(dtype, (dtype if regression else tf.int32)))
            self.tf_data_iterator = self.tf_train_dataset.prefetch(2).make_initializable_iterator()
            if tf_dataset is not None:
                self.tf_data_iterator = tf_dataset.make_one_shot_iterator()
            ph_data, ph_labels = self.tf_data_iterator.get_next()
            print("data iterator inst., time: ", time.time()-gstart) 


            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder_with_default(ph_data, (self.batch_size, M_0, nfeature), 'data')
                if self.dense:
                    self.ph_labels = tf.placeholder_with_default(ph_labels, (self.batch_size, M_0), 'labels')
                else:
                    self.ph_labels = tf.placeholder_with_default(ph_labels, (self.batch_size), 'labels')
                self.ph_training = tf.placeholder(tf.bool, (), 'training')

            print("inputs, time: ", time.time()-gstart)
            # Model.
            op_data = self.ph_data
            op_logits, self.op_descriptor = self.inference(op_data, self.ph_training)
            print("inference done, time: ", time.time()-gstart)
            if self.dense and (np.asarray(self.p)>1).any():
                op_logits = self.upward(op_logits, self.ph_training)
                print("decoder done, time: ", time.time()-gstart)
            self.op_loss = self.loss(op_logits, self.ph_labels, self.regularization, op_data, extra_loss=self.extra_loss, regression=regression)
            print("loss done, time: ", time.time()-gstart)
            self.op_train = self.training(self.op_loss)
            print("training done, time: ", time.time()-gstart)
            self.op_prediction = self.prediction(op_logits)
            self.op_probabilities = self.probabilities(op_logits)
            self.op_labels = self.ph_labels
            print("op end done, time: ", time.time()-gstart)
            
            # Metrics
            with tf.name_scope('metrics'):
                if regression:
                    self.tf_mre, self.tf_mre_update = tf.metrics.mean_relative_error(self.ph_labels, self.op_prediction, 
                                                             tf.abs(tf.clip_by_value(self.ph_labels, 1, np.inf)), name='metrics_mre')
                    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics_mre")
                elif self.dense:
                    self.tf_accuracy, self.tf_accuracy_update = tf.metrics.average_precision_at_k(tf.to_int64(self.ph_labels), 
                                                                                      op_logits, 
                                                                                      3,
                                                                                      name='metrics_map')
                else:
                    self.tf_accuracy, self.tf_accuracy_update = tf.metrics.accuracy(self.ph_labels, self.op_prediction,
                                                                                    name='metrics_acc')
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")

            print("metrics done, time: ", time.time()-gstart)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            self.op_metrics_init = tf.local_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()
        print("all done, time: ", time.time()-gstart)

    def inference(self, data, training):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout and
            batch normalization.
            True: the model is run for training.
            False: the model is run for evaluation.
        """
        # TODO: optimizations for sparse data
        logits, descriptors = self._inference(data, training)
        return logits, descriptors
    
    def upward(self, logits, training):
        seg_map = self._decoder(logits, training)
        return seg_map

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            if self.regression:
                prediction = tf.squeeze(logits)
            else:
                prediction = tf.argmax(logits, axis=-1)
            return prediction
        
    ## add triplet_loss
    def triplet_loss(self, descriptor, labels):
        with tf.name_scope('triplet_loss'):
            norm = l2_normalize(descriptor, axis=-1)
            triplet_loss = triplet_semihard_loss(labels, norm)
            triplet_loss = tf.where(tf.is_nan(triplet_loss),
                            tf.zeros_like(triplet_loss),
                            triplet_loss)

        tf.summary.scalar('loss/triplet_loss', triplet_loss)
        return triplet_loss

    def loss(self, logits, labels, regularization, data ,extra_loss=False, regression=False):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            if regression:
                with tf.name_scope('MSE'):
                    predictions = tf.squeeze(logits)
#                     if self.M:
#                         labels = tf.expand_dims(labels, axis=-1)
                    if hasattr(self, 'train_mask'):
                        predictions = predictions * data[..., -2]
                        labels = labels * data[..., -1]
                    mse = tf.losses.mean_squared_error(labels, predictions)
                    loss = mse
            else:
                with tf.name_scope('cross_entropy'):
                    labels = tf.to_int64(labels)
                    labels_onehot = tf.one_hot(labels, 3)
#                     weights = tf.constant([[0.00102182, 0.95426438, 0.04471379]])
                    if self.weighted:
                        weights = tf.constant([[0.34130685, 318.47388343,  14.93759951]])
                        batch_weights = tf.reshape(tf.matmul(tf.reshape(labels_onehot, [-1,3]), tf.transpose(weights)), 
                                                   [self.batch_size, self.L[0].shape[0]])
#                     batch_weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                    if self.weighted:
                        cross_entropy = tf.multiply(batch_weights, cross_entropy)  
#                     cross_entropy = tf.reduce_sum(cross_entropy*batch_weights) / self.batch_size
                    cross_entropy = tf.reduce_mean(cross_entropy)
                    loss = cross_entropy
            with tf.name_scope('regularization'):
                n_weights = np.sum(self.regularizers_size)
                regularization *= tf.add_n(self.regularizers) / n_weights
            loss = loss + regularization
            if extra_loss:
                loss += self.triplet_loss(self.op_descriptor, labels)

            # Summaries for TensorBoard.
            if regression:
                tf.summary.scalar('loss/mse', mse)
            else:
                tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            return loss

    def training(self, loss):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(self.startstep, name='global_step', trainable=False)
            learning_rate = self.scheduler(global_step)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            optimizer = self.optimizer(learning_rate)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # Add control dependencies to compute gradients and moving averages (batch normalization).
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([op_gradients] + update_ops):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(graph=self.graph, config=config)
            #print(self._get_path('checkpoints'))
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, stddev=0.1, regularization=True):
        initial = tf.truncated_normal_initializer(0, stddev=stddev)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var) / stddev**2)
            self.regularizers_size.append(np.prod(shape))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=False):
        initial = tf.constant_initializer(0)
        # initial = tf.truncated_normal_initializer(0, stddev=1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            self.regularizers_size.append(np.prod(shape))
        tf.summary.histogram(var.op.name, var)
        return var


class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        batch_norm: apply batch normalization after filtering (boolean vector)
        L: List of Graph Laplacians. Size M x M.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        pool: pooling, e.g. max or average.
        activation: non-linearity, e.g. relu, elu, leaky_relu.
        statistics: layer which computes statistics from feature maps for the network to be invariant to translation and rotation.
            * None: no statistical layer (default)
            * 'mean': compute the mean of each feature map
            * 'var': compute the variance of each feature map
            * 'meanvar': compute the mean and variance of each feature map
            * 'histogram': compute a learned histogram of each feature map

    Training parameters:
        num_epochs:     Number of training epochs.
        scheduler:      Learning rate scheduler: function that takes the current step and returns the learning rate.
        optimizer:      Function that takes the learning rate and returns a TF optimizer.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
        profile:        Whether to profile compute time and memory usage. Needs libcupti in LD_LIBRARY_PATH.
        debug:          Whether the model should be debugged via Tensorboard.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, L, F, K, p, batch_norm, M,
                num_epochs, scheduler, optimizer, num_feat_in=1, tf_dataset=None, restore=False,
                conv='chebyshev5', pool='max', activation='relu', statistics=None, Fseg=None, dtype=tf.float32,
                regularization=0, dropout=1, batch_size=128, eval_frequency=200, regression=False, dense=False,
                weighted=False, mask=None, extra_loss=False, dropFilt=1, dir_name='', profile=False, debug=False):
        super(cgcnn, self).__init__()

        # Verify the consistency w.r.t. the number of layers.
        if not len(L) == len(F) == len(K) == len(p) == len(batch_norm):
            print(len(L), len(F), len(K), len(p), len(batch_norm))
            raise ValueError('Wrong specification of the convolutional layers: '
                             'parameters L, F, K, p, batch_norm, must have the same length.')
        if not np.all(np.array(p) >= 1):
            raise ValueError('Down-sampling factors p should be greater or equal to one.')
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        if not np.all(np.mod(p_log2, 1) == 0) and self.sampling != 'icosahedron':
            raise ValueError('Down-sampling factors p should be powers of two.')
        if len(M) == 0 and p[-1] != 1 and self.sampling != 'icosahedron' and not dense:
            raise ValueError('Down-sampling should not be used in the last '
                             'layer if no fully connected layer follows.')
        if mask and not isinstance(mask, list):
            raise ValueError('Must provide a list of mask for training and validation.')

        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]     # Laplacian size is npix x npix
        j = 0
        self.L = L

        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        M_last = M_0
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else num_feat_in
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if not (i == Ngconv-1 and len(M) == 0):  # No bias if it's a softmax.
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            if batch_norm[i]:
                print('    batch normalization')

        if Ngconv:
            M_last = L[-1].shape[0] * F[-1] // p[-1]

        if statistics is not None:
            print('  Statistical layer: {}'.format(statistics))
            if statistics is 'mean':
                M_last = F[-1]
                print('    representation: 1 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'var':
                M_last = F[-1]
                print('    representation: 1 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'meanvar':
                M_last = 2 * F[-1]
                print('    representation: 2 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'histogram':
                nbins = 20
                M_last = nbins * F[-1]
                print('    representation: {} * {} = {}'.format(nbins, F[-1], M_last))
                print('    weights: {} * {} = {}'.format(nbins, F[-1], M_last))
                print('    biases: {} * {} = {}'.format(nbins, F[-1], M_last))

        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            if i < Nfc - 1:  # No bias if it's a softmax.
                print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i]
        
        ## TODO: rewrite this before when using dense predictions

        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.Fseg = Fseg
        self.num_epochs = num_epochs
        self.scheduler, self.optimizer = scheduler, optimizer
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.batch_norm = batch_norm
        self.dir_name = dir_name
        self.filter = getattr(self, conv)
        self.pool = getattr(self, 'pool_' + pool)
        self.unpool = getattr(self, 'unpool_' + pool)
        self.activation = getattr(tf.nn, activation)
        self.statistics = statistics
        self.profile, self.debug = profile, debug
        self.dropFilt = dropFilt
        self.extra_loss = extra_loss
        self.regression = regression
        self.dense = dense
        self.weighted = weighted
        self.dtype = dtype
        self.restore = restore
        if mask:
            self.train_mask = mask[0]
            self.val_mask = mask[1]

        # Build the computational graph.
        self.build_graph(M_0, num_feat_in, tf_dataset, regression, dtype=dtype)

        # show_all_variables()

    def chebyshev5(self, x, L, Fout, K, training=False):
        fstart = time.time()
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        L = tf.cast(L, self.dtype)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        W = self._weight_variable_cheby(K, Fin, Fout, regularization=True)
        # Drop filters of the convolutional layer
        if self.dropFilt<1:
            W = tf.reshape(W, [Fin, K, Fout])
            W = tf.transpose(W, perm=[1,0,2])
            mask = tf.ones((Fin, Fout))
            dropout = tf.cond(training, lambda: float(self.dropFilt), lambda: 1.0)
            mask = tf.nn.dropout(mask, dropout) * (dropout)
            W = tf.multiply(W, mask)
            W = tf.transpose(W, perm=[1,0,2])
            W = tf.reshape(W, [Fin*K, Fout])
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def _weight_variable_cheby(self, K, Fin, Fout, regularization=True):
        """Xavier like weight initializer for Chebychev coefficients."""
        stddev = 1 / np.sqrt(Fin * (K + 0.5) / 2)
        return self._weight_variable([Fin*K, Fout], stddev=stddev, regularization=regularization)

    def monomials(self, x, L, Fout, K, training=False):
        r"""Convolution on graph with monomials."""
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to monomial basis.
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        for k in range(1, K):
            x1 = tf.sparse_tensor_dense_matmul(L, x0)  # M x Fin*N
            x = concat(x, x1)
            x0 = x1
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        W = self._weight_variable([Fin*K, Fout], regularization=True)
        # Drop filters of the convolutional layer
        if self.dropFilt<1:
            W = tf.reshape(W, [Fin, K, Fout])
            W = tf.transpose(W, perm=[1,0,2])
            mask = tf.ones((Fin, Fout))
            dropout = tf.cond(training, lambda: float(self.dropFilt), lambda: 1.0)
            mask = tf.nn.dropout(mask, dropout) * (dropout)
            W = tf.multiply(W, mask)
            W = tf.transpose(W, perm=[1,0,2])
            W = tf.reshape(W, [Fin*K, Fout])
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def bias(self, x):
        """Add one bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return x + b

    def pool_max(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            if self.sampling is 'equiangular':
                N, M, F = x.get_shape()
                N, M, F = int(N), int(M), int(F)
                x = tf.reshape(x,[N,int((M/self.ratio)**0.5), int((M*self.ratio)**0.5), F])
                x = tf.nn.max_pool(x, ksize=[1,p**0.5,p**0.5,1], strides=[1,p**0.5,p**0.5,1], padding='SAME')
                return tf.reshape(x, [N, -1, F])
            elif self.sampling  is 'icosahedron':
                return x[:, :p, :]
            else:
                x = tf.expand_dims(x, 3)  # N x M x F x 1
                x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
                return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def pool_average(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            if self.sampling is 'equiangular':
                N, M, F = x.get_shape()
                N, M, F = int(N), int(M), int(F)
#                 print(M, (M/self.ratio)**0.5, (M*self.ratio)**0.5)
                x = tf.reshape(x,[N,int((M/self.ratio)**0.5), int((M*self.ratio)**0.5), F])
                x = tf.nn.avg_pool(x, ksize=[1,p**0.5,p**0.5,1], strides=[1,p**0.5,p**0.5,1], padding='SAME')
                return tf.reshape(x, [N, -1, F])
            elif self.sampling  is 'icosahedron':
                return x[:, :p, :]
            else:
                x = tf.expand_dims(x, 3)  # N x M x F x 1
                x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
                return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x
    

    def learned_histogram(self, x, bins=20, initial_range=2):
        """A learned histogram layer.

        The center and width of each bin is optimized.
        One histogram is learned per feature map.
        """
        # Shape of x: #samples x #nodes x #features.
        n_features = int(x.get_shape()[2])
        centers = tf.linspace(-float(initial_range), initial_range, bins, name='range')
        centers = tf.expand_dims(centers, axis=1)
        centers = tf.tile(centers, [1, n_features])  # One histogram per feature channel.
        centers = tf.Variable(
            tf.reshape(tf.transpose(centers), shape=[1, 1, n_features, bins]),
            name='centers',
            dtype=tf.float32)
        width = 4 * initial_range / bins  # 50% overlap between bins.
        widths = tf.get_variable(
            name='widths',
            shape=[1, 1, n_features, bins],
            dtype=tf.float32,
            initializer=tf.initializers.constant(value=width, dtype=tf.float32))
        x = tf.expand_dims(x, axis=3)
        # All are rank-4 tensors: samples, nodes, features, bins.
        widths = tf.abs(widths)
        dist = tf.abs(x - centers)
        hist = tf.reduce_mean(tf.nn.relu(1 - dist * widths), axis=1) * (bins/initial_range/4)
        return hist

    def batch_normalization(self, x, training, momentum=0.9):
        """Batch norm layer."""
        # Normalize over all but the last dimension, that is the features.
        return tf.layers.batch_normalization(x,
                                             axis=-1,
                                             momentum=momentum,
                                             epsilon=1e-5,
                                             center=False,  # Done by bias.
                                             scale=False,  # Done by filters.
                                             training=training)

    def fc(self, x, Mout, bias=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable_fc(int(Min), Mout, regularization=True)
        y = tf.matmul(x, W)
        if bias:
            y += self._bias_variable([Mout], regularization=False)
        return y

    def _weight_variable_fc(self, Min, Mout, regularization=True):
        """Xavier like weight initializer for fully connected layer."""
        stddev = 1 / np.sqrt(Min)
        return self._weight_variable([Min, Mout], stddev=stddev, regularization=regularization)

    def _inference(self, x, training):
        infstart = time.time()
        self.pool_layers = []
        self.pool_layers.append(x)
        # Graph convolutional layers.
        # x = tf.expand_dims(x, 2)  # N x M x F=1        # or N x M x F=num_features_in
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
#                 print("conv{}".format(i), x.shape)
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i], training)
#                 self.pool_layers.append(x)
                print("filter{}, time: ".format(i), time.time()-infstart)
                if i == len(self.p)-1 and len(self.M) == 0:
#                     self.pool_layers.append(x)
                    break  # That is a linear layer before the softmax.
                if self.batch_norm[i]:
                    x = self.batch_normalization(x, training)
                print("bn{}, time: ".format(i), time.time()-infstart)
                x = self.bias(x)
                x = self.activation(x)
#                 self.pool_layers.append(x)
                print("relu{}, time: ".format(i), time.time()-infstart)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])
                print("pooling{}, time: ".format(i), time.time()-infstart)
                self.pool_layers.append(x)

        descriptor = x
        # Statistical layer (provides invariance to translation and rotation).
        with tf.variable_scope('stat'):
            n_samples, n_nodes, n_features = x.get_shape()
            if self.statistics is None:
                if self.M:# or self.regression:
                    x = tf.reshape(x, [int(n_samples), int(n_nodes * n_features)])
                else:
                    pass
            elif self.statistics is 'mean':
                x, _ = tf.nn.moments(x, axes=1)
            elif self.statistics is 'var':
                _, x = tf.nn.moments(x, axes=1)
            elif self.statistics is 'meanvar':
                mean, var = tf.nn.moments(x, axes=1)
                x = tf.concat([mean, var], axis=1)
            elif self.statistics is 'histogram':
                n_bins = 20
                x = self.learned_histogram(x, n_bins)
                x = tf.reshape(x, [int(n_samples), n_bins * int(n_features)])
            elif self.statistics is 'max':
                x = tf.reduce_max(x, axis=1)
            else:
                raise ValueError('Unknown statistical layer {}'.format(self.statistics))
        
#         descriptor = x

        # Fully connected hidden layers.
        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = self.activation(x)
                dropout = tf.cond(training, lambda: float(self.dropout), lambda: 1.0)
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        if len(self.M) != 0:
            with tf.variable_scope('logits'):
                x = self.fc(x, self.M[-1], bias=False)
#         print("end down pass")
        return x, descriptor
    
    def _decoder(self, x, training):
        # transpose filter
        decstart = time.time()
        for i in range(1, 1+len(self.p)):
#             print(self.p[-i])
            if self.p[-i]>1:
                with tf.variable_scope('upconv{}'.format(len(self.p)-i)):
                    with tf.name_scope('pooling'):
                        if self.sampling == 'icosahedron':
                            try:
                                p = self.p[-i-1]
                            except:
                                p = 10 * 4 ** 5 + 2 # self.p[-i]
                        else:
                            p = self.p[-i]
                        x = self.unpool(x, p)
                        print('unpool{}, time: '.format(len(self.p)-i), time.time()-decstart)
                    with tf.name_scope('up-conv'):
                        x = self.filter(x, self.L[-i], self.F[-i], self.K[-i], training)
                        x = self.bias(x)
                        print('upconv{}, time: '.format(len(self.p)-i), time.time()-decstart)
#                 x = gen_nn_ops._max_pool_grad(x, self.pool_layers[-i], self.pool_layers[-i], [1,p,1,1], [1,p,1,1],'SAME')
                    x = tf.concat([x, self.pool_layers[-i]], axis=-1)
                with tf.variable_scope('deconv{}'.format(len(self.p)-i)):
                    with tf.name_scope('filter'):
                        try:
                            x = self.filter(x, self.L[-i], self.F[-i], self.K[-i], training)
                            if self.batch_norm[-i]:
                                x = self.batch_normalization(x, training)
                            x = self.bias(x)
                            x = self.activation(x)
                            print('deconv{}, time: '.format(len(self.p)-i), time.time()-decstart)
                        except:
                            raise ValueError("Down-sampling should not be used in the first layers if training for segmentation task")
            else:
                with tf.variable_scope('deconv{}'.format(len(self.p)-i)):
                    with tf.name_scope('filter'):
                        try:
                            x = self.filter(x, self.L[-i], self.F[-i], self.K[-i], training)
                            if self.batch_norm[-i]:
                                x = self.batch_normalization(x, training)
                            x = self.bias(x)
                            x = self.activation(x)
                            print('deconv{}, time: '.format(len(self.p)-i), time.time()-decstart)
                        except:
                            x = self.filter(x, self.L[-i], self.Fseg, self.K[-i], training)
                            x = self.bias(x)
                            x = self.activation(x)
                            print('deconv{}'.format(len(self.p)-i), x.shape)
        
#         print("end up pass")
        with tf.variable_scope('outconv'):
            x = self.filter(x, self.L[-i], self.Fseg, 1, training)
        print('end decoder, time: ', time.time()-decstart)

        return x
        

    def unpool_average(self, x, p):
        if self.sampling is 'equiangular':
            from tensorflow.keras.backend import repeat_elements
#             raise NotImplementedError('unpooling with equiangular sampling is not yet implemented')
            N, M, F = x.shape
            N, M, F = int(N), int(M), int(F)
            x = tf.reshape(x,[N,int((M/self.ratio)**0.5), int((M*self.ratio)**0.5), F])
            x = repeat_elements(repeat_elements(x, int(p**0.5), axis=1), int(p**0.5), axis=2)
            return tf.reshape(x, [N, -1, F])
        elif self.sampling is 'icosahedron':
            N, M, F = x.shape
            return tf.pad(x, tf.constant([[0,0],[0,int(p-M)],[0,0]]), "CONSTANT", constant_values=1)
#             one_pad = tf.ones((N, p-M, F))
#             return tf.concat([x, one_pad], axis=1)
            #return x[:, :p, :]
        N, M, F = x.shape  # N x M x F
        x = tf.tile(tf.expand_dims(x, 2), [1,1,p,1]) 
        return tf.reshape(x, [N, M*p, F]) # N x M*p x F
    
    def unpool_max(self, x, p):
        if self.sampling is 'equiangular':
            raise NotImplementedError('unpooling max with equiangular sampling is not yet implemented')
        elif self.sampling is 'icosahedron':
            raise NotImplementedError('unpooling max with icosahedron sampling is not yet implemented')
        # TODO: keep in memory the true position of max pool
        N, M, F = x.shape  # N x M x F
        zeros_pad = tf.zeros([N, M, p-1, F])
        x = tf.concat([tf.expand_dims(x, 2), zeros_pad], axis=2)
        return tf.reshape(x, [N, M*p, F]) # N x M*p x F

    def get_filter_coeffs(self, layer, ind_in=None, ind_out=None):
        """Return the Chebyshev filter coefficients of a layer.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        K, Fout = self.K[layer-1], self.F[layer-1]
        trained_weights = self.get_var('conv{}/weights'.format(layer))  # Fin*K x Fout
        trained_weights = trained_weights.reshape((-1, K, Fout))
        if layer >= 2:
            Fin = self.F[layer-2]
            assert trained_weights.shape == (Fin, K, Fout)

        # Fin x K x Fout => K x Fout x Fin
        trained_weights = trained_weights.transpose([1, 2, 0])
        if ind_in:
            trained_weights = trained_weights[:, :, ind_in]
        if ind_out:
            trained_weights = trained_weights[:, ind_out, :]
        return trained_weights

    def plot_chebyshev_coeffs(self, layer, ind_in=None, ind_out=None,  ax=None, title='Chebyshev coefficients - layer {}'):
        """Plot the Chebyshev coefficients of a layer.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        title : figure title
        """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        trained_weights = self.get_filter_coeffs(layer, ind_in, ind_out)
        K, Fout, Fin = trained_weights.shape
        ax.plot(trained_weights.reshape((K, Fin*Fout)), '.')
        ax.set_title(title.format(layer))
        return ax


class deepsphere(cgcnn):
    """
    Spherical convolutional neural network based on graph CNN

    The following are hyper-parameters of the spherical layers.
    They are lists, which length is equal to the number of gconv layers.
        nsides: NSIDE paramter of the healpix package
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        batch_norm: apply batch norm at the end of the filter (bool vector)

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        pool: pooling, e.g. max or average.
        activation: non-linearity, e.g. relu, elu, leaky_relu.
        statistics: layer which computes statistics from feature maps for the network to be invariant to translation and rotation.
            * None: no statistical layer (default)
            * 'mean': compute the mean of each feature map
            * 'var': compute the variance of each feature map
            * 'meanvar': compute the mean and variance of each feature map
            * 'histogram': compute a learned histogram of each feature map

    Training parameters:
        num_epochs:     Number of training epochs.
        scheduler:      Learning rate scheduler: function that takes the current step and returns the learning rate.
        optimizer:      Function that takes the learning rate and returns a TF optimizer.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
        profile:        Whether to profile compute time and memory usage. Needs libcupti in LD_LIBRARY_PATH.
        debug:          Whether the model should be debugged via Tensorboard.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, nsides, indexes=None, use_4=False, sampling='healpix', std=None, full=False, new=True, n_neighbors=8, **kwargs):
        # nsides is bandwidth if sampling is equiangular (SOFT)
        L, p = utils.build_laplacians(nsides, indexes=indexes, use_4=use_4, sampling=sampling, 
                                      std=std, full=full, new=new, n_neighbors=n_neighbors)
        self.sampling = sampling
        if sampling == 'equiangular':
            if isinstance(nsides[0], tuple):
                self.ratio = nsides[0][1]/nsides[0][0]
            else:
                self.ratio = 1.
        self.nsides = nsides
        self.pygsp_graphs = [None] * len(nsides)
        super(deepsphere, self).__init__(L=L, p=p, **kwargs)

    def get_gsp_filters(self, layer,  ind_in=None, ind_out=None):
        """Get the filter as a pygsp format

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        from pygsp import filters

        trained_weights = self.get_filter_coeffs(layer, ind_in, ind_out)
        nside = self.nsides[layer-1]
        if self.pygsp_graphs[layer-1] is None:
            if self.sampling is 'healpix':
                self.pygsp_graphs[layer-1] = utils.healpix_graph(nside=nside)
            elif self.sampling is 'equiangular':
                self.pygsp_graphs[layer-1] = utils.equiangular_graph(bw=nside)
            elif self.sampling is 'icosahedron':
                self.pygsp_graphs[layer-1] = utils.icosahedron_graph(order=nside)
            else:
                raise valueError('Unknown sampling: '+self.sampling)
            self.pygsp_graphs[layer-1].estimate_lmax()
        return filters.Chebyshev(self.pygsp_graphs[layer-1], trained_weights)

    def plot_filters_spectral(self, layer,  ind_in=None, ind_out=None, ax=None, **kwargs):
        """Plot the filter of a special layer in the spectral domain.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        """
        import matplotlib.pyplot as plt

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)

        if ax is None:
            ax = plt.gca()
        filters.plot(sum=False, ax=ax, **kwargs)

        return ax

    def plot_filters_section(self, layer,  ind_in=None, ind_out=None, ax=None, **kwargs):
        """Plot the filter section on the sphere

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        """
        from . import plot

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)
        fig = plot.plot_filters_section(filters, order=self.K[layer-1], **kwargs)
        return fig

    def plot_filters_gnomonic(self, layer,  ind_in=None, ind_out=None, **kwargs):
        """Plot the filter localization on gnomonic view.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        from . import plot

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)
        fig = plot.plot_filters_gnomonic(filters, order=self.K[layer-1], **kwargs)

        return fig
    
class flexPartSphere(cgcnn):
    """
    try to feed random part of sphere
    """
    def __init__(self, **kwargs):
        super(flexPartSphere, self).__init__(**kwargs)
        raise NotImplementedError("pooling random part of sphere not implemented yet")
        
    def pool_any_part_max(self, x, Nside, theta, phi):
        pix, weights = hp.get_interp_weights(Nside, theta, phi, nest=True, lonlat=True)
        indexes = np.unique(pix)
        size = dataset_temp.shape
        size = list(size)
        size[1] = len(indexes) # hp.nside2npix(Nside)
        size = tuple(size)
        new_map = np.zeros(size)
        # new_map[new_map==0] = hp.UNSEEN
        pool_fun = getattr(np, pool)
        for i, index in enumerate(indexes):
            pl = np.where(pix==index)
            wght = 1/(weights[pl]+1e-8)
            wght[wght>1] = 1
            data_p = wght[np.newaxis,:,np.newaxis] * dataset_temp[:, pl[1], :]
            new_map[:,i,:] = pool_fun(data_p, axis=1)
        return new_map
    
    def pool_part_max(self, x, p, Nside, index):
        """Max pooling of size p on partial sphere. Sould be a power of 2."""
        if p > 1:
            full_map = tf.ones([x.shape[0], hp.nside2npix(Nside), x.shape[2]]) * -1e8
            full_map[index] = x
            full_map = tf.expand_dims(full_map, 3)
            full_map = tf.nn.max_pool(full_map, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            x = tf.squeeze(full_map, [3])
            x = x[index]
            return x
            # pool over full range of index instead of matrix
            # split(np.arange(Nside), p)
            # split only in index
            # max(x[split])
            
            pass  # use full nan maps?
        else:
            return x

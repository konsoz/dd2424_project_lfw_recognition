import Dataset
import os
import sys
import tensorflow as tf
import numpy as np
import logging as log
# import timeit
import argparse

from sklearn import metrics

from Dataset import IMG_SIZE, RE_IMG_SIZE

TRAIN_IMAGE_DIR = os.getcwd() + '/data/lfw-deepfunneled'
TEST_IMAGE_DIR = os.getcwd() + '/data/lfw-deepfunneled'  # TODO CREATE TEST SET?
CKPT_DIR = 'ckpt_dir'
MODEL_CKPT = 'ckpt_dir/model.cktp'

### Parameters for Logistic Regression ###
BATCH_SIZE = 32

### Network Parameters ###
n_input = IMG_SIZE ** 2
n_channels = 3
dropout = 0.8  # Dropout, probability to keep units

"""FROM CIFAR GUIDE"""
# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
import re

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


"""FROM CIFAR GUIDE"""


class ConvNet(object):
    ## Constructor to build the model for the training ##
    def __init__(self, **kwargs):

        params = set(['learning_rate', 'max_epochs', 'display_step', 'std_dev', 'dataset_train', 'dataset_valid', 'dataset_test'])

        # initialize all allowed keys to false
        self.__dict__.update((key, False) for key in params)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in params)

        if (self.dataset_train != False and self.dataset_valid != False):
            # Load the Training Set
            self.train_imgs_lab = Dataset.loadDataset(self.dataset_train)
            self.valid_imgs_lab = Dataset.loadDataset(self.dataset_valid)
        else:
            # Load the Test Set
            self.test_imgs_lab = Dataset.loadDataset(self.dataset_test)


        # Graph input
        self.img_pl = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, n_channels])
        self.label_pl = tf.placeholder(tf.float32, [None, Dataset.NUM_LABELS])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


    # Batch function for Training - give the next batch of images and labels
    def BatchIterator(self, imbs_lab, batch_size):
        imgs = []
        labels = []

        for img, label in imbs_lab:
            imgs.append(img)
            labels.append(label)
            if len(imgs) == batch_size:
                yield imgs, labels
                imgs = []
                labels = []
        if len(imgs) > 0:
            yield imgs, labels

    """
    Create AlexNet model
    """

    def conv2d(self, name, l_input, w, b, s):
        pre_activation = tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='SAME'), b)
        return tf.nn.relu(pre_activation, name=name)

    def max_pool(self, name, l_input, k, s):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

    def norm(self, name, l_input, lsize):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=2e-05, beta=0.75, name=name)

    def simple_model(self, _images_u8, _dropout):

        #Reshape input image batch
        img_padded_or_cropped = tf.image.resize_images(_images_u8, [RE_IMG_SIZE, RE_IMG_SIZE])
        _images_f32 = tf.image.convert_image_dtype(img_padded_or_cropped, tf.float32)

        _X = tf.reshape(_images_f32, shape=[-1, RE_IMG_SIZE, RE_IMG_SIZE, 3])


        #Batch normalization on images, TODO add fc layer after this?
        #b_norm = tf.contrib.layers.batch_norm(_X, center=True, scale=True, is_training=True, scope='bn')

        # Convolution Layer 1
        with tf.variable_scope('conv1') as scope:
            #wc1 = tf.Variable(tf.random_normal([5, 5, n_channels, 64], stddev=self.std_dev))
            wc1 = _variable_with_weight_decay('weights',shape=[5, 5, 3, 16], stddev=5e-2, wd=0.0)
            #bc1 = tf.Variable(tf.random_normal([64]))
            bc1 = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
            conv1 = self.conv2d(scope.name, _X, wc1, bc1, s=1)
            print("conv1.shape: ", conv1.get_shape())

        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1, k=3, s=2)
        print("pool1.shape:", pool1.get_shape())

        # Apply Normalization
        norm1 = self.norm('norm1', pool1, lsize=4)
        print("norm1.shape:", norm1.get_shape())


        # Convolution Layer 2
        with tf.variable_scope('conv2') as scope:
            #wc2 = tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=self.std_dev))
            wc2 = _variable_with_weight_decay('weights', shape=[5, 5, 16, 16], stddev=5e-2, wd=0.0)
            #bc2 = tf.Variable(tf.random_normal([64]))
            bc2 = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
            conv2 = self.conv2d(scope.name, norm1, wc2, bc2, s=1)
            print("conv2.shape:", conv2.get_shape())

        # Apply Normalization
        norm2 = self.norm('norm2', conv2, lsize=4)
        print("norm2.shape:", norm2.get_shape())

        # Max Pooling (down-sampling)
        pool2 = self.max_pool('pool2', norm2, k=3, s=2)
        print("pool5.shape:", pool2.get_shape())


        # Fully connected layer 1
        with tf.variable_scope('fc1') as scope:
            pool2_shape = pool2.get_shape().as_list()
            dim = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
            reshape = tf.reshape(pool2, [-1, dim])

            #wd = tf.Variable(tf.random_normal([dim2, 384]))
            wd = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
            #bd = tf.Variable(tf.random_normal([384]))
            bd = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshape, wd) + bd, name=scope.name)  # Relu activation
            print("reshape.shape:", reshape.get_shape())
            print("fc1.shape:", fc1.get_shape())

        # Fully connected layer 2
        with tf.variable_scope('fc2') as scope:
            wfc = tf.Variable(tf.random_normal([384, 192], stddev=self.std_dev))
            wfc = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
            #bfc = tf.Variable(tf.random_normal([192]))
            bfc = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            fc2 = tf.nn.relu(tf.matmul(fc1, wfc) + bfc, name=scope.name)  # Relu activation
            print("fc2.shape:", fc2.get_shape())

        # Output, class prediction LOGITS
        with tf.variable_scope('logits') as scope:
            #wout = tf.Variable(tf.random_normal([192, Dataset.NUM_LABELS], stddev=self.std_dev))
            wout = _variable_with_weight_decay('weights', [192, Dataset.NUM_LABELS], stddev=1 / 192.0, wd=0.0)
            #bout = tf.Variable(tf.random_normal([Dataset.NUM_LABELS]))
            bout = _variable_on_cpu('biases', [Dataset.NUM_LABELS], tf.constant_initializer(0.0))
            out = tf.matmul(fc2, wout) + bout


        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()

        # The function returns the Logits to be passed to softmax and the Softmax for the PREDICTION
        return out

    # Method for training the model and testing its accuracy
    def training(self):

        # Launch the graph
        with tf.Session() as sess:

            ############################# Construct model: prepare logits, loss and optimizer #############################

            # logits: unnormalized log probabilities
            logits = self.simple_model(self.img_pl, self.keep_prob)

            # loss: cross-entropy between the target and the softmax activation function applied to the model's prediction
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label_pl))
            tf.summary.scalar("cross-entropy_for_loss", loss)
            # optimizer: find the best gradients of the loss with respect to each of the variables
            train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1).minimize(loss)

            print(logits.get_shape(), self.label_pl.get_shape())

            ######## Evaluate model: the degree to which the result of the prediction conforms to the correct value ########

            # list of booleans
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.label_pl, 1))
            # [True, False, True, True] -> [1,0,1,1] -> 0.75
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

            # Initializing the variables
            init = tf.global_variables_initializer()
            # Run the Op to initialize the variables.
            sess.run(init)
            summary_writer = tf.summary.FileWriter(CKPT_DIR, graph=sess.graph)

            ##################################### Training the model ################################################

            # collect imgs for validation
            validation_imgs_batch = [b for i, b in enumerate(self.BatchIterator(self.valid_imgs_lab, BATCH_SIZE))]

            # Run for epoch
            for epoch in range(self.max_epochs):
                print("epoch = %d" % epoch)
                log.info('Epoch %s' % epoch)
                self.train_imgs_lab = Dataset.loadDataset(self.dataset_train)

                # Loop over all batches
                for step, elems in enumerate(self.BatchIterator(self.train_imgs_lab, BATCH_SIZE)):
                    print("\nstep = %d" % step)
                    log.info('step %s' % step)
                    ### from iterator return batch lists ###
                    batch_imgs_train, batch_labels_train = elems
                    _, train_acc, train_loss = sess.run([train_step, accuracy, loss],
                                                        feed_dict={self.img_pl: batch_imgs_train,
                                                                   self.label_pl: batch_labels_train,
                                                                   self.keep_prob: 1.0})

                    print("Training Accuracy = " + "{:.5f}".format(train_acc))
                    print("Training Loss = " + "{:.6f}".format(train_loss))
                    log.info("Training Accuracy = " + "{:.5f}".format(train_acc))
                    log.info("Training Loss = " + "{:.6f}".format(train_loss))

            print("Optimization Finished!")

            # Save the models to disk
            save_model_ckpt = self.saver.save(sess, MODEL_CKPT)
            print("Model saved in file %s" % save_model_ckpt)

            ############################################### Metrics ##############################################

            y_p = tf.argmax(logits, 1)  # the value predicted

            target_names = ['class %d' % i for i in range(Dataset.NUM_LABELS)]
            list_pred_total = []
            list_true_total = []

            # Accuracy Precision Recall F1-score by VALIDATION IMAGES
            for step, elems in enumerate(validation_imgs_batch):
                batch_imgs_valid, batch_labels_valid = elems
                valid_acc, y_pred = sess.run([accuracy, y_p], feed_dict={self.img_pl: batch_imgs_valid,
                                                                         self.label_pl: batch_labels_valid,
                                                                         self.keep_prob: 1.0})
                log.info("Validation accuracy = %.5f" % (valid_acc))
                list_pred_total.extend(y_pred)
                y_true = np.argmax(batch_labels_valid, 1)
                list_true_total.extend(y_true)

            # Classification Report
            print(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))
            print("ACCURACY ON VALIDATION: %f" %metrics.accuracy_score(list_true_total, list_pred_total))
            log.info('\n' + metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))

            # Network Input Values
            log.info("Learning Rate %d" % (self.learning_rate))
            log.info("Number of epochs %d" % (self.max_epochs))
            log.info("Standard Deviation %d" % (self.std_dev))

    def prediction(self):
        with tf.Session() as sess:

            ### Construct model
            pred = self.simple_model(self.img_pl, self.weights, self.biases, self.keep_prob)

            ### Restore model
            ckpt = tf.train.get_checkpoint_state("ckpt_dir")
            if (ckpt):
                self.saver.restore(sess, MODEL_CKPT)
                print("Model restored")
            else:
                print("No model checkpoint found to restore - ERROR")
                return

            ############################################### Metrics ##############################################

            y_p = tf.argmax(pred, 1)  # the value predicted

            target_names = ['class 0', 'class 1', 'class 2']
            list_pred_total = []
            list_true_total = []

            # Accuracy Precision Recall F1-score by TEST IMAGES
            for step, elems, _ in enumerate(self.BatchIterator(self.test_imgs_lab, BATCH_SIZE)):
                batch_imgs_test, batch_labels_test = elems
                print(batch_imgs_test[0])
                print(batch_labels_test[0])

                y_pred = sess.run(y_p, feed_dict={self.img_pl: batch_imgs_test, self.keep_prob: 1.0})
                # print(len(y_pred))
                list_pred_total.extend(y_pred)
                y_true = np.argmax(batch_labels_test, 1)
                # print(len(y_true))
                list_true_total.extend(y_true)

            # Classification Report
            print(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))
            log.info(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))

            # Network Input Values
            log.info("Learning Rate %d" % self.learning_rate)
            log.info("Number of epochs %d" % self.max_epochs)
            log.info("Standard Deviation %d" % self.std_dev)


### MAIN ###
def main():
    np.random.seed(7)

    parser = argparse.ArgumentParser(description='A convolutional neural network for image recognition')
    subparsers = parser.add_subparsers()

    training_args = [
        (['-lr', '--learning-rate'], {'help': 'learning rate', 'type': float, 'default': 0.001}),
        (['-e', '--epochs'], {'help': 'epochs', 'type': int, 'default': 5}),
        (['-ds', '--display-step'], {'help': 'display step', 'type': int, 'default': 10}),
        (['-sd', '--std-dev'], {'help': 'std-dev', 'type': float, 'default': 1.0}),
        (['-dtr', '--dataset_training'],
         {'help': 'dataset training file', 'type': str, 'default': 'images_shuffled.pkl'})
    ]

    test_args = [
        (['-dts', '--dataset_test'], {'help': 'dataset test file', 'type': str, 'default': 'images_test_dataset.pkl'})
    ]

    # parser train
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    for arg in training_args:
        parser_train.add_argument(*arg[0], **arg[1])

    # parser preprocessing training data
    parser_preprocess = subparsers.add_parser('preprocessing_training')
    parser_preprocess.set_defaults(which='preprocessing_training')
    parser_preprocess.add_argument('-f', '--file', help='output training file', type=str, default='images_dataset.pkl')
    parser_preprocess.add_argument('-s', '--shuffle', help='shuffle training dataset', action='store_true')
    parser_preprocess.set_defaults(shuffle=False)

    # parser preprocessing test data
    parser_preprocess = subparsers.add_parser('preprocessing_test')
    parser_preprocess.set_defaults(which='preprocessing_test')
    parser_preprocess.add_argument('-t', '--test', help='output test file', type=str, default='images_test_dataset.pkl')

    # parser predict
    parser_predict = subparsers.add_parser('predict')
    parser_predict.set_defaults(which='predict')
    for arg in test_args:
        parser_predict.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()

    # FILE LOG
    log.basicConfig(filename='FileLog.log', level=log.INFO, format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w+")

    # TRAINING & PREDICTION
    if args.which in ('train', 'predict'):

        # t = timeit.timeit("Dataset.loadDataset(TRAIN_IMAGE_DIR)", setup="from __main__ import *")

        if args.which == 'train':
            # TRAINING
            # create the object ConvNet for training
            conv_net = ConvNet(learning_rate=args.learning_rate, max_epochs=args.epochs, display_step=args.display_step,
                               std_dev=args.std_dev, dataset_train="images_dataset.pkl_train",  dataset_valid="images_dataset.pkl_valid")
            # count total number of imgs in training
            train_img_count = Dataset.getNumImages(TRAIN_IMAGE_DIR)
            log.info("Training set num images = %d" % train_img_count)
            conv_net.training()
        else:
            # PREDICTION
            # create the object ConvNet for test
            conv_net = ConvNet(dataset_test=args.dataset_test)
            # count total number of imgs in test
            test_img_count = Dataset.getNumImages(TEST_IMAGE_DIR)
            log.info("Test set num images = %d" % test_img_count)
            conv_net.prediction()

    # PREPROCESSING TRAINING
    elif args.which == 'preprocessing_training':
        if args.shuffle:
            l = [i for i in Dataset.loadDataset('images_dataset.pkl')]
            np.random.shuffle(l)
            Dataset.saveShuffle(l)
        else:
            Dataset.saveDataset2(TRAIN_IMAGE_DIR, args.file)

    # PREPROCESSING TEST
    elif args.which == 'preprocessing_test':
        Dataset.saveDataset(TEST_IMAGE_DIR, args.test)


if __name__ == '__main__':
    main()

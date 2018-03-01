# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import os

import time

import input_data
import model
import tl_files

N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
Train_Proportion = 0.8  # Proportion of the data to be used for training
Valid_Proportion = 0.1
Test_Proportion = 0.1  # Proportion of the data to be used for testing

Enable_Debug = True


def main(_):
    model_name = "food2_model.ckpt"

    if Enable_Debug == False:
        data_dir = os.path.join(FLAGS.buckets, )
        final_dir = os.path.join(FLAGS.checkpointDir, )
        log_dir = os.path.join(FLAGS.checkpointDir, )
    else:
        data_dir = os.path.join(FLAGS.buckets, )
        final_dir = os.path.join(FLAGS.checkpointDir, )
        log_dir = os.path.join(FLAGS.checkpointDir, )

    print("data dir: " + data_dir)
    print("final dir: " + final_dir)

    x_train, y_train, x_valid, y_valid, x_test, y_test = input_data.load_image_path(data_dir, Valid_Proportion,
                                                                                    Test_Proportion)

    # train
    batch_size = 64
    n_epoch = 500
    learning_rate = 0.0001
    print_freq = 1
    n_step_epoch = int((len(y_train) + len(y_valid) + len(y_test)) / batch_size)
    n_step = n_epoch * n_step_epoch
    print('learning_rate: %f' % learning_rate)
    print('batch_size: %d' % batch_size)
    print('n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    train_batch_run, train_label_batch_run = input_data.get_batch(x_train,
                                                                  y_train,
                                                                  IMG_W,
                                                                  IMG_H,
                                                                  batch_size)
    train_logits_run, net = model.inference(train_batch_run, batch_size, N_CLASSES)
    train_loss_run = model.losses(train_logits_run, train_label_batch_run)
    train_op_run = model.trainning(train_loss_run, learning_rate)
    # train_acc_run = model.evaluation(train_logits_run, train_label_batch_run)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step = 0
        for epoch in range(n_epoch):
            start_time = time.time()
            train_loss, train_acc, n_batch = 0, 0, 0
            for s in range(n_step_epoch):
                err, _ = sess.run([train_loss_run, train_op_run])
                step += 1
                train_loss += err
                n_batch += 1

                # 50倍打印频率的整数倍step，写进TensorBorder一次
                if step + 1 == 1 or (step + 1) % (print_freq * 50) == 0:
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

            # 每轮epoch打印一次该epoch的loss
            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d : Step %d-%d of %d took %fs" % (
                    epoch, step - n_step_epoch, step, n_step, time.time() - start_time))
            print("   train loss: %f" % (train_loss / n_batch))
            # print("   train acc: %f" % (train_acc / n_batch))

        # if (epoch + 1) % (print_freq * 50) == 0:# 每50倍轮epoch保存计算图
        print("Save model " + "!" * 10)
        saver = tf.train.Saver()
        save_path = saver.save(sess, final_dir + model_name, global_step=epoch + 1)

        tl_files.save_npz(net.all_params, name='model.npz', sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)

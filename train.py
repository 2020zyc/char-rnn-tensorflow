from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    #数据目录
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    #保存模型
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    #hidden大小
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    #层数
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    #模型类型
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    #minibatch大小
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    #RNN序列长度
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    #迭代次数
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    #保存频率，checkpoint
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    #梯度修剪
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    #学习率
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    #衰减率
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')      
    #初始化模型                 
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    #下面一行从命令行解析实值参数，填充到args参数结构中
    args = parser.parse_args()
    train(args)

def train(args):
    #args中已经有命令行的赋值
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    #数据加载器会分析词汇量
    args.vocab_size = data_loader.vocab_size
    
    # check compatibility if training is continued from previously saved model
    # 检查新旧版本（模型）的兼容性
    if args.init_from is not None:
        # check if all necessary files exist 
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme
        
        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl')) as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagreee on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagreee on dictionary mappings!"
        
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)
        
    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()

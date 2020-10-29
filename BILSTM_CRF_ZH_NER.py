#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, logging
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.utils import np_utils, plot_model
from keras_contrib import metrics
from keras_contrib import losses
from keras.models import Sequential, load_model, save_model, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Input, TimeDistributed, Dropout
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import pickle
import numpy as np


class data_prossor(object):
	'''数据处理'''
	path = 'data/'
	maxlen = 100
	
	@staticmethod
	def load_data():
		ptrain = os.path.join(data_prossor.path, 'train.txt')  # 训练集 'data/train.txt'
		pval = os.path.join(data_prossor.path, 'test.txt')  # 验证集 'data/test.txt'
		
		with open(ptrain, 'r', ) as fd, open(pval, 'r', ) as fdv:
			train, vocab, labels = [], [], []
			#  ps = '继 O\n去 O\n年 O...滩 I-address\n！ O''
			for ps in fd.read().strip().split('\n\n'):
				p = []
				for s in ps.strip().split('\n'):  # s='继 O'
					wt = s.split()  # wt=['继', 'O']
					p.append(wt)  # p=[['继', 'O'], ['去', 'O']]
					vocab.append(wt[0])  # vocab= ['继']
					labels.append(wt[1])  # labels= ['O']
				train.append(p)
			
			val = [[s.split() for s in p.split('\n')] for p in fdv.read().split('\n\n')]
			vocab = dict([(w, i) for i, w in enumerate(list(set(vocab)))])
			
			if not os.path.exists(os.path.join(data_prossor.path, 'vocab.pkl')):
				pickle.dump(vocab, open(os.path.join(data_prossor.path, 'vocab.pkl'), 'wb'))
			else:
				vocab = pickle.load(open(os.path.join(data_prossor.path, 'vocab.pkl'), 'rb'))
		
		labels = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
		train_x, train_y = data_prossor.econde_data(train, vocab, labels)
		val_x, val_y = data_prossor.econde_data(val, vocab, labels)
		return train_x, train_y, val_x, val_y, vocab, labels
	
	@staticmethod
	def econde_data(data, vocab, labels, onehot=False):
		
		x = [[vocab.get(it[0], 1) for it in item if len(it) == 2] for item in data]
		y = [[labels.index(it[1]) for it in item if len(it) == 2] for item in data]
		x = pad_sequences(x, data_prossor.maxlen)
		y = pad_sequences(y, data_prossor.maxlen, value=-1)
		
		if onehot:
			y = np_utils.to_categorical(y, len(labels))
		else:
			y = y.reshape((y.shape[0], y.shape[1], 1))
		
		print(x.shape, y.shape)
		return x, y
	
	@staticmethod
	def parse_text(text, vocab):
		ttext = [vocab.get(w, -1) for w in list(text)]
		ttext = pad_sequences([ttext], data_prossor.maxlen)
		print(ttext)
		return ttext


class BiLSTM_CRF_model(object):
	def __init__(self, labels, vocab):
		self.max_len = 100
		self.max_wnum = len(vocab)
		self.embd_dim = 200
		self.drop_rate = 0.5
		self.batch_size = 64
		self.epochs = 10
		self.lstmunit = 64
		self.label_num = len(labels)
		self.model_path = 'model/'
	
	def model_create(self):
		inputs = Input(shape=(data_prossor.maxlen,), dtype='int32')
		x = Embedding(self.max_wnum,
					  self.embd_dim,
					  mask_zero=True)(inputs)
		x = Bidirectional(LSTM(self.lstmunit, return_sequences=True))(x)
		x = Dropout(self.drop_rate)(x)
		# Bi-LSTM展开输出
		x = Dense(self.label_num)(x)
		crf = CRF(units=self.label_num, sparse_target=True)(x)
		model = Model(inputs=inputs, outputs=crf)
		model.summary()
		model.compile(optimizer='adam',
					  loss=losses.crf_loss,
					  metrics=[metrics.crf_accuracy])
		plot_model(model, to_file="{}{}".format(self.model_path, 'bilstm-crf.png'))
		return model
	
	def model_fit(self, model, train, val):
		early_stopping = EarlyStopping(monitor='crf_loss',
									   patience=5,
									   verbose=1, )
		model_checkpoint = ModelCheckpoint(filepath=self.model_path + 'bilstm_crf.h5',
										   save_best_only=True,
										   verbose=1)
		tensor_board = TensorBoard(log_dir='./logs',
								   batch_size=self.batch_size,
								   write_images=True)
		
		# 在这里使用校验数据会出现val_loss=nan的问题，小批量数据不会出现，采用softmax也不会出现
		# model.fit(train[0], train[1], validation_data=[val[0], val[1]],
		#           batch_size=self.batch_size, epochs=self.epochs,
		#           shuffle=True, callbacks=[early_stopping, model_checkpoint, tensor_board])
		
		model.fit(train[0], train[1],
				  validation_split=0.2,
				  batch_size=self.batch_size,
				  epochs=self.epochs,
				  shuffle=True,
				  callbacks=[early_stopping, model_checkpoint, tensor_board])
	
	def model_predict(self, model, text, vocab):
		ttext = data_prossor.parse_text(text, vocab)
		
		raw = model.predict(ttext)[0][-len(text):]
		# [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 1,...
		result_ = [np.argmax(row) for row in raw]
		# ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', ...]
		result = [labels[i] for i in result_]
		
		per, org, loc = '', '', ''
		for s, t in zip(list(text), result[0][-len(text):]):
			print(s, t)
			if t in ('B-PER', 'I-PER'):
				per += ' ' + s if (t == 'B-PER') else s
			if t in ('B-ORG', 'I-ORG'):
				org += ' ' + s if (t == 'B-ORG') else s
			if t in ('B-LOC', 'I-LOC'):
				loc += ' ' + s if (t == 'B-LOC') else s
		ner = 'PER:' + per + '\nORG:' + org + '\nLOC：' + loc
		return ner


if __name__ == '__main__':
	train_x, train_y, val_x, val_y, vocab, labels = data_prossor.load_data()
	
	BiLSTM_CRF = BiLSTM_CRF_model(labels, vocab)
	model = BiLSTM_CRF.model_create()
	if os.path.exists(BiLSTM_CRF.model_path + 'bilstm_crf.h5'):
		model.load_weights(BiLSTM_CRF.model_path + 'bilstm_crf.h5')
	else:
		BiLSTM_CRF.model_fit(model, (train_x, train_y), (val_x, val_y))
	
	text = '中华人民共和国国务院总理周恩来在外交部长陈毅，副部长王东的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
	ner = BiLSTM_CRF.model_predict(model, text, vocab)
	print(ner)

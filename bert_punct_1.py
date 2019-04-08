
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import bert
from bert import tokenization
from bert import modeling
from bert import optimization
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.layers.python.layers import initializers
import tf_metrics
import json


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
	"data_dir", "./raw_data",
	"The input datadir. ex) 'NERdata'"
)

flags.DEFINE_string(
	"output_dir", "./output_serving_1",
	"The output path for saving the model"
)

flags.DEFINE_string(
	'export_dir', './export_model_1',
	'The output path to save a model for TF serving'
)


flags.DEFINE_string(
	"init_checkpoint", None,
	"Initial checkpoint for the bert model"
)

flags.DEFINE_string(
	"data_config_path", "data.conf",
	"The path for saving the configuration"
)

flags.DEFINE_bool(
	"do_train", True,
	"Whether to run training."
)

flags.DEFINE_bool(
	"do_predict", True,
	"Whether to run prediction."
)

flags.DEFINE_bool(
	"do_export", True,
	"Export model for serving"
)


flags.DEFINE_integer(
	"max_seq_length", 128,
	"The maximum sequence length after bert tokenization"
)

flags.DEFINE_integer(
	"batch_size", 8,
	"The batch size used for training, evaluating and predicting"
)

flags.DEFINE_float(
	"learning_rate", 5e-5,
	"The initial learning rate"
)

flags.DEFINE_float(
	"num_epochs", 4.0,
	"Total number of training epochs to perform"
)

flags.DEFINE_float(
	"warmup_proportion", 0.1,
	"Proportion of training for which to perform warmup of learning rate"
)

flags.DEFINE_float(
	"bert_dropout_rate", 0.2,
	"Proportion of dropout for bert"
)
flags.DEFINE_bool(
	"use_lstm", True,
	"Whether to use lstm."
)

flags.DEFINE_integer(
	"lstm_size", 256,
	"size of lstm units"
)

flags.DEFINE_float(
	"lstm_dropout_rate", 0.2,
	"Proportion of dropout for lstm"
)


flags.DEFINE_bool(
	"use_crf", True,
	"Whether to use crf."
)


flags.DEFINE_integer(
	"save_checkpoints_steps", 500,
	"Number of steps after which to save the checkpoint"
)

flags.DEFINE_integer(
	"save_summary_steps", 50,
	"Number of steps after which to save the summary"
)

flags.DEFINE_integer(
	"keep_checkpoint_max", 5,
	"The maximum number of checkpoints to save"
)


class InputExample(object):
	
	def __init__(self, guid, text_a, text_b=None, label=None):
		
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

class InputFeatures(object):
	
	def __init__(self, input_ids, input_mask, segment_ids, label_ids):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_ids = label_ids


class NerProcessor(object):
	def get_train_examples(self, data_dir):
		return self._create_example(
			self._read_data(data_dir, 'train'), 'train'
		)
	

	def get_dev_examples(self, data_dir):
		return self._create_example(
			self._read_data(data_dir, 'dev'), 'dev'
		)

	def get_test_examples(self, data_dir):
		return self._create_example(
			self._read_data(data_dir, 'test'), 'test'
		)

	def get_labels(self):
		return ["0", "1", "2", "3", "X", "[pad]"]
	
	def _create_example(self, data, set_type):
		examples = []
		for i, (sent, lab) in enumerate(zip(data['sentence'],data['label'])):
			guid = "%s-%s" % (set_type, i)
			text_a = tokenization.convert_to_unicode(sent)
			label = tokenization.convert_to_unicode(lab)
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
		return examples


	def _read_data(self, data_dir, dataset):

		data = {}
		data['sentence'] = []
		data['label'] = []

		with open(os.path.join(data_dir,"{}.word.txt".format(dataset)), "r") as f:
			for line in f:
				data["sentence"].append(line.strip())
	
		with open(os.path.join(data_dir,"{}.label.txt".format(dataset)), "r") as f:
			for line in f:
				data["label"].append(line.strip())
		
		return data


def create_tokenizer_from_hub_module():
	with tf.Graph().as_default():
		bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
		tokenization_info =  bert_module(signature="tokenization_info", as_dict=True)
		with tf.Session() as sess: 
			vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
												 tokenization_info["do_lower_case"]])
	return bert.tokenization.FullTokenizer(
		vocab_file=vocab_file, do_lower_case=do_lower_case )


def convert_feature_to_tf_example(feature):
	def create_int_features(values):
		f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
		return f

	features = collections.OrderedDict()
	features["input_ids"] = create_int_features(feature.input_ids)
	features["input_mask"] = create_int_features(feature.input_mask)
	features["segment_ids"] = create_int_features(feature.segment_ids)
	features["label_ids"] = create_int_features(feature.label_ids)
	tf_example = tf.train.Example(features=tf.train.Features(feature=features))
	return tf_example


EOS = ['1','3']
PUNCTUATION_LABELS = ['1','2','3']
def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
	
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	writer = tf.python_io.TFRecordWriter(output_file)

	idx = 0
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info("Converting example %d of %d" % (ex_index, len(examples)))

		textlist = example.text_a.split(' ')
		labellist = example.label.split(' ')
		tokens_a = []
		tokens_b = None
		labels = []
	
		for i, word in enumerate(textlist):
			token = tokenizer.tokenize(word)
			tokens_a.extend(token)
			label_1 = labellist[i]
			for m in range(len(token)):
				if m == 0:
					labels.append(label_1)
				else:
					labels.append("X")

		
		tokens = []
		segment_ids = []
		label_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)
		label_ids.append(label_map["[pad]"])
		
		skip_until_eos = False
		last_eos_idx = 0
				
		for token, label in zip(tokens_a, labels):

			if skip_until_eos:
				if label in EOS:
					skip_until_eos = False
			
				continue
		
			elif label in PUNCTUATION_LABELS:
			
				if label in EOS:
					last_eos_idx = len(tokens)

				tokens.append(token)    
				segment_ids.append(0)
				label_ids.append(label_map[label])

			else:
				tokens.append(token)    
				segment_ids.append(0)
				label_ids.append(label_map[label])


			if len(tokens) == max_seq_length-1:
				assert len(tokens) ==  len(label_ids),"#words: %d; #punctuations: %d" % (len(tokens), len(label_ids))

				if last_eos_idx == 0:
					skip_until_eos = True

					tokens = []
					segment_ids = []
					label_ids = []
					tokens.append("[CLS]")
					segment_ids.append(0)
					label_ids.append(label_map["[pad]"])

				else:

					tokens.append("[SEP]")
					segment_ids.append(0)
					label_ids.append(label_map["[pad]"])

					input_ids = tokenizer.convert_tokens_to_ids(tokens)
					input_mask = [1] * len(input_ids)

					assert len(input_ids) == max_seq_length
					assert len(input_mask) == max_seq_length
					assert len(segment_ids) == max_seq_length
					assert len(label_ids) == max_seq_length

					if idx < 5:
						tf.logging.info("*** Example ***")
						tf.logging.info("guid: %s" % (example.guid))
						tf.logging.info("tokens: %s" % " ".join(
							[tokenization.printable_text(x) for x in tokens]))
						tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
						tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
						tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
						tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
						#tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

					feature = InputFeatures(
						input_ids=input_ids,
						input_mask=input_mask,
						segment_ids=segment_ids,
						label_ids=label_ids,
						#label_mask = label_mask
					)
					tf_example = convert_feature_to_tf_example(feature)
					writer.write(tf_example.SerializeToString())
					idx += 1
					tokens = ["[CLS]"] + tokens[last_eos_idx+1:-1]
					segment_ids = [0] + segment_ids[last_eos_idx+1:-1]
					label_ids = [label_map["[pad]"]] + label_ids[last_eos_idx+1:-1]

					
				last_eos_idx = 0

		if len(tokens) > 0:

			tokens.append("[SEP]")
			segment_ids.append(0)
			label_ids.append(label_map["[pad]"])

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)
			while len(input_ids) < max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)
				label_ids.append(label_map["[pad]"])


			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length
			assert len(label_ids) == max_seq_length

			if idx < 5:
				tf.logging.info("*** Example ***")
				tf.logging.info("guid: %s" % (example.guid))
				tf.logging.info("tokens: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens]))
				tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
				tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
				tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
				tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
				#tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

			feature = InputFeatures(
				input_ids=input_ids,
				input_mask=input_mask,
				segment_ids=segment_ids,
				label_ids=label_ids,
				#label_mask = label_mask
			)
			tf_example = convert_feature_to_tf_example(feature)
			writer.write(tf_example.SerializeToString())
			idx += 1

	return idx

def convert_example_to_features(example, label_list, max_seq_length, tokenizer, start_index):
	
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	textlist = example.text_a.split(' ')
	labellist = example.label.split(' ')
	tokens_a = []
	tokens_b = None
	labels = []
	labels_mask_1 = []

	for i, word in enumerate(textlist[start_index:], start_index):
		token = tokenizer.tokenize(word)
		tokens_a.extend(token)
		label_1 = labellist[i]
		for m in range(len(token)):
			if m == 0:
				labels.append(label_1)
				labels_mask_1.append(1)
			else:
				labels.append("X")
				labels_mask_1.append(0)

	
	# features = []
	# imp_labels = []
	
	tokens = []
	segment_ids = []
	label_ids = []
	label_mask = []
	
	tokens.append("[CLS]")
	segment_ids.append(0)
	label_ids.append(label_map["[pad]"])
	label_mask.append(0)

			
	for i, (token, label) in enumerate(zip(tokens_a, labels)):

		tokens.append(token)    
		segment_ids.append(0)
		label_ids.append(label_map[label])
		label_mask.append(labels_mask_1[i])			


		if len(tokens) == max_seq_length-1:
			assert len(tokens) ==  len(label_ids),"#words: %d; #punctuations: %d" % (len(tokens), len(label_ids))

			tokens.append("[SEP]")
			segment_ids.append(0)
			label_ids.append(label_map["[pad]"])
			label_mask.append(0)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)

			feature = InputFeatures(
				input_ids=input_ids,
				input_mask=input_mask,
				segment_ids=segment_ids,
				label_ids=label_ids,
				#label_mask = label_mask
			)
			tf_example = convert_feature_to_tf_example(feature)
			return tf_example, label_mask
			# features.append(tf_example)
			# imp_labels.append(label_mask)
			

			# tokens = []
			# segment_ids = []
			# label_ids = []
			# label_mask = []
			
			# tokens.append("[CLS]")
			# segment_ids.append(0)
			# label_ids.append(label_map["[pad]"])
			# label_mask.append(0)

			
	if len(tokens) > 0:

		tokens.append("[SEP]")
		segment_ids.append(0)
		label_ids.append(label_map["[pad]"])
		label_mask.append(0)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)
			label_ids.append(label_map["[pad]"])
			label_mask.append(0)

		feature = InputFeatures(
			input_ids=input_ids,
			input_mask=input_mask,
			segment_ids=segment_ids,
			label_ids=label_ids,
		)
		tf_example = convert_feature_to_tf_example(feature)
		# features.append(tf_example)
		# imp_labels.append(label_mask)
	
	return tf_example, label_mask

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, word_length=None):
	
	name_to_feature ={
		"input_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"input_mask": tf.FixedLenFeature([seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"label_ids": tf.FixedLenFeature([seq_length], tf.int64)
	
	}

	
	def _decode_record(record, name_to_features):
		example = tf.parse_single_example(record, name_to_features)
		for name in list(example.keys()):
			t = example[name]
			if t.dtype == tf.int64:
				t = tf.to_int32(t)
			example[name] = t

		return example


	def input_fn(params):
		batch_size = params["batch_size"]
		d = tf.data.TFRecordDataset(input_file)
		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=100)
		d = d.apply(tf.data.experimental.map_and_batch(
			lambda record: _decode_record(record, name_to_feature),
			batch_size = batch_size,
			drop_remainder=drop_remainder
		))
		return d

	return input_fn


def create_model(is_training, input_ids, input_mask, segment_ids, labels, num_labels):
	
	bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
							 trainable=True)
	bert_inputs = dict(input_ids=input_ids,
					   input_mask=input_mask,
					   segment_ids=segment_ids)
	
	bert_output = bert_module(
		inputs=bert_inputs,
		signature="tokens",
		as_dict=True)
	
	output_layer = bert_output["sequence_output"]
	output_layer = tf.layers.dropout(output_layer, rate=FLAGS.bert_dropout_rate, training=is_training)

	hidden_size = output_layer.shape[-1].value
	seq_length = FLAGS.max_seq_length

	used = tf.sign(tf.abs(input_mask))
	lengths = tf.reduce_sum(used, reduction_indices=1)

	def bi_lstm_fused(inputs, lengths, rnn_size, is_training, dropout_rate, scope='bi_lstm_fused'):
		with tf.variable_scope(scope):
			t = tf.transpose(inputs, perm=[1,0,2])
			lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
			lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
			lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
			output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
			output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths)
			outputs = tf.concat([output_fw, output_bw], axis=-1)
			outputs = tf.transpose(outputs, perm=[1,0,2])
			return tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

	def lstm_layer(inputs, lengths, is_training):
		rnn_output = tf.identity(inputs)
		for i in range(2):
			scope = 'bi_lstm_fused-%s' % i
			rnn_output = bi_lstm_fused(rnn_output,
				lengths,
				rnn_size=FLAGS.lstm_size,
				is_training=is_training,
				dropout_rate=FLAGS.lstm_dropout_rate,
				scope=scope)

		return rnn_output

	def project_layer(inputs, out_dim, seq_length, scope='project'):

		with tf.variable_scope(scope):
			in_dim = inputs.shape[-1].value
			weight = tf.get_variable("W", 
									[in_dim, out_dim], dtype=tf.float32,
									initializer=initializers.xavier_initializer())
			bias = tf.get_variable("b", 
									[out_dim], dtype=tf.float32,
									initializer=tf.zeros_initializer())
			t_output = tf.reshape(inputs, [-1, in_dim])

			output = tf.matmul(t_output, weight)
			output = tf.nn.bias_add(output, bias)
		
			output = tf.reshape(output, [-1, seq_length, out_dim])
			return output


	def loss_layer(logits, labels, num_labels, lengths, input_mask, scope='loss'):
		
		with tf.variable_scope(scope):  
			trans = tf.get_variable(
				"transitions",
				shape=[num_labels, num_labels],
				initializer=initializers.xavier_initializer())

			if FLAGS.use_crf:
				print("****Using CRF****")
				log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
					inputs=logits,
					tag_indices=labels,
					transition_params=trans,
					sequence_lengths=lengths)
				per_example_loss = -log_likelihood
				loss = tf.reduce_mean(per_example_loss)

			else:
				log_probs = tf.nn.log_softmax(logits, axis=-1)
				one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
				cross_entropy = -tf.reduce_sum(one_hot_labels * log_probs,reduction_indices=2)
				cross_entropy *= tf.to_float(input_mask)
				per_example_loss = tf.reduce_sum(cross_entropy, reduction_indices=1)
				per_example_loss /= tf.cast(lengths, tf.float32)
				loss = tf.reduce_mean(per_example_loss)
			
			return loss, per_example_loss, trans


	if FLAGS.use_lstm:
		print("****Using BILstm****")
		output1 = lstm_layer(output_layer, lengths, is_training)
		logits = project_layer(output1, num_labels, seq_length)
	else:
		logits = project_layer(output_layer, num_labels, seq_length)

	
	loss, per_example_loss, trans = loss_layer(logits, labels, num_labels, lengths, input_mask)
	

	if FLAGS.use_crf:
		prediction, _ = tf.contrib.crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=lengths)
	
	else:
		probabilities = tf.nn.softmax(logits, axis=-1)
		prediction = tf.argmax(probabilities, axis=-1, output_type=tf.int32)

	prediction *= input_mask

	print('#' * 20)
	print('shape of output_layer:', output_layer.shape)
	print('hidden state size:%d' % hidden_size)
	print('seq_length:%d' % seq_length)
	print('shape of logit', logits.shape)
	print('shape of loss', loss.shape)
	print('shape of per_example_loss', per_example_loss.shape)
	print('num labels:%d' % num_labels)
	print('#' * 20)
	
	return (loss, per_example_loss, logits, prediction)

def model_fn_builder(num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps):
	
	def model_fn(features, labels, mode, params):
		
		tf.logging.info("*** Features ***")
		for name in sorted(features.keys()):
			tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		label_ids = features["label_ids"]
		
		
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		
		(loss, per_example_loss, logits, predicted_labels) = create_model(
			is_training, input_ids, input_mask, segment_ids, label_ids, num_labels)
		
		
		tf.summary.scalar('loss', loss)
		tvars = tf.trainable_variables()
		if init_checkpoint:
			(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
																									   init_checkpoint)
			tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
			tf.logging.info("**** Trainable Variables ****")
			for var in tvars:
				init_string = ""
				if var.name in initialized_variable_names:
					init_string = ", *INIT_FROM_CKPT*"
				tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
								init_string)

		output_spec = None
		
		if mode == tf.estimator.ModeKeys.TRAIN:
			train_op = bert.optimization.create_optimizer(
				loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
			
			logging_hook = tf.train.LoggingTensorHook({"batch_loss" : loss}, every_n_iter=10)
			output_spec = tf.estimator.EstimatorSpec(mode=mode,
				loss=loss,
				train_op=train_op,
				training_hooks=[logging_hook])


		
		else: 
			
			def metric_fn(label_ids, predicted_labels, per_example_loss):
				precision = tf_metrics.precision(label_ids, predicted_labels, num_labels, [1,2,3],
												 average="macro")
				recall = tf_metrics.recall(label_ids, predicted_labels, num_labels, [1,2,3],
												 average="macro")
				f1 = tf_metrics.f1(label_ids, predicted_labels, num_labels, [1,2,3],
												 average="macro")
				loss = tf.metrics.mean(values=per_example_loss)

		
				return {
					"eval_precision": precision,
					"eval_recall": recall,
					"eval_f": f1
				}
			
			eval_metrics = metric_fn(label_ids, predicted_labels, per_example_loss)
			
			
			if mode == tf.estimator.ModeKeys.EVAL:
				output_spec = tf.estimator.EstimatorSpec(mode=mode,
					loss=loss,
					eval_metric_ops=eval_metrics)
			else:
				
				predictions = {
					'labels': predicted_labels
				}
				output_spec = tf.estimator.EstimatorSpec(mode=mode,
					predictions=predictions)

		return output_spec
		
	return model_fn



def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)
	processor = NerProcessor()

	label_list = processor.get_labels()
	tokenizer = create_tokenizer_from_hub_module()

	run_config = tf.estimator.RunConfig(
		model_dir=FLAGS.output_dir,
		save_checkpoints_steps=FLAGS.save_checkpoints_steps,
		keep_checkpoint_max=FLAGS.keep_checkpoint_max,
		save_summary_steps=FLAGS.save_summary_steps)

	train_examples = None
	num_train_steps = None
	num_warmup_steps = None

	if os.path.exists(FLAGS.data_config_path):
		with open(FLAGS.data_config_path) as fd:
			data_config = json.load(fd)
	else:
		data_config = {}

	if FLAGS.do_train:
		if len(data_config) == 0:
			train_examples = processor.get_train_examples(FLAGS.data_dir)
			train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
			length_data = filed_based_convert_examples_to_features(
				train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
			
			num_train_steps = int((length_data/FLAGS.batch_size)*FLAGS.num_epochs)
			num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
			
			#num_train_steps = 500
			#num_warmup_steps = 100

			data_config['num_train_steps'] = num_train_steps
			data_config['num_warmup_steps'] = num_warmup_steps
			data_config['num_train_size'] = length_data
			data_config['train.tf_record_path'] = train_file
		else:
			num_train_steps = int(data_config['num_train_steps'])
			num_warmup_steps = int(data_config['num_warmup_steps'])
			train_file = data_config.get('train.tf_record_path')

	model_fn = model_fn_builder(
		num_labels=len(label_list),
		init_checkpoint=FLAGS.init_checkpoint,
		learning_rate=FLAGS.learning_rate,
		num_train_steps=num_train_steps,
		num_warmup_steps=num_warmup_steps)

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		config=run_config,
		params={"batch_size":FLAGS.batch_size})

	if FLAGS.do_train:
		num_train_size = data_config['num_train_size']
		tf.logging.info("**** Running training ****")
		tf.logging.info("  Num examples = %d", num_train_size)
		tf.logging.info("  Batch size = %d", FLAGS.batch_size)
		tf.logging.info("  Num steps = %d", num_train_steps)

		train_input_fn = file_based_input_fn_builder(
			input_file=train_file,
			seq_length=FLAGS.max_seq_length,
			is_training=True,
			drop_remainder=True)

		if data_config.get('eval.tf_record_path', '') == '':
			eval_examples = processor.get_dev_examples(FLAGS.data_dir)
			eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
			length_eval = filed_based_convert_examples_to_features(
				eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
			data_config['eval.tf_record_path'] = eval_file
			data_config['num_eval_size'] = length_eval
		else:
			eval_file = data_config['eval.tf_record_path']
		
		num_eval_size = data_config.get('num_eval_size', 0)
		tf.logging.info("***** Running evaluation *****")
		tf.logging.info("  Num examples = %d", num_eval_size)
		tf.logging.info("  Batch size = %d", FLAGS.batch_size)
		
		eval_input_fn = file_based_input_fn_builder(
			input_file=eval_file,
			seq_length=FLAGS.max_seq_length,
			is_training=False,
			drop_remainder=False)
		# train and evaluate 
		#hook = tf.contrib.estimator.stop_if_no_decrease_hook(
		#   estimator, 'eval_f', 3000, min_steps=30000, run_every_secs=360)
		#train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[hook])
		train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
		eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=120)
		tp = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
		result = tp[0]


		output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
		if result:
			with open(output_eval_file, "w") as writer:
				tf.logging.info("***** Eval results *****")
				for key in sorted(result.keys()):
					tf.logging.info("  %s = %s", key, str(result[key]))
					writer.write("%s = %s\n" % (key, str(result[key])))

	if not os.path.exists(FLAGS.data_config_path):
		with open(FLAGS.data_config_path, 'a') as fd:
			json.dump(data_config, fd)

	if FLAGS.do_predict:
		predict_examples = processor.get_test_examples(FLAGS.data_dir)
		predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
		length_predict = filed_based_convert_examples_to_features(predict_examples, label_list,
			FLAGS.max_seq_length, tokenizer,
			predict_file)

		tf.logging.info("**** Running prediction *****")
		tf.logging.info("  Num examples = %d", length_predict)
		tf.logging.info(" Batch size = %d", FLAGS.batch_size)

		predict_input_fn = file_based_input_fn_builder(
			input_file=predict_file,
			seq_length=FLAGS.max_seq_length,
			is_training=False,
			drop_remainder=False)

		predicted_result = estimator.evaluate(input_fn=predict_input_fn, steps=None)
		output_eval_file = os.path.join(FLAGS.output_dir, "predicted_results.txt")
		with open(output_eval_file, "w") as writer:
			tf.logging.info("**** Predicted results *****")
			for key in sorted(predicted_result.keys()):
				tf.logging.info(" %s = %s", key, str(predicted_result[key]))
				writer.write("%s = %s\n" % (key, str(predicted_result[key])))

	if FLAGS.do_export:

		def serving_input_fn():
			with tf.variable_scope("foo"):
				feature_spec = {
					"input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
					"input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
					"segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
					"label_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
				}
				serialized_tf_example = tf.placeholder(dtype=tf.string,
					shape=[None],
					name='input_example_tensor')


				receiver_tensors = {'examples': serialized_tf_example}
				features = tf.parse_example(serialized_tf_example, feature_spec)
				for name in list(features.keys()):
					t = features[name]
					if t.dtype == tf.int64:
						t = tf.to_int32(t)
					features[name] = t

				return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

		estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn)


if __name__ == '__main__':
	tf.app.run()
	 

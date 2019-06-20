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
import tf_metrics
import json


flags = tf.flags
FLAGS = flags.FLAGS 

flags.DEFINE_string(
	"data_dir", "./raw_data",
	"The input datadir. ex) 'NERdata'"
)

flags.DEFINE_string(
	"bert_config_file","uncased_L-12_H-768_A-12/bert_config.json" ,
	"The config json file corresponding to the pre-trained BERT model"
)

flags.DEFINE_string(
	"vocab_file", "uncased_L-12_H-768_A-12/vocab.txt",
	"The vocabulary file that the BERT model was trained on"
)

flags.DEFINE_string(
	"init_checkpoint", "uncased_L-12_H-768_A-12/bert_model.ckpt",
	"Initial checkpoint for the bert model"
)

flags.DEFINE_bool(
	"do_lower_case", True,
	"Whether to lower case the input text."
)

flags.DEFINE_string(
	"output_dir", "./output",
	"The output path for saving the model"
)

flags.DEFINE_string(
	'export_dir', './export',
	'The output path to save a model for TF serving'
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
	"learning_rate", 2e-5,
	"The initial learning rate"
)

flags.DEFINE_float(
	"num_epochs", 5.0,
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
	"lstm_size", 200,
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
	"save_checkpoints_steps", 250,
	"Number of steps after which to save the checkpoint"
)

flags.DEFINE_integer(
	"save_summary_steps", 50,
	"Number of steps after which to save the summary"
)

flags.DEFINE_integer(
	"keep_checkpoint_max", 2,
	"The maximum number of checkpoints to save"
)

class InputExample(object):
	"""
		Convert each example to Input example object
	"""

	def __init__(self, guid, text_a, text_b=None, label=None):
		
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

class InputFeatures(object):
	
	"""
		Store the features as Input Feature object 
	"""

	def __init__(self, input_ids, input_mask, segment_ids, label_ids):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_ids = label_ids
		
class NerProcessor(object):


	def get_train_examples(self, data_dir):
		
		"""
			Generates a list of training examples from the data_dir 

			Args:
				data_dir: The name of the directory containing the training data
			
			Returns:
				examples: A list of Input Examples containing the parsed training data
		"""      


		return self._create_example(
			self._read_data(data_dir, 'train'), 'train'
		)
	

	def get_dev_examples(self, data_dir):
		
		"""
			Generates a list of validation examples from the data_dir 

			Args:
				data_dir: The name of the directory containing the validation data
			
			Returns:
				examples: A list of Input Examples containing the parsed validation data
		"""      

		return self._create_example(
			self._read_data(data_dir, 'dev'), 'dev'
		)

	def get_test_examples(self, data_dir):
		
		"""
			Generates a list of testing examples from the data_dir 

			Args:
				data_dir: The name of the directory containing the testing data
			
			Returns:
				examples: A list of Input Examples containing the parsed testing data
		"""

		return self._create_example(
			self._read_data(data_dir, 'test'), 'test'
		)

	def get_labels(self):
		
		"""
			Returns:
				A list of all the labels for the BERT model
			
			Labels:
				0:Non punctuation
				1:Period 
				2:Comma
				3:Question Mark
				4:Sub-tokens from BERT  
				5:CLS, SEP and padding 
				sentences to make them equal to max_length          
		
		"""

		#Remove 3 if using chinese
		return ["0", "1", "2", "3","X", "[pad]"]
	
	def _create_example(self, data, set_type):
		"""
			Generates a list of Input Example objects from data

			Args:
				data: A dictionary containing a list of corresponding 
					sentence and labels
				set_type: A string mentioning the type of dataset. 
					Value can be equal to train, dev or test

			Returns:
				examples: A list of Input Example objects containing the parsed data
		"""
		examples = []
		for i, (sent, lab) in enumerate(zip(data['sentence'],data['label'])):
			guid = "%s-%s" % (set_type, i)
			text_a = tokenization.convert_to_unicode(sent)
			label = tokenization.convert_to_unicode(lab)
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
		return examples


	def _read_data(self, data_dir, dataset):

		""" 
			Reads data from the data_dir
			
			Returns:
				data: A dictionary containing a list of corresponding 
					sentence and labels

		"""
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

def convert_feature_to_tf_example(feature):
	
	"""
		Convert features object to tensorflow example
	"""

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

#Symbols that denote end of a sentence
#Remove 3 if using Chinese text 
EOS = ['1','3']

def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
	
	"""
		Converts examples to tensorflow examples and 
		write them using TFRecordWriter to output_file
		
		Basic difference with BERT is that one single 
		Input example can be converted to multiple 
		tf_examples.

		Args:
			examples: A list of Input Example objects
			label_list:  The list of all the possible labels
			max_seq_length: Length of each feature for BERT model
			tokenizer: BERT tokenizer
			output_file: File for writing the tf_examples
		
		Returns:
			idx: The count of the tf_examples written
			  
	"""

	#label_map is a dictionary for converting labels to ids
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i


	writer = tf.python_io.TFRecordWriter(output_file)
	
	#variable to count the number of input examples
	idx = 0
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info("Converting example %d of %d" % (ex_index, len(examples)))

		
		textlist = example.text_a.split(' ')
		labels = example.label.split(' ')

		label_ids = []
		tokens = []
		segment_ids = []
		
		#Add CLS token  
		tokens.append("[CLS]")
		segment_ids.append(0)
		label_ids.append(label_map["[pad]"])

		#used for skipping sentences with more than 128 tokens
		skip_until_eos = False
		last_eos_idx = 0

		for i, word in enumerate(textlist):
			bert_tokens = tokenizer.tokenize(word)
			for j, bert_token in enumerate(bert_tokens):
				
				#Skip words till the labels are not in EOS
				if skip_until_eos:
					if labels[i] in EOS:
						skip_until_eos = False

					continue

				if labels[i] in EOS:
					last_eos_idx = len(tokens)

				tokens.append(bert_token)
				segment_ids.append(0)
				
				if j == len(bert_tokens)-1:
					#label the last token as the true label
					label_ids.append(label_map[labels[i]])	
				else:
					#label rest of the tokens as X
					label_ids.append(label_map["X"])

				if len(tokens) == max_seq_length-1:
					assert len(tokens) ==  len(label_ids) ,"#words: %d; #punctuations: %d" \
					% (len(tokens), len(label_ids))

					#if example has a sentence that is more than 128 tokens, skip it    
					if last_eos_idx == 0:
						skip_until_eos = True

						label_ids = []
						tokens = []
						segment_ids = []
						tokens.append("[CLS]")
						segment_ids.append(0)
						label_ids.append(label_map["[pad]"])
					
					#Else add the sequence of 128 tokens with labels to the output file
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

						if idx < 10:
							tf.logging.info("*** Example ***")
							tf.logging.info("guid: %s" % (example.guid))
							tf.logging.info("tokens: %s" % " ".join(
								[tokenization.printable_text(x) for x in tokens]))
							tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
							tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
							tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
							tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))


						feature = InputFeatures(
							input_ids=input_ids,
							input_mask=input_mask,
							segment_ids=segment_ids,
							label_ids=label_ids
						)
						tf_example = convert_feature_to_tf_example(feature)
						writer.write(tf_example.SerializeToString())
						idx += 1

						#Continue to process the rest of the example
						tokens = ["[CLS]"] + tokens[last_eos_idx+1:-1]
						segment_ids = [0] + segment_ids[last_eos_idx+1:-1]
						label_ids = [label_map["[pad]"]] + label_ids[last_eos_idx+1:-1]

					last_eos_idx = 0 
		
		#if the tokens are less than 128 for tf_example. Pad it with pad tokens.
		if len(tokens) > 2:

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
				
			if idx < 10:
				tf.logging.info("*** Example ***")
				tf.logging.info("guid: %s" % (example.guid))
				tf.logging.info("tokens: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens]))
				tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
				tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
				tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
				tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
			
			feature = InputFeatures(
				input_ids=input_ids,
				input_mask=input_mask,
				segment_ids=segment_ids,
				label_ids=label_ids
			)
			tf_example = convert_feature_to_tf_example(feature)
			writer.write(tf_example.SerializeToString())
			idx += 1
	
	return idx

def convert_example_to_features(example, label_list, max_seq_length, tokenizer, start_index):
	

	"""
		Function used for serving
		Similar to filed_based_convert_examples_to_features
	
	"""
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	textlist = example.text_a.split(' ')
	labels = example.label.split(' ')


	label_ids = []
	#label_mask used to keep count of the relevant labels
	label_mask = []
	tokens = []
	segment_ids = []
	
	#Add CLS token  
	tokens.append("[CLS]")
	segment_ids.append(0)
	label_ids.append(label_map["[pad]"])
	label_mask.append(0)

	
	for i, word in enumerate(textlist[start_index:], start_index):
		bert_tokens = tokenizer.tokenize(word)
		for j, bert_token in enumerate(bert_tokens):
			
			tokens.append(bert_token)
			segment_ids.append(0)
			label_ids.append(label_map["[pad"])
			
			if j == len(bert_token)-1:
				#mark the last token as relevant
				label_mask.append(1)	
			else:
				#mark rest of the tokens as irrelevant
				label_ids.append(0)

			if len(tokens) == max_seq_length-1:
				assert len(tokens) ==  len(label_ids) ,"#words: %d; #punctuations: %d" \
				% (len(tokens), len(label_ids))

				tokens.append("[SEP]")
				segment_ids.append(0)
				label_ids.append(label_map["[pad]"])
				label_mask.append(0)

				input_ids = tokenizer.convert_tokens_to_ids(tokens)
				input_mask = [1] * len(input_ids)

				assert len(input_ids) == max_seq_length
				assert len(input_mask) == max_seq_length
				assert len(segment_ids) == max_seq_length
				assert len(label_ids) == max_seq_length

				feature = InputFeatures(
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=segment_ids,
					label_ids=label_ids
				)
				tf_example = convert_feature_to_tf_example(feature)
				return tf_example, label_mask	

	
	if len(tokens) > 2:
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

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		assert len(label_ids) == max_seq_length
			
		tf_example = convert_feature_to_tf_example(feature)
		
	return tf_example, label_mask

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
	"""
		Function that builds the input_fn for estimator
		Converts tf_example from input_file to tensors 
		and batches them for input_fn


	"""

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


def create_model(bert_config, is_training, features, num_labels):

	input_ids = features["input_ids"]
	input_mask = features["input_mask"]
	segment_ids = features["segment_ids"]
	labels = features["label_ids"]
	
	#Use this if using tensorflow hub
	#bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
	#						 trainable=True)
	#bert_inputs = dict(input_ids=input_ids,
	#				   input_mask=input_mask,
	#				   segment_ids=segment_ids)
	#bert_output = bert_module(
	#	inputs=bert_inputs,
	#	signature="tokens",
	#	as_dict=True)
	#output_layer = bert_output["sequence_output"]
	

	model = modeling.BertModel(
		config=bert_config,
		is_training=is_training,
		input_ids=input_ids,
		input_mask=input_mask,
		token_type_ids=segment_ids,
	)


	output_layer = model.get_sequence_output()
	output_layer = tf.layers.dropout(output_layer, rate=FLAGS.bert_dropout_rate, training=is_training)
	# dim = (batch_size, sequence_length, 768)

	hidden_size = output_layer.shape[-1].value
	seq_length = FLAGS.max_seq_length

	used = tf.sign(tf.abs(input_mask))
	lengths = tf.reduce_sum(used, reduction_indices=1)

	#BiLSTM block
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

	#Calls the BiLSTM block several times and stacks them together
	#default is 2
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

	#Project layer used to calculate the logits
	def project_layer(inputs, out_dim, seq_length, scope='project'):

		with tf.variable_scope(scope):
			in_dim = inputs.shape[-1].value
			t_output = tf.reshape(inputs, [-1, in_dim])
			output = tf.layers.dense(inputs, out_dim,
				kernel_initializer=tf.contrib.layers.xavier_initializer())
			output = tf.reshape(output, [-1, seq_length, out_dim])
			return output

	#Cross entropy or CRF loss
	def loss_layer(logits, labels, num_labels, lengths, input_mask, scope='loss'):
		
		with tf.variable_scope(scope):  
			trans = tf.get_variable(
				"transitions",
				shape=[num_labels, num_labels],
				initializer=tf.contrib.layers.xavier_initializer())
			if FLAGS.use_crf:
				tf.logging.info("*** Using CRF ***")
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
				
				#Use for weighted cross entropy loss
				#class_weights = tf.constant([0.38, 2.52, 3.08, 4.35, 2.22, 2.69])
				#label_weights =  tf.reduce_sum(class_weights * one_hot_labels,reduction_indices=2) 
				#cross_entropy = FLAGS.alpha * label_weights * cross_entropy + (1-FLAGS.alpha) * cross_entropy 	
				
				cross_entropy = -tf.reduce_sum(one_hot_labels * log_probs,reduction_indices=2)
				cross_entropy *= tf.to_float(input_mask)
				per_example_loss = tf.reduce_sum(cross_entropy, reduction_indices=1)
				per_example_loss /= tf.cast(lengths, tf.float32)
							
				loss = tf.reduce_mean(per_example_loss)
				
			return loss, per_example_loss, trans

	if FLAGS.use_lstm:
		tf.logging.info("*** Using LSTM ***")
		output1 = lstm_layer(output_layer, lengths, is_training)
	else:
		output1 = output_layer
	

	logits = project_layer(output1, num_labels, seq_length)
	loss, per_example_loss, trans = loss_layer(logits, labels, num_labels, lengths, input_mask)
	
	if FLAGS.use_crf:
		prediction, _ = tf.contrib.crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=lengths)
	
	else:
		probabilities = tf.nn.softmax(logits, axis=-1)
		prediction = tf.argmax(probabilities, axis=-1, output_type=tf.int32)

	prediction *= input_mask
	return (loss, per_example_loss, logits, prediction)

#model_fn that needs to be initialized for the estimator api
def model_fn_builder(bert_config, num_labels, init_checkpoint, starter_learning_rate, num_train_steps, num_warmup_steps):
	
	def model_fn(features, labels, mode, params):
		
		tf.logging.info("*** Features ***")
		for name in sorted(features.keys()):
			tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		
		(loss, per_example_loss, logits, predicted_labels) = create_model(
			bert_config, is_training, features, num_labels)
		
		
		#Used for initializing BERT model from the checkpoint
		tvars = tf.trainable_variables()
		if init_checkpoint:
			(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
																									   init_checkpoint)
			tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
			tf.logging.info("**** Loading from checkpoint ****")
			
		output_spec = None
		
		if mode == tf.estimator.ModeKeys.TRAIN:
			
			#learning rate: linear warmup with expontial decay
			global_step = tf.train.get_or_create_global_step()
			learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.9,   
				staircase=True)
					
			global_steps_int = tf.cast(global_step, tf.int32)
			warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
			global_steps_float = tf.cast(global_steps_int, tf.float32)
			warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
			warmup_percent_done = global_steps_float / warmup_steps_float
			warmup_learning_rate = starter_learning_rate * warmup_percent_done
			is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
			learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
			
			# Adam optimizer with correct L2 weight decay
			optimizer = optimization.AdamWeightDecayOptimizer(
				learning_rate=learning_rate,
				weight_decay_rate=0.01,
				beta_1=0.9,
				beta_2=0.999,
				epsilon=1e-6,
				exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
			
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 2.0)
			train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
			new_global_step = global_step + 1
			train_op = tf.group(train_op, [global_step.assign(new_global_step)])
			
			tf.summary.scalar('learning_rate', learning_rate)
			logging_hook = tf.train.LoggingTensorHook({"batch_loss" : loss}, every_n_iter=10)
			output_spec = tf.estimator.EstimatorSpec(mode=mode,
				loss=loss,
				train_op=train_op,
				training_hooks=[logging_hook])


		
		else: 
			
			#metric_fn that uses tf_metrics for calculating f1, precision and recall
			#for the punctuation clasess
			def metric_fn(label_ids, predicted_labels, input_mask, num_labels):
				
				label_ids = tf.boolean_mask(label_ids, input_mask)
				predicted_labels = tf.boolean_mask(predicted_labels, input_mask)
				
				precision = tf_metrics.precision(label_ids, predicted_labels, num_labels, [1,2],
												 average="macro")
				recall = tf_metrics.recall(label_ids, predicted_labels, num_labels, [1,2],
												 average="macro")
				f1 = tf_metrics.f1(label_ids, predicted_labels, num_labels, [1,2],
												 average="macro")
				
		
				return {
					"eval_precision": precision,
					"eval_recall": recall,
					"eval_f": f1
				}
			

			input_mask = features["input_mask"]
			label_ids = features["label_ids"]
			eval_metrics = metric_fn(label_ids, predicted_labels, input_mask, num_labels)
			
			
			if mode == tf.estimator.ModeKeys.EVAL:
				output_spec = tf.estimator.EstimatorSpec(mode=mode,
					loss=loss,
					eval_metric_ops=eval_metrics)
			else:
				
				predictions = {
					'labels': predicted_labels
				}
				output_spec = tf.estimator.EstimatorSpec(mode=mode,
					predictions=predictions,
					eval_metric_ops=eval_metrics)

		return output_spec
		
	return model_fn

		

def main(_):
	
	tf.logging.set_verbosity(tf.logging.INFO)
	
	#Initializing the preprocessing objecr
	processor = NerProcessor()

	#Getting config from file
	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
	
	label_list = processor.get_labels()
	
	#tokenizer from file
	tokenizer = tokenization.FullTokenizer(
		vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

	#Initial configuration for the estimator
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

	#Training
	if FLAGS.do_train:
		if len(data_config) == 0:
			train_examples = processor.get_train_examples(FLAGS.data_dir)
			train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
			num_train_size = filed_based_convert_examples_to_features(
				train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
			data_config['num_train_size'] = num_train_size
			data_config['train.tf_record_path'] = train_file
		else:
			num_train_size = data_config['num_train_size']
			train_file = data_config['train.tf_record_path']
		
		num_train_steps = int((num_train_size/FLAGS.batch_size)*FLAGS.num_epochs)
		num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
	
		num_train_steps = 1000
		num_warmup_steps = 100

	model_fn = model_fn_builder(
		bert_config=bert_config,
		num_labels=len(label_list),
		init_checkpoint=FLAGS.init_checkpoint,
		starter_learning_rate=FLAGS.learning_rate,
		num_train_steps=num_train_steps,
		num_warmup_steps=num_warmup_steps)

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		config=run_config,
		params={"batch_size":FLAGS.batch_size})

	if FLAGS.do_train:
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
			num_eval_size = filed_based_convert_examples_to_features(
					eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
			data_config['eval.tf_record_path'] = eval_file
			data_config['num_eval_size'] = num_eval_size
		else:
			eval_file = data_config['eval.tf_record_path']
			num_eval_size = data_config['num_eval_size']
		
		tf.logging.info("***** Running evaluation *****")
		tf.logging.info("  Num examples = %d", num_eval_size)
		tf.logging.info("  Batch size = %d", FLAGS.batch_size)
		
		eval_input_fn = file_based_input_fn_builder(
			input_file=eval_file,
			seq_length=FLAGS.max_seq_length,
			is_training=False,
			drop_remainder=False)

		#Dumping the data_config so that the model 
		#can read the serialized data directly
		if not os.path.exists(FLAGS.data_config_path):
			with open(FLAGS.data_config_path, 'a') as fd:
				json.dump(data_config, fd)
		
		# train and evaluate 
		#hook = tf.contrib.estimator.stop_if_no_decrease_hook(
		#   estimator, 'eval_f', 3000, min_steps=30000, run_every_secs=360)
		#train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[hook])
		train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
		eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=120)
		tp = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
		result = tp[0]
		
		output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
		#Writing the results of the validation set
		if result:
			with open(output_eval_file, "w") as writer:
				tf.logging.info("***** Eval results *****")
				for key in sorted(result.keys()):
					tf.logging.info("  %s = %s", key, str(result[key]))
					writer.write("%s = %s\n" % (key, str(result[key])))

	

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

		
		predicted_result = estimator.predict(input_fn=predict_input_fn)
		for x in predicted_result:
			print(x)
		output_eval_file = os.path.join(FLAGS.output_dir, "predicted_results.txt")
		#Writing the results of the validation set
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
	


		








	
						
		
		


		
	



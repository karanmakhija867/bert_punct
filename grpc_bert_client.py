from __future__ import print_function

import argparse 
import time

import tensorflow as tf
import grpc
import bert
from bert import tokenization
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from bert_SUD import NerProcessor, InputFeatures, convert_feature_to_tf_example, convert_example_to_features

from flask import Flask, render_template,request, jsonify
import numpy as np
import re

VOCAB_FILE = "uncased_L-24_H-1024_A-16/vocab.txt"
UNCASED = True

app = Flask(__name__)

def process_input(input_sen):
	regex = re.compile('[%s]'% re.escape('.,?'))
	new_input = regex.sub('',input_sen)
	return new_input.lower().strip()

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/punctuate',methods=['POST'])
def punctuate():
	
	text = request.form.get('input_text',0,type=str)
	output_sent = run(text)
	data = jsonify(result=output_sent)
	return data



def generate_output(words,predictions):

	"""
		Generate the output in the required format
	"""

	idx2Punc = dict()
	idx2Punc[1] = '.'
	idx2Punc[2] = ','
	idx2Punc[3] = '?'

	output = ""
	for i in range(0,len(words)):
		punc = predictions[i]
		if (i==0 or output[-2]=='.' or output[-2]=='?') and punc in idx2Punc:
			output += words[i].capitalize()+idx2Punc[punc]+" "
		elif i==0 or output[-2]=='.' or output[-2]=='?' :
			output += words[i].capitalize()+" "
		elif punc in idx2Punc:
			output += words[i]+idx2Punc[punc]+" "
		else:
			output += words[i]+" "

	return output

def run(input_sent, host='0.0.0.0', port='8500', model='bert_punct', signature_name='serving_default'):
	
	channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
	
	processor = NerProcessor()
	label_list = processor.get_labels()

	new_input = process_input(input_sent)
	num_words = len(new_input.split())
	label = ''
	for i in range(0, num_words-1):
		label += '0 '
	label += '0'
	
	data = {}
	data['sentence'] = [new_input]
	data['label'] = [label]
	

	#data['sentence'] = ['hello and welcome to the book quiz the final contest in this series we started with eight teams of famous faces and now we\'re left with the best two who have gamely battled their way through some pretty tough questions to get here in just under half an hour i\'ll be declaring one team the series champions our first finalists defeated miranda sawyer and anthony horowitz in their opening match and then margaret jay and david nicholls in their semi final amassing a total of one hundred ten points in the two matches despite lots of squabbling wendy and giles got their act together in the final round so please welcome back wendy holden and giles coren their opponents got here by knocking out simon hoggart and al kennedy in their first match and india knight and james delingpole in the semi final with a combined score of one hundred nineteen points last time david was marvellous on marlow whilst daisy managed to identify for whom the bell tolls in the picture round without having the faintest idea how she did it or indeed why she did it please welcome the other finalists david aaronovitch and daisy goodman both teams should know the drill by now but i\'ll just remind them that for their opening round they\'ll get two works of literature to identify and then i\'ll ask them to make a connection between the two two points for a correct answer if you get it wrong i hand it to the opposing team for a single point first quotation goes to wendy and giles it\'s from a work first published in nineteen eighty four i want you to tell me which novelist wrote this']
	#data['label'] = ['2 0 0 0 0 0 2 0 0 0 0 0 1 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 2 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 2 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 1 0 2 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 2 0 2 0 0 0 1 0 0 0 0 2 0 0 0 0 1 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0']

	max_seq_length = 128
	inputExample = processor._create_example(data, 'test')
	
	prediction = []
	len_example = len(data['sentence'][0].split())
	start_index = 0 
	while start_index < len_example:
		tf_example, label_mask = convert_example_to_features(inputExample[0], label_list, max_seq_length, tokenizer, start_index)
		model_input = tf_example.SerializeToString()
		model_request = predict_pb2.PredictRequest()
		model_request.model_spec.name = model
		model_request.model_spec.signature_name = signature_name

		dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
		tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
		tensor_proto = tensor_pb2.TensorProto(
			dtype=types_pb2.DT_STRING,
			tensor_shape=tensor_shape_proto,
			string_val=[model_input])

		model_request.inputs['examples'].CopyFrom(tensor_proto)
		result = stub.Predict(model_request, 10.0)
		
		result = tf.make_ndarray(result.outputs['labels'])
		masks = np.array(label_mask)
		new_result= result[0][masks==1]
		
		indices = np.sort(np.append(np.where(new_result==1),np.where(new_result==3)))
		if len(indices) == 0:
			index = len(new_result)
		else: 
			index = indices[-1]+1   
		prediction += new_result[:index].tolist()
		start_index += index

	words = data['sentence'][0].split()
	output = generate_output(words, prediction)
	return output



	
if __name__ == '__main__':
	
	tokenizer = tokenization.FullTokenizer(
		vocab_file=VOCAB_FILE, do_lower_case=UNCASED)
	app.run(debug=True)







	


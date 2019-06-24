

## Deploying model for punctuation detection
Take the files and folders from export directory. This includes:

 - assets
 - variables
 - saved_model.pb
 
 Create the following file structure **models_for_serving/bert_punct/1** and store the files in that folder.
 > mkdir -p ./model_for_serving/bert_punct/1
 
 > cp -R ./export_model/1529121297/ ./models_for_serving/bert_punct/1

Ensure you have docker and clone the tensorflow serving docker image 
>docker pull tensorflow/serving

To run the model server for serving execute the following command
>sudo docker run --rm  -v ${PWD}/models_for_serving:/models   -e MODEL_NAME='bert_punct' -p 8500:8500  --name punct tensorflow/serving 
 
To run the gRPc client execute 
>python grpc_bert_client.py

This will create flask app that can be accessed on **http://localhost:5000**






from src.bert_te import BertArgumentStructure
from src.data import Data
from src.utility import handle_errors 

from src.loading_utils import load_model

from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
import logging


logging.basicConfig(datefmt='%H:%M:%S', level=logging.DEBUG)


app = Flask(__name__)
metrics = PrometheusMetrics(app)
model = load_model(config_file_path="config/config.json")

@metrics.summary('requests_by_status', 'Request latencies by status',
                 labels={'status': lambda r: r.status_code})
@metrics.histogram('requests_by_status_and_path', 'Request latencies by status and path',
                   labels={'status': lambda r: r.status_code, 'path': lambda: request.path})
@handle_errors
@app.route('/bert-te', methods = ['GET', 'POST'])
def bertte():
	if request.method == 'POST':
		file_obj = request.files['file']
		# data = Data(file_obj)
		result = BertArgumentStructure(
			file_obj=file_obj, model=model
		).get_argument_structure()

		return result
	
	if request.method == 'GET':
		info = """The Inference Identifier is a component of AMF that detects argument relations between propositions. 
		This implementation utilises the Hugging Face implementation of BERT for textual entailment. 
		The model is fine-tuned to recognize inferences, conflicts, and non-relations. 
		It accepts xIAF as input and returns xIAF as output. 
		This component can be integrated into the argument mining pipeline alongside a segmenter."""
		return info


@handle_errors
@app.route('/bert-te-from-json-to-json', methods=['GET', 'POST'])
def bertte_from_json():
	if request.method == 'POST':
		xaif_dict = request.json

		result = BertArgumentStructure(
			file_obj=None, model=model
		).get_argument_structure_from_json(xaif_dict=xaif_dict)

		return result

	if request.method == 'GET':
		info = """The Inference Identifier is a component of AMF that detects argument relations between propositions. 
		This implementation utilises the Hugging Face implementation of BERT for textual entailment. 
		The model is fine-tuned to recognize inferences, conflicts, and non-relations. 
		It accepts xIAF as input and returns xIAF as output. 
		This component can be integrated into the argument mining pipeline alongside a segmenter."""
		return info

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5002"), debug=False)	  

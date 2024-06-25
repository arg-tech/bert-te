
from src.bert_te import BertArgumentStructure 
from src.utility import handle_errors 
from src.models import Model

from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
import logging
import json




logging.basicConfig(datefmt='%H:%M:%S', level=logging.DEBUG)

def load_config(file_path):
        """Load the contents of the config.json file to get the model files."""
        with open(file_path, 'r') as config_file:
            config_data = json.load(config_file)
            return config_data.get('model_path')

app = Flask(__name__)
metrics = PrometheusMetrics(app)

config_file_path = "config/config.json"
model_path = load_config(config_file_path)
model = Model(model_path)

@metrics.summary('requests_by_status', 'Request latencies by status',
                 labels={'status': lambda r: r.status_code})
@metrics.histogram('requests_by_status_and_path', 'Request latencies by status and path',
                   labels={'status': lambda r: r.status_code, 'path': lambda: request.path})
@handle_errors
@app.route('/bert-te', methods = ['GET', 'POST'])
def bertte():
	if request.method == 'POST':
		file_obj = request.files['file']
		result = BertArgumentStructure(file_obj,model).get_argument_structure()

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

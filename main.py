
from src.bert_te import get_argument_structure
from src.data import Data
from src.utility import handle_errors 

from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
import logging
logging.basicConfig(datefmt='%H:%M:%S',
                    level=logging.DEBUG)


app = Flask(__name__)
metrics = PrometheusMetrics(app)
@app.route('/propositionUnitizer-01', methods = ['GET', 'POST'])
@metrics.summary('requests_by_status', 'Request latencies by status',
                 labels={'status': lambda r: r.status_code})
@metrics.histogram('requests_by_status_and_path', 'Request latencies by status and path',
                   labels={'status': lambda r: r.status_code, 'path': lambda: request.path})
@handle_errors
@app.route('/bert-te', methods = ['GET', 'POST'])
def bertte():
	if request.method == 'POST':
		file_obj = request.files['file']
		data = Data(file_obj)
		result=get_argument_structure(file_obj)
		return result
	
	if request.method == 'GET':
		info = """The Inference Identifier is a component of AMF that detects argument relations between propositions. 
		This implementation utilizes the Hugging Face implementation of BERT for textual entailment. The model is fine-tuned to recognize inferences, conflicts, and non-relations. 
		It accepts xIAF as input and returns xIAF as output. 
		This component can be integrated into the argument mining pipeline alongside a segmenter."""
		return info	
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5002"), debug=False)	  

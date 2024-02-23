import json
from src.models import Model
from src.data import Data, AIF
from src.templates import BertTEOutput

class BertArgumentStructure:
	def __init__(self,file_obj):
		self.file_obj = file_obj
		self.config_file_path = "'config/config.json'"
		self.model_path = self.load_config(self.config_file_path)
		self.model = Model(self.model_path)

	def load_config(self, file_path):
		"""Load the contents of the config.json file."""
		with open(file_path, 'r') as config_file:
			config_data = json.load(config_file)
			return config_data.get('model_path')

	def get_argument_structure(self, ):
		data = Data(self.file_obj)
		if not data.is_valid_json(): 
			return "Invalid input"
		
		x_aif = data.get_aif()
		aif = x_aif.get('AIF', {})
		nodes, edges = aif.get('nodes', []), aif.get('edges', [])		
		if 'nodes' not in aif or 'locutions' not in aif or 'edges' not in aif:
			return "Invalid json-aif"		
		propositions_id_pairs = {}
		for node in nodes:
			if node.get('type') == "I":
				proposition = node.get('text', '').strip()
				if proposition:
					node_id = node.get('nodeID')
					propositions_id_pairs[node_id] = proposition
        
		checked_pairs = set()
		for prop1_node_id, prop1 in propositions_id_pairs.items():
			for prop2_node_id, prop2 in propositions_id_pairs.items():
				if prop1_node_id != prop2_node_id:
					pair1 = (prop1_node_id, prop2_node_id)
					pair2 = (prop2_node_id, prop1_node_id)
					if pair1 not in checked_pairs and pair2 not in checked_pairs:
						checked_pairs.add(pair1)
						checked_pairs.add(pair2)
						prediction = self.model.predict((prop1, prop2))
						nodes, edges = AIF.create_entry(nodes, edges, prediction, prop1_node_id, prop2_node_id)
		return BertTEOutput.format_output(nodes, edges, x_aif, aif)






  






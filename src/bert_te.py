
import json

from src.models import Model
from src.utility import get_next_max_id
from src.data import Data, AIF
from src.templates import BertTEOutput




# Load the contents of the config.json file
with open('config/config.json', 'r') as config_file:
    config_data = json.load(config_file)
model_path = config_data['model_path']
model = Model(model_path)
	
def get_argument_structure(file_obj):
	data = Data(file_obj)
	if  data.is_valid_json(): 
		extended_json_aif = data.get_aif()
		json_dict = extended_json_aif['AIF']
		if 'nodes' in json_dict and 'locutions' in json_dict and 'edges' in json_dict:
			nodes, edges = json_dict['nodes'], json_dict['edges']
			propositions_all, propositions_id = [], {}
			for nodes_entry in nodes:
				original_node_id = nodes_entry['nodeID']
				if nodes_entry['type'] == "I":
					proposition = nodes_entry['text'].strip()								
					if proposition != "":
						if proposition not in propositions_all:
							propositions_all.append(proposition)
							propositions_id.update({original_node_id:proposition})							
			connected_propositions_dict = {}
			for proposition1_node_ID, prop1 in propositions_id.items():
				for  proposition2_node_ID, prop2 in propositions_id.items():
					if proposition1_node_ID != proposition2_node_ID:
						key1 = f"{proposition1_node_ID}and{proposition2_node_ID}"
						key2 = f"{proposition2_node_ID}and{proposition1_node_ID}"						
						if key1 not in connected_propositions_dict and key2 not in connected_propositions_dict:
							connected_propositions_dict[key1] = key2
							connected_propositions_dict[key2] = key1
							proposition_pairs = (prop1, prop2)
						prediction = model.predict(proposition_pairs)
						nodes, edges = AIF.create_entry(nodes, edges, prediction, proposition1_node_ID, proposition2_node_ID)
			return BertTEOutput.format_output(nodes, edges, json_dict, extended_json_aif)

		else:
			return("Invalid json-aif")
	else:
		return("Invalid input")





  

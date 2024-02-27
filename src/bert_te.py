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
        """Load the contents of the config.json file to get the model files."""
        with open(file_path, 'r') as config_file:
            config_data = json.load(config_file)
            return config_data.get('model_path')

    def get_argument_structure(self):
        """Retrieve the argument structure from the input data."""
        data = self.get_json_data()
        if not data:
            return "Invalid input"
        
        x_aif = data.get_aif()
        aif = x_aif.get('AIF', {})
        if not self.is_valid_aif(aif):
            return "Invalid json-aif"

        propositions_id_pairs = self.get_propositions_id_pairs(aif)
        self.update_node_edge_with_relations(propositions_id_pairs, aif)

        return self.format_output(x_aif, aif)

    def get_json_data(self):
        """Retrieve JSON data from the file."""
        data = Data(self.file_obj)
        return data if data.is_valid_json() else None

    def is_valid_aif(self, aif):
        """Check if the AIF data is valid."""
        return 'nodes' in aif and 'edges' in aif

    def get_propositions_id_pairs(self, aif):
        """Extract proposition ID pairs from the AIF data."""
        propositions_id_pairs = {}
        for node in aif.get('nodes', []):
            if node.get('type') == "I":
                proposition = node.get('text', '').strip()
                if proposition:
                    node_id = node.get('nodeID')
                    propositions_id_pairs[node_id] = proposition
        return propositions_id_pairs
    
    def update_node_edge_with_relations(self, propositions_id_pairs, aif):
        """
        Update the nodes and edges in the AIF structure to reflect the new relations between propositions.
        """
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
                        AIF.create_entry(aif['nodes'], aif['edges'], prediction, prop1_node_id, prop2_node_id)

    def format_output(self, x_aif, aif):
        """Format the output data."""
        return BertTEOutput.format_output(x_aif['AIF']['nodes'], x_aif['AIF']['edges'], x_aif, aif)

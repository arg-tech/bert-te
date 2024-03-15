from src.data import Data, AIF

from itertools import combinations


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

class BertArgumentStructure:
    def __init__(self,file_obj, model, batch_size=8):
        self.file_obj = file_obj

        self.model = model
        self.batch_size = batch_size

    def get_argument_structure_from_json(self, xaif_dict):
        aif = xaif_dict.get('AIF', {})

        propositions_id_pairs = self.get_propositions_id_pairs(aif)
        self.update_node_edge_with_relations(propositions_id_pairs, aif)

        return self.format_output(xaif_dict, aif)


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

        node_ids_combs = list(combinations(
            list(propositions_id_pairs.keys()), 2
        ))


        for batch_node_pairs in divide_chunks(node_ids_combs, self.batch_size):

            batch_proposition_pairs = [
                [propositions_id_pairs[node_id_1], propositions_id_pairs[node_id_2]]
                for node_id_1, node_id_2 in batch_node_pairs
            ]
            batch_preds = self.model.predict_pairs_batch(
                proposition_pairs=batch_proposition_pairs
            )

            for node_ids_pair, prediction in zip(batch_node_pairs, batch_preds):
                AIF.create_entry(
                    aif['nodes'], aif['edges'],
                    prediction, node_ids_pair[0], node_ids_pair[1]
                )

    def format_output(self, x_aif, aif):
        """Format the output data."""

        xaif_output = {}
        xaif_output["nodes"] = x_aif['AIF']['nodes'].copy()
        xaif_output["edges"] = x_aif['AIF']['edges'].copy()
        xaif_output["AIF"] = aif.copy()
        return xaif_output

        # return BertTEOutput.format_output(x_aif['AIF']['nodes'], x_aif['AIF']['edges'], x_aif, aif)

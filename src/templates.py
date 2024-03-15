import json

class BertTEOutput:
    @staticmethod
    def format_output(nodes, edges, aif={}, x_aif={}):

        aif['nodes'] = nodes
        aif['edges'] = edges
        x_aif['AIF'] = aif.copy()
        return json.dumps(x_aif)


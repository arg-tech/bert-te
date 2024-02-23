
from transformers import BartForSequenceClassification, BartTokenizer

class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForSequenceClassification.from_pretrained(model_path)
        self.RA_TRESHOLD = 80
        self.CA_TRESHOLD = 10

    def predict(self, proposition_pair):
        proposition1, proposition2 = proposition_pair
        return self._post_process(proposition1, proposition2)

    def _get_prob(self, text1, text2):
        input_ids = self.tokenizer.encode(text1, text2, return_tensors='pt')
        logits = self.model(input_ids)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_prob = probs[:, 1].item() * 100
        return true_prob

    def _post_process(self, text1, text2):
        true_prob, arg_rel = 0.0, "None"
        true_prob1 = self._get_prob(text1, text2)
        true_prob2 = self._get_prob(text2, text1)
        true_prob = max(true_prob1, true_prob2)
        
        if true_prob > self.RA_THRESHOLD:
            arg_rel = "RA"
        elif true_prob < self.CA_THRESHOLD:
            arg_rel = "CA"
        else:
            arg_rel = "None"

        return arg_rel


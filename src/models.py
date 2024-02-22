
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

    def _post_process(self, text1, text2):
        arg_rel = "None"
        total_entailment2 = []
        true_prob = 0.0

        input_ids = self.tokenizer.encode(text1, text2, return_tensors='pt')
        logits = self.model(input_ids)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_prob1 = probs[:, 1].item() * 100
        input_ids = self.tokenizer.encode(text2, text1, return_tensors='pt')
        logits = self.model(input_ids)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_prob2 = probs[:, 1].item() * 100
        true_prob = max(true_prob1, true_prob2)
        total_entailment2.append(true_prob)
        if total_entailment2[0] > self.RA_TRESHOLD:
            arg_rel = "RA"
        elif total_entailment2[0] < self.CA_TRESHOLD :
            arg_rel = "CA"

        return arg_rel


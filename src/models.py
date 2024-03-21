from transformers import BartForSequenceClassification, BartTokenizer
from typing import List, Tuple
from transformers import BatchEncoding

class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForSequenceClassification.from_pretrained(model_path)
        self.RA_THRESHOLD = 80
        self.CA_THRESHOLD = 10

    def predict_pair(self, proposition_pair):
        proposition1, proposition2 = proposition_pair
        return self._post_process(proposition1, proposition2)


    def _tokenize_pairs_batch(self, proposition_pairs: List[List[str]]) -> Tuple[BatchEncoding]:

        # tokenize in provided order: [[t1, t2], [t3, t4]]
        encoded_input_original_pair_order = self.tokenizer.batch_encode_plus(
            proposition_pairs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # tokenize in reversed order of text pairs:  [[t2, t1], [t4, t3]]
        reversed_order_proposition_pairs = [
            text_pair[::-1] for text_pair in proposition_pairs
        ]
        encoded_input_reversed_pair_order = self.tokenizer.batch_encode_plus(
            reversed_order_proposition_pairs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        return encoded_input_reversed_pair_order, encoded_input_original_pair_order

    def _make_decision(self, prob: float) -> str:

        if prob > self.RA_THRESHOLD:
            arg_rel = "RA"
        elif prob < self.CA_THRESHOLD:
            arg_rel = "CA"
        else:
            arg_rel = "None"

        return arg_rel

    def predict_pairs_batch(self, proposition_pairs: List[List[str]]) -> List[str]:

        encoded_input_reversed_pair_order, encoded_input_original_pair_order = self._tokenize_pairs_batch(
            proposition_pairs=proposition_pairs
        )

        preds_reversed_pair_order = self.model(**encoded_input_reversed_pair_order)
        preds_original_pair_order = self.model(**encoded_input_original_pair_order)

        entail_contradiction_logits_reversed_pair_order = preds_reversed_pair_order.logits[:, [0, 2]].softmax(dim=1)
        entail_contradiction_logits_original_pair_order = preds_original_pair_order.logits[:, [0, 2]].softmax(dim=1)

        true_probs_reversed_pair_order = entail_contradiction_logits_reversed_pair_order[:, 1] * 100
        true_probs_original_pair_order = entail_contradiction_logits_original_pair_order[:, 1] * 100

        max_true_probs = [
            max([prob_rev, prob_orig]) for prob_rev, prob_orig in zip(
                true_probs_reversed_pair_order, true_probs_original_pair_order
            )
        ]

        preds_decisions = [
            self._make_decision(prob) for prob in max_true_probs
        ]

        return preds_decisions





    def _get_prob(self, text1, text2):
        input_ids = self.tokenizer.encode(text1, text2, return_tensors='pt')
        logits = self.model(input_ids)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_prob = probs[:, 1].item() * 100
        return true_prob

    def _post_process(self, text1, text2):

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


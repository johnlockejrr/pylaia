from typing import Any, Dict, List

import numpy as np
import torch
from torchaudio.models.decoder import ctc_decoder

from laia.losses.ctc_loss import transform_batch


class CTCLanguageDecoder:
    """
    Initialize a CTC decoder with n-gram language modeling.
    Args:
        language_model_path (str): path to a KenLM or ARPA language model
        lexicon_path (str): path to a lexicon file containing the possible words and corresponding spellings.
            Each line consists of a word and its space separated spelling. If `None`, uses lexicon-free
            decoding.
        tokens_path (str): path to a file containing valid tokens. If using a file, the expected
            format is for tokens mapping to the same index to be on the same line
        language_model_weight (float): weight of the language model.
        blank_token (str): token representing the blank/ctc symbol
        unk_token (str): token representing unknown characters
        sil_token (str): token representing the space character
    """

    def __init__(
        self,
        language_model_path: str,
        lexicon_path: str,
        tokens_path: str,
        language_model_weight: float = 1.0,
        blank_token: str = "<ctc>",
        unk_token: str = "<unk>",
        sil_token: str = "<space>",
        temperature: float = 1.0,
    ):
        self.decoder = ctc_decoder(
            lm=language_model_path,
            lexicon=lexicon_path,
            tokens=tokens_path,
            lm_weight=language_model_weight,
            blank_token=blank_token,
            unk_word=unk_token,
            sil_token=sil_token,
            nbest=10,
        )
        self.temperature = temperature
        self.language_model_weight = language_model_weight

    def __call__(
        self,
        features: Any,
    ) -> Dict[str, List]:
        """
        Decode a feature vector using n-gram language modelling.
        Args:
            features (Any): feature vector of size (n_frame, batch_size, n_tokens).
                Can be either a torch.tensor or a torch.nn.utils.rnn.PackedSequence
        Returns:
            out (Dict[str, List]): a dictionary containing the hypothesis (the list of decoded tokens).
                There is no character-based probability.
        """

        # Get the actual size of each feature in the batch
        batch_features, batch_sizes = transform_batch(features)
        batch_features = batch_features.detach()

        # Reshape from (n_frame, batch_size, n_tokens) to (batch_size, n_frame, n_tokens)
        batch_features = batch_features.permute((1, 0, 2))

        # Apply temperature scaling
        batch_features = batch_features / self.temperature

        # Apply log softmax
        batch_features = torch.nn.functional.log_softmax(batch_features, dim=-1)

        # No GPU support for torchaudio's ctc_decoder
        device = torch.device("cpu")
        batch_features = batch_features.to(device)
        if isinstance(batch_sizes, list):
            batch_sizes = torch.tensor(batch_sizes)
            batch_sizes.to(device)

        # Decode
        hypotheses = self.decoder(batch_features, batch_sizes)

        # Format the output
        out = {}
        out["hyp"] = [hypothesis[0].tokens.tolist() for hypothesis in hypotheses]

        # Normalize confidence score
        # out["prob-htr"] = [
        #     CTCLanguageDecoder.compute_scores_from_initial_matrix(
        #         tokens=hypothesis[0].tokens.tolist(),
        #         timesteps=hypothesis[0].timesteps.tolist(),
        #         probs=features,
        #     )
        #     for hypothesis, features in zip(hypotheses, batch_features)
        # ]
        out["prob-htr"] = [
            CTCLanguageDecoder.compute_scores_from_nbest(hypothesis)
            for hypothesis in hypotheses
        ]
        return out

    @staticmethod
    def compute_scores_from_initial_matrix(tokens, timesteps, probs):
        """
        Compute confidence scores using probabilities from PyLaia

        :param tokens: tokens from CTCHypothesis
        :param timesteps: timesteps from CTCHypothesis
        :param probs: initial probability matrix
        :return: mean confidence score
        """
        sequence_prob = []
        for token, timestep in zip(tokens[1:-1], timesteps[1:-1]):
            sequence_prob.append(probs[timestep - 1, token].exp())
        return np.mean(sequence_prob)

    @staticmethod
    def compute_scores_from_nbest(nbest_hypothesis):
        """
        Compute confidence scores using n best hypotheses

        :param nbest_hypothesis: a CTCHypothesis object with nbest>=10
        :return: confidence score
        """
        nbest_scores = [np.exp(hyp.score) for hyp in nbest_hypothesis]
        return nbest_scores[0] / sum(nbest_scores)

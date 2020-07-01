import pickle
from typing import List, Dict, Tuple

import numpy

from configs import PreprocessingConfig
from dataset import Vocabulary
from utils.common import PAD, SOS, EOS, FROM_TOKEN, PATH_TYPES, TO_TOKEN


class BufferedPathContext:
    """Class for storing buffered path contexts
    labels: numpy array [max_target_parts + 1; buffer size]
    from\to_tokens: numpy array [max_name_parts + 1; buffer_size * paths_for_labels (unique per sample)]
    path_types: numpy array [max_path_length + 1; buffer_size * paths_for_labels (unique per sample)]
    paths_for_labels: list [buffer size] -- number of paths for each label

    +1 for SOS token, put EOS if enough space
    """

    def __init__(
        self,
        config: PreprocessingConfig,
        vocab: Vocabulary,
        labels: List[List[int]],
        from_tokens: List[List[List[int]]],
        path_types: List[List[List[int]]],
        to_tokens: List[List[List[int]]],
    ):
        if not (len(from_tokens) == len(path_types) == len(to_tokens)):
            raise ValueError(f"Unequal sizes of array with path parts")
        if len(labels) != len(from_tokens):
            raise ValueError(f"Number of labels is different to number of paths")

        self.paths_for_label = [len(pc) for pc in from_tokens]
        self._end_idx = numpy.cumsum(self.paths_for_label).tolist()
        self._start_idx = [0] + self._end_idx[:-1]

        buffer_size = len(labels)
        self.labels = numpy.empty((config.max_target_parts + 1, buffer_size), dtype=numpy.int)
        self.from_tokens = numpy.empty((config.max_name_parts + 1, self._end_idx[-1]), dtype=numpy.int)
        self.path_types = numpy.empty((config.max_path_length + 1, self._end_idx[-1]), dtype=numpy.int)
        self.to_tokens = numpy.empty((config.max_name_parts + 1, self._end_idx[-1]), dtype=numpy.int)

        cur_path_idx = 0
        for sample in range(buffer_size):
            self.labels[:, sample] = self._prepare_to_store(labels[sample], config.max_target_parts, vocab.label_to_id)
            for ft, pt, tt in zip(from_tokens[sample], path_types[sample], to_tokens[sample]):
                self.from_tokens[:, cur_path_idx] = self._prepare_to_store(ft, config.max_name_parts, vocab.token_to_id)
                self.path_types[:, cur_path_idx] = self._prepare_to_store(pt, config.max_path_length, vocab.type_to_id)
                self.to_tokens[:, cur_path_idx] = self._prepare_to_store(tt, config.max_name_parts, vocab.token_to_id)
                cur_path_idx += 1

    @staticmethod
    def _prepare_to_store(values: List[int], max_len: int, to_id: Dict) -> List[int]:
        used_len = min(len(values), max_len)
        result = [to_id[SOS]] + values[:used_len]
        if used_len < max_len:
            result.append(to_id[EOS])
            used_len += 1
        result += [to_id[PAD]] * (max_len - used_len)
        return result

    def dump(self, path: str):
        with open(path, "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    def __len__(self):
        return len(self.paths_for_label)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, numpy.ndarray], numpy.ndarray, int]:
        path_slice = slice(self._start_idx[idx], self._end_idx[idx])
        return (
            {
                FROM_TOKEN: self.from_tokens[:, path_slice],
                PATH_TYPES: self.path_types[:, path_slice],
                TO_TOKEN: self.to_tokens[:, path_slice],
            },
            self.labels[:, [idx]],
            self.paths_for_label[idx],
        )
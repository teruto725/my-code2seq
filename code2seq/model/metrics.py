from torchmetrics import Metric
from nltk import bleu_score
import torch
import numpy as np
import textdistance


class Perfect(Metric):
    """
    完全一致
    """
    def __init__(self, dist_sync_on_step = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.correct += torch.sum(torch.prod(preds == target, axis = 0))
        self.total += preds.shape[1]
    def compute(self):
        return self.correct.float() / self.total



class BLEU(Metric):
    """
    BLEU4のスコア
    """
    def __init__(self, dist_sync_on_step = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        catted = torch.stack([preds,target]).cpu().numpy().astype(int).astype(str)
        stack = 0
        for i in range(catted.shape[2]):# batch size
            stack+=( self._calc_bleu(catted[:,:,i]))
            
        self.correct += torch.tensor(stack, dtype=torch.int)
        self.total += preds.shape[1]
        
        
    def compute(self):
        return self.correct.float() / self.total
    

    def _calc_bleu(self,col:np.ndarray):
        score = bleu_score.sentence_bleu([col[0].tolist()], col[1].tolist(), smoothing_function=bleu_score.SmoothingFunction().method1)
        return score


class Leivensitein(Metric):
    """
    Leivensitein
    """
    def __init__(self, dist_sync_on_step = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        catted = torch.stack([preds,target]).cpu().numpy().astype(int)
        
        stack = 0
        for i in range(catted.shape[2]):# batch size
            stack+=( self._calc_score(catted[:,:,i]))
            
        self.correct += torch.tensor(stack, dtype=torch.int)
        self.total += preds.shape[1]
        
        
    def compute(self):
        return self.correct.float() / self.total
    

    def _calc_score(self,col:np.ndarray):
        s1 = col[0].tolist()
        s2 = col[1].tolist()
        score = textdistance.levenshtein.distance(s1,s2)/max(len(s1),len(s2))
        return score
    
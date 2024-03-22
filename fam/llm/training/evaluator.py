import numpy as np
import fam.llm.training.dist as dist_utils

class Evaluator(object):
    def __init__(self):
        self.reset()

    def update(self, results):
        for k in self.fields:
            self.results[k] += results[k]

    def synchronize_between_processes(self):
        all_results = dist_utils.all_gather(self.results)
        merged_results = {}
        for r in all_results:
            for k in r.keys():
                merged_results.setdefault(k, [])
                merged_results[k] += r[k]
        for k in merged_results.keys():
            merged_results[k] = np.array(merged_results[k])
        self.results = merged_results

    def summarize(self):
        #! With multi-gpu, the dataloader duplicates examples if the number of examples is not divisible by the batch size.
        if dist_utils.is_main_process():
            return {k: np.mean(self.results[k]) for k in self.fields}

        else:
            assert False, "This if function should not be called."

    def reset(self):        
        self.results = {}
        self.fields = [
            "loss",
        ]

        for f in self.fields:
            self.results.setdefault(f, [])

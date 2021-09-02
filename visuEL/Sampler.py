import math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.neighbors import NearestNeighbors
from pm4py.objects.log.importer.xes import importer as xes_importer

class CminSampler:
    def __init__(self, p=0, heuristic_presampling=5000, random_seed=1):
        """
        Reduce the size of an event logs, while making sure that the selected traces are representatives
        :param seqs: list of list representing the event logs (e.g., [[1,2],[1,2,3],[1,2]]
        :param p: Size of the event logs once sampled. If set to 1, the size is automatically choosen as a function of the original size
        :param heuristic_presampling: Apply a random sampling (proportional to variants' size) to make the process faster.
        :param random_seed: Given the same seed and identical event logs, make sure the sampling, forces the same output
        """
        self.p = p
        self.heuristic_presampling = heuristic_presampling
        self.random_seed = random_seed
        self.seqs, self.variants, self.multiplicity, self.capacity = [None]*4

    def load_from_seq(self, seqs):
        self._load([[str(y) for y in x] for x in seqs])

    def load_from_xes(self, path):
        log = xes_importer.apply(path)
        self.load_from_pm4py(log)

    def load_from_pm4py(self, log_object):
        seq = [[e['concept:name'] for e in trace] for trace in log_object]
        self.load_from_seq(seq)

    def load_from_df(self, df, case_col, activity_col):
        seq = df.groupby(case_col)[activity_col].agg(list).tolist()
        self.load_from_seq(seq)

    def _load(self, seqs):
        self.seqs = seqs
        self.variants = pd.Series(self.seqs).value_counts()

        # If size is not reasonable or empty, we automatically choose the size
        if self.p<1 or self.p>self.variants.shape[0] or not self.p:
            self.p = self.auto_size()

        # Count per variants
        self.multiplicity = self.variants.values

        # n. traces assigned to each rep'
        self.capacity = math.floor(self.multiplicity.sum()/self.p)

        # Expected Occurrence Reduction reduces the number of operations needed (see paper)
        ratio = self.multiplicity.sum()/self.p
        count = self.multiplicity / ratio
        self.eo_counts = np.floor(count) # Add frequently appearing variants
        self.multiplicity = ((count - self.eo_counts) * ratio).round(0).astype(int) # Update the multiplicity

    def auto_size(self):
        """
        Define p based on the number of traces
        """
        return int(math.ceil(math.log(float(self.variants.sum()),1.5)))

    def sample(self):
        """
        Select the p most representative traces
        """
        if len(self.seqs) == 1:
            return self.seqs
        sampler = self.samplingWithEucl()
        seqs = [y for i, x in enumerate(sampler) if x > 0 for y in [self.variants.index.tolist()[i]]*int(x)]

        return seqs

    def samplingWithEucl(self):
        """
        Run the sampling algorithm in the Euclidean space
        """

        data = self.buildSignature()
        data = data.repeat(self.multiplicity, axis=0)

        original_seq_index = np.arange(self.multiplicity.shape[0]).repeat(self.multiplicity)
        not_assigned = np.ones(data.shape[0]).astype(bool)

        if self.heuristic_presampling:
            if data.shape[0] > self.heuristic_presampling:
                np.random.seed(self.random_seed)
                r = np.random.choice(not_assigned.shape[0], not_assigned.shape[0] - self.heuristic_presampling, replace=False)
                not_assigned[r] = False
                self.capacity = math.floor(self.heuristic_presampling / self.p)

        output_count = []
        while len(output_count) != self.p - self.eo_counts.sum():
            i_not_assigned = np.where(not_assigned==True)[0]
            ldata = data[not_assigned,:]

            if ldata.shape[0] > 2:
                neigh = NearestNeighbors(n_neighbors=min(self.capacity,ldata.shape[0]))
                neigh.fit(ldata)
                D, I = neigh.kneighbors(ldata, return_distance=True)
                best_id = D.sum(axis=1).argmin()
                closest_to_best = I[best_id,:]

                output_count.append(original_seq_index[i_not_assigned[best_id]])
                not_assigned[i_not_assigned[closest_to_best]] = False
            else:
                # Add random seq (because choosing the most central from less than 3 obs is not possible)
                missing = self.p - len(output_count) - self.eo_counts.sum()
                np.random.seed(self.random_seed)
                output_count.extend(np.random.choice(original_seq_index[i_not_assigned], int(missing)).tolist())

        return np.bincount(output_count, minlength=self.multiplicity.shape[0]) + self.eo_counts

    def buildSignature(self):
        """
        Extract features using ngrams and reduce dimensionality with SVD
        """
        cv = CountVectorizer(ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False, max_features=1024)
        seqs = [['$$START$$']+list([str(y) for y in x])+['$$END$$'] for x in list(self.variants.keys())]
        data = cv.fit_transform(seqs)
        data = TruncatedSVD(min(8, int(data.shape[1]/2)+1), random_state=0).fit_transform(data).astype(np.float32)
        return data



import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
from py2opt.routefinder import RouteFinder
import random2
from itertools import combinations
import editdistance
import os
import sys

class CminSampler:
    def __init__(self, variants, p=0):
        """
        Parameters
        ----------
        variants : pd.Series
            variants is a pd.Series where index=variant and value=count
        p : int
            Number of samples to return. If set to zero, it will automatically choose the size using self.auto_size()
        """
        self.variants = variants

        if p<1 or p>variants.shape[0]:
            p = self.auto_size()
        self.p = p

        self.multiplicity = variants.values
        self.variant_seq = variants.index.tolist()

        # capacity it the number of traces assigned to each rep'
        self.capacity = math.floor(self.multiplicity.sum()/self.p)

        # Expected occurrence reduction reduces the number of operations needed (see paper)
        self.eo_counts = np.floor(((self.multiplicity / self.multiplicity.sum())) * self.p)

    def auto_size(self):
        """
        Define p based on the number of traces
        """
        return int(round(math.log(float(self.variants.sum()),1.5),0))

    def sample(self, tsp_sort=True):
        """
        Select the p most representative traces
        -------
        list
            a list (size p) of list of the most representative traces
        """
        sampler = self.samplingWithEucl()
        seqs = [y for i, x in enumerate(sampler) if x > 0 for y in [self.variant_seq[i]]*int(x)]
        if tsp_sort:
            sys.stdout = open(os.devnull, 'w')
            seqs = self.tsp_sorting(seqs)
            sys.stdout = sys.__stdout__
        return seqs

    def samplingWithEucl(self, max_seq=10000):
        """ Run the sampling algorithm in the Euclidean space

        Parameters
        ----------
        max_seq : int
            Heuristic to speed-up the sampling. It will randomly pick {max_seq} traces
        """
        data = self.buildSignature()
        data = data.repeat(self.multiplicity, axis=0)

        original_seq_index = np.arange(self.multiplicity.shape[0]).repeat(self.multiplicity)
        not_assigned = np.ones(data.shape[0]).astype(bool)

        if data.shape[0] > max_seq:
            np.random.seed(0)
            r = np.random.choice(not_assigned.shape[0], not_assigned.shape[0]-max_seq, replace=False)
            not_assigned[r] = False
            self.capacity = math.floor(max_seq/self.p)

        output_count = []
        while len(output_count) != self.p - self.eo_counts.sum():
            i_not_assigned = np.where(not_assigned==True)[0]
            ldata = data[not_assigned,:]
            index = faiss.IndexFlatL2(ldata.shape[1])   # build the index
            index.add(ldata)                            # add vectors to the index
            D, I = index.search(ldata, self.capacity)
            best_id = D.sum(axis=1).argmin()
            closest_to_best = I[best_id,:]

            output_count.append(original_seq_index[i_not_assigned[best_id]])
            not_assigned[i_not_assigned[closest_to_best]] = False

        return np.bincount(output_count, minlength=self.multiplicity.shape[0]) + self.eo_counts

    def buildSignature(self):
        """
        Extract features using ngrams and reduce dimensionality with SVD
        """
        cv = CountVectorizer(ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False, max_features=1024)
        data = cv.fit_transform([['$$START$$']+list(x)+['$$END$$'] for x in list(self.variants.keys())])
        data = TruncatedSVD(min(64, int(data.shape[1]/2)+1), random_state=0).fit_transform(data).astype(np.float32)
        return data

    def tsp_sorting(self, seqs):
        dist_mat = self.buildDistanceMatrix(seqs)
        random2.seed(1)
        route_finder = RouteFinder(dist_mat, None, iterations=5)
        best_distance, best_route = route_finder.solve()

        d = []
        for i in range(len(best_route)):
            current = best_route[i]
            try:
                next = best_route[i+1]
            except:
                next = best_route[0]
            d.append(dist_mat[current][next])
        cut = np.array(d).argmax()
        best_route = best_route[cut+1:]+best_route[:cut+1]

        return [seqs[int(x)] for x in best_route]


    def buildDistanceMatrix(self, seq):
        m = np.zeros([len(seq), len(seq)])
        for x, y in combinations(range(0,len(seq)), 2):
            d = editdistance.eval(seq[x], seq[y]) / max([len(seq[x]), len(seq[y])])
            m[x,y] = d
            m[y,x] = d
        for x in range(len(seq)):
            m[x,x] = 0
        return m.astype(np.float64)

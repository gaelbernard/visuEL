from visuEL import Vis
from visuEL import CminSampler
import numpy as np
if __name__ == '__main__':
    n_activities = 2
    n_seq = 10000
    max_length=10
    seqs = []
    for _ in range(n_seq):
        l = np.random.randint(max_length)+1
        seqs.append(np.random.randint(0,n_activities,l).tolist())

    #seqs = [[1,2,3,4,54,6,7],[1,2,3,34,5,6,7],[1,2,3,43,5,6,7],[1,2,3,4,5,6,7],[1,2,3,24,5,6,7],[1,2,3,14,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,432,3,4,5,6,7],[8,9,465,434,5635,8327],[1,2,3,34,5,6,7],[1,2,32,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[8,9,465,434,535,8327],[8,9,465,434,535,8327]]

    #seqs = seqs + seqs + seqs + seqs
    #seqs = seqs + seqs + seqs + seqs
    #seqs = seqs + seqs + seqs + seqs
    #seqs = seqs + seqs + seqs + seqs
    print (len(seqs))


    sampler = CminSampler()
    sampler.load_from_seq(seqs)
    sampled_seq = sampler.sample()

    vis = Vis(title='{} cases'.format(len(seqs)))
    vis.load_from_seq(sampled_seq)
    vis.save_svg('seqlegend', 'test_seqlegend.svg')
    vis.save_svg('legend', 'test_legend.svg')
    vis.save_svg('seq', 'test_seq.svg')
    print (vis.get_svg('seqlegend'))


from visuEL import Vis
from visuEL import CminSampler
if __name__ == '__main__':
    seqs = [[0,"dsadnsajdnsajkndkjsa ndjksandkjasnkjdnsakj","hello",3,45,1,4,3,2],[0,3,"hello",3,"hello",3,"hello",332,23,23],[32,32,32,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,4532,3,"hello",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,432,3,"hel4lo",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"he5llo",],[32,32,432,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,4342,3,"hello",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"hello",],[32,322,332,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"hello",],[32,322,32,32,324,3,24,32,423,423,432,3,"h2ello",21,"hello",332,"hello",],[2,3],[2,3],[2,3,"hello",],[2,3,1],[2,3,"hello",3],[2,34,2],[5,3,4,53,2],[4,3,2],[2,3,2],["hello","hello",3,23,4,5],["hello",7,3,2],[2,3],["hello",3,4,52,3],[2,9,"hello",],[43,4,52,3,4],[2,"hello",2,3],[2,3,2],[2,3,2],[2,4,2],['2',3,"hello","hell6o",],["hello",1,3,2],[25,3,3,4,5],[0,"dsadnsajdnsajkndkjsa ndjksandkjasnkjdnsakj","hello",3,45,1,4,3,2],[0,3,"hello",3,"hello",3,"hello",332,23,23],[32,32,32,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,4532,3,"7hello",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,4123,2432,3,"hel4lo",21,"hello",32,"hello",],[32,32,32,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"he5llo",],[32,32,432,332,324,3,24,32,423,423,432,3,"hello",21,"hel6lo",32,"hello",],[32,32,32,32,324,3,24,32,423,423,4342,3,"hello",21,"hello",32,"h4ello",],[32,32532,32,324,3,24,32,423,423,432,3,"hello",21,"hello",32,"hello",],[32,322,332,32,324,3,24,32,423,423,432,34,"hello",21,"hello",32,"hello",],[32,322,32,32,324,3,24,32,423,423,432,3,"h2ello",21,"hello",332,"hello",],[2,3],[2,3],[2,3,"hello",],[2,3,1],[2,3,"hello",3],[2,34,2],[5,3,4,53,2],[4,3,2],[2,3,2],["hello","hello",3,23,4,5],["hello",7,3,2],[2,3],["hello",3,4,52,3],[2,9,"hello",],[43,4,52,3,4],[2,"hello",2,3],[2,3,2],[2,3,2],[2,4,2],['2',3,"hello","hell6o",],["hello",1,3,2],[25,3,3,4,5],]
    seqs = seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs
    seqs = seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs
    seqs = seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs
    #seqs = seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs + seqs

    # Create a sequence of activities
    seqs = [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2,3],[1,2,3],[1,4,3],[1,6,3,4,3,1,2,3,1],[1,6,3,4,3,1,2,3,1],[1,6,3,4,3,1,2,3,1],[1,6,3,4,3,1,2,3,1]]

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


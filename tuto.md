# visuEL 

##Tutorial: Clustering Visualization
visuEL is particularly useful for visualizing several clusters of event logs. In this specific use case, having a single legend shared across several clusters eases the visual inspections of the clusters. This is what we will achieve in this tutorial. For instruction on how to install visuEL, please refer to [this page](README.md) 

## Step 1: load the event logs
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from visuEL import Vis
from visuEL import CminSampler

# Folder where the SVGs will be exported
path_folder = '.'

# Loading the event logs (from a dataframe)
df = pd.read_csv('/Users/gbernar1/Documents/Dev/0_data/xes-standard-dataset/bpi_2020/PermitLog.csv')
seqs = df.groupby('Case ID')['Activity'].agg(list).tolist()

# Cluster the traces (not specific to visuEL, all we need is to get some labels)
cv = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda doc: doc, lowercase=False, max_features=1024)
data = cv.fit_transform([['$$START$$'] + list(x) + ['$$END$$'] for x in list(seqs)])
data = TruncatedSVD(min(8, int(data.shape[1] / 2) + 1), random_state=0).fit_transform(data).astype(np.float32)
clusterer = KMeans(n_clusters=12)
labels = clusterer.fit_predict(data)

# Sample the full event logs
sampler = CminSampler()
sampler.load_from_seq(seqs)
sampled_seq = sampler.sample()

# Create three SVGs for the full event logs : (1) only the legend, (2) only the sequence, (3) both
vis = Vis(title='{} cases'.format(len(seqs)))
vis.load_from_seq(sampled_seq)
vis.save_svg('legend', '{}/legend.svg'.format(path_folder))
vis.save_svg('seq', '{}/full_event_logs.svg'.format(path_folder))
vis.save_svg('seqlegend', '{}/full_event_logs_with_legend.svg'.format(path_folder))

# Save the legend (that we will use for all the clusters)
lgd = vis.get_legend()

# Sample and export SVG for all clusters
for l in np.unique(labels):
    name = 'cluster_{}'.format(l)
    seqs_in_cluster = [seqs[x] for x in np.where(labels == l)[0]]

    # Sample the sequence in the cluster
    sampler = CminSampler()
    sampler.load_from_seq(seqs_in_cluster)
    sampled_seq_cluster = sampler.sample()
    
    # Create a visualization
    # Note that we use the legend that was create for the full event logs
    vis = Vis(title='{} cases'.format(len(seqs_in_cluster)), legend=lgd)
    vis.load_from_seq(sampled_seq_cluster)
    
    # Export the SVG (without the legend)
    path = '{}/{}.svg'.format(path_folder,l)
    vis.save_svg('seq', '{}/{}.svg'.format(path_folder, name))
```

After running this code, several SVGs will be created in the {path_folder}. 

## Contact
Please do not hesitate to contact me at visuel_contact@pm.me

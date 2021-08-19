import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
import drawSvg as draw
from visuEL.Sampler import CminSampler

class Vis:
    width_in_block = 20
    padding = 1
    square_s = 50

    def __init__(self, mapping_name=None, legend=True, max_n_color=6):
        self.mapping_name = mapping_name
        self.legend = legend
        self.max_n_color = int(max_n_color)
        self.svg = None
        self.activities_df = None

    def _load(self, variants):
        sampler = CminSampler(variants, 0)
        sampling = sampler.sample()
        self.activities_df = self.activity_definition(sampling)
        self.svg = self._build_svg(sampling)


    def load_variants(self, variants):
        self._load(variants)

    def load_seqs(self, seqs):
        seqs = [[str(y) for y in x] for x in seqs]
        self._load(pd.Series(seqs).value_counts())

    def activity_definition(self, sampling):

        # Count activities
        y = pd.Series([y for x in sampling for y in x]).value_counts().to_frame()

        # Prepare dataframe
        y.columns = ['count']
        y.index.name = 'name'
        y = y.reset_index()
        y = y.sort_values(by=['count','name'], ascending=[False, True])

        # Assign color
        n_color = min(self.max_n_color, y.shape[0])
        palette = plt.get_cmap('magma', n_color)
        colors = [rgb2hex(palette(x)) for x in range(n_color)]
        y['color'] = colors[-1]
        y.loc[y.iloc[0:n_color].index, 'color'] = colors

        # Map potential name
        mapping = {x:x for x in y['name'].tolist()}
        if self.mapping_name:
            for k,v in self.mapping_name.items():
                if k in mapping.keys():
                    mapping[k] = v
        y.index = y['name'].tolist()
        y['o_name'] = y['name'].copy()
        y['name'] = y['name'].map(mapping)
        y['ranking'] = np.arange(y.shape[0])

        y.loc[y.iloc[n_color-1:].index, 'name'] = '{} hidden...'.format(y.shape[0]-n_color+1)
        y = y.to_dict(orient='index')

        return y

    def _build_svg(self, sampling):
        n = min(self.max_n_color, len(self.activities_df))
        if self.legend:
            n += n+3
        height = n*(self.padding+self.square_s)
        width = self.width_in_block * self.square_s

        d = draw.Drawing(width, height, origin=(0,0), displayInline=False)
        print (self.activities_df)
        for row, trace in enumerate(sampling):
            for col, activity in enumerate(trace):
                top = height - ((row*self.square_s) + (row*self.padding))
                if len(trace) > self.width_in_block:
                    if col == len(trace) - 2:
                        left = (self.width_in_block - 2)*self.square_s
                        r = self.square_s/15
                        d.append(draw.Circle(left+(self.square_s/2), top+(self.square_s/2), r, fill='#000000'))
                        d.append(draw.Circle(left+(self.square_s/2)-(r*3), top+(self.square_s/2), r, fill='#000000'))
                        d.append(draw.Circle(left+(self.square_s/2)+(r*3), top+(self.square_s/2), r, fill='#000000'))
                        continue
                    if col == len(trace)-1:
                        left = (self.width_in_block - 1)*self.square_s
                        d.append(draw.Rectangle(left, top, self.square_s, self.square_s, fill=self.activities_df[trace[-1]]['color']))
                        continue
                    if col > self.width_in_block - 3:
                        continue
                left = col*self.square_s
                d.append(draw.Rectangle(left, top, self.square_s, self.square_s, fill=self.activities_df[activity]['color']))
        if self.legend:

            font_size = self.square_s*0.6
            t = (top - ((self.square_s + self.padding)*2))
            d.append(draw.Text('LEGEND (top activities)', font_size, 0, t+(self.square_s*0.2), fill='black'))

            for i, v in enumerate(self.activities_df.values()):
                if v['ranking']>=self.max_n_color:
                    continue
                t = (top - ((self.square_s + self.padding)*(v['ranking']+3)))
                d.append(draw.Rectangle(0, t, self.square_s*0.8, self.square_s*0.8, fill=v['color']))
                d.append(draw.Text(v['name'], font_size, self.square_s, t+(self.square_s/5), fill='black'))

        return d

    def get_svg(self):
        return self.svg.asSvg()

    def save_svg(self,filename, folder=''):
        self.svg.saveSvg('{}{}.svg'.format(folder, filename))
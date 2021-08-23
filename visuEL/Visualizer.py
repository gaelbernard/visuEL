import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
import svgwrite

class Vis:
    padding = 5
    square_s = 50

    def __init__(self, seqs, max_n_color=6, legend=None, mapping_name=None, title=None, width_in_block=20):
        """
        Produce SVG visualizations for event logs
        :param seqs: event logs represented as list of lists
        :param max_n_color: maximum number of colors, only the {max_n_color}
                            most appearing activities will have their own color
        :param legend:  For some analysis, it could be useful to have two event logs that
                        shares the same legend (e.g., when doing clustering); i.e.,
                        two visualizations will share the same colors and are hence comparable.
                        This param allow to apply an existing legend to the graph.
                        The legend is extracted from the graph using self.get_legend().
                        If none, the legend is built automatically by looking at the
                        most occuring activities
        :param mapping_name: In a web service context, we can save some space if we provide an
                             index for the activity name instead of the real name;
                             i.e., [['this is activity 1','activity2']] becomes [[1,2]]
                             mapping_name allows to map the real name to the index so the
                             legend will show the real name.
                             this is how the mapping_name would look like:
                             {1:'this is activity 1', 2:'this is activity 2'}
        :param title:   Optional title at the top of the visualization
        :param width_in_block:  Max number of activities per sequence (default: 20).
                                If a trace contains more than {width_in_block}, it will
                                be truncated like [1,2,3,4,5,...,10] (with width_in_block=7)
        """
        self.seqs = self.load_seqs(seqs)
        self.max_n_color = int(max_n_color)
        self.legend = legend
        self.mapping_name = mapping_name
        self.title = title
        self.width_in_block = width_in_block

        if not self.legend:
            self.legend = self.extract_legend(self.seqs)
        else:
            self._add_missing_activity_to_legend()

        self.svg = {}
        # We create three kind of SVGs
        for type in ['seq','legend','seqlegend']:
            self.svg[type] = self._build_svg(type)


    def _add_missing_activity_to_legend(self):
        '''
        When using an existing legend, we should make sure that all
        activities from the current event logs are also added to the legend
        (with the color 'others')
        :return:
        '''
        last_row = None
        highest_ranking = 0
        for v in self.legend.values():
            if v['ranking']>highest_ranking:
                highest_ranking = v['ranking']
                last_row = v
        for k,v in self.extract_legend(self.seqs).items():
            if k not in self.legend:
                self.legend[k] = last_row
                self.legend[k]['o_name'] = k

    def load_seqs(self, seqs):
        return [[str(y) for y in x] for x in seqs]

    def extract_legend(self, seqs):
        '''
        Assign a color to each activities.
        The top {self.max_n_color-1} activities that occurs the most will
        have a distinct color while the remaining one will get the label 'others'
        :param seqs: event logs as list of lists
        :return: a dictionary
        '''

        # Count activities
        y = pd.Series([y for x in seqs for y in x]).value_counts().to_frame()

        # Prepare dataframe
        y.columns = ['count']
        y.index.name = 'name'
        y = y.reset_index()
        y = y.sort_values(by=['count','name'], ascending=[False, True])

        # Assign color
        n_color = min(self.max_n_color, y.shape[0])
        palette = plt.get_cmap('magma', n_color+1)
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

        if y.shape[0] > self.max_n_color:
            y.loc[y.iloc[n_color-1:].index, 'name'] = '+ {} others...'.format(y.shape[0]-n_color+1)
            y.loc[y.iloc[n_color-1:].index, 'color'] = '#eee'
        return y.to_dict(orient='index')

    def _build_svg(self, type):
        '''
        Draw the SVG.
        :param type: There are three types of SVGs:
                    1. 'legend': show only the legend
                    2. 'seq': show only the sequence of activities
                    3. 'seqlegend': show both (1) and (2)
        :return: a svgwrite object
        '''

        n = 0
        if type in ['seq','seqlegend']:
            n += len(self.seqs)   # For the sequence
            n += 1              # For the number of cases at the top
        if type in ['legend','seqlegend']:
            # For the legend
            n += min(self.max_n_color, len(self.legend)) + 2

        height = n*(self.padding+self.square_s)
        width = self.width_in_block * self.square_s
        font_size = self.square_s*0.7


        d = svgwrite.Drawing('test.svg', profile='tiny', size=(width, height))

        if type in ['seq','seqlegend']:
            d.add(d.text(self.title,  insert=(5, 25), fill='black', font_size=font_size,))

            for row, trace in enumerate(self.seqs):
                for col, activity in enumerate(trace):
                    top = (((row+(self.title is not None))*self.square_s) + ((row)*self.padding+1))
                    if len(trace) > self.width_in_block:
                        if col == len(trace) - 2:
                            left = (self.width_in_block - 2)*self.square_s
                            r = self.square_s/15
                            d.add(d.circle((left+(self.square_s/2), top+(self.square_s/2)), r, fill='#000000'))
                            d.add(d.circle((left+(self.square_s/2)-(r*3), top+(self.square_s/2)), r, fill='#000000'))
                            d.add(d.circle((left+(self.square_s/2)+(r*3), top+(self.square_s/2)), r, fill='#000000'))
                            continue
                        if col == len(trace)-1:
                            left = (self.width_in_block - 1)*self.square_s
                            d.add(d.rect((left, top), (self.square_s, self.square_s), fill=self.legend[trace[-1]]['color']))
                            continue
                        if col > self.width_in_block - 3:
                            continue
                    left = col*self.square_s
                    d.add(d.rect((left, top), (self.square_s, self.square_s), fill=self.legend[activity]['color']))

        if type in ['legend','seqlegend']:
            top = 1 if type == 'legend' else len(self.seqs) + 2
            for i, v in enumerate(self.legend.values()):
                if v['ranking']>=self.max_n_color:
                    continue
                t = (self.square_s + self.padding)*(v['ranking']+top)
                d.add(d.rect((0, t), (self.square_s*0.8, self.square_s*0.8), fill=v['color']))
                d.add(d.text(v['name'],  insert=(self.square_s, t+(self.square_s*0.6)), fill='black', font_size=font_size,))
        d.viewbox(0, 0, width, height)
        d.fit(horiz='left', vert='top', scale='meet')
        return d

    def get_svg(self, type):
        """
        Return a SVG in text format
        :param type: There are three types of SVGs:
                    1. 'legend': show only the legend
                    2. 'seq': show only the sequence of activities
                    3. 'seqlegend': show both (1) and (2)
        :return: text in xml format
        """
        return self.svg[type].tostring()

    def get_activities(self):
        '''
        Return the legend, useful when we want to apply it to another visualizations.
        :return: dictionary
        '''
        return self.legend

    def save_svg(self, type, path):
        '''
        Save the SVG to the specified path
        :param type: There are three types of SVGs:
                    1. 'legend': show only the legend
                    2. 'seq': show only the sequence of activities
                    3. 'seqlegend': show both (1) and (2)
        :param path: path
        '''
        with open(path, 'w') as f:
            f.write(self.get_svg(type))

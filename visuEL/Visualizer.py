import numpy as np
import pandas as pd
import copy
import svgwrite
from pm4py.objects.log.importer.xes import importer as xes_importer
import sys, os
from py2opt.routefinder import RouteFinder
from itertools import combinations
import editdistance
import random2
class Vis:
    padding = 5
    square_s = 50
    margin_legend = 30
    colors = ['#000000','#3b156e','#8b2d80','#dc4b6a','#fc9f72']
    other_color = '#eeeeee'

    def __init__(self, legend=None, mapping_name=None, title=None, width_in_block=20, tsp_sort=True, random_seed=1):
        """
        Produce SVG visualizations for event logs
        :param seqs: event logs represented as list of lists
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
        :param tsp_sort: If set to true: Sort the sequence by similarity using a TSP algorithm
        :param random_seed: Given the same seed and identical event logs, make sure the sampling, forces the same output
        """
        self.seqs = None
        self.max_n_color = len(self.colors)
        self.legend = legend
        self.mapping_name = mapping_name
        self.title = title
        self.width_in_block = width_in_block
        self.tsp_sort = tsp_sort
        self.random_seed = random_seed
        self.svg = {}

    def _load(self, seqs):

        self.seqs = seqs
        if not self.legend:
            self.legend = self.extract_legend(seqs)
        else:
            self._add_missing_activity_to_legend()

        if self.tsp_sort and len(seqs)>2:
            self.seqs = self.tsp_sorting(seqs)

        # We create three kind of SVGs
        for type in ['seq','legend','seqlegend']:
            self.svg[type] = self._build_svg(type)

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
        y['color'] = self.other_color
        y.loc[y.iloc[:self.max_n_color].index, 'color'] = self.colors[:y.shape[0]]

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
            n_other = y.shape[0]-self.max_n_color
            if n_other>1:
                y.loc[y.iloc[self.max_n_color:].index, 'name'] = '+ {} others...'.format(n_other)
            y.loc[y.iloc[self.max_n_color:].index, 'color'] = self.other_color
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
        if type in ['legend','seqlegend']:
            # For the legend
            u_name = {x['name'] for x in self.legend.values()}
            n += len(u_name)
        n += self.title is not None

        height = n*(self.padding+self.square_s)
        if type == 'seqlegend':
            height += self.margin_legend

        width = self.width_in_block * self.square_s
        font_size = self.square_s*0.7


        d = svgwrite.Drawing('test.svg', profile='tiny', size=(width, height))
        d.add(d.rect((0, 0), (width, height), fill='#ffffff'))

        if type in ['seq','seqlegend']:
            if self.title is not None:
                d.add(d.text(self.title,  insert=(0, 35), fill='black', font_size=font_size,))

            for row, trace in enumerate(self.seqs):
                for col, activity in enumerate(trace):
                    top = (((row+(self.title is not None))*self.square_s) + ((row)*self.padding))
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
            top = 0 if type == 'legend' else len(self.seqs)
            top += self.title is not None

            for i, v in enumerate(self.legend.values()):
                if v['ranking']>=self.max_n_color+1:
                    continue
                t = (self.square_s + self.padding)*(v['ranking']+top)
                if type == 'seqlegend':
                    t += self.margin_legend

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

    def get_legend(self):
        '''
        Return the legend, useful when we want to apply it to another visualizations.
        :return: dictionary
        '''
        return copy.deepcopy(self.legend)

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

    def buildDistanceMatrix(self, seq):
        '''
        Build distance matrix between sequences using the normalize edit distance
        :param seq: event logs provided as list of list
        :return: Matrix of size (len(seqs)^2)
        '''
        m = np.zeros([len(seq), len(seq)])
        for x, y in combinations(range(0,len(seq)), 2):
            d = editdistance.eval(seq[x], seq[y]) / max([len(seq[x]), len(seq[y])])
            m[x,y] = d
            m[y,x] = d
        for x in range(len(seq)):
            m[x,x] = 0
        return m.astype(np.float64)

    def tsp_sorting(self, seqs):
        '''
        Order the sequence by similarty using a TSP algorithm
        :param seqs: event logs represented as list of lists
        :return:
        '''
        colored_seqs = [[self.legend[e]['color'] for i, e in enumerate(t) if i < self.width_in_block] for t in seqs]
        dist_mat = self.buildDistanceMatrix(colored_seqs)
        if len({''.join(x) for x in colored_seqs}) < 2:
            return seqs
        random2.seed(self.random_seed)
        sys.stdout = open(os.devnull, 'w')
        route_finder = RouteFinder(dist_mat, None, iterations=5)
        best_distance, best_route = route_finder.solve()
        sys.stdout = sys.__stdout__

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
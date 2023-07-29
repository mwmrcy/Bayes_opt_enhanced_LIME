import string
import matplotlib.pyplot as plt
import numpy as np
from utils import InvalidExplanationMode

from io import open
import os
import os.path
import json




from sklearn.utils import check_random_state

#Helper function to generate random div ids. This is useful for embedding HTML into ipython notebooks.
def id_generator(size=15, random_state=None):
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))

#Class for mapping features to the specific domain.
class DomainMapper(object):
    def __init__(self):
        pass

    #Maps the feature ids to concrete names.
    def map_exp_ids(self, exp, **kwargs):
        return exp

    def visualize_instance_html(self,exp,label,div_name,exp_object_name,**kwargs):
        """Produces html for visualizing the instance.
        Default behaviour does nothing. Subclasses can implement this as they
        see fit.
        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             kwargs: optional keyword arguments
        Returns:
             js code for visualizing the instance
        """
        return ''


#Object returned by explainers.
class Explanation(object):

    def __init__(self,
                 domain_mapper,
                 mode='classification',
                 class_names=None,
                 random_state=None):
        self.random_state = random_state
        self.mode = mode
        self.domain_mapper = domain_mapper
        self.local_exp = {}
        self.intercept = {}
        self.score = {}
        self.local_pred = {}
        self.scaled_data = None
        if mode == 'classification':
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None
        elif mode == 'regression':
            self.class_names = ['negative', 'positive']
            self.predicted_value = None
            self.min_value = 0.0
            self.max_value = 1.0
            self.dummy_label = 1
        else:
            raise InvalidExplanationMode('Invalid explanation mode "{}". '
                            'Should be either "classification" '
                            'or "regression".'.format(mode))

    def available_labels(self):
        try:
            assert self.mode == "classification"
        except AssertionError:
            raise NotImplementedError('Not supported for regression explanations.')
        else:
            ans = self.top_labels if self.top_labels else self.local_exp.keys()
            return list(ans)

    def as_list(self, label=1, **kwargs):
        """Returns the explanation as a list.
        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
                Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            list of tuples (representation, weight), where representation is
            given by domain_mapper. Weight is a float.
        """
        label_to_use = label if self.mode == "classification" else self.dummy_label
        ans = self.domain_mapper.map_exp_ids(self.local_exp[label_to_use], **kwargs)
        ans = [(x[0], float(x[1])) for x in ans]
        return ans

    def as_list_one(self, label=1, **kwargs):
        label_to_use = label if self.mode == "classification" else self.dummy_label
        ans = self.domain_mapper.map_exp_ids(self.local_exp[label_to_use], **kwargs)

        return ans

    def as_list_zero(self, label=0, **kwargs):
        label_to_use = label if self.mode == "classification" else self.dummy_label
        ans = self.domain_mapper.map_exp_ids(self.local_exp[label_to_use], **kwargs)

        return ans

    def as_map(self):
        return self.local_exp

    def as_pyplot_figure(self, label=0, type='h', **kwargs):
        exp = self.as_list(label=label, **kwargs)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['purple' if x > 0 else 'orange' for x in vals]
        pos = np.arange(len(exp)) + .5
        if type == 'h':
            plt.barh(pos, vals, align='center', color=colors)
            plt.yticks(pos, names)
        else:
            plt.bar(pos, vals, align='center', color=colors)
            plt.xticks(pos, names, rotation=90)

        if self.mode == "classification":
            title = 'Local explanation for class %s' % self.class_names[label]
        else:
            title = 'Local explanation'
        plt.title(title)
        return fig, names

    def as_pyplot_to_figure(self, type='h', name = None, label='0', lp=None, **kwargs):
        if label == '0':
            exp = self.as_list_zero(label=0, **kwargs)
        else:
            exp = self.as_list_one(label=1, **kwargs)
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['blue' if x > 0 else 'yellow' for x in vals]
        #pos = np.arange(len(exp)) + .2
        pos = np.arange(len(exp)) / 3

        fig = plt.figure(figsize=(4, 2))
        if type == 'h':
            plt.barh(pos, vals, align='center', color=colors, height=0.2)
            plt.yticks(pos, names)
        else:
            plt.bar(pos, vals, align='center', color=colors)
            plt.xticks(pos, names, rotation=90)

        if self.mode == "classification":
            title = 'Local explanation for class: %s' % self.class_names[int(label)]
        else:
            title = 'Local explanation'
        plt.title(title)
        #plt.suptitle('Sup title', y=1.05, fontsize=18)
        #plt.savefig(str(name) + ".png")
        filename= 'results/' + str(name)+".pdf"
        plt.savefig(filename, bbox_inches='tight')
        return fig, names
    def show_in_notebook(self,
                         labels=None,
                         predict_proba=True,
                         show_predicted_value=True,
                         **kwargs):
        """Shows html explanation in ipython notebook.
        See as_html() for parameters.
        This will throw an error if you don't have IPython installed"""

        from IPython.core.display import display, HTML
        display(HTML(self.as_html(labels=labels,
                                  predict_proba=predict_proba,
                                  show_predicted_value=show_predicted_value,
                                  **kwargs)))


    def as_html(self,
                labels=None,
                predict_proba=True,
                show_predicted_value=True,
                **kwargs):
        """Returns the explanation as an html page.
        Args:
            labels: desired labels to show explanations for (as barcharts).
                If you ask for a label for which an explanation wasn't
                computed, will throw an exception. If None, will show
                explanations for all available labels. (only used for classification)
            predict_proba: if true, add  barchart with prediction probabilities
                for the top classes. (only used for classification)
            show_predicted_value: if true, add  barchart with expected value
                (only used for regression)
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            code for an html page, including javascript includes.
        """

        def jsonize(x):
            return json.dumps(x, ensure_ascii=False)

        if labels is None and self.mode == "classification":
            labels = self.available_labels()

        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'),
                      encoding="utf8").read()

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        random_id = id_generator(size=15, random_state=check_random_state(self.random_state))
        out += u'''
        <div class="lime top_div" id="top_div%s"></div>
        ''' % random_id

        predict_proba_js = ''
        if self.mode == "classification" and predict_proba:
            predict_proba_js = u'''
            var pp_div = top_div.append('div')
                                .classed('lime predict_proba', true);
            var pp_svg = pp_div.append('svg').style('width', '100%%');
            var pp = new lime.PredictProba(pp_svg, %s, %s);
            ''' % (jsonize([str(x) for x in self.class_names]),
                   jsonize(list(self.predict_proba.astype(float))))

        predict_value_js = ''
        if self.mode == "regression" and show_predicted_value:
            # reference self.predicted_value
            # (svg, predicted_value, min_value, max_value)
            predict_value_js = u'''
                    var pp_div = top_div.append('div')
                                        .classed('lime predicted_value', true);
                    var pp_svg = pp_div.append('svg').style('width', '100%%');
                    var pp = new lime.PredictedValue(pp_svg, %s, %s, %s);
                    ''' % (jsonize(float(self.predicted_value)),
                           jsonize(float(self.min_value)),
                           jsonize(float(self.max_value)))

        exp_js = '''var exp_div;
            var exp = new lime.Explanation(%s);
        ''' % (jsonize([str(x) for x in self.class_names]))

        if self.mode == "classification":
            for label in labels:
                exp = jsonize(self.as_list(label))
                exp_js += u'''
                exp_div = top_div.append('div').classed('lime explanation', true);
                exp.show(%s, %d, exp_div);
                ''' % (exp, label)
        else:
            exp = jsonize(self.as_list())
            exp_js += u'''
            exp_div = top_div.append('div').classed('lime explanation', true);
            exp.show(%s, %s, exp_div);
            ''' % (exp, self.dummy_label)

        raw_js = '''var raw_div = top_div.append('div');'''

        if self.mode == "classification":
            html_data = self.local_exp[labels[0]]
        else:
            html_data = self.local_exp[self.dummy_label]

        raw_js += self.domain_mapper.visualize_instance_html(
                html_data,
                labels[0] if self.mode == "classification" else self.dummy_label,
                'raw_div',
                'exp',
                **kwargs)
        out += u'''
        <script>
        var top_div = d3.select('#top_div%s').classed('lime top_div', true);
        %s
        %s
        %s
        %s
        </script>
        ''' % (random_id, predict_proba_js, predict_value_js, exp_js, raw_js)
        out += u'</body></html>'

        return out

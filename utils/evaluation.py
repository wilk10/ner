import pandas
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.data import Data


class Evaluation:
    def __init__(self, items, verbose):
        self.items = items
        self.verbose = verbose
        self.bioconcepts = Data().bioconcepts

    def classify_text_characters_as_annotated_or_not(self, item, flag):
        assert flag in ['true', 'pred']
        classifications = np.zeros(len(item['text']))
        for annotation in item[flag]:
            tag = annotation['tag'].upper().strip()
            bioconcept_index = self.bioconcepts.index(tag) + 1
            classifications[annotation['start']:annotation['end']] = bioconcept_index
        return classifications

    @staticmethod
    def show_evaluations_one_by_one(ys_by_bioconcepts):
        data = {
            'entity': [entity for _, ys in ys_by_bioconcepts.items() for entity in ys['entity']],
            'bioconcept': [bioconcept for bioconcept, ys in ys_by_bioconcepts.items() for _ in ys['entity']],
            'true': [y for _, ys in ys_by_bioconcepts.items() for y in ys['true']],
            'pred': [y for _, ys in ys_by_bioconcepts.items() for y in ys['pred']]}
        df = pandas.DataFrame(data)
        print(f'{df.to_string()}\n----------------------\n')
        input()

    def calculate_ys_by_bioconcept(self):
        ys_by_bioconcepts = {bioconcept: {'true': [], 'pred': [], 'entity': []} for bioconcept in self.bioconcepts}
        for item in self.items:
            for flag in ['true', 'pred']:
                other_flag = 'pred' if flag == 'true' else 'true'
                other_classifications = self.classify_text_characters_as_annotated_or_not(item, other_flag)
                for annotation in item[flag]:
                    bioconcept = annotation['tag'].upper().strip()
                    text = item['text']
                    entity = f"{text[annotation['start']:annotation['end']]}".strip()
                    relevant_classifications = other_classifications[annotation['start']:annotation['end']]
                    undetected_entity = all([clas == 0 for clas in relevant_classifications])
                    if undetected_entity:
                        ys_by_bioconcepts[bioconcept][flag].append(1)
                        ys_by_bioconcepts[bioconcept][other_flag].append(0)
                        ys_by_bioconcepts[bioconcept]['entity'].append(entity)
                    elif flag == 'true':
                        counter = Counter(relevant_classifications)
                        sorted_classes = [clas for clas, count in counter.most_common() if clas != 0]
                        main_class = int(sorted_classes[0])
                        other_bioconcept = self.bioconcepts[main_class - 1]
                        if other_bioconcept == bioconcept:
                            ys_by_bioconcepts[bioconcept][flag].append(1)
                            ys_by_bioconcepts[bioconcept][other_flag].append(1)
                            ys_by_bioconcepts[bioconcept]['entity'].append(entity)
                        else:
                            ys_by_bioconcepts[bioconcept][flag].append(1)
                            ys_by_bioconcepts[bioconcept][other_flag].append(0)
                            ys_by_bioconcepts[bioconcept]['entity'].append(entity)
            if self.verbose:
                self.show_evaluations_one_by_one(ys_by_bioconcepts)
        return ys_by_bioconcepts

    def run(self):
        ys_by_bioconcept = self.calculate_ys_by_bioconcept()
        metrics_by_bioconcept = dict.fromkeys(self.bioconcepts)
        for bioconcept in self.bioconcepts:
            y_true = ys_by_bioconcept[bioconcept]['true']
            y_pred = ys_by_bioconcept[bioconcept]['pred']
            metrics = {
                'f1': f1_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred)}
            metrics_by_bioconcept[bioconcept] = metrics
        df = pandas.DataFrame(metrics_by_bioconcept)
        average_f1 = np.mean(df.T.f1) * 100
        print(df.T)
        print(f'\naverage f1 score: {average_f1:.0f}%')

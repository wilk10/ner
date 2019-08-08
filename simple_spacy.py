import re
import spacy
from utils.data import Data
from utils.evaluation import Evaluation


class SimpleSpaCy:
    def __init__(self):
        self.data = Data()
        self.entities_by_bioconcept = self.data.learn_training_entries()
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def check_valid_year(num):
        has_4_digits = len(num) == 4
        try:
            int(num)
        except ValueError:
            is_int = False
        else:
            is_int = True
        is_valid = num[:2] == '19' or num[:2] == '20'
        return has_4_digits and is_int and is_valid

    @staticmethod
    def check_prevalence(num):
        return '/' in num or '.' in num

    @staticmethod
    def find_matches_and_make_annotations(entry, text, bioconcept):
        matches = [match for match in re.finditer(entry, text)]
        annotations = []
        for match in matches:
            annotation = {'tag': bioconcept, 'start': match.start(), 'end': match.end()}
            annotations.append(annotation)
        return annotations

    def fit_to_validation(self):
        output_json = {'result': []}
        results = []
        for kingdom in ['animal', 'plant']:
            validation_data = self.data.read_json(kingdom, 'validation')
            for item in validation_data['result']:
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content'].lower()
                output_item = {'example': item['example'], 'results': {'annotations': [], 'classifications': []}}
                doc = self.nlp(text)
                nouns = list(set([chunk.text for chunk in doc.noun_chunks]))
                nums = [token.lemma_ for token in doc if token.pos_ == "NUM"]
                for bioconcept in Data.BIOCONCEPTS_BY_KINGDOM[kingdom]:
                    for noun in nouns:
                        if noun in self.entities_by_bioconcept[bioconcept]:
                            annotations = self.find_matches_and_make_annotations(noun, text, bioconcept)
                            output_item['results']['annotations'].extend(annotations)
                    for num in nums:
                        is_valid_year = self.check_valid_year(num)
                        add_year = bioconcept == 'YEAR' and is_valid_year
                        is_possible_prevalence = self.check_prevalence(num)
                        add_prevalence = bioconcept == 'PREVALENCE' and is_possible_prevalence
                        if add_year or add_prevalence:
                            annotations = self.find_matches_and_make_annotations(num, text, bioconcept)
                            output_item['results']['annotations'].extend(annotations)
                result = {
                    'text': text,
                    'true': item['results']['annotations'],
                    'pred': output_item['results']['annotations']}
                results.append(result)
                output_json['result'].append(output_item)
        return results, output_json

    def run(self):
        items, _ = self.fit_to_validation()
        evaluation = Evaluation(items, verbose=False)
        evaluation.run()


if __name__ == '__main__':
    SimpleSpaCy().run()

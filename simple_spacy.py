import re
import spacy
from utils.data import Data
from utils.evaluation import Evaluation


class SimpleSpaCy:
    def __init__(self):
        self.data = Data()
        self.entities_by_bioconcept = self.data.learn_training_entries()
        self.nlp = spacy.load("en_core_web_sm")

    def fit_to_validation(self):
        output_json = {'result': []}
        results = []
        for flag in ['animal', 'plant']:
            validation_data = self.data.read_json(flag, 'validation')
            for item in validation_data['result']:
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content'].lower()
                output_item = {'example': item['example'], 'results': {'annotations': [], 'classifications': []}}
                doc = self.nlp(text)
                nouns = list(set([chunk.text for chunk in doc.noun_chunks]))
                for bioconcept in Data.BIOCONCEPTS_BY_FLAG[flag]:
                    for noun in nouns:
                        if noun in self.entities_by_bioconcept[bioconcept]:
                            matches = [match for match in re.finditer(noun, text)]
                            for match in matches:
                                annotation = {'tag': bioconcept, 'start': match.start(), 'end': match.end()}
                                output_item['results']['annotations'].append(annotation)
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

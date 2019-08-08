import re
import json
import spacy
from utils.data import Data
from utils.eppo import Eppo
from utils.evaluation import Evaluation


class SpacyWithAPIs:
    MODEL = 'en_core_web_sm'
    NOUNS_NOT_IN_EPPO_FILE_NAME = 'nouns_not_in_eppo.json'

    def __init__(self):
        self.data = Data()
        self.entities_by_bioconcept = self.data.learn_training_entries()
        self.nlp = spacy.load(self.MODEL)
        self.eppo = Eppo(time_to_sleep=0.1)
        self.taxonomies_by_bioconcept = self.get_taxonomies_by_bioconcept()
        self.nouns_not_in_eppo_path = self.data.cwd / self.NOUNS_NOT_IN_EPPO_FILE_NAME
        self.nouns_not_in_eppo = self.load_nouns_not_in_eppo()

    def get_taxonomies_by_bioconcept(self):
        with open(str(self.eppo.entity_taxonomies_by_bioconcept_path), encoding='utf-8') as f:
            entity_taxonomies_by_bioconcept = json.load(f)
        taxonomies_by_bioconcept = {bioconcept: [] for bioconcept in entity_taxonomies_by_bioconcept.keys()}
        for bioconcept, entity_taxonomies in entity_taxonomies_by_bioconcept.items():
            for _, taxonomy in entity_taxonomies:
                if taxonomy not in taxonomies_by_bioconcept[bioconcept]:
                    taxonomies_by_bioconcept[bioconcept].append(taxonomy)
        return taxonomies_by_bioconcept

    def load_nouns_not_in_eppo(self):
        with open(str(self.nouns_not_in_eppo_path), encoding='utf-8') as f:
            nouns_not_in_eppo = json.load(f)
        return nouns_not_in_eppo

    def save_nouns_not_in_eppo(self):
        with open(str(self.nouns_not_in_eppo_path), 'w', encoding='utf-8') as f:
            json.dump(self.nouns_not_in_eppo, f, ensure_ascii=False, indent=4)

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
        try:
            matches = [match for match in re.finditer(entry, text)]
        except re.error:
            matches = []
        annotations = []
        for match in matches:
            annotation = {'tag': bioconcept, 'start': match.start(), 'end': match.end()}
            annotations.append(annotation)
        return annotations

    '''
    def add_acronyms(self, noun, text, annotations, n_char_search=10):
        for annotation in annotations:
            end_search_index = min(annotation['end'] + n_char_search, len(text))
            following_text = text[annotation['end']:end_search_index]
            if '(' in text and ')' in text:
                import pdb
                pdb.set_trace()

            if '(' in following_text and ')' in following_text:
                pass
        return annotations
    '''

    def add_partial_initials(self, input_noun, other_nouns, text, bioconcept, annotations):
        noun_chunks = input_noun.split(' ')
        more_entities = []
        if len(noun_chunks) > 1:
            for i, this_chunk in enumerate(noun_chunks):
                if '(' not in this_chunk and ')' not in this_chunk:
                    initialised_chunk = this_chunk[0] + '.'
                    new_chunks = noun_chunks[:i] + [initialised_chunk] + noun_chunks[i+1:]
                    partially_initialised_noun = ' '.join(new_chunks)
                    if partially_initialised_noun not in other_nouns:
                        more_entities.append(partially_initialised_noun)
        for entity in more_entities:
            new_annotations = self.find_matches_and_make_annotations(entity, text, bioconcept)
            annotations.extend(new_annotations)
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
                            if bioconcept == 'PATHOGENIC_ORGANISMS':
                                annotations = self.add_partial_initials(noun, nouns, text, bioconcept, annotations)
                            #annotations = self.add_acronyms(noun, text, annotations)
                            output_item['results']['annotations'].extend(annotations)
                        elif bioconcept in ['PLANT_SPECIES', 'PLANT_PEST'] and noun not in self.nouns_not_in_eppo:
                            level1_taxonomy = self.eppo.get_eppo_code_and_taxonomy(noun)
                            if level1_taxonomy is not None:
                                if level1_taxonomy in self.taxonomies_by_bioconcept[bioconcept]:
                                    new_annotations = self.find_matches_and_make_annotations(noun, text, bioconcept)
                                    output_item['results']['annotations'].extend(new_annotations)
                                    print(f'{noun} ({bioconcept})')
                            else:
                                self.nouns_not_in_eppo.append(noun)
                    for num in nums:
                        is_valid_year = self.check_valid_year(num)
                        add_year = bioconcept == 'YEAR' and is_valid_year
                        is_possible_prevalence = self.check_prevalence(num)
                        add_prevalence = bioconcept == 'PREVALENCE' and is_possible_prevalence
                        if add_year or add_prevalence:
                            annotations = self.find_matches_and_make_annotations(num, text, bioconcept)
                            output_item['results']['annotations'].extend(annotations)
                    self.save_nouns_not_in_eppo()
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
    SpacyWithAPIs().run()

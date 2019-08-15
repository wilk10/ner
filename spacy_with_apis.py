import re
import json
import spacy
from utils.data import Data
from utils.eppo import Eppo
from utils.evaluation import Evaluation


class SpacyWithAPIs:
    MODEL = 'en_core_web_sm'
    ACCEPTED_EPPO_NOUNS_FILE_NAME = 'accepted_nouns_from_eppo.json'
    NOT_ACCEPTED_EPPO_NOUNS_FILE_NAME = 'nouns_in_eppo_but_not_accepted.json'
    OUTPUT_FILE_NAME = 'predictions.json'
    MANUALLY_EXCLUDED_EPPO_RESULTS = ['none', 'may', 'pest', 'the pest', 'argentina', 'sao', 'florida']

    def __init__(self):
        self.data = Data()
        self.entities_by_bioconcept = self.data.learn_training_entries()
        self.nlp = spacy.load(self.MODEL)
        self.eppo = Eppo(time_to_sleep=0.1)
        self.taxonomies_by_bioconcept = self.get_taxonomies_by_bioconcept()
        self.accepted_eppo_nouns_path = self.data.dict_dir / self.ACCEPTED_EPPO_NOUNS_FILE_NAME
        self.accepted_eppo_nouns_by_bioconcept = self.data.load_json(self.accepted_eppo_nouns_path)
        self.not_accepted_eppo_nouns_path = self.data.dict_dir / self.NOT_ACCEPTED_EPPO_NOUNS_FILE_NAME
        self.not_accepted_eppo_nouns_by_bioconcept = self.data.load_json(self.not_accepted_eppo_nouns_path)
        self.output_path = self.data.dict_dir / self.OUTPUT_FILE_NAME

    def get_taxonomies_by_bioconcept(self):
        with open(str(self.eppo.entity_taxonomies_by_bioconcept_path), encoding='utf-8') as f:
            entity_taxonomies_by_bioconcept = json.load(f)
        taxonomies_by_bioconcept = {bioconcept: [] for bioconcept in entity_taxonomies_by_bioconcept.keys()}
        for bioconcept, entity_taxonomies in entity_taxonomies_by_bioconcept.items():
            for _, taxonomy in entity_taxonomies:
                if taxonomy not in taxonomies_by_bioconcept[bioconcept]:
                    taxonomies_by_bioconcept[bioconcept].append(taxonomy)
        return taxonomies_by_bioconcept

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
        entry = entry.replace('.', '\.')
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

    def add_partial_initials(self, input_noun, text, bioconcept, annotations):
        noun_chunks = input_noun.split(' ')
        more_entities = []
        if len(noun_chunks) > 1:
            for i, this_chunk in enumerate(noun_chunks):
                if '(' not in this_chunk and ')' not in this_chunk:
                    initialised_chunk = this_chunk[0] + '.'
                    new_chunks = noun_chunks[:i] + [initialised_chunk] + noun_chunks[i+1:]
                    partially_initialised_noun = ' '.join(new_chunks)
                    more_entities.append(partially_initialised_noun)
        for entity in more_entities:
            new_annotations = self.find_matches_and_make_annotations(entity, text, bioconcept)
            annotations.extend(new_annotations)
        return annotations

    def add_eppo_data(self, noun, text, bioconcept, jsons_to_save):
        annotations = []
        if noun not in self.eppo.nouns_not_in_eppo and noun not in self.MANUALLY_EXCLUDED_EPPO_RESULTS:
            if noun not in self.accepted_eppo_nouns_by_bioconcept[bioconcept]:
                if noun not in self.not_accepted_eppo_nouns_by_bioconcept[bioconcept]:
                    level1_taxonomy = self.eppo.get_eppo_code_and_taxonomy(noun)
                    if level1_taxonomy is not None:
                        if level1_taxonomy in self.taxonomies_by_bioconcept[bioconcept]:
                            annotations = self.find_matches_and_make_annotations(noun, text, bioconcept)
                            self.accepted_eppo_nouns_by_bioconcept[bioconcept].append(noun)
                        else:
                            self.not_accepted_eppo_nouns_by_bioconcept[bioconcept].append(noun)
                    else:
                        self.eppo.nouns_not_in_eppo.append(noun)
                    jsons_to_save = True
            else:
                annotations = self.find_matches_and_make_annotations(noun, text, bioconcept)
        return annotations, jsons_to_save

    @staticmethod
    def clean_annotations(annotations, text, buffer=0):
        sorted_annotations = sorted(annotations, key=lambda a: a['start'])
        cleaned_annotations = [sorted_annotations[0]]
        for i, annotation in enumerate(sorted_annotations[1:]):
            prev_annotation = cleaned_annotations[-1]
            if annotation['start'] > prev_annotation['end'] + buffer:
                cleaned_annotations.append(annotation)
            elif annotation['start'] - prev_annotation['end'] > 0 and annotation['tag'] == prev_annotation['tag']:
                if ',' not in text[prev_annotation['end']:annotation['start']]:
                    annotation = {'tag': annotation['tag'], 'start': prev_annotation['start'], 'end': annotation['end']}
                cleaned_annotations.append(annotation)
            else:
                annotation_length = annotation['end'] - annotation['start']
                previous_length = prev_annotation['end'] - prev_annotation['start']
                if annotation_length > previous_length:
                    cleaned_annotations[-1] = annotation
        return cleaned_annotations

    def fit_to_validation(self):
        output_json = {'result': []}
        results = []
        jsons_to_save = False
        for kingdom in ['animal', 'plant']:
            validation_data = self.data.read_json(kingdom, 'validation')
            for item in validation_data['result']:
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content'].lower()
                output_item = {'example': item['example'], 'results': {'annotations': [], 'classifications': []}}
                doc = self.nlp(text)
                nouns = list(set([chunk.text for chunk in doc.noun_chunks]))
                nums = list(set([token.lemma_ for token in doc if token.pos_ == "NUM"]))
                for bioconcept in Data.BIOCONCEPTS_BY_KINGDOM[kingdom]:
                    for noun in nouns:
                        if noun in self.entities_by_bioconcept[bioconcept]:
                            annotations = self.find_matches_and_make_annotations(noun, text, bioconcept)
                            if bioconcept == 'PATHOGENIC_ORGANISMS':
                                annotations = self.add_partial_initials(noun, text, bioconcept, annotations)
                            #annotations = self.add_acronyms(noun, text, annotations)
                            output_item['results']['annotations'].extend(annotations)
                        elif bioconcept in ['PLANT_SPECIES', 'PLANT_PEST']:
                            eppo_annotations, jsons_to_save = self.add_eppo_data(noun, text, bioconcept, jsons_to_save)
                            output_item['results']['annotations'].extend(eppo_annotations)
                    for num in nums:
                        is_valid_year = self.check_valid_year(num)
                        add_year = bioconcept == 'YEAR' and is_valid_year
                        is_possible_prevalence = self.check_prevalence(num)
                        add_prevalence = bioconcept == 'PREVALENCE' and is_possible_prevalence
                        if add_year or add_prevalence:
                            year_prevalence_annotations = self.find_matches_and_make_annotations(num, text, bioconcept)
                            output_item['results']['annotations'].extend(year_prevalence_annotations)
                if jsons_to_save:
                    self.data.save_json(self.eppo.nouns_not_in_eppo_path, self.eppo.nouns_not_in_eppo)
                    self.data.save_json(self.accepted_eppo_nouns_path, self.accepted_eppo_nouns_by_bioconcept)
                    self.data.save_json(self.not_accepted_eppo_nouns_path, self.not_accepted_eppo_nouns_by_bioconcept)
                    jsons_to_save = False
                annotations = output_item['results']['annotations']
                if annotations:
                    cleaned_annotations = self.clean_annotations(annotations, text)
                    output_item['results']['annotations'] = cleaned_annotations
                result = {
                    'text': text,
                    'true': item['results']['annotations'],
                    'pred': output_item['results']['annotations']}
                results.append(result)
                output_json['result'].append(output_item)
        self.data.save_json(self.output_path, output_json)
        return results

    def run(self):
        results = self.fit_to_validation()
        evaluation = Evaluation(results, verbose=False)
        evaluation.run()


if __name__ == '__main__':
    SpacyWithAPIs().run()

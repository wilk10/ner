import re
from spacy_deep_learning import SpacyDeepLearning
from utils.data import Data
from utils.eppo import Eppo
from utils.cat_life import CatLife


class Normalisation:
    NOMENCLATURE_BY_KINGDOM = {'plant': 'EPPO Global Database', 'animal': 'Catalogue of Life'}

    def __init__(self):
        self.data = Data()
        self.eppo = Eppo(time_to_sleep=0)
        self.cat_life = CatLife(time_to_sleep=0)

    def get_eppo_code(self, entity):
        concept_id = None
        response = self.eppo.get_search_result(entity, typeorg=1)
        if response is not None:
            latin_or_english_names = [result for result in response if result['lang'] in ['la', 'en']]
            if latin_or_english_names:
                concept_id = latin_or_english_names[0]['eppocode']
        return concept_id

    @staticmethod
    def find_matching_full_version_entity(text, entity):
        for char in ['.', ',', ':', ';', '(', ')', '[', ']']:
            text = text.replace(char, '')
        full_version_entity = None
        entity_words = entity.lower().split()
        text_words = text.lower().split()
        for j, word in enumerate(text_words):
            if word == entity_words[-1] and j != 0:
                previous_word = text_words[j-1]
                if previous_word[0] == entity_words[0][0] and len(previous_word) > 2:
                    full_version_entity = ' '.join([previous_word, entity_words[-1]])
        return full_version_entity

    def get_concept_id(self, entity, bioconcept, text):
        if bioconcept == 'PLANT_SPECIES':
            concept_id = self.get_eppo_code(entity)
            if concept_id is None:
                entity_words = entity.split()
                if len(entity_words) == 2 and entity_words[0][1] == '.':
                    full_entity = self.find_matching_full_version_entity(text, entity)
                    if full_entity is not None:
                        print(f'!!! found matching full entity: {entity} -> {full_entity}')
                        concept_id = self.get_eppo_code(full_entity)
            print(f'ENTITY: {entity}. conceptID: {concept_id}')
        else:
            assert bioconcept == 'TARGET_SPECIES'
            response = self.cat_life.get_response(entity)
            # if any result, need to look for the ones that have a name that equals the entity.
            #   having the entity contained in the name is not enough
            # another check is "source_database" == "ITIS Global: The Integrated Taxonomic Information System"
            #   or at least give more weight to those results
            # the look for the ID in either the result itself, or inside "accepted_name" dictionary
            import pdb
            pdb.set_trace()
            concept_id = None
        return concept_id

    def run(self):
        n_items_by_bioconcept = {'LOCATION': 20000, 'YEAR': 7000}
        spacy_deep_learning = SpacyDeepLearning(n_items_by_bioconcept)
        model_dir_by_bioconcept = spacy_deep_learning.train_models_or_get_dirs()
        for kingdom in ['plant', 'animal']:
            validation_data = self.data.read_json(kingdom, 'validation')
            kingdom_classifications = []
            for i, item in enumerate(validation_data['result']):
                if 'content' not in item['example'].keys():
                    continue
                item_text = item['example']['content']
                print(f'{i}: {item_text}')
                predictions = spacy_deep_learning.make_predictions(kingdom, model_dir_by_bioconcept, item_text)
                item_classifications = []
                for prediction in predictions:
                    bioconcept = prediction['tag'].upper()
                    if bioconcept in ['PLANT_SPECIES', 'TARGET_SPECIES']:
                        named_entity = item_text[prediction['start']:prediction['end']]
                        concept_id = self.get_concept_id(named_entity, bioconcept, item_text)
                        classification = {
                            'named_entity': named_entity,
                            'Nomenclature': self.NOMENCLATURE_BY_KINGDOM[kingdom],
                            'conceptID': concept_id}
                        item_classifications.append(classification)
                input()
                kingdom_classifications.append(item_classifications)
            print(kingdom_classifications)


if __name__ == '__main__':
    Normalisation().run()

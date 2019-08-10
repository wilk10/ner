import os
import time
import requests
import numpy as np
import urllib.parse
from utils.data import Data


class Eppo:
    URL = 'https://data.eppo.int/api/rest/1.0'
    OUTPUT_FILE_NAME = 'entity_taxonomy_by_bioconcept.json'
    NOUNS_NOT_IN_EPPO_FILE_NAME = 'nouns_not_in_eppo.json'
    EPPO_DATATYPES = ['GAF', 'PFL', 'GAI', 'SIT', 'SFT', 'SPT']
    CLASS_BY_LANGUAGE = {'LA': 1, 'EN': 2}
    EPPO_INFO_KEYS = ['categorization', 'distribution', 'pests', 'hosts']

    def __init__(self, time_to_sleep=0.5):
        self.time_to_sleep = time_to_sleep
        self.token = os.getenv('EPPO_TOKEN', '')
        self.data = Data()
        self.nouns_not_in_eppo_path = self.data.dict_dir / self.NOUNS_NOT_IN_EPPO_FILE_NAME
        self.nouns_not_in_eppo = self.data.load_json(self.nouns_not_in_eppo_path)
        self.entity_taxonomies_by_bioconcept_path = self.data.dict_dir / self.OUTPUT_FILE_NAME

    @staticmethod
    def check_response(response, is_search):
        if isinstance(response, list) and not response:
            result = None
        elif isinstance(response, list) and isinstance(response[0], dict):
            result = response
        elif isinstance(response, dict) and is_search:
            result = None
            print(response)
        else:
            result = response
        return result

    def make_call(self, url, params, is_search=False):
        parsed_params = urllib.parse.urlencode(params)
        response = requests.get(url, params=parsed_params).json()
        time.sleep(self.time_to_sleep)
        return self.check_response(response, is_search)

    def get_search_result(self, entity):
        params = {'authtoken': self.token, 'kw': entity, 'searchfor': 3, 'searchmode': 1, 'typeorg': 0}
        url = self.URL + f'/tools/search'
        return self.make_call(url, params, is_search=True)

    def get_taxonomy(self, eppo_code):
        params = {'authtoken': self.token}
        url = self.URL + f'/taxon/{eppo_code}/taxonomy'
        response = self.make_call(url, params)
        if response is not None:
            level = response[0]['level']
            assert level == 1, f'level {level} is not 1'
            return response

    def get_eppocode_infos(self, eppo_code):
        params = {'authtoken': self.token}
        url = self.URL + f'/taxon/{eppo_code}'
        response = self.make_call(url, params)
        if response is not None:
            assert isinstance(response, dict)
            return [response['attached_infos'][key] for key in self.EPPO_INFO_KEYS]
        else:
            return [np.nan] * 4

    @staticmethod
    def clean_the_entity(entity):
        chunks = entity.split(' ')
        clean_chunks = [chunk for chunk in chunks if chunk != 'the']
        return ' '.join(clean_chunks)

    @staticmethod
    def check_valid_keyword(entity):
        is_long_enough = len(entity) >= 3
        has_valid_characters = all([char not in ['(', ')', '[', ']', '{', '}', '\\', ','] for char in entity])
        has_no_digits = all([not char.isdigit() for char in entity])
        return is_long_enough and has_valid_characters and has_no_digits

    def get_eppo_code_and_taxonomy(self, entity):
        clean_entity = self.clean_the_entity(entity)
        valid_keyword = self.check_valid_keyword(clean_entity)
        if valid_keyword:
            response = self.get_search_result(clean_entity)
            eppo_code = response[0]['eppocode']
            if eppo_code is not None:
                taxonomy = self.get_taxonomy(eppo_code)
                level1_taxonomy = taxonomy[0]['prefname']
                if level1_taxonomy is not None:
                    return level1_taxonomy
        return None

    def save_taxonomy_data_to_json(self):
        entities_by_bioconcept = self.data.learn_training_entries()
        level1_taxonomy_by_bioconcept = {bioconcept: [] for bioconcept in Data.BIOCONCEPTS_BY_KINGDOM['plant']}
        level1_taxonomy_by_bioconcept['PATHOGENIC_ORGANISMS'] = []
        for bioconcept, entities in entities_by_bioconcept.items():
            if bioconcept in level1_taxonomy_by_bioconcept.keys():
                print(f'working on {bioconcept}')
                for entity in entities:
                    level1_taxonomy = self.get_eppo_code_and_taxonomy(entity)
                    if level1_taxonomy is not None:
                        level1_taxonomy_by_bioconcept[bioconcept].append([entity, level1_taxonomy])
                self.data.save_json(self.entity_taxonomies_by_bioconcept_path, level1_taxonomy_by_bioconcept)

    def get_eppo_search_xs(self, result):
        eppo_code = result['eppocode']
        if result['type'] not in self.EPPO_DATATYPES:
            datatype = 6
        else:
            datatype = self.EPPO_DATATYPES.index(result['type'])
        is_preferred = result['ispreferred']
        language = result['lang']
        language_class = 0 if language not in self.CLASS_BY_LANGUAGE.keys() else self.CLASS_BY_LANGUAGE[language]
        is_active = result['isactive']
        return [eppo_code, datatype, is_preferred, language_class, is_active]

    def return_features(self, entity, n_columns):
        if entity not in self.nouns_not_in_eppo and len(entity) >= 3:
            response = self.get_search_result(entity)
            if response is not None:
                eppo_xs = [len(response)]
                eppo_search_xs = self.get_eppo_search_xs(response[0])
                eppo_xs.extend(eppo_search_xs)
                eppo_code = eppo_search_xs[0]
                taxonomy = self.get_taxonomy(eppo_code)
                if taxonomy:
                    lowest_level_taxonomy = taxonomy[-1]['level']
                else:
                    lowest_level_taxonomy = 0
                eppo_xs.append(lowest_level_taxonomy)
                eppo_info_xs = self.get_eppocode_infos(eppo_code)
                eppo_xs.extend(eppo_info_xs)
            else:
                eppo_xs = [0] + [np.nan] * (n_columns - 1)
        else:
            eppo_xs = [0] + [np.nan] * (n_columns - 1)
        return eppo_xs


if __name__ == '__main__':
    Eppo().save_taxonomy_data_to_json()

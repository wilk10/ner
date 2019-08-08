import os
import json
import pathlib


class Data:
    DATA_DIR_NAME = 'Challenge_9934213_data_v2'
    DIR_NAME_BY_KINGDOM = {'animal': 'DATASET_AHAW_for_Challenge', 'plant': 'DATASET_PLH_for_Challenge'}
    BIOCONCEPTS_BY_KINGDOM = {
        'plant': ['PLANT_PEST', 'PLANT_SPECIES', 'PLANT_DISEASE_COMMNAME'],
        'animal': ['PATHOGENIC_ORGANISMS', 'TARGET_SPECIES', 'LOCATION', 'PREVALENCE', 'YEAR', 'ANMETHOD']}
    DICTIONARIES_DIR_NAME = 'dictionaries'
    HELP_DICT_NAME = 'excluded_bioconcepts_by_entity.json'

    def __init__(self):
        self.cwd = pathlib.Path.cwd()
        self.dict_dir = self.cwd / self.DICTIONARIES_DIR_NAME
        self.bioconcepts = [bc for kingdom, bioconcepts in self.BIOCONCEPTS_BY_KINGDOM.items() for bc in bioconcepts]
        self.excluded_bioconcepts_by_entity = self.load_help_dict()

    def load_help_dict(self):
        file_path = self.dict_dir / self.HELP_DICT_NAME
        with open(str(file_path)) as f:
            data = json.load(f)
        return data

    def get_target_dir(self, kingdom):
        assert kingdom in ['animal', 'plant']
        dir_name = self.DIR_NAME_BY_KINGDOM[kingdom]
        return self.cwd / self.DATA_DIR_NAME / dir_name

    @staticmethod
    def get_file_path(target_dir, file_name_chunk):
        files = os.listdir(str(target_dir))
        target_files = [file for file in files if file_name_chunk in file]
        assert len(target_files) == 1
        target_file = target_files[0]
        return target_dir / target_file

    def read_text(self, kingdom, phase):
        assert phase in ['training', 'validation']
        target_dir = self.get_target_dir(kingdom)
        file_path = self.get_file_path(target_dir, f'{phase}set_text')
        with open(str(file_path), 'r') as f:
            entries = [entry.rstrip() for entry in f]
        return entries

    def read_json(self, kingdom, phase):
        assert phase in ['training', 'validation']
        target_dir = self.get_target_dir(kingdom)
        file_path = self.get_file_path(target_dir, f'{phase}set_annotations')
        with open(str(file_path), encoding='utf-8') as f:
            data = json.load(f)
        return data

    def clean_training_data(self, entities_by_bioconcept):
        unique_entities_by_bioconcept = {
            bioconcept: list(set(entities)) for bioconcept, entities in entities_by_bioconcept.items()}
        clean_entities_by_bioconcept = {bioconcept: [] for bioconcept in self.bioconcepts}
        for bioconcept, entities in unique_entities_by_bioconcept.items():
            for entity in entities:
                if entity not in self.excluded_bioconcepts_by_entity.keys():
                    clean_entities_by_bioconcept[bioconcept].append(entity)
                elif bioconcept not in self.excluded_bioconcepts_by_entity[entity]:
                    clean_entities_by_bioconcept[bioconcept].append(entity)
        return clean_entities_by_bioconcept

    def learn_training_entries(self):
        entities_by_bioconcept = {bioconcept: [] for bioconcept in self.bioconcepts}
        for kingdom in ['animal', 'plant']:
            training_data = self.read_json(kingdom, 'training')
            for item in training_data['result']:
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content']
                annotations = item['results']['annotations']
                for annotation in annotations:
                    named_entity = f"{text[annotation['start']:annotation['end']]}"
                    clean_named_entity = named_entity.lower().strip()
                    bioconcept = annotation['tag'].upper().strip()
                    if clean_named_entity:
                        entities_by_bioconcept[bioconcept].append(clean_named_entity)
        clean_entities_by_bioconcept = self.clean_training_data(entities_by_bioconcept)
        return clean_entities_by_bioconcept

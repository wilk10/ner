import os
import json
import pathlib


class Data:
    DATA_DIR_NAME = 'Challenge_9934213_data_v2'
    DIR_NAME_BY_FLAG = {'animal': 'DATASET_AHAW_for_Challenge', 'plant': 'DATASET_PLH_for_Challenge'}
    BIOCONCEPTS_BY_FLAG = {
        'plant': ['PLANT_PEST', 'PLANT_SPECIES', 'PLANT_DISEASE_COMMNAME'],
        'animal': ['PATHOGENIC_ORGANISMS', 'TARGET_SPECIES', 'LOCATION', 'PREVALENCE', 'YEAR', 'ANMETHOD']}

    def __init__(self):
        self.cwd = pathlib.Path.cwd()

    def get_target_dir(self, flag):
        assert flag in ['animal', 'plant']
        dir_name = self.DIR_NAME_BY_FLAG[flag]
        return self.cwd / self.DATA_DIR_NAME / dir_name

    @staticmethod
    def get_file_path(target_dir, file_name_chunk):
        files = os.listdir(str(target_dir))
        target_files = [file for file in files if file_name_chunk in file]
        assert len(target_files) == 1
        target_file = target_files[0]
        return target_dir / target_file

    def read_text(self, flag, phase):
        assert phase in ['training', 'validation']
        target_dir = self.get_target_dir(flag)
        file_path = self.get_file_path(target_dir, f'{phase}set_text')
        with open(str(file_path), "r") as f:
            entries = [entry.rstrip() for entry in f]
        return entries

    def read_json(self, flag, phase):
        assert phase in ['training', 'validation']
        target_dir = self.get_target_dir(flag)
        file_path = self.get_file_path(target_dir, f'{phase}set_annotations')
        with open(str(file_path), encoding='utf-8') as f:
            data = json.load(f)
        return data

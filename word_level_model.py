import spacy
import pandas
import numpy as np
from utils.data import Data
from utils.eppo import Eppo
from utils.cat_life import CatLife
from utils.evaluation import Evaluation


class WordLevelModel:
    MODEL = 'en_core_web_sm'
    COLUMNS_BY_SOURCE = {
        'y': ['PLPE', 'PLSP', 'PLDI', 'PAOR', 'TASP', 'LOCA', 'PREV', 'YEAR', 'ANME'],
        'general': ['kingdom', 'item_id', 'entity_id', 'length', 'n_occurr', 'in_first_100'],
        'span': ['sentiment', 'n_tokens', 'vector_norm'],
        'token': [
            'token_i', 'any_digits', 'all_digits', 'any_like_num', 'any_title', 'any_oov', 'any_propn', 'any_verb',
            'any_nsubj', 'any_dobj', 'any_compound'],
        'eppo': [
            'eppo_n_results', 'eppo_code', 'eppo_datatype', 'is_preferred', 'lang', 'active', 'taxonomy_level',
            'n_categ', 'n_distrib', 'n_pests', 'n_hosts'],
        'cat': ['cat_n_results', 'accepted_name', 'is_extinct']}
    DF_FILE_NAME = 'entities_dataframe.csv'
    COLUMNS_TO_NOT_ANALYSE = ['kingdom', 'item_id', 'entity_id', 'eppo_code']

    def __init__(self):
        self.data = Data()
        self.entities_by_bioconcept = self.data.learn_training_entries()
        self.nlp = spacy.load(self.MODEL)
        self.eppo = Eppo(time_to_sleep=0.1)
        self.cat = CatLife(time_to_sleep=0)
        self.columns = [column for _, columns in self.COLUMNS_BY_SOURCE.items() for column in columns]
        self.entities_df_path = self.data.cwd / self.DF_FILE_NAME

    def find_tag_if_any(self, entity, annotated_entities, tags):
        tag = None
        if entity in annotated_entities:
            tag = tags[annotated_entities.index(entity)]
            if entity in self.data.excluded_bioconcepts_by_entity.keys():
                if tag == self.data.excluded_bioconcepts_by_entity[entity]:
                    tag = None
        for annotated_entity in annotated_entities:
            if entity in annotated_entity or annotated_entity in entity:
                tag = tags[annotated_entities.index(annotated_entity)]
        return tag

    def get_ys(self, tag):
        if tag is not None:
            bits = tag.split('_')[:2]
            if len(bits) == 2:
                y_name = ''.join([bit[:2] for bit in bits])
            else:
                assert len(bits) == 1
                y_name = bits[0][:4]
            ys = [1 if y_col == y_name else 0 for y_col in self.COLUMNS_BY_SOURCE['y']]
        else:
            ys = np.zeros(len(self.COLUMNS_BY_SOURCE['y']))
        return list(ys)

    def extract_data_nouns(self, span, tag, doc, k, i, j):
        ys = self.get_ys(tag)
        chunks = [chunk.lower_ for chunk in doc.noun_chunks]
        general_xs = [k, i, j, len(span.text), chunks.count(span.lower_), int(span.lower_ in doc.text[:100].lower())]
        span_xs = [span.sentiment, len([token for token in span]), span.vector_norm]
        tokens = [token for token in span]
        token_1_xs = [
            tokens[0].i, any([t.is_digit for t in tokens]), all([t.is_digit for t in tokens]),
            any([t.like_num for t in tokens]), any([t.is_title for t in tokens]), any([t.is_oov for t in tokens])]
        token_1_xs = [int(val) for val in token_1_xs]
        token_2_xs = [
            any([t.pos == 'PROPN' for t in tokens]), any([t.pos == 'VERB' for t in tokens]),
            any([t.dep == 'nsubj' for t in tokens]), any([t.dep == 'dobj' for t in tokens]),
            any([t.dep == 'compound' for t in tokens])]
        token_2_xs = [int(val) for val in token_2_xs]
        eppo_xs = self.eppo.return_features(span.lower_, len(self.COLUMNS_BY_SOURCE['eppo']))
        cat_response = self.cat.get_response(span.lower_)
        cat_xs = [cat_response['total_number_of_results'], np.nan, np.nan]
        if 'results' in cat_response.keys():
            cat_results = cat_response['results'][0]
            if 'name_status' in cat_results.keys():
                cat_xs[1] = int(cat_results['name_status'] == 'accepted name')
            if 'is_extinct' in cat_results.keys():
                cat_xs[2] = int(cat_results['is_extinct'] == 'true')
        data = ys + general_xs + span_xs + token_1_xs + token_2_xs + eppo_xs + cat_xs
        return pandas.Series(data, index=self.columns)

    def extract_data_nums(self, token, tag, doc, k, i, j):
        ys = self.get_ys(tag)
        chunks = [token.lower_ for token in doc]
        general_xs = [k, i, j, len(token.text), chunks.count(token.lower_), int(token.lower_ in doc.text[:100].lower())]
        span_xs = [token.sentiment, 1, token.vector_norm]
        token_1_xs =[token.i, token.is_digit, token.is_digit, token.like_num, token.is_title, token.is_oov]
        token_1_xs = [int(val) for val in token_1_xs]
        token_2_xs = [
            token.pos == 'PROPN', token.pos == 'VERB', token.dep == 'nsubj', token.dep == 'dobj',
            token.dep == 'compound']
        token_2_xs = [int(val) for val in token_2_xs]
        eppo_xs = [np.nan] * len(self.COLUMNS_BY_SOURCE['eppo'])
        cat_xs = [np.nan] * len(self.COLUMNS_BY_SOURCE['cat'])
        data = ys + general_xs + span_xs + token_1_xs + token_2_xs + eppo_xs + cat_xs
        return pandas.Series(data, index=self.columns)

    def extract_training_features(self):
        df = pandas.read_csv(self.entities_df_path, index_col=0)
        #df = pandas.DataFrame(columns=self.columns)
        for kingdom in ['animal', 'plant']:
            training_data = self.data.read_json(kingdom, 'training')
            for i, item in enumerate(training_data['result']):
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content']
                doc = self.nlp(text)
                nouns = [chunk for chunk in doc.noun_chunks]
                nums = [token for token in doc if token.pos_ == "NUM"]
                annotations = item['results']['annotations']
                sorted_annotations = sorted(annotations, key=lambda a: a['start'])
                annotated_entities = [text[a['start']:a['end']].lower().strip() for a in sorted_annotations]
                tags = [a['tag'].upper().strip() for a in sorted_annotations]
                entities_by_flag = {"nouns": nouns, "nums": nums}
                j = 0
                for flag, entities in entities_by_flag.items():
                    for entity in entities:
                        j += 1
                        if len(df[(df.kingdom == kingdom) & (df.item_id == i) & (df.entity_id == j)]) > 0:
                            continue
                        print(f'{kingdom}s: item {i} entity {j}: {entity.lower_}')
                        tag = self.find_tag_if_any(entity.lower_, annotated_entities, tags)
                        if flag == "nouns":
                            row = self.extract_data_nouns(entity, tag, doc, kingdom, i, j)
                        else:
                            row = self.extract_data_nums(entity, tag, doc, kingdom, i, j)
                        row.name = entity.lower_
                        df = df.append(row)
                df.to_csv(self.entities_df_path)
        return df

    def fit_to_validation(self, df):
        output_json = {'result': []}
        results = []
        jsons_to_save = False
        for kingdom in ['animal', 'plant']:
            validation_data = self.data.read_json(kingdom, 'validation')
            for item in validation_data['result']:
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content']
                output_item = {'example': item['example'], 'results': {'annotations': [], 'classifications': []}}
                doc = self.nlp(text)
                # make features

    def run(self):
        df = self.extract_training_features()
        results = self.fit_to_validation(df)
        evaluation = Evaluation(results, verbose=False)
        evaluation.run()


if __name__ == '__main__':
    WordLevelModel().run()

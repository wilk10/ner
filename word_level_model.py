import spacy
import pandas
import argparse
import numpy as np
import statsmodels.api
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils.data import Data
from utils.eppo import Eppo
from utils.cat_life import CatLife
from utils.evaluation import Evaluation


class WordLevelModel:
    MODEL = 'en_core_web_sm'
    COLUMNS_BY_SOURCE = {
        'info': ['entity', 'kingdom', 'item_id', 'entity_id'],
        'y': ['PLPE', 'PLSP', 'PLDI', 'PAOR', 'TASP', 'LOCA', 'PREV', 'YEAR', 'ANME'],
        'general': ['length', 'n_occurr', 'in_first_100'],
        'span': ['sentiment', 'n_tokens', 'vector_norm'],
        'token': [
            'token_i', 'any_digits', 'all_digits', 'any_like_num', 'any_title', 'any_oov', 'any_propn', 'any_verb',
            'any_nsubj', 'any_dobj', 'any_compound'],
        'eppo': [
            'eppo_n_results', 'eppo_code', 'eppo_type', 'is_preferred', 'lang', 'active', 'taxonomy', 'n_categ',
            'n_distrib', 'n_pests', 'n_hosts'],
        'cat': ['cat_n_results', 'is_accepted', 'is_extinct']}
    DF_FILE_NAME = 'entities_dataframe.csv'
    COLUMNS_TO_NOT_ANALYSE = ['kingdom', 'item_id', 'entity_id', 'eppo_code']
    COLUMNS_TO_EXCLUDE = ['sentiment', 'any_oov']
    MANUALLY_EXCLUDED_FEATURES_BY_Y = {'YEAR': ['n_pests'], 'PAOR': ['n_categ', 'n_hosts'], 'PLPE': ['n_distrib']}

    def __init__(self, sampling='NONE', feature_selection='AUTO', do_extract_features='NO'):
        self.sampling = sampling.upper()
        assert self.sampling in ['UP', 'DOWN', 'SMOTE', 'NONE']
        self.feature_selection = feature_selection.upper()
        assert self.feature_selection in ['AUTO', 'MANUAL']
        do_extract_features_input = do_extract_features.upper()
        assert do_extract_features_input in ['YES', 'NO']
        self.do_extract_features = True if do_extract_features_input == 'YES' else False
        self.data = Data()
        self.entities_by_bioconcept = self.data.learn_training_entries()
        self.nlp = spacy.load(self.MODEL)
        self.eppo = Eppo(time_to_sleep=0)
        self.cat = CatLife(time_to_sleep=0)
        self.columns = [column for _, columns in self.COLUMNS_BY_SOURCE.items() for column in columns]
        self.entities_df_path = self.data.cwd / self.DF_FILE_NAME
        self.df = self.get_training_df()
        pandas.set_option("display.max_rows", 500)
        pandas.set_option("display.max_columns", 50)
        pandas.set_option('display.width', 200)

    def get_training_df(self):
        if self.do_extract_features:
            if self.entities_df_path.exists():
                input_df = pandas.read_csv(self.entities_df_path, index_col=0)
            else:
                input_df = pandas.DataFrame(columns=self.columns)
            df = self.extract_training_features(input_df)
        else:
            columns_to_use = [column for column in self.columns if column not in self.COLUMNS_TO_EXCLUDE]
            df = pandas.read_csv(self.entities_df_path, index_col=0, usecols=columns_to_use)
        return df

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
        info = [span.lower_, k, i, j]
        ys = self.get_ys(tag)
        chunks = [chunk.lower_ for chunk in doc.noun_chunks]
        general_xs = [len(span.text), chunks.count(span.lower_), int(span.lower_ in doc.text[:100].lower())]
        span_xs = [span.sentiment, len([token for token in span]), span.vector_norm]
        tokens = [token for token in span]
        token_1_xs = [
            tokens[0].i, any([t.is_digit for t in tokens]), all([t.is_digit for t in tokens]),
            any([t.like_num for t in tokens]), any([t.is_title for t in tokens]), any([t.is_oov for t in tokens])]
        token_1_xs = [int(val) for val in token_1_xs]
        token_2_xs = [
            any([t.pos_ == 'PROPN' for t in tokens]), any([t.pos_ == 'VERB' for t in tokens]),
            any([t.dep_ == 'nsubj' for t in tokens]), any([t.dep_ == 'dobj' for t in tokens]),
            any([t.dep_ == 'compound' for t in tokens])]
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
        data = info + ys + general_xs + span_xs + token_1_xs + token_2_xs + eppo_xs + cat_xs
        return pandas.Series(data, index=self.columns)

    def extract_data_nums(self, token, tag, doc, k, i, j):
        info = [token.lower_, k, i, j]
        ys = self.get_ys(tag)
        chunks = [token.lower_ for token in doc]
        general_xs = [len(token.text), chunks.count(token.lower_), int(token.lower_ in doc.text[:100].lower())]
        span_xs = [token.sentiment, 1, token.vector_norm]
        token_1_xs =[token.i, token.is_digit, token.is_digit, token.like_num, token.is_title, token.is_oov]
        token_1_xs = [int(val) for val in token_1_xs]
        token_2_xs = [
            token.pos_ == 'PROPN', token.pos_ == 'VERB', token.dep_ == 'nsubj', token.dep_ == 'dobj',
            token.dep_ == 'compound']
        token_2_xs = [int(val) for val in token_2_xs]
        eppo_xs = [0] + [np.nan] * (len(self.COLUMNS_BY_SOURCE['eppo']) - 1)
        cat_xs = [0] + [np.nan] * (len(self.COLUMNS_BY_SOURCE['cat']) - 1)
        data = info + ys + general_xs + span_xs + token_1_xs + token_2_xs + eppo_xs + cat_xs
        return pandas.Series(data, index=self.columns)

    def extract_training_features(self, df):
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
                        df = df.append(row, ignore_index=True)
                df.to_csv(self.entities_df_path)
        return df

    '''
    def explore_features(self):
        for column in self.df.columns:
            print(f'{self.df[column].value_counts()}\n')
        for i, this_y in enumerate(self.COLUMNS_BY_SOURCE['y']):
            print(f'{this_y}')
            other_ys = [y for y in self.COLUMNS_BY_SOURCE['y'] if y != this_y]
            selected_columns = [column for column in self.df.columns if column not in other_ys]
            kingdom = 'plant' if i <= 2 else 'animal'
            kingdom_df = self.df.loc[self.df.kingdom == kingdom]
            y_df = kingdom_df.loc[:, selected_columns]
            for column in selected_columns:
                group_by = y_df.groupby([column, this_y]).size()
                print(group_by)
    '''

    def fix_data(self):
        corrected_df = self.df.copy()
        for column in ['lang', 'active', 'taxonomy', 'n_categ', 'n_distrib', 'n_pests', 'n_hosts', 'is_accepted']:
            column_data = self.df[column]
            corrected_data = column_data.fillna(0)
            corrected_df[column] = corrected_data
        for column in ['is_preferred', 'is_extinct']:
            column_data = self.df[column]
            corrected_data = column_data.fillna(1)
            corrected_df[column] = corrected_data
        eppo_type_data = self.df['eppo_type']
        eppo_type_data += 1
        eppo_type_data.loc[eppo_type_data == 7.0] = 0
        eppo_type_data = eppo_type_data.fillna(0)
        corrected_df['eppo_type'] = eppo_type_data
        return corrected_df

    def investigate_shit_features(self, kingdom_df):
        kingdom = list(set(kingdom_df['kingdom']))[0]
        training_data = self.data.read_json(kingdom, 'training')
        sampled_item_ids = np.random.choice(range(0, len(training_data['result'])), 10)
        for item_id in sampled_item_ids:
            item = training_data['result'][item_id]
            if 'content' not in item['example'].keys():
                continue
            text = item['example']['content']
            item_df = kingdom_df.loc[kingdom_df['item_id'] == item_id]
            annotations = item['results']['annotations']
            sorted_annotations = sorted(annotations, key=lambda a: a['start'])
            annotated_entities = [text[a['start']:a['end']].lower().strip() for a in sorted_annotations]
            tags = [a['tag'].upper().strip() for a in sorted_annotations]
            print(text)
            print(item_df)
            print(annotated_entities)
            print(tags)
            import pdb
            pdb.set_trace()

    @staticmethod
    def remove_correlated_features(df, threshold=0.7):
        columns_to_exclude = []
        correlation_matrix = df.corr()
        y_column = correlation_matrix.columns[0]
        for column in correlation_matrix.columns:
            other_columns = [c for c in correlation_matrix.columns if c != column]
            series = correlation_matrix[column][other_columns]
            features_to_check = [f for f, corr in series.items() if corr > threshold or corr < -threshold] + [column]
            if len(features_to_check) > 1:
                correlations_with_y = correlation_matrix.loc[y_column, features_to_check]
                id_max_correlation = np.abs(correlations_with_y).values.argmax()
                column_to_keep = correlations_with_y.index.values[id_max_correlation]
                columns_to_exclude_now = [i for i in correlations_with_y.index.values if i != column_to_keep]
                columns_to_exclude.extend(columns_to_exclude_now)
        columns_to_exclude_final = list(set(columns_to_exclude))
        columns_to_keep = [c for c in df.columns if c not in columns_to_exclude_final]
        final_df = df.loc[:, columns_to_keep]
        assert final_df.columns[0] == y_column
        print(f'{y_column}: excluded these columns for high correlation: {columns_to_exclude_final}')
        return final_df

    def preprocess_dataset_and_split(self, kingdom_df, selected_columns):
        y_full_df = kingdom_df.loc[:, selected_columns]
        selected_df = y_full_df.loc[:, selected_columns].drop_duplicates()
        this_y = selected_df.columns[0]
        if this_y in self.MANUALLY_EXCLUDED_FEATURES_BY_Y.keys():
            manually_excluded_features = self.MANUALLY_EXCLUDED_FEATURES_BY_Y[this_y]
            custom_columns = [c for c in selected_df.columns if c not in manually_excluded_features]
            selected_df = selected_df.loc[:, custom_columns]
        y_df = self.remove_correlated_features(selected_df)
        y = y_df[this_y]
        x_columns = [c for c in y_df.columns if c != this_y]
        X = y_df.loc[:, x_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_x_y_from_resampled_df(df, y_name):
        print(df[y_name].value_counts())
        y = df[y_name]
        X = df.drop(y_name, axis=1)
        return X, y

    def up_or_down_sample(self, X, y):
        df = pandas.concat([X, y], axis=1)
        negative_df = df[df[y.name] == 0]
        positive_df = df[df[y.name] == 1]
        if self.sampling == 'UP':
            n_samples = max(len(negative_df) // 2, len(positive_df))
            positive_df_upsampled = resample(positive_df, replace=True, n_samples=n_samples, random_state=42)
            resampled_df = pandas.concat([negative_df, positive_df_upsampled])
            output_X, output_y = self.get_x_y_from_resampled_df(resampled_df, y.name)
        elif self.sampling in ['DOWN', 'SMOTE']:
            factor = 2 if self.sampling == 'DOWN' else 4
            n_samples = min(len(positive_df) * factor, len(negative_df))
            negative_df_downsampled = resample(negative_df, replace=False, n_samples=n_samples, random_state=42)
            resampled_df = pandas.concat([negative_df_downsampled, positive_df])
            output_X, output_y = self.get_x_y_from_resampled_df(resampled_df, y.name)
            if self.sampling == 'SMOTE':
                smote = SMOTE(random_state=42, sampling_strategy=(2 / factor))
                smote_X, smote_y = smote.fit_resample(output_X, output_y)
                output_X = pandas.DataFrame(smote_X, columns=X.columns)
                output_y = pandas.DataFrame(smote_y, columns=[y.name])
                print(f'output_X.shape: {output_X.shape} | output_y.shape: {output_y.shape}')
        else:
            assert self.sampling == 'NONE'
            output_X = X
            output_y = y
        return output_X, output_y

    def auto_or_manual_feature_selection(self, estimator, X, y):
        if self.feature_selection == 'AUTO':
            rfe = RFE(estimator, 7)
            rfe = rfe.fit(X, y)
            remaining_features = X.columns[rfe.support_]
            output_X = X.loc[:, remaining_features]
        else:
            assert self.feature_selection == 'MANUAL'
            output_X = X
        return output_X

    @staticmethod
    def fit_and_check_singular_matrix(logit_model, X, y):
        try:
            result = logit_model.fit(maxiter=200)
        except np.linalg.linalg.LinAlgError:
            df = X.copy()
            df[y.name] = y
            for col in X.columns:
                print(df.groupby([y.name, col]).size())
            import pdb
            pdb.set_trace()
        return result

    def train(self, df):
        metrics_by_y = dict.fromkeys(self.COLUMNS_BY_SOURCE['y'])
        for i, this_y in enumerate(self.COLUMNS_BY_SOURCE['y']):
            print(f'\n{this_y}\n')
            other_ys = [y for y in self.COLUMNS_BY_SOURCE['y'] if y != this_y]
            selected_columns = [
                c for c in df.columns
                if c not in other_ys and c not in self.COLUMNS_TO_EXCLUDE and c not in self.COLUMNS_TO_NOT_ANALYSE]
            kingdom = 'plant' if i <= 2 else 'animal'
            kingdom_df = df.loc[df.kingdom == kingdom]
            if kingdom == 'animal':
                self.investigate_shit_features(kingdom_df)
            X_train, X_test, y_train, y_test = self.preprocess_dataset_and_split(kingdom_df, selected_columns)
            X_train, y_train = self.up_or_down_sample(X_train, y_train)
            logreg = LogisticRegression(solver='liblinear')
            filtered_X_train = self.auto_or_manual_feature_selection(logreg, X_train, y_train)
            logit_model = statsmodels.api.Logit(y_train, filtered_X_train)
            result = self.fit_and_check_singular_matrix(logit_model, filtered_X_train, y_train)
            print(result.summary2())
            print('-------------------------------------------------------------')
            logreg.fit(filtered_X_train, y_train)
            y_pred = logreg.predict(X_test.loc[:, filtered_X_train.columns])
            metrics = Evaluation.calculate_metrics(y_test, y_pred)
            metrics_by_y[this_y] = metrics
        metrics_df = pandas.DataFrame(metrics_by_y)
        average_f1 = np.mean(metrics_df.T.f1) * 100
        print(metrics_df.T)
        print(f'average f1 score: {average_f1:.0f}%')

    '''
    def fit_to_validation(self):
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
    '''

    def run(self):
        #self.explore_features()
        df = self.fix_data()
        self.train(df)
        '''
        results = self.fit_to_validation()
        evaluation = Evaluation(results, verbose=False)
        evaluation.run()
        '''


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('sampling', type=str)
    parser.add_argument('feature_selection', type=str)
    parser.add_argument('do_extract_features', type=str, nargs='?', default='NO')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    WordLevelModel(**vars(arguments)).run()

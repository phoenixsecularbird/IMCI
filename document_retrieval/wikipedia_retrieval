import json
import re
import tqdm
import sqlite3
import unicodedata
import nltk
import wikipedia
from allennlp.predictors import Predictor
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}

# database_path
DATABASE_PATH = ""

# parser_path
PARSER_PATH = ""


def is_number(str1):
    return str1.isdigit()


def normalize(text):
    return unicodedata.normalize('NFD', text)


class DocDB(object):

    def __init__(self, db_path=None):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        return self.path

    def close(self):
        self.connection.close()

    def get_doc_ids(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE id = ?", (normalize(doc_id),))
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]


class FeverDocDB(DocDB):

    def __init__(self, path=None):
        super().__init__(path)

    def get_doc_lines(self, doc_id):
        return self.get_doc_text(doc_id)

    def get_non_empty_doc_ids(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results


class Doc_Retrieval:

    def __init__(self, database_path, add_claim=False, k_wiki_results=None):
        self.db = FeverDocDB(database_path)
        self.add_claim = add_claim
        self.k_wiki_results = k_wiki_results
        self.proter_stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.wordpunct_tokenize
        self.predictor = Predictor.from_path(PARSER_PATH)

    def get_NP(self, tree, nps):

        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    nps.append(tree['word'])
                    self.get_NP(tree['children'], nps)
                else:
                    self.get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_NP(sub_tree, nps)

        # 如果某个名词短语是纯粹的数字，或者在停用词列表里，则将其剔除
        new_nps = []
        for np in nps:
            if is_number(np) or np.lower() in STOPWORDS:
                continue
            new_nps.append(np)
        return new_nps

    def get_subjects(self, tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])

        new_subjects = []
        for subject in subjects:
            if is_number(subject) or subject.lower() in STOPWORDS:
                continue
            new_subjects.append(subject)

        return new_subjects

    def get_noun_phrases(self, line):

        claim = line['claim']
        tokens = self.predictor.predict(claim)

        nps = []
        tree = tokens['hierplane_tree']['root']
        noun_phrases = self.get_NP(tree, nps)

        subjects = self.get_subjects(tree)
        for subject in subjects:
            if len(subject) > 0:
                noun_phrases.insert(0, subject)

        if self.add_claim:
            noun_phrases.insert(0, claim)

        new_phrases = list(set(noun_phrases))
        new_phrases.sort(key=noun_phrases.index)

        return new_phrases, subjects

    def check_wiki_result(self, claim_words, wiki_results):

        checked = []

        for page in wiki_results:

            page = page.replace(" ", "_")
            page = page.replace("(", "-LRB-")
            page = page.replace(")", "-RRB-")
            page = page.replace(":", "-COLON-")
            if 'disambiguation' in page:
                continue

            page = normalize(page)
            processed_page = re.sub("-LRB-.*?-RRB-", "", page)
            processed_page = re.sub("_", " ", processed_page)
            processed_page = re.sub("-COLON-", ":", processed_page)
            processed_page = processed_page.replace("-", " ")
            processed_page = processed_page.replace("–", " ")
            processed_page = processed_page.replace(".", "")

            page_words = [self.proter_stemm.stem(word.lower()) for word in self.tokenizer(processed_page) if
                          len(word) > 0]

            if all([item in claim_words for item in page_words]):
                if not is_number(processed_page):
                        if ':' in page:
                            page = page.replace(":", "-COLON-")
                        checked.append(page)
        return checked

    def get_doc_for_claim(self, noun_phrases, claim_words):

        predicted_pages = []
        for np in noun_phrases:

            if len(np) > 300:
                continue

            try:
                docs = wikipedia.search(np)
                docs = self.check_wiki_result(claim_words, docs)
                if self.k_wiki_results is not None:
                    predicted_pages.extend(docs[:self.k_wiki_results])
                else:
                    predicted_pages.extend(docs)
            except:
                continue

        new_pages = list(set(predicted_pages))
        new_pages.sort(key=predicted_pages.index)

        return new_pages

    def np_conc(self, noun_phrases):

        predicted_pages = []
        for np in noun_phrases:
            page = np.replace('( ', '-LRB-')
            page = page.replace(' )', '-RRB-')
            page = page.replace(' - ', '-')
            page = page.replace(' :', '-COLON-')
            page = page.replace(' ,', ',')
            page = page.replace(" 's", "'s")
            page = page.replace(' ', '_')

            if len(page) < 1:
                continue
            doc_lines = self.db.get_doc_lines(page)
            if doc_lines is not None and 'disambiguation' not in page:
                predicted_pages.append(page)
        return predicted_pages

    def exact_match(self, line):

        noun_phrases, subjects = self.get_noun_phrases(line)

        claim = normalize(line['claim'])
        claim = claim.replace(".", "")
        claim = claim.replace("-", " ")
        words = [self.proter_stemm.stem(word.lower()) for word in self.tokenizer(claim)]
        words = set(words)
        predicted_pages = self.np_conc(noun_phrases)

        wiki_results = self.get_doc_for_claim(noun_phrases, words)
        predicted_pages.extend(wiki_results)

        new_pages = list(set(predicted_pages))
        new_pages.sort(key=predicted_pages.index)

        return new_pages


def main(dataset_file, k_wiki_results=10):

    doc_retriever = Doc_Retrieval(DATABASE_PATH, add_claim=True, k_wiki_results=k_wiki_results)

    with open(dataset_file) as f:
        dataset_list = [json.loads(line) for line in f]

    with ThreadPool(processes=10) as p:

        docs = [p.apply_async(doc_retriever.exact_match, args=(item,)) for item in tqdm(dataset_list)]

        for i, doc in enumerate(tqdm(docs)):
            dataset_list[i]["predicted_docids"] = doc.get()

    return dataset_list


if __name__ == '__main__':

    dataset_file = ""
    k_wiki_results = 10
    main(dataset_file, k_wiki_results)

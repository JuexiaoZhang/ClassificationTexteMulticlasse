import numpy as np
import pandas as pd
import re,string,language_check
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')


class DataGenerator:
    """
        La définition du DataGenerator: data preprocessing et feature engineering

        Args:
            trainfilename (String): L'adresse de Train set
            testfilename (String): L'adresse de Test set

        Attributes:
            word_count_vector(Dataframe): Caractéristique construites par des vecteurs de mots
            handmade_features(Dataframe): Caractéristiques extraites manuellement

       """
    def __init__(self, trainfilename,testfilename):
        self.trainfilename = trainfilename
        self.testfilename = testfilename
        self.get_df_from_path()

        # Feature Engineering
        self.dict_word_about_lang = {
            'word_country_about_GER': ['germany', 'austria', 'liechtenstein', 'switzerland', 'luxembourg', 'belgium'],
            'word_country_about_TUR': ['turkey', 'cyprus'],
            'word_country_about_CHI': ['china', 'singapore'],
            'word_country_about_TEL': ['india'],
            'word_country_about_ARA': ['algeria', 'bahrain', 'chad', 'comoros', 'djibouti', 'egypt', 'iraq', 'jordan',
                                       'kuwait', 'lebanon', 'libya', 'mauritania', 'morocco', 'oman', 'qatar',
                                       'saudi arabia', 'somalia', 'sudan', 'syria', 'tunisia', 'united arab emirates',
                                       'yemen', 'palestine', 'somaliland', 'western sahara', 'zanzibar', 'tanzania',
                                       'eritrea', 'mali', 'niger', 'senegal'],
            'word_country_about_SPA': ['spain', 'andorra', 'gibraltar', 'equatorial guinea', 'morocco', 'argentina',
                                       'bolivia', 'chile', 'colombia', 'costa rica', 'cuba', 'dominican republic',
                                       'ecuador', 'equatorial guinea', 'el ealvador', 'guatemala', 'honduras', 'mexico',
                                       'nicaragua', 'panama', 'paraguay', 'peru', 'puerto pico', 'uruguay',
                                       'venezuela'],
            'word_country_about_HIN': ['india', 'nepal', 'mauritius'],
            'word_country_about_JPN': ['japan', 'palau', 'angaur'],
            'word_country_about_KOR': ['korea', 'north'],
            'word_country_about_FRE': ['belgium', 'benin', 'burkina', 'burundi', 'cameroon', 'canada', 'central',
                                       'chad', 'comoros', 'ivoire', 'djibouti', 'france', 'guinea', 'haiti',
                                       'luxembourg', 'madagascar', 'mali', 'monaco', 'niger', 'congo', 'rwanda',
                                       'senegal', 'seychelles', 'switzerland', 'togo', 'vanuatu'],
            'word_country_about_ITA': ['switzerland', 'italy', 'vatican'],
            'word_region_about_GER': ['berlin', 'hamburg', 'munich', 'cologne', 'frankfurt', 'essen', 'dortmund',
                                      'stuttgart', 'dusseldorf', 'bremen', 'vienna', 'graz', 'linz', 'salzburg',
                                      'ruggell', 'schellenberg', 'gamprin', 'eschen', 'mauren', 'zurich', 'geneva',
                                      'basel', 'lausanne', 'bern', 'winterthur', 'lucerne', 'gallen',
                                      'esch-sur-alzette', 'differdange', 'dudelange', 'ettelbruck', 'diekirch', 'wiltz',
                                      'echternach', 'rumelange', 'grevenmacher', 'antwerp', 'ghent', 'charleroi',
                                      'liège', 'brussels'],
            'word_region_about_TUR': ['ankara', 'edirne', 'eskisehir', 'bursa', 'mardin', 'istanbul', 'antalya',
                                      'izmir', 'canakkale', 'trabzon', 'nicosia', 'limassol', 'larnaca', 'famagusta',
                                      'paphos', 'kyrenia'],
            'word_region_about_CHI': ['hongkong', 'macau', 'tibet', 'taiwan', 'beijing', 'shanghai', 'shenzhen',
                                      'guangzhou'],
            'word_region_about_TEL': ['mumbai', 'bangalore', 'kolkata', 'delhi', 'chennai', 'hyderabad', 'agra',
                                      'kochi', 'chandigarh', 'mysore'],
            'word_region_about_ARA': ['algiers', 'oran', 'manama', 'riffa', 'muharraq', 'djamena', 'moroni', 'cairo',
                                      'baghdad', 'mosul', 'amman', 'zarqa', 'al', 'ahmadi', 'beyrouth', 'tripoli',
                                      'nouakchott', 'nouadhibou', 'casablanca', 'tangier', 'marrakesh', 'rabat',
                                      'dhalouf', 'adam', 'abha', 'mogadishu', 'khartoum', 'sudan', 'aleppo', 'damascus',
                                      'tunis', 'sfax', 'dubai', 'dhabi', 'sharjah', 'sana', 'kabira', 'abaarso',
                                      'ayourou', 'bamako', 'sikasso', 'dakar'],
            'word_region_about_SPA': ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'louis'],
            'word_region_about_HIN': ['madrid', 'barcelona', 'valencia', 'vella', 'escaldes-engordany', 'bata',
                                      'malabo', 'casablanca', 'rabat', 'fes', 'buenos', 'cordoba', 'rosario', 'sierra',
                                      'alto', 'arica', 'iquique', 'bogota', 'medellin', 'jos', 'puerto', 'havana',
                                      'santiago', 'domingo', 'macorís', 'guayaquil', 'quito', 'salvador', 'soyapango',
                                      'salvador', 'soyapango', 'guatemala', 'quetzaltenango', 'tegucigalpa', 'pedro',
                                      'mexico', 'iztapalapa', 'leon', 'granada', 'panama', 'miguelito', 'asuncion',
                                      'ciudad', 'lima', 'arequipa', 'juan', 'bayamon', 'montevideo', 'salto', 'caracas',
                                      'maracaibo'],
            'word_region_about_JPN': ['sapporo', 'tokyo', 'yokohama', 'nagoya', 'kyoto', 'nara', 'osaka', 'kobe',
                                      'airai'],
            'word_region_about_KOR': ['seoul', 'busan', 'incheon', 'daegu', 'pyongyang', 'hamhung', 'chongjin', 'nampo',
                                      'chaoxian'],
            'word_region_about_FRE': ['antwerp', 'ghent', 'cotonou', 'porto-novo', 'ouagadougou', 'bobo-dioulasso',
                                      'bujumbura', 'gitega', 'douala', 'yaounde', 'taronto', 'montreal', 'baugui',
                                      'bimbo', 'djamena', 'moundou', 'moroni', 'moutsamoudou', 'abidjan', 'abobo',
                                      'djibouti', 'sabieh', 'paris', 'marseille', 'lyon', 'camayenne', 'conakry',
                                      'port-au-prince', 'delmas', 'luxembourg', 'antananarivo', 'toamasina', 'bamako',
                                      'sikasso', 'monaco', 'cario', 'niamey', 'zinder', 'kinshasa', 'lubumbashi',
                                      'kigali', 'butare', 'dakar', 'pikine', 'victoria', 'boileau', 'zurich', 'geneva',
                                      'basel', 'lome', 'sokode', 'vila', 'lugaville'],
            'word_region_about_ITA': ['zurich', 'geneva', 'basel', 'lausanne', 'bern', 'winterthur', 'lucerne',
                                      'mesolcina', 'bregaglia', 'poschiavo', 'marino', 'rome', 'milan', 'naples',
                                      'turin', 'palermo', 'genoa', 'bologna', 'florence']} # Dictionnaire des mots concernant des langues maternelles
        self.build_word_count_vector()
        self.build_handmade_feature()
        self.allfeatures = np.concatenate((self.word_count_vector.toarray(), self.handmade_features), axis=1)



    def get_df_from_path(self):
        """
        Importer des données

        """
        with open('./'+self.trainfilename, 'r') as f:
            x = f.readlines()
        self.df_train = pd.DataFrame(x)
        self.train_size = self.df_train.shape[0]

        with open('./'+self.testfilename, 'r') as f:
            x = f.readlines()
        self.df_test = pd.DataFrame(x)
        self.test_size = self.df_test.shape[0]

        self.df = pd.concat([self.df_train, self.df_test])
        self.df.columns = ['raw_data']
        self.df['lable'] = [item[1:4] for item in list(self.df['raw_data'])]
        self.df['text'] = [item[6::] for item in list(self.df['raw_data'])]
        self.df['text_preproc'] = self.df['text'].apply(lambda x: self.pre_process(x))

    def build_word_count_vector(self):
        """
        Construire un vecteur de mot

        """
        docs = self.df['text_preproc'].tolist() # get the text column
        '''
        Another test ：create a vocabulary of words, ignore words that appear in 85% of documents, eliminate stop words
        cv=CountVectorizer(max_df=0.85,stop_words='english')
        Cepandant, avec les parametres max_df=0.85 et stop_words='english', le score devient pire.
        '''
        cv = CountVectorizer()
        self.word_count_vector = cv.fit_transform(docs)

    def build_handmade_feature(self):
        """
        Extraire manuellement des caractéristiques

        """

        # Reindex le dataframe self.df
        self.df = self.df.reset_index(drop=True)

        # Feature statistique
        print('Creating Statistical Features... ')
        self.df['wordList_lower'] = [self.preprocess_lower(item) for item in self.df['text']]
        self.df['wordList_upper'] = [self.preprocess_upper(item) for item in self.df['text']]
        self.df['sentenceList'] = self.df['text'].apply(sent_tokenize)
        self.df['punctuations'] = self.df['text'].apply(lambda x: "".join(_ for _ in x if _ in string.punctuation))

        # Combien de char par text
        print('--- Number of letters')
        self.df['char_count'] = self.df['text'].apply(len)

        # Combien de mot par text
        print('--- Number of words')
        self.df['word_count'] = [len(item) for item in self.df['wordList_lower']]

        # La longueur moyenne des mots dans le texte
        print('--- Average word length')
        self.df['word_density'] = self.df['char_count'] / self.df['word_count']

        # Nombre de punctuation dans le texte
        print('--- Number of punctuations')
        self.df['punctuation_count'] = [len(item) for item in self.df['punctuations']]
        self.df['punctuation_count_apostrophe'] = self.df['punctuations'].apply(lambda x: self.count_punctuation(x, '\''))
        self.df['punctuation_count_bracket_right'] = self.df['punctuations'].apply(
            lambda x: self.count_punctuation(x, ')'))
        self.df['punctuation_count_bracket_left'] = self.df['punctuations'].apply(lambda x: self.count_punctuation(x, '('))
        self.df['punctuation_count_exclamation'] = self.df['punctuations'].apply(lambda x: self.count_punctuation(x, '!'))
        self.df['punctuation_count_question'] = self.df['punctuations'].apply(lambda x: self.count_punctuation(x, '?'))
        self.df['punctuation_count_dash'] = self.df['punctuations'].apply(lambda x: self.count_punctuation(x, '-'))

        # Analyse de sentiment
        print('--- Sentiment analysis')
        sentiments = ['polarity_negative', 'polarity_neutre', 'polarity_positive', 'subjectivity_objective',
                      'subjectivity_subjective'] # Etiquette de sentiment
        df_sentiment = self.build_df_sentiments(self.df, sentiments)
        self.df = pd.concat([self.df, pd.DataFrame(df_sentiment, columns=sentiments)], axis=1)

        # Combien de phrase par text
        print('--- Number of sentences')
        self.df['sentence_count'] = [len(item) for item in self.df['sentenceList']]

        # Longeur moyenne de phrase par text (en char/mot)
        print('--- Average sentence length')
        self.df['sentence_density_char'] = self.df['char_count'] / self.df['sentence_count']
        self.df['sentence_density_word'] = self.df['word_count'] / self.df['sentence_count']

        # Nombre de mot commencé par majuscule
        print('--- Number of words started with uppercase')
        self.df['upper_case_word_count'] = self.df['wordList_upper'].apply(
            lambda x: len([wrd for wrd in x if not wrd.islower()]))

        # Combien d‘article par text （a, an, the)
        print('--- Number of articles (a, an, the)')
        self.df['quantity_a'] = [self.quantity_of_word('a', item) for item in self.df['wordList_lower']]
        self.df['quantity_an'] = [self.quantity_of_word('an', item) for item in self.df['wordList_lower']]
        self.df['quantity_the'] = [self.quantity_of_word('the', item) for item in self.df['wordList_lower']]

        # Analyse des mots concernant des langues maternelles
        print('--- Words relating to mother tongues')
        list_word_about_lang = list(self.dict_word_about_lang.keys())
        df_word_about_langue = self.build_df_word_about_langue(self.df, list_word_about_lang)
        self.df = pd.concat([self.df, pd.DataFrame(df_word_about_langue, columns=list_word_about_lang)], axis=1)

        '''
        # Certaines caractéristiques qui prennent beaucoup de temps à extraire ont été commentées :
        # pos-tagger
        print('--- Pos-tagger')
        self.df['noun_count'] = self.df['wordList_lower'].apply(lambda x: self.check_pos_tag(x, 'noun'))
        self.df['verb_count'] = self.df['wordList_lower'].apply(lambda x: self.check_pos_tag(x, 'verb'))
        self.df['adj_count'] = self.df['wordList_lower'].apply(lambda x: self.check_pos_tag(x, 'adj'))
        self.df['adv_count'] = self.df['wordList_lower'].apply(lambda x: self.check_pos_tag(x, 'adv'))
        self.df['pron_count'] = self.df['wordList_lower'].apply(lambda x: self.check_pos_tag(x, 'pron'))

        # Combien de Stopword
        print('--- Stopwords')
        self.df['stopwords_count'] = self.df['wordList_lower'].apply(
            lambda x: len([wrd for wrd in x if wrd in stopwords.words('english')]))
        # Fréquence des stopwords
        self.df['stopwords_frequency'] = self.df['stopwords_count'] / self.df['word_count']
        
        # Faute grammaire
        self.df['mistake_rules'] = [self.collect_ruleid(item) for item in self.df['text'].values]
        self.df['mistake_count'] = self.df['mistake_rules'].apply(lambda x: len(x))
        df_mistake_rules, allrules = self.build_df_mistake_rules(self.df)
        self.df = pd.concat([self.df, pd.DataFrame(df_mistake_rules, columns=allrules)], axis=1)
        self.df.drop('mistake_rules', axis=1, inplace=True)
        '''

        self.handmade_features = self.df.iloc[:, list(self.df.columns).index('char_count'):]

    def pre_process(self,text):
        """
        Supprimer les caractères spéciaux

        Args:
            text (string): Chaîne à traiter
        """
        text = text.lower() # lowercase
        text = re.sub("</?.*?>", " <> ", text) # remove tags
        text = re.sub("(\\d|\\W)+", " ", text) # remove special characters and digits
        return text

    #def preprocess_sentence(sent):
    #    sent_tokenize_list = sent_tokenize(sent)
    #    return sent

    # Compter le nombre de punctuation "punc"
    def count_punctuation(self,x, punc):
        count = 0
        for i in x:
            if i == punc:
                count += 1
        return count

    # Combien mot "word" par text
    def quantity_of_word(self,word, line):
        count = 0
        for i in line:
            if i == word:
                count += 1
        return count

    # Obtenir le nombre de tag de text en utilisant nltk.pos_tag()
    def check_pos_tag(self,x, tag):
        # Dictionnaire du pos-tagger
        pos_family = {
            'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
            'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
            'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adj': ['JJ', 'JJR', 'JJS'],
            'adv': ['RB', 'RBR', 'RBS', 'WRB']
        }
        cnt = 0
        try:
            wiki = nltk.pos_tag(x)
            for tup in wiki:
                ppo = tup[1]
                if ppo in pos_family[tag]:
                    cnt += 1
        except:
            pass
        return cnt

    # Chercher le mot 'value' sur la dictionnaire "dict", retourner les keys
    def get_key(self, dict, value):
        keys = []
        for i in self.dict_word_about_lang.items():
            if value in i[1]:
                keys.append(i[0])
        return keys

    '''
    # Lister toutes les types de faute grammaire dans un text
    def collect_ruleid(self,text):
        tool = language_check.LanguageTool('en-US')
        matches = tool.check(text)
        ruleid = []
        for i in matches:
            ruleid.append(i.ruleId)
        return ruleid
    '''

    # Création du dataframe sur l'analyse de sentiment,shape = (nombre d'échantillon, len(sentiments))
    def build_df_sentiments(self,df,sentiments):
        df_sentiment = []
        for text in df['text']:
            l = [0 for item in range(len(sentiments))]
            blob = TextBlob(text)
            score_polarity = np.mean([x.polarity for x in blob.sentences])  # la moyenne du socre polarity
            score_subjectivity = np.mean([x.subjectivity for x in blob.sentences])  # la moyenne du socre subjectivity
            # distinguer l'émotion
            if score_polarity < -0.3:
                l[sentiments.index('polarity_negative')] = 1
            elif score_polarity <= 0.3:
                l[sentiments.index('polarity_neutre')] = 1
            else:
                l[sentiments.index('polarity_positive')] = 1
            # distinguer la subjectivité
            if score_subjectivity >= 0.5:
                l[sentiments.index('subjectivity_objective')] = 1
            else:
                l[sentiments.index('subjectivity_subjective')] = 1
            df_sentiment.append(l)
        return df_sentiment



    # Creation du dataframe contenant le nombre des mots concernant une langue maternelles dans un texte
    def build_df_word_about_langue(self,df,list_langue):
        df_word_about_langue = []
        list_word_about_lang = list(self.dict_word_about_lang.keys())
        for list_word in df['wordList_lower']:
            l = [0 for item in range(len(list_word_about_lang))]
            for word in list_word:
                keys = self.get_key(self.dict_word_about_lang, word)
                for key in keys:
                    l[list_word_about_lang.index(key)] += 1
            df_word_about_langue.append(l)
        return df_word_about_langue

    def build_df_mistake_rules(self,df):
        allrules = []
        for rules in df['mistake_rules']:
            for rule in rules:
                if rule not in allrules:
                    allrules.append(rule)
        df_rules = []
        for rules in df['mistake_rules']:
            l = [0 for item in range(len(allrules))]
            for rule in rules:
                l[allrules.index(rule)] += 1
            df_rules.append(l)
        return df_rules, allrules

    def preprocess_lower(self,sent):
        # Uppercase to lowercase
        sent = sent.lower()
        # Transformez la ponctuation en espaces
        sent = sent.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # Transformer une phrase en une liste de mot
        sent = nltk.word_tokenize(sent)
        return sent

    def preprocess_upper(self,sent):
        sent = sent.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        sent = nltk.word_tokenize(sent)
        return sent





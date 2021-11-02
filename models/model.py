from sklearn.naive_bayes import BernoulliNB
import pickle
import pandas as pd


class Model:
    """
    La définition du Machine Learning Model

    Args:
        data (DataGenerator): Données prétraitées par le data generator

    """

    def __init__(self,data):
        self.allfeatures = data.allfeatures
        self.df= data.df
        self.train_size = data.train_size
        self.test_size = data.test_size
        self.trainning()

    def trainning(self):
        """
        Construire le modèle et entraîner les données

        """
        train_X = self.allfeatures[0:self.train_size]
        train_Y = self.df.iloc[0:self.train_size]['lable'].tolist()

        clf = BernoulliNB()
        clf.fit(train_X, train_Y)
        #BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        self.train_score = clf.score(train_X, train_Y)

        # Save the model
        pkl_filename = "./data/working/models/train_model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)

    def predict(self):
        """
        Prédisez la nationalité sur Test Set et écrivez le résultat dans le fichier rendu.txt

        """
        # Load model from file
        pkl_filename = "./data/working/models/train_model.pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        test_X = self.allfeatures[self.train_size:self.train_size+self.test_size]

        list_result = model.predict(test_X)
        df_result = self.df.iloc[self.train_size:self.train_size+self.test_size].copy()
        df_result['predicted_result'] = ['(' + item + ') ' for item in list_result]
        df_result['rendu'] = df_result["predicted_result"] + df_result["text"]
        pd.DataFrame(df_result['rendu']).to_csv('./data/output/rendu.txt', header=False, index=False, sep=' ')


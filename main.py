from utils.utils import get_args
from loaders.data_generator import DataGenerator
from models.model import Model

def main():
    # capture the config path from the run arguments
    # then process the json configuration file

    try:
        args = get_args()
        trainfilename = args.trainfilename
        testfilename = args.testfilename
    except:
        print("missing or invalid arguments")
        exit(0)

    print("Data preprocessing...")
    data = DataGenerator(trainfilename,testfilename)
    print("Model training...")
    model = Model(data)
    print("Model training score: ",model.train_score)
    print("Predicting...")
    model.predict()
    print("The result is saved in the folder: data/output")


if __name__ == '__main__':
    main()

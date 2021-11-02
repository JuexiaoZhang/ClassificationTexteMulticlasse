import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('trainfilename', type=str, help='path of train.txt')
    argparser.add_argument('testfilename', type=str, help='path of test.txt')
    args = argparser.parse_args()

    return args


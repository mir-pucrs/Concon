# -*- coding: utf-8 -*-
import logging
import argparse
from nltk.tokenize import sent_tokenize
from os.path import realpath, splitext

# Set argparser.
parser = argparse.ArgumentParser(description='Process a file path.')

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('logs/conversion.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def convert_to_sentences(input_file):
    """
    :param input_file: String containing a path to a file. 
    :type input_file: str
    :return: None 
    """

    # Create output file.
    input_file = realpath(input_file)

    basename, ext = splitext(input_file)
    output_file = basename + "_sentences" + ext

    # Read text.
    logging.info("Reading file - {}".format(input_file))
    try:
        text = open(input_file, 'r').read()
    except:
        logger.exception("Can't open the file.")

    # Convert into sentences.
    logging.info('Converting text into a set of sentences.')
    sentences = sent_tokenize(text)

    with open(output_file, 'w') as w_file:
        logging.info('Writing sentences to the new file.')
        for sent in sentences:
            # Run over the sentences saving them in the output_file.
            w_file.write(sent + "\n")

    w_file.close()

    return output_file


if __name__ == "__main__":

    parser.add_argument('file_path', type=str, help='Please, provide a valid path to a file containing text.')

    args = parser.parse_args()

    file_path = args.file_path

    print convert_to_sentences(file_path)

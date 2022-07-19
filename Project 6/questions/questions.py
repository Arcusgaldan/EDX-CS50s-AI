import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files_dict = {}
    for file in [cur_file for cur_file in os.listdir(directory) if os.path.isfile(os.path.join(directory, cur_file)) and os.path.splitext(os.path.join(directory, cur_file))[1] == ".txt"]:
        with open(os.path.join(directory, file), 'r', encoding="utf8") as file_reader:
            files_dict[file] = file_reader.read()
            
    return files_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.tokenize.word_tokenize(document)
    tokens = [token.lower() for token in tokens if token not in string.punctuation and token not in nltk.corpus.stopwords.words("english")]
    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    for doc in documents:
        for word in documents[doc]:
            if word in idfs:
                continue
            total_docs = len(documents)
            appearance_docs = len([document for document in documents if word in documents[document]])
            idf = math.log(total_docs/appearance_docs)
            idfs[word] = idf
    
    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    ranking = {}
    for document in files:
        doc_score = 0
        common_words = set(files[document]).intersection(query)
        for common_word in common_words:
            tf = files[document].count(common_word)
            doc_score += tf * idfs[common_word]
        ranking[document] = doc_score
    ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    ranking = [ranked[0] for ranked in ranking]
    return ranking[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ranking = {}
    for sentence in sentences:
        common_words = set(sentences[sentence]).intersection(query)
        idf = 0
        qt_density = 0
        for cw in common_words:
            idf += idfs[cw]
        for word in sentences[sentence]:
            if word in query:
                qt_density += 1
        qt_density = qt_density / len(sentences[sentence])
        ranking[sentence] = (idf, qt_density)
    ranking = dict(sorted(ranking.items(), key=lambda x: x[1][1], reverse=True))
    ranking = sorted(ranking.items(), key=lambda x: x[1][0], reverse=True)
    ranking = [ranked[0] for ranked in ranking]
    return ranking[:n]

if __name__ == "__main__":
    main()

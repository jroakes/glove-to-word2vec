# -*- coding: utf-8 -*-
"""
word2vec embeddings start with a line with the number of lines (tokens?) and 
the number of dimensions of the file. This allows gensim to allocate memory 
accordingly for querying the model. Larger dimensions mean larger memory is 
held captive. Accordingly, this line has to be inserted into the GloVe 
embeddings file.
"""

import os
import shutil
import smart_open
from sys import platform

import gensim


def prepend_line(infile, outfile, line):
	""" 
	Function use to prepend lines using bash utilities in Linux. 
	(source: http://stackoverflow.com/a/10850588/610569)
	"""
	with open(infile, 'r') as old:
		with open(outfile, 'w') as new:
			new.write(str(line) + "\n")
			shutil.copyfileobj(old, new)

def prepend_slow(infile, outfile, line):
	"""
	Slower way to prepend the line by re-creating the inputfile.
	"""
	with open(infile, 'r') as fin:
		with open(outfile, 'w') as fout:
			fout.write(line + "\n")
			for line in fin:
				fout.write(line)

def get_lines(glove_file_name):
    """Return the number of vectors and dimensions in a file in GloVe format."""
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_lines = sum(1 for line in f)
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims
	
# Input: GloVe Model File
# More models can be downloaded from http://nlp.stanford.edu/projects/glove/
glove_file="glove.6B.300d.txt"

num_lines, dims = get_lines(glove_file)

# Output: Gensim Model text format.
gensim_file='glove_model2.txt'
gensim_first_line = "{} {}".format(num_lines, dims)

# Prepends the line.
if platform == "linux" or platform == "linux2":
	prepend_line(glove_file, gensim_file, gensim_first_line)
else:
	prepend_slow(glove_file, gensim_file, gensim_first_line)

# Demo: Loads the newly created glove_model.txt into gensim API.
model=gensim.models.Word2Vec.load_word2vec_format(gensim_file,binary=False) #GloVe Model

print model.most_similar(positive=['australia'], topn=10)
print model.similarity('woman', 'man')

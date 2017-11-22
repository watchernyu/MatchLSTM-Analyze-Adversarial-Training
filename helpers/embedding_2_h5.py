# encoding: utf-8
import zipfile
import numpy as np
import h5py

#convert glove pretrained embedding (in .zip format) into .h5 (high dimensional array format) files

def export_data_h5(vocabulary, embedding_matrix, output='embedding.h5'):
    f = h5py.File(output, "w")
    compress_option = dict(compression="gzip", compression_opts=9, shuffle=True)
    words_flatten = '\n'.join(vocabulary)
    f.attrs['vocab_len'] = len(vocabulary)
    dt = h5py.special_dtype(vlen=str)
    _dset_vocab = f.create_dataset('words_flatten', (1, ), dtype=dt, **compress_option)
    _dset_vocab[...] = [words_flatten]
    _dset = f.create_dataset('embedding', embedding_matrix.shape, dtype=embedding_matrix.dtype, **compress_option)
    _dset[...] = embedding_matrix
    f.flush()
    f.close()


def glove_export(embedding_file): # take in a zip file
    with zipfile.ZipFile(embedding_file) as zf:
        #for each file in the .zip file, export a .h5 file
        for name in zf.namelist():
            vocabulary = []
            embeddings = []
            with zf.open(name) as f:
                for line in f:
                    vals = line.split(' ')
                    vocabulary.append(vals[0])
                    embeddings.append([float(x) for x in vals[1:]])
            export_data_h5(vocabulary, np.array(embeddings, dtype=np.float32), output=name + ".h5")

# the file name here is hardcoded
# now changed to use 6B so that things are faster
# might change back later... #TODO
if __name__ == '__main__':
    glove_export('glove.6B.300d.zip')




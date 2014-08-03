import pyviennacl as p
import numpy as np

def read_mtx(fname, dtype=p.float32, sparse_type=p.CompressedMatrix):
    """
    Read a MatrixMarket file. Assume coordinate format. Very crude!
    """
    fd = open(fname)
    lines = list(map(lambda x: x.strip().split(" "), fd.readlines()))
    ln = -1
    for line in lines:
        ln += 1
        if line[ln][0] == "%":
            continue
        else:
            break
    n = int(lines[ln][0])
    m = int(lines[ln][1])
    try: nnz = int(lines[ln][2])
    except: nnz = n * m
    if m == 1:
        vec_type = np.result_type(dtype).type
        values = list(map(lambda x: vec_type(" ".join(x)), lines[ln+1:]))
        values = np.array(values)
        vec = p.Vector(values, dtype=dtype)
        return vec
    else:
        mat = sparse_type(n, m, nnz, dtype=dtype)
        mat_type = p.np_result_type(mat).type
        def assign(l):
            try:
                i, j, v = int(l[0]), int(l[1]), mat_type(l[2])
                mat.insert(i-1, j-1, v)
            except ValueError:
                pass
        result = list(map(assign, lines[ln+1:]))
        return mat


def read_vector(fname, dtype=np.float32):
    fd = open(fname)
    lines = list(map(lambda x: x.strip().split(" "), fd.readlines()))
    count = int(lines[0][0])
    vector = list(map(lambda x: p.np_result_type(dtype).type(x), lines[1]))
    vector = p.Vector(vector, dtype = dtype)
    if vector.size != count:
        raise Exception("Sizes %d and %d do not match!" % (vector.size, count))
    return vector


import pyviennacl as p

def read_mtx(fname, dtype):
    """
    Read a MatrixMarket file. Assume coordinate format, and double precision.
    
    Very crude!
    """
    fd = open(fname)
    lines = map(lambda x: x.strip().split(" "), fd.readlines())
    ln = -1
    for line in lines:
        ln += 1
        if line[ln][0] == "%":
            continue
        else:
            break
    n = int(lines[ln][0])
    m = int(lines[ln][1])
    nnz = int(lines[ln][2])
    mat = p.CompressedMatrix(n, m, nnz, dtype=dtype)
    mat_type = p.np_result_type(mat).type
    def assign(l):
        try:
            i, j, v = int(l[0]), int(l[1]), mat_type(l[2])
            mat[i-1, j-1] = v
        except ValueError:
            pass
    map(assign, lines[ln+1:])
    return mat

def read_vector(fname, dtype):
    fd = open(fname)
    lines = map(lambda x: x.strip().split(" "), fd.readlines())
    count = int(lines[0][0])
    vector = list(map(lambda x: p.np_result_type(dtype).type(x), lines[1]))
    vector = p.Vector(vector, dtype = dtype)
    if vector.size != count:
        raise Exception("Sizes %d and %d do not match!" % (vector.size, count))
    return vector
    

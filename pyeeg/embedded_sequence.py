import numpy


def embed_seq(X, Tau, D):
    """Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension DE. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.

    Parameters
    ----------

    X
        list

        a time series

    Tau
        integer

        the lag or delay when building embedding sequence

    D
        integer

        the embedding dimension

    Returns
    -------

    Y
        2-D list

        embedding matrix built

    Examples
    ---------------
    >>> import pyeeg
    >>> a=range(0,9)
    >>> pyeeg.embed_seq(a,1,4)
    array([[ 0.,  1.,  2.,  3.],
           [ 1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.],
           [ 3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.],
           [ 5.,  6.,  7.,  8.]])
    >>> pyeeg.embed_seq(a,2,3)
    array([[ 0.,  2.,  4.],
           [ 1.,  3.,  5.],
           [ 2.,  4.,  6.],
           [ 3.,  5.,  7.],
           [ 4.,  6.,  8.]])
    >>> pyeeg.embed_seq(a,4,1)
    array([[ 0.],
           [ 1.],
           [ 2.],
           [ 3.],
           [ 4.],
           [ 5.],
           [ 6.],
           [ 7.],
           [ 8.]])

    """
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return numpy.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

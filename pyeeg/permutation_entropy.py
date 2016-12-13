import numpy
from .embedded_sequence import embed_seq


def permutation_entropy(x, n, tau):
    """Compute Permutation Entropy of a given time series x, specified by
    permutation order n and embedding lag tau.

    Parameters
    ----------

    x
        list

        a time series

    n
        integer

        Permutation order

    tau
        integer

        Embedding lag

    Returns
    ----------

    PE
       float

       permutation entropy

    Notes
    ----------
    Suppose the given time series is X =[x(1),x(2),x(3),...,x(N)].
    We first build embedding matrix Em, of dimension(n*N-n+1),
    such that the ith row of Em is x(i),x(i+1),..x(i+n-1). Hence
    the embedding lag and the embedding dimension are 1 and n
    respectively. We build this matrix from a given time series,
    X, by calling pyEEg function embed_seq(x,1,n).

    We then transform each row of the embedding matrix into
    a new sequence, comprising a set of integers in range of 0,..,n-1.
    The order in which the integers are placed within a row is the
    same as those of the original elements:0 is placed where the smallest
    element of the row was and n-1 replaces the largest element of the row.

    To calculate the Permutation entropy, we calculate the entropy of PeSeq.
    In doing so, we count the number of occurrences of each permutation
    in PeSeq and write it in a sequence, RankMat. We then use this sequence to
    calculate entropy by using Shannon's entropy formula.

    Permutation entropy is usually calculated with n in range of 3 and 7.

    References
    ----------
    Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural
    complexity measure for time series." Physical Review Letters 88.17
    (2002): 174102.


    Examples
    ----------
    >>> import pyeeg
    >>> x = [1,2,4,5,12,3,4,5]
    >>> pyeeg.permutation_entropy(x,5,1)
    2.0

    """

    PeSeq = []
    Em = embed_seq(x, tau, n)

    for i in range(0, len(Em)):
        r = []
        z = []

        for j in range(0, len(Em[i])):
            z.append(Em[i][j])

        for j in range(0, len(Em[i])):
            z.sort()
            r.append(z.index(Em[i][j]))
            z[z.index(Em[i][j])] = -1

        PeSeq.append(r)

    RankMat = []

    while len(PeSeq) > 0:
        RankMat.append(PeSeq.count(PeSeq[0]))
        x = PeSeq[0]
        for j in range(0, PeSeq.count(PeSeq[0])):
            PeSeq.pop(PeSeq.index(x))

    RankMat = numpy.array(RankMat)
    RankMat = numpy.true_divide(RankMat, RankMat.sum())
    EntropyMat = numpy.multiply(numpy.log2(RankMat), RankMat)
    PE = -1 * EntropyMat.sum()

    return PE

import numpy
from .embedded_sequence import embed_seq


def samp_entropy(X, M, R):
    """Computer sample entropy (SampEn) of series X, specified by M and R.

    SampEn is very close to ApEn.

    Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
    embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
    Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
    are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
    function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
    difference with Em is that the length of each embedding sequence is M + 1

    Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
    are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
    Em[j] is defined as 1) the maximum difference of their corresponding scalar
    components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
    1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
    between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
    value of R is defined as 20% - 30% of standard deviation of X.

    Pick Em[i] as a template, for all j such that 0 < j < N - M , we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k.

    We repeat the same process on Emp and obtained Cmp[i], 0 < i < N - M.

    The SampEn is defined as log(sum(Cm)/sum(Cmp))

    References
    ----------

    Costa M, Goldberger AL, Peng C-K, Multiscale entropy analysis of biological
    signals, Physical Review E, 71:021906, 2005

    See also
    --------
    ap_entropy: approximate entropy of a time series

    """

    N = len(X)

    Em = embed_seq(X, 1, M)
    A = numpy.tile(Em, (len(Em), 1, 1))
    B = numpy.transpose(A, [1, 0, 2])
    D = numpy.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = numpy.max(D, axis=2) <= R
    numpy.fill_diagonal(InRange, 0)  # Don't count self-matches

    Cm = InRange.sum(axis=0)  # Probability that random M-sequences are in range
    Dp = numpy.abs(
        numpy.tile(X[M:], (N - M, 1)) - numpy.tile(X[M:], (N - M, 1)).T
    )

    Cmp = numpy.logical_and(Dp <= R, InRange[:-1, :-1]).sum(axis=0)

    # Avoid taking log(0)
    Samp_En = numpy.log(numpy.sum(Cm + 1e-100) / numpy.sum(Cmp + 1e-100))

    return Samp_En

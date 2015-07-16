# coding=UTF-8

"""Copyleft 2010-2015 Forrest Sheng Bao http://fsbao.net
   Copyleft 2010 Xin Liu
   Copyleft 2014-2015 Borzou Alipour Fard

PyEEG, a Python module to extract EEG feature.

Project homepage: http://pyeeg.org

**Data structure**

PyEEG only uses standard Python and numpy data structures,
so you need to import numpy before using it.
For numpy, please visit http://numpy.scipy.org

**Naming convention**

I follow "Style Guide for Python Code" to code my program
http://www.python.org/dev/peps/pep-0008/

Constants: UPPER_CASE_WITH_UNDERSCORES, e.g., SAMPLING_RATE, LENGTH_SIGNAL.

Function names: lower_case_with_underscores, e.g., spectrum_entropy.

Variables (global and local): CapitalizedWords or CapWords, e.g., Power.

If a variable name consists of one letter, I may use lower case, e.g., x, y.

Functions listed alphabetically
--------------------------------------------------

"""
import numpy


# ####################### Begin function definitions #######################

def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse.

    Parameters
    ----------

    X

        list

        a time series

    Returns
    -------
    H

        float

        Hurst exponent

    Notes
    --------
    Author of this function is Xin Liu

    Examples
    --------

    >>> import pyeeg
    >>> from numpy.random import randn
    >>> a = randn(4096)
    >>> pyeeg.hurst(a)
    0.5057444

    """

    N = len(X)

    T = numpy.array([float(i) for i in xrange(1, N + 1)])
    Y = numpy.cumsum(X)
    Ave_T = Y / T

    S_T = numpy.zeros((N))
    R_T = numpy.zeros((N))
    for i in xrange(N):
        S_T[i] = numpy.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = numpy.log(R_S)
    n = numpy.log(T).reshape(N, 1)
    H = numpy.linalg.lstsq(n[1:], R_S[1:])[0]
    return H[0]


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
    N = len(X)

    if D * Tau > N:
        print "Cannot build such a matrix, because D * Tau > N"
        exit()

    if Tau < 1:
        print "Tau has to be at least 1"
        exit()

    Y = numpy.zeros((N - (D - 1) * Tau, D))
    for i in xrange(0, N - (D - 1) * Tau):
        for j in xrange(0, D):
            Y[i][j] = X[i + j * Tau]
    return Y


def in_range(Template, Scroll, Distance):
    """Determines whether one vector is the the range of another vector.

    The two vectors should have equal length.

    Parameters
    -----------------
    Template
        list
        The template vector, one of two vectors being compared

    Scroll
        list
        The scroll vector, one of the two vectors being compared

    Distance
        float
        Two vectors match if their distance is less than D


    Notes
    -------
    The distance between two vectors can be defined as Euclidean distance
    according to some publications.

    The two vector should of equal length

    """

    for i in range(0, len(Template)):
            if abs(Template[i] - Scroll[i]) > Distance:
                return False
    return True
    """ Desperate code, but do not delete
    def bit_in_range(Index):
        if abs(Scroll[Index] - Template[Bit]) <=  Distance :
            print "Bit=", Bit, "Scroll[Index]", Scroll[Index], \
             "Template[Bit]", Template[Bit], \
             "abs(Scroll[Index] - Template[Bit])",\
             abs(Scroll[Index] - Template[Bit])
            return Index + 1 # move

    Match_No_Tail = range(0, len(Scroll) - 1) # except the last one
    #print Match_No_Tail

    # first compare Template[:-2] and Scroll[:-2]

    # every bit of Template is in range of Scroll
    for Bit in xrange(0, len(Template) - 1):
        Match_No_Tail = filter(bit_in_range, Match_No_Tail)
        print Match_No_Tail

    # second and last, check whether Template[-1] is in range of Scroll and
    #    Scroll[-1] in range of Template

    # 2.1 Check whether Template[-1] is in the range of Scroll
    Bit = - 1
    Match_All =  filter(bit_in_range, Match_No_Tail)

    # 2.2 Check whether Scroll[-1] is in the range of Template
    # I just write a  loop for this.
    for i in Match_All:
        if abs(Scroll[-1] - Template[i] ) <= Distance:
            Match_All.remove(i)

    return len(Match_All), len(Match_No_Tail)
    """


def bin_power(X, Band, Fs):
    """Compute power in each frequency bin specified by Band from FFT result of
    X. By default, X is a real signal.

    Note
    -----
    A real signal can be synthesized, thus not real.

    Parameters
    -----------

    Band
        list

        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.

        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.

     X
        list

        a 1-D real time series.

    Fs
        integer

        the sampling rate in physical frequency

    Returns
    -------

    Power
        list

        spectral power in each frequency bin.

    Power_ratio
        list

        spectral power in each frequency bin normalized by total power in ALL
        frequency bins.

    """

    C = numpy.fft.fft(X)
    C = abs(C)
    Power = numpy.zeros(len(Band) - 1)
    for Freq_Index in xrange(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[numpy.floor(
                Freq / Fs * len(X)
            ): numpy.floor(Next_Freq / Fs * len(X))]
        )
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio


def pfd(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's difference function.

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    if D is None:
        D = numpy.diff(X)
        D = D.tolist()
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in xrange(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return numpy.log10(n) / (
        numpy.log10(n) + numpy.log10(n / n + 0.4 * N_delta)
    )


def hfd(X, Kmax):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter
    """
    L = []
    x = []
    N = len(X)
    for k in xrange(1, Kmax):
        Lk = []
        for m in xrange(0, k):
            Lmk = 0
            for i in xrange(1, int(numpy.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / numpy.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(numpy.log(numpy.mean(Lk)))
        x.append([numpy.log(float(1) / k), 1])

    (p, r1, r2, s) = numpy.linalg.lstsq(x, L)
    return p[0]


def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """

    if D is None:
        D = numpy.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = numpy.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(numpy.array(X) ** 2)
    M4 = 0
    for i in xrange(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return numpy.sqrt(M2 / TP), numpy.sqrt(
        float(M4) * TP / M2 / M2
    )  # Hjorth Mobility and Complexity


def spectral_entropy(X, Band, Fs, Power_Ratio=None):
    """Compute spectral entropy of a time series from either two cases below:
    1. X, the time series (default)
    2. Power_Ratio, a list of normalized signal power in a set of frequency
    bins defined in Band (if Power_Ratio is provided, recommended to speed up)

    In case 1, Power_Ratio is computed by bin_power() function.

    Notes
    -----
    To speed up, it is recommended to compute Power_Ratio before calling this
    function because it may also be used by other functions whereas computing
    it here again will slow down.

    Parameters
    ----------

    Band
        list

        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.

        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.

     X
        list

        a 1-D real time series.

    Fs
        integer

        the sampling rate in physical frequency

    Returns
    -------

    As indicated in return line

    See Also
    --------
    bin_power: pyeeg function that computes spectral power in frequency bins

    """

    if Power_Ratio is None:
        Power, Power_Ratio = bin_power(X, Band, Fs)

    Spectral_Entropy = 0
    for i in xrange(0, len(Power_Ratio) - 1):
        Spectral_Entropy += Power_Ratio[i] * numpy.log(Power_Ratio[i])
    Spectral_Entropy /= numpy.log(
        len(Power_Ratio)
    )  # to save time, minus one is omitted
    return -1 * Spectral_Entropy


def svd_entropy(X, Tau, DE, W=None):
    """Compute SVD Entropy from either two cases below:
    1. a time series X, with lag tau and embedding dimension dE (default)
    2. a list, W, of normalized singular values of a matrix (if W is provided,
    recommend to speed up.)

    If W is None, the function will do as follows to prepare singular spectrum:

        First, computer an embedding matrix from X, Tau and DE using pyeeg
        function embed_seq():
                    M = embed_seq(X, Tau, DE)

        Second, use scipy.linalg function svd to decompose the embedding matrix
        M and obtain a list of singular values:
                    W = svd(M, compute_uv=0)

        At last, normalize W:
                    W /= sum(W)

    Notes
    -------------

    To speed up, it is recommended to compute W before calling this function
    because W may also be used by other functions whereas computing it here
    again will slow down.
    """

    if W is None:
        Y = embed_seq(X, Tau, DE)
        W = numpy.linalg.svd(Y, compute_uv=0)
        W /= sum(W)  # normalize singular values

    return -1 * sum(W * numpy.log(W))


def fisher_info(X, Tau, DE, W=None):
    """Compute SVD Entropy from either two cases below:
    1. a time series X, with lag tau and embedding dimension dE (default)
    2. a list, W, of normalized singular values of a matrix (if W is provided,
    recommend to speed up.)

    If W is None, the function will do as follows to prepare singular spectrum:

        First, computer an embedding matrix from X, Tau and DE using pyeeg
        function embed_seq():
                    M = embed_seq(X, Tau, DE)

        Second, use scipy.linalg function svd to decompose the embedding matrix
        M and obtain a list of singular values:
                    W = svd(M, compute_uv=0)

        At last, normalize W:
                    W /= sum(W)

    Notes
    -------------

    To speed up, it is recommended to compute W before calling this function
    because W may also be used by other functions whereas computing it here
    again will slow down.
    """

    if W is None:
        Y = embed_seq(X, Tau, DE)
        W = numpy.linalg.svd(Y, compute_uv=0)
        W /= sum(W)  # normalize singular values

    return -1 * sum(W * numpy.log(W))


def ap_entropy(X, M, R):
    """Computer approximate entropy (ApEN) of series X, specified by M and R.

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

    Pick Em[i] as a template, for all j such that 0 < j < N - M + 1, we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k. The probability that a random row in Em matches Em[i] is
    \simga_1^{N-M+1} k[i] / (N - M + 1), thus sum(k)/ (N - M + 1),
    denoted as Cm[i].

    We repeat the same process on Emp and obtained Cmp[i], but here 0<i<N-M
    since the length of each sequence in Emp is M + 1.

    The probability that any two embedding sequences in Em match is then
    sum(Cm)/ (N - M +1 ). We define Phi_m = sum(log(Cm)) / (N - M + 1) and
    Phi_mp = sum(log(Cmp)) / (N - M ).

    And the ApEn is defined as Phi_m - Phi_mp.


    Notes
    -----

    #. Please be aware that self-match is also counted in ApEn.
    #. This function now runs very slow. We are still trying to speed it up.

    References
    ----------

    Costa M, Goldberger AL, Peng CK, Multiscale entropy analysis of biological
    signals, Physical Review E, 71:021906, 2005

    See also
    --------
    samp_entropy: sample entropy of a time series

    Notes
    -----
    Extremely slow implementation. Do NOT use if your dataset is not small.

    """
    N = len(X)

    Em = embed_seq(X, 1, M)
    Emp = embed_seq(X, 1, M + 1)  # try to only build Emp to save time

    Cm, Cmp = numpy.zeros(N - M + 1), numpy.zeros(N - M)
    # in case there is 0 after counting. Log(0) is undefined.

    for i in xrange(0, N - M):
        # print i
        for j in xrange(i, N - M):  # start from i, self-match counts in ApEn
            # compare N-M scalars in each subseq v 0.01b_r1
            # if max(abs(Em[i]-Em[j])) <= R:
            if in_range(Em[i], Em[j], R):
                Cm[i] += 1  # Xin Liu
                Cm[j] += 1
                if abs(Emp[i][-1] - Emp[j][-1]) <= R:  # check last one
                    Cmp[i] += 1
                    Cmp[j] += 1
        if in_range(Em[i], Em[N - M], R):
            Cm[i] += 1
            Cm[N - M] += 1
        # try to count Cm[j] and Cmp[j] as well here

        # index from 0, so N-M+1 is N-M  v 0.01b_r1
        # if max(abs(Em[N-M]-Em[N-M])) <= R:
    # for Cm, there is one more iteration than Cmp
    # if in_range(Em[i], Em[N - M], R):
    #     Cm[N - M] += 1 # cross-matches on Cm[N - M]

    Cm[N - M] += 1  # Cm[N - M] self-matches
    # import code;code.interact(local=locals())
    Cm /= (N - M + 1)
    Cmp /= (N - M)
    # import code;code.interact(local=locals())
    Phi_m, Phi_mp = sum(numpy.log(Cm)), sum(numpy.log(Cmp))

    Ap_En = (Phi_m - Phi_mp) / (N - M)

    return Ap_En


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


    Notes
    -----
    Extremely slow computation. Do NOT use if your dataset is not small and you
    are not patient enough.

    """

    N = len(X)

    Em = embed_seq(X, 1, M)
    Emp = embed_seq(X, 1, M + 1)

    Cm, Cmp = numpy.zeros(N - M - 1) + 1e-100, numpy.zeros(N - M - 1) + 1e-100
    # in case there is 0 after counting. Log(0) is undefined.

    for i in xrange(0, N - M):
        for j in xrange(i + 1, N - M):  # no self-match
            # if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1
            if in_range(Em[i], Em[j], R):
                Cm[i] += 1
#            if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
                if abs(Emp[i][-1] - Emp[j][-1]) <= R:  # check last one
                    Cmp[i] += 1

    Samp_En = numpy.log(sum(Cm) / sum(Cmp))

    return Samp_En


def dfa(X, Ave=None, L=None):
    """Compute Detrended Fluctuation Analysis from a time series X and length of
    boxes L.

    The first step to compute DFA is to integrate the signal. Let original
    series be X= [x(1), x(2), ..., x(N)].

    The integrated signal Y = [y(1), y(2), ..., y(N)] is obtained as follows
    y(k) = \sum_{i=1}^{k}{x(i)-Ave} where Ave is the mean of X.

    The second step is to partition/slice/segment the integrated sequence Y
    into boxes. At least two boxes are needed for computing DFA. Box sizes are
    specified by the L argument of this function. By default, it is from 1/5 of
    signal length to one (x-5)-th of the signal length, where x is the nearest
    power of 2 from the length of the signal, i.e., 1/16, 1/32, 1/64, 1/128,
    ...

    In each box, a linear least square fitting is employed on data in the box.
    Denote the series on fitted line as Yn. Its k-th elements, yn(k),
    corresponds to y(k).

    For fitting in each box, there is a residue, the sum of squares of all
    offsets, difference between actual points and points on fitted line.

    F(n) denotes the square root of average total residue in all boxes when box
    length is n, thus
    Total_Residue = \sum_{k=1}^{N}{(y(k)-yn(k))}
    F(n) = \sqrt(Total_Residue/N)

    The computing to F(n) is carried out for every box length n. Therefore, a
    relationship between n and F(n) can be obtained. In general, F(n) increases
    when n increases.

    Finally, the relationship between F(n) and n is analyzed. A least square
    fitting is performed between log(F(n)) and log(n). The slope of the fitting
    line is the DFA value, denoted as Alpha. To white noise, Alpha should be
    0.5. Higher level of signal complexity is related to higher Alpha.

    Parameters
    ----------

    X:
        1-D Python list or numpy array
        a time series

    Ave:
        integer, optional
        The average value of the time series

    L:
        1-D Python list of integers
        A list of box size, integers in ascending order

    Returns
    -------

    Alpha:
        integer
        the result of DFA analysis, thus the slope of fitting line of log(F(n))
        vs. log(n). where n is the

    Examples
    --------
    >>> import pyeeg
    >>> from numpy.random import randn
    >>> print pyeeg.dfa(randn(4096))
    0.490035110345

    Reference
    ---------
    Peng C-K, Havlin S, Stanley HE, Goldberger AL. Quantification of scaling
    exponents and crossover phenomena in nonstationary heartbeat time series.
    _Chaos_ 1995;5:82-87

    Notes
    -----

    This value depends on the box sizes very much. When the input is a white
    noise, this value should be 0.5. But, some choices on box sizes can lead to
    the value lower or higher than 0.5, e.g. 0.38 or 0.58.

    Based on many test, I set the box sizes from 1/5 of    signal length to one
    (x-5)-th of the signal length, where x is the nearest power of 2 from the
    length of the signal, i.e., 1/16, 1/32, 1/64, 1/128, ...

    You may generate a list of box sizes and pass in such a list as a
    parameter.

    """

    X = numpy.array(X)

    if Ave is None:
        Ave = numpy.mean(X)

    Y = numpy.cumsum(X)
    Y -= Ave

    if L is None:
        L = numpy.floor(len(X) * 1 / (
            2 ** numpy.array(range(4, int(numpy.log2(len(X))) - 4)))
        )

    F = numpy.zeros(len(L))  # F(n) of different given box length n

    for i in xrange(0, len(L)):
        n = int(L[i])                        # for each box length L[i]
        if n == 0:
            print "time series is too short while the box length is too big"
            print "abort"
            exit()
        for j in xrange(0, len(X), n):  # for each box
            if j + n < len(X):
                c = range(j, j + n)
                # coordinates of time in the box
                c = numpy.vstack([c, numpy.ones(n)]).T
                # the value of data in the box
                y = Y[j:j + n]
                # add residue in this box
                F[i] += numpy.linalg.lstsq(c, y)[1]
        F[i] /= ((len(X) / n) * n)
    F = numpy.sqrt(F)

    Alpha = numpy.linalg.lstsq(numpy.vstack(
        [numpy.log(L), numpy.ones(len(L))]
    ).T, numpy.log(F))[0][0]

    return Alpha


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
    Suppose the given time series is X =[x(1),x(2),x(3),x(N)].
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
    calculate entropy by using Shanons entropy formula.

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
        x= PeSeq[0]
        for j in range(0, PeSeq.count(PeSeq[0])):
            PeSeq.pop(PeSeq.index(x))

    RankMat = numpy.array(RankMat)
    RankMat = numpy.true_divide(RankMat, RankMat.sum())
    EntropyMat = numpy.multiply(numpy.log2(RankMat), RankMat)
    PE = -1 * EntropyMat.sum()

    return PE


def information_based_similarity(x, y, n):
    """Calculates the information based similarity of two time series x
    and y.

    Parameters
    ----------

    x

        list

        a time series

    y

        list

        a time series

    n

        integer

        word order


    Returns
    ----------
    IBS

        float

        Information based similarity


    Notes
    ----------
    Information based similarity is a measure of dissimilarity between
    two time series. Let the sequences be x and y. Each sequence is first
    replaced by its first ordered difference(Encoder). Calculating the
    Heaviside of the resulting sequences, we get two binary sequences,
    SymbolicSeq. Using PyEEG function, embed_seq, with lag of 1 and dimension
    of n, we build an embedding matrix from the latter sequence.

    Each row of this embedding matrix is called a word. Information based
    similarity measures the distance between two sequence by comparing the
    rank of words in the sequences; more explicitly, the distance, D, is
    calculated using the formula:

    "1/2^(n-1) * sum( abs(Rank(0)(k)-R(1)(k)) * F(k) )" where Rank(0)(k)
    and Rank(1)(k) are the rank of the k-th word in each of the input
    sequences. F(k) is a modified "shannon" weighing function that increases
    the weight of each word in the calculations when they are more frequent in
    the sequences.

    It is advisable to calculate IBS for numerical sequences using 8-tupple
    words.

    References
    ----------
    Yang AC, Hseu SS, Yien HW, Goldberger AL, Peng CK: Linguistic analysis of
    the human heartbeat using frequency and rank order statistics. Phys Rev
    Lett 2003, 90: 108103


    Examples
    ----------
    >>> import pyeeg
    >>> from numpy.random import randn
    >>> x = randn(100)
    >>> y = randn(100)
    >>> pyeeg.information_based_similarity(x,y,8)
    0.64512947848249214

    """

    Wordlist = []
    Space = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Sample = [0, 1]

    if (n == 1):
        Wordlist = Sample

    if (n == 2):
        Wordlist = Space

    elif (n > 1):
        Wordlist = Space
        Buff = []
        for k in range(0, n - 2):
            Buff = []

            for i in range(0, len(Wordlist)):
                Buff.append(tuple(Wordlist[i]))
            Buff = tuple(Buff)

            Wordlist = []
            for i in range(0, len(Buff)):
                for j in range(0, len(Sample)):
                    Wordlist.append(list(Buff[i]))
                    Wordlist[len(Wordlist) - 1].append(Sample[j])

    Wordlist.sort()

    Input = [[], []]
    Input[0] = x
    Input[1] = y

    SymbolicSeq = [[], []]
    for i in range(0, 2):
        Encoder = numpy.diff(Input[i])
        for j in range(0, len(Input[i]) - 1):
            if(Encoder[j] > 0):
                SymbolicSeq[i].append(1)
            else:
                SymbolicSeq[i].append(0)

    Wm = []
    Wm.append(embed_seq(SymbolicSeq[0], 1, n).tolist())
    Wm.append(embed_seq(SymbolicSeq[1], 1, n).tolist())

    Count = [[], []]
    for i in range(0, 2):
        for k in range(0, len(Wordlist)):
            Count[i].append(Wm[i].count(Wordlist[k]))

    Prob = [[], []]
    for i in range(0, 2):
        Sigma = 0
        for j in range(0, len(Wordlist)):
            Sigma += Count[i][j]
        for k in range(0, len(Wordlist)):
            Prob[i].append(numpy.true_divide(Count[i][k], Sigma))

    Entropy = [[], []]
    for i in range(0, 2):
        for k in range(0, len(Wordlist)):
            if (Prob[i][k] == 0):
                Entropy[i].append(0)
            else:
                Entropy[i].append(Prob[i][k] * (numpy.log2(Prob[i][k])))

    Rank = [[], []]
    Buff = [[], []]
    Buff[0] = tuple(Count[0])
    Buff[1] = tuple(Count[1])
    for i in range(0, 2):
        Count[i].sort()
        Count[i].reverse()
        for k in range(0, len(Wordlist)):
            Rank[i].append(Count[i].index(Buff[i][k]))
            Count[i][Count[i].index(Buff[i][k])] = -1

    IBS = 0
    Z = 0
    n = 0
    for k in range(0, len(Wordlist)):
        if ((Buff[0][k] != 0) & (Buff[1][k] != 0)):
            F = -Entropy[0][k] - Entropy[1][k]
            IBS += numpy.multiply(numpy.absolute(Rank[0][k] - Rank[1][k]), F)
            Z += F
        else:
            n += 1

    IBS = numpy.true_divide(IBS, Z)
    IBS = numpy.true_divide(IBS, len(Wordlist) - n)

    return IBS

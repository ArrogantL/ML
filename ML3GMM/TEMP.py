def calPKX(pk, means, vars, X, k):
    """
    calculating P(K=k|X)
    :param pk:blend weight
    :param means:gaussians' means
    :param vars:gaussians' variances
    :param X:target sample
    :param k:target class
    :return:[[P(k|X)]]
    """
    assert k < len(pk)
    pkx = 0
    lgpxk = 0
    lgpxilist = []
    power0 = (2 * math.pi) ** (-len(X) / 2)
    for i in range(len(pk)):
        mean = means[i]
        var = vars[i]
        # use log to ease overflow
        lgpxi = multivariate_normal.logpdf(X, mean=mean, cov=var) + log(pk[i])
        lgpxilist.append(lgpxi)
        if i == k:
            lgpxk = lgpxi
    for lgpxi in lgpxilist:
        pkx += exp(lgpxi - lgpxk)

    pkx = 1 / pkx
    return pkx
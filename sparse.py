import numpy as np

def get_target_sparsity(m):
    """
    Get expected matrix sparsity, chosen to be O(log n).
    """
    return np.log(m.shape[0])


def get_sparsity_pattern(m):
    """
    Prepare in O(n log n) time a sparsity pattern over the n x n matrix with a
    pseudorandom generator.
    """
    target_sparsity = get_target_sparsity(m)

    # procedure produces an equivalent distribution of 1s and 0s as sampling individual
    # matrix elements i.i.d. from binomial distribution

    # since we'll take half of the generated indices, we set the probability of a nonzero
    # element to be double the target sparsity
    p_one = min(2 * target_sparsity / m.shape[0], 1.0)

    # for each row, sample the binomial distribution to get the number of nonzero indices
    # matches in expectation get_target_sparsity(m), i.e. O(log n)
    # reference the upper triangular indices according to the lower triangular indices
    # can be done efficiently by mapping indices instead of copying matrix elements

    one_filter = np.zeros(m.shape)
    for i in range(m.shape[0]):
        # find O(log n) indices
        num_nonzero = np.random.randint(m.shape[0],
                                        size=np.random.binomial(m.shape[0], p_one))
        one_filter[i][num_nonzero] = 1
    one_filter = np.tril(one_filter) + np.tril(one_filter, -1).T

    # make sure the diagonal is ones
    np.fill_diagonal(one_filter, 1)

    return one_filter

def sparsify_unbiased(m, sparsity_pattern):
    """
    Sparsify NTK matrix `m` using a given sparsity pattern.
    Used for the fully-connected network.
    """
    return m * sparsity_pattern


def sparsify_biased(m, sparsity_pattern, t0, t1):
    """
    Sparsify NTK matrix `m` using a given sparsity pattern, then additionally sparsify by
    setting elements below `t0` and `t1` in classes 0 and 1 respectively to 0.
    Used for the convolutional network.
    """
    # 对角线元素不变，把非对角线元素中较小的置为0
    class_0, class_1 = block_diagonal(m)
    one_filter = sparsity_pattern * ((m > t0) * class_0 + (m > t1) * class_1)
    np.fill_diagonal(one_filter, 1)

    kernel_train_sparse = m * one_filter

    # 对主对角线元素进行修正
    # we expect a factor of ~target_sparsity by Gershgorin's theorem
    # empirically, the well-conditioning of the kernel makes it scale better than this
    f = 0.76 * get_target_sparsity(m) ** 0.9
    conditioning = f * np.diag(kernel_train_sparse) * np.eye(kernel_train_sparse.shape[0])
    kernel_train_conditioned = kernel_train_sparse + conditioning
    return kernel_train_conditioned
import math
import numpy as np
import scipy as sp

class KernelMatrix(object):
    """
    implement the Graph Neural Tangent Kernel
    """
    def __init__(self, num_layers, num_mlp_layers, jk, scale):
        """
        num_layers: number of layers in the neural networks (including the input layer)
        num_mlp_layers: number of MLP layers
        jk: a bool variable indicating whether to add jumping knowledge
        scale: the scale used aggregate neighbors [uniform, degree]
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.scale = scale
        assert(scale in ['uniform', 'degree'])

        self.sigma_w2 = 2.
        self.sigma_b2 = 0.01


    def __adj_diag_nngp(self, S, adj_block, N, scale_mat):
        # S = S / num_feat
        # return adj_block.dot(S.reshape(-1)).reshape(N, N) * self.sigma_w2 + self.sigma_b2
        return adj_block.dot(S.reshape(-1)).reshape(N, N) * scale_mat

    def __next_diag_nngp(self, S):
        diag = np.sqrt(np.diag(S))

        # dense
        S = S * self.sigma_w2

        # relu
        sqrt_xx_yy = diag[:, None] * diag[None, :]
        S = S / sqrt_xx_yy
        S = np.clip(S, -1, 1)  # normalization
        theta = np.arccos(S)
        S = self.sigma_w2 * sqrt_xx_yy * (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / 2. / np.pi
        return S, diag

    def __adj_nngp(self, S, adj_block, N1, N2, scale_mat):
        """
        go through one adj layer, for all elements
        """
        return adj_block.dot(S.reshape(-1)).reshape(N1, N2) * scale_mat

    def __next_nngp(self, S, diag1, diag2):
        # dense
        S = S * self.sigma_w2

        # relu
        sqrt_xx_yy = diag1[:, None] * diag2[None, :]
        S = S / sqrt_xx_yy
        S = np.clip(S, -1, 1)  # normalization
        theta = np.arccos(S)
        S = self.sigma_w2 * sqrt_xx_yy * (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / 2. / np.pi

        return S


    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = np.sqrt(np.diag(S)) # the sqrt of the diagonal entries of sigma --> to make the diag of the Λ to be 1 in the later
        S = S / diag[:, None] / diag[None, :]
        S = np.clip(S, -1, 1)  # normalization
        # dot sigma
        DS = (math.pi - np.arccos(S)) / math.pi  # dot_sigma of next mlp layer
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi # sigma of next mlp layer
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag

    def __adj_diag(self, S, adj_block, N, scale_mat):
        """
        go through one adj layer
        S: the covariance
        adj_block: the adjacency relation
        N: number of vertices
        scale_mat: scaling matrix
        """
        return adj_block.dot(S.reshape(-1)).reshape(N, N) * scale_mat

    def __next(self, S, diag1, diag2):
        """
        go through one normal layer, for all elements
        """
        S = S / diag1[:, None] / diag2[None, :]
        S = np.clip(S, -1, 1)
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, DS
    
    def __adj(self, S, adj_block, N1, N2, scale_mat):
        """
        go through one adj layer, for all elements
        """
        return adj_block.dot(S.reshape(-1)).reshape(N1, N2) * scale_mat
      
    def diag(self, g, A):
        """
        compute the diagonal element of GNTK for graph `g` with adjacency matrix `A`
        g: graph g
        A: adjacency matrix
        """
        N = A.shape[0]  # num of nodes of graph g

        # calculate c_u * c_u'
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / np.array(np.sum(A, axis=1) * np.sum(A, axis=0))

        diag_list = []  # 储存l=1~L-1个block的相同graph的r = 0~R-1的sigma的对角线元素的sqrt, len = (L-1)*R
        diag_nngp_list = []
        nngp_xx_list = []
        adj_block = sp.sparse.kron(A, A)

        # input covariance
        sigma = np.matmul(g.node_features, g.node_features.T)   # 计算sigma^{0}
        sigma = self.__adj_diag(sigma, adj_block, N, scale_mat) # 计算G = G'时的sigma^{1}_{0}
        ntk = np.copy(sigma)

        nngp = np.copy(sigma)
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self.__next_diag(sigma) # sigma of next mlp layer, dot_sigma of next mlp layer, the sqrt of the diagonal entries of sigma
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma

                nngp, diag_nngp = self.__next_diag_nngp(nngp)
                diag_nngp_list.append(diag_nngp)
            # if not last layer
            if layer != self.num_layers - 1:
                ########
                nngp_xx_list.append(nngp)
                nngp_old = np.copy(nngp)
                nngp = np.matmul(np.matmul(nngp, nngp), nngp.T)
                ntk = 2 * nngp + np.matmul(np.matmul(nngp_old, ntk*2), nngp_old.T)
                ########

                sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
                ntk = self.__adj_diag(ntk, adj_block, N, scale_mat)  # compute sigma^{l+1}_{0} and ntk^{l+1}_{0}

                nngp = self.__adj_diag_nngp(nngp, adj_block, N, scale_mat)
        return diag_list, diag_nngp_list, nngp_xx_list

    def kernel(self, g1, g2, diag_list1, diag_list2, A1, A2, diag_list1_nngp, diag_list2_nngp, nngp_xx_list1, nngp_xx_list2):
        """
        compute the kernel value \Theta(g1, g2)
        g1: graph1
        g2: graph2
        diag_list1, diag_list2: g1, g2's the diagonal elements of covariance matrix in all layers
        A1, A2: g1, g2's adjacency matrix
        """

        n1 = A1.shape[0]  # num of nodes of g1
        n2 = A2.shape[0]  # num of nodes of g2
        
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / np.array(np.sum(A1, axis=1) * np.sum(A2, axis=0))
        
        adj_block = sp.sparse.kron(A1, A2)
        
        jump_ntk = 0
        sigma = np.matmul(g1.node_features, g2.node_features.T)  # 计算时的sigma^{0}
        jump_ntk += sigma
        sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)  # 计算sigma^{1}_{0}
        ntk = np.copy(sigma)

        nngp = np.copy(sigma)
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma = self.__next(sigma, 
                                    diag_list1[(layer - 1) * self.num_mlp_layers + mlp_layer],
                                    diag_list2[(layer - 1) * self.num_mlp_layers + mlp_layer])  # sigma of next mlp layer, dot_sigma of next mlp layer
                ntk = ntk * dot_sigma + sigma  # ntk

                nngp = self.__next_nngp(nngp,
                                        diag_list1_nngp[(layer - 1) * self.num_mlp_layers + mlp_layer],
                                        diag_list2_nngp[(layer - 1) * self.num_mlp_layers + mlp_layer])
            jump_ntk += ntk
            # if not last layer
            if layer != self.num_layers - 1:
                ########
                nngp = np.matmul(np.matmul(nngp_xx_list1[(layer - 1)], nngp), nngp_xx_list2[(layer - 1)].T)
                ntk = 2 * nngp + np.matmul(np.matmul(nngp_xx_list1[(layer - 1)], ntk*2), nngp_xx_list2[(layer - 1)].T)
                ########

                sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
                ntk = self.__adj(ntk, adj_block, n1, n2, scale_mat)  # compute sigma^{l+1}_{0} and ntk^{l+1}_{0}

                nngp = self.__adj_nngp(nngp, adj_block, n1, n2, scale_mat)


        if self.jk:
            return np.sum(jump_ntk)
        else:
            return np.sum(ntk)

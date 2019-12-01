from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import cfg
from datasets.dataset_info import KITTICategory

from models.model_util import get_box3d_corners_helper
from models.model_util import huber_loss

from models.common import Conv1d, Conv2d, DeConv1d, init_params, XConv
from models.common import softmax_focal_loss_ignore, get_accuracy

from ops.query_depth_point.query_depth_point import QueryDepthPoint
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair
from models.box_transform import size_decode, size_encode, center_decode, center_encode, angle_decode, angle_encode
#####################
from util_funcs import knn_indices_func_gpu, knn_indices_func_cpu, UFloatTensor, ULongTensor
from model_util import Conv, SepConv, Dense, EndChannels
from typing import Tuple, Callable, Optional
import matplotlib.pyplot as plt
from itertools import product, groupby
from torch import FloatTensor


NUM_SIZE_CLUSTER = len(KITTICategory.CLASSES)
MEAN_SIZE_ARRAY = KITTICategory.MEAN_SIZE_ARRAY
'''
class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.  """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int,
                 P : int, C_mid : int, depth_multiplier : int) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
        :param C_mid: Dimensionality of lifted point features.
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution.
        """
        super(XConv, self).__init__()

        self.P = P
        self.K = K
        self.dims = dims

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers
        self.dense1 = Dense(dims, C_mid)
        self.dense2 = Dense(C_mid, C_mid)

        # Layers to generate X
        # print("in_channels", dims, "out_channels", K*K, "C_in", C_in)
        self.x_trans = nn.Sequential(
            EndChannels(Conv(
                in_channels = dims,
                out_channels = K*K,
                kernel_size = (1, K),
                with_bn = False
            )),
            Dense(K*K, K*K, with_bn = False),
            Dense(K*K, K*K, with_bn = False, activation = None)
        )

        self.end_conv = EndChannels(SepConv(
            # XXX
            # in_channels = C_mid + C_in,
            in_channels = C_mid,
            out_channels = C_out,
            kernel_size = (1, K),
            depth_multiplier = depth_multiplier
        )).cuda()

    def forward(self, x : Tuple[UFloatTensor,            # (N, P, dims)
                                UFloatTensor,            # (N, P, K, dims)
                                Optional[UFloatTensor]]  # (N, P, K, C_in)
               ) -> UFloatTensor:                        # (N, K, C_out)
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x
        if fts is not None:
            assert(rep_pt.size()[0] == pts.size()[0] == fts.size()[0])  # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1] == fts.size()[1])  # Check P is equal.
            assert(pts.size()[2] == fts.size()[2] == self.K)            # Check K is equal.
            assert(fts.size()[3] == self.C_in)                          # Check C_in is equal.
        else:
            assert(rep_pt.size()[0] == pts.size()[0])                   # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1])                   # Check P is equal.
            assert(pts.size()[2] == self.K)                             # Check K is equal.
        assert(rep_pt.size()[2] == pts.size()[3] == self.dims)          # Check dims is equal.

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim = 2)  # (N, P, 1, dims)

        # Move pts to local coordinate system of rep_pt.
        # t0 = time.time()
        pts_local = pts - p_center  # (N, P, K, dims)
        # print("pts_local", pts_local.shape)
        # print("localizing", time.time() - t0)
        # pts_local = self.pts_layernorm(pts - p_center)

        # Individually lift each point into C_mid space.
        # t0 = time.time()
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted  = self.dense2(fts_lifted0)  # (N, P, K, C_mid)
        # print("lifting", time.time() - t0)

        # t0 = time.time()
        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)
        # print("cat", time.time() - t0)

        # Learn the (N, K, K) X-transformation matrix.
        # t0 = time.time()
        X_shape = (N, P, self.K, self.K)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)
        # print("X-CONV", time.time() - t0)


        # Weight and permute fts_cat with the learned X.
        # t0 = time.time()
        fts_X = torch.matmul(X, fts_cat)
        # print("matmul", time.time() - t0)
        # t0 = time.time()
        fts_p = self.end_conv(fts_X).squeeze(dim = 2)
        # print("end-conv", time.time() - t0)
        return fts_p # [:,rand_idx_inv,:]

class PointCNN(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                                            UFloatTensor,  # (N, x, dims)
                                            int, int],
                                           ULongTensor],    # (N, P, K)
                 sampling_method : str = "fast_fps") -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        :param r_indices_func: Selector function of the type,
          INPUTS
          rep_pts : Representative points.
          pts  : Point cloud.
          K : Number of points for each region.
          D : "Spread" of neighboring points.
          OUTPUT
          pts_idx : Array of indices into pts such that pts[pts_idx] is the set
          of points in the "region" around rep_pt.
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4
        depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)
        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.K = K
        self.D = D
        self.P = P
        self.sampling_method = sampling_method

    def select_region(self, pts : UFloatTensor,  # (N, x, dims)
                      pts_idx : ULongTensor      # (N, P, K)
                     ) -> UFloatTensor:          # (P, K, dims)
        """
        Selects neighborhood points based on output of r_indices_func.
        :param pts: Point cloud to select regional points from.
        :param pts_idx: Indices of points in region to be selected.
        :return: Local neighborhoods around each representative point.
        """
        regions = torch.stack([
            pts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))
        ], dim = 0)
        return regions

    def forward(self,
                x : Tuple[FloatTensor,        # (N, P, dims)
                          FloatTensor,        # (N, x, dims)
                          FloatTensor],       # (N, x, C_in)
                pts_idx : FloatTensor = None  # TODO
               ) -> FloatTensor:              # (N, P, C_out)
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :param x: (rep_pts, pts, fts) where
          - rep_pts: Representative points.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the
          feature associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated to rep_pts.
        """
        if len(x) == 2:
            pts, fts = x

            if 0 < self.P < pts.size()[1]:
                # Select random set of indices of subsampled points.
                if self.sampling_method == "rand":
                    idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
                    rep_pts = pts[:,idx,:]
                elif self.sampling_method == "fps":
                    # t0 = time.time()
                    idx = self.batch_fps(pts, self.P)
                    rep_pts = torch.stack([pts[n][i,:] for n,i in enumerate(idx)])
                    # print("BATCH FPS:", time.time() - t0)
                elif self.sampling_method == "fast_fps":
                    # t0 = time.time()
                    idx = self.fast_fps(pts, self.P)
                    # print("FPS", time.time() - t0)
                    rep_pts = torch.stack([pts[n][i,:] for n,i in enumerate(idx)])
                else:
                    raise ValueError("Unrecognized sampling method %s" % self.sampling_method)
            else:
                # All input points are representative points.
                rep_pts = pts
        else:
            rep_pts, pts, fts = x
        fts = self.dense(fts) if fts is not None else fts

        # This step takes ~97% of the time. Prime target for optimization: KNN on GPU.
        if type(pts_idx) == type(None):
            # t0 = time.time()
            pts_idx = self.r_indices_func(rep_pts, pts)
            # print("KNN:", time.time() - t0)
            # print(pts_idx.size())
        else:
            pts_idx = pts_idx[:,:,:self.K*self.D:self.D].cuda()
        # -------------------------------------------------------------------------- #

        pts_regional = self.select_region(pts, pts_idx)

        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))

        return (rep_pts, fts_p) if len(x) == 2 else fts_p

    def batch_fps(self, batch_pts, K):
        """ Found here: 
        https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
        """

        if isinstance(batch_pts, torch.autograd.Variable):
            batch_pts = batch_pts.data
        if batch_pts.is_cuda:
            batch_pts = batch_pts.cpu()

        calc_distances = lambda p0, pts: ((p0 - pts)**2).sum(dim = 1)

        def fps(x):
            pts, K = x
            D = pts.size()[1]
            farthest_idx = torch.IntTensor(K)
            farthest_idx.zero_()
            farthest_idx[0] = np.random.randint(len(pts))
            distances = calc_distances(pts[farthest_idx[0]], pts)

            for i in range(1, K):
                farthest_idx[i] = torch.max(distances, dim = 0)[1]
                farthest_pts = pts[farthest_idx[i]]
                distances = torch.min(distances, calc_distances(farthest_pts, pts))

            return farthest_idx

        batch_pts = list(map(fps, [(pts,K) for pts in batch_pts]))
        return torch.stack(batch_pts, dim = 0).long().cuda()

    def fast_fps(self, batch_pts, K):

        cell_size = torch.FloatTensor([1.2, 1.2, 1.2]).cuda()

        def fps(x):
            pts, K = x
            N = len(pts)

            lower = torch.min(pts, dim = 0)[0]
            upper = torch.max(pts, dim = 0)[0]
            dims = upper - lower
            idx_collapse = (dims / cell_size).int() + 1
            idx_collapse[0] = idx_collapse[1] * idx_collapse[2]
            idx_collapse[1] = idx_collapse[2]
            idx_collapse[2] = 1

            bin_idx = ((pts - upper.cuda()) / cell_size).int()
            bin_idx *= idx_collapse
            bin_idx = torch.sum(bin_idx, dim = 1)
            sorted_bins, p_idx = torch.sort(bin_idx, dim = 0)

            densities = [len(list(group)) for key, group in groupby(sorted_bins.tolist())]

            bin_prob = 1.0 / len(densities)
            p_probs = []

            for d in densities:
                single_bins = [bin_prob / d] * d
                p_probs += single_bins

            return torch.from_numpy(np.random.choice(p_idx, size = K, replace = False, p = p_probs))

        batch_pts = list(map(fps, [(pts,K) for pts in batch_pts]))
        return torch.stack(batch_pts, dim = 0).long().cuda()

class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                                            UFloatTensor,  # (N, x, dims)
                                            int, int],
                                           ULongTensor],   # (N, P, K)
                 sampling_method : str = "rand") -> None:
        """ See documentation for PointCNN. """
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P
        self.sampling_method = sampling_method

    def forward(self, x : Tuple[UFloatTensor,  # (N, x, dims)
                                UFloatTensor]  # (N, x, dims)
               ) -> Tuple[UFloatTensor,        # (N, P, dims)
                          UFloatTensor]:       # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            if self.sampling_method == "rand":
                idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
                rep_pts = pts[:,idx,:]
            elif self.sampling_method == "fps":
                # t0 = time.time()
                idx = self.batch_fps(pts, self.P)
                rep_pts = torch.stack([pts[n][i,:] for n,i in enumerate(idx)])
                # print("BATCH FPS:", time.time() - t0)
            else:
                raise ValueError("Unrecognized sampling method %s" % self.sampling_method)
        else:
            # All input points are representative points.
            rep_pts = pts
        # t0 = time.time()
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        # print("TOTAL:", time.time() - t0)
        return rep_pts, rep_pts_fts

    def batch_fps(self, batch_pts, K):
        """ Found here: 
        https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
        """

        if isinstance(batch_pts, torch.autograd.Variable):
            batch_pts = batch_pts.data
        if batch_pts.is_cuda:
            batch_pts = batch_pts.cpu()

        calc_distances = lambda p0, pts: ((p0 - pts)**2).sum(dim = 1)

        def fps(x):
            pts, K = x
            D = pts.size()[1]
            farthest_idx = torch.IntTensor(K)
            farthest_idx.zero_()
            farthest_idx[0] = np.random.randint(len(pts))
            distances = calc_distances(pts[farthest_idx[0]], pts)

            for i in range(1, K):
                farthest_idx[i] = torch.max(distances, dim = 0)[1]
                farthest_pts = pts[farthest_idx[i]]
                distances = torch.min(distances, calc_distances(farthest_pts, pts))

            return farthest_idx

        batch_pts = list(map(fps, [(pts,K) for pts in batch_pts]))
        return torch.stack(batch_pts, dim = 0).long().cuda()

class PCNNModule(nn.Module):
    def __init__(self):
        super(PCNNModule, self).__init__()
        self.pointnet1 = PointCNN(C_in=1, C_out=128, dims=3, K=8, D=2, P=-1, r_indices_func=knn_indices_func_gpu, sampling_method="fast_fps")
        self.pointnet2 = PointCNN(C_in=1, C_out=128, dims=3, K=8, D=4, P=-1, r_indices_func=knn_indices_func_gpu, sampling_method="fast_fps")
        self.pointnet3 = PointCNN(C_in=1, C_out=256, dims=3, K=8, D=4, P=-1, r_indices_func=knn_indices_func_gpu, sampling_method="fast_fps")
        self.pointnet4 = PointCNN(C_in=1, C_out=512, dims=3, K=8, D=4, P=-1, r_indices_func=knn_indices_func_gpu, sampling_method="fast_fps")

    def forward(self, point_cloud, sample_pc, feat=None, one_hot_vec=None):
        pc = point_cloud
        pc1 = sample_pc[0].permute(0, 2, 1)
        pc2 = sample_pc[1].permute(0, 2, 1)
        pc3 = sample_pc[2].permute(0, 2, 1)
        pc4 = sample_pc[3].permute(0, 2, 1)
        
        feat1 = self.pointnet1((pc1, None))[1].permute(0, 2, 1)
        feat2 = self.pointnet2((pc2, None))[1].permute(0, 2, 1)
        feat3 = self.pointnet3((pc3, None))[1].permute(0, 2, 1)
        feat4 = self.pointnet4((pc4, None))[1].permute(0, 2, 1)

        if one_hot_vec is not None:
            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat1.shape[-1])
            feat1 = torch.cat([feat1, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat2.shape[-1])
            feat2 = torch.cat([feat2, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat3.shape[-1])
            feat3 = torch.cat([feat3, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat4.shape[-1])
            feat4 = torch.cat([feat4, one_hot], 1)

        return feat1, feat2, feat3, feat4
'''
# single scale PointNet module
class PointNetModule(nn.Module):
    def __init__(self, Infea, mlp, dist, nsample, use_xyz=True, use_feature=True):
        super(PointNetModule, self).__init__()
        self.dist = dist
        self.nsample = nsample
        self.use_xyz = use_xyz

        if Infea > 0:
            use_feature = True
        else:
            use_feature = False

        self.use_feature = use_feature

        self.query_depth_point = QueryDepthPoint(dist, nsample)

        if self.use_xyz:
            self.conv1 = Conv2d(Infea + 3, mlp[0], 1)
        else:
            self.conv1 = Conv2d(Infea, mlp[0], 1)

        self.conv2 = Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = Conv2d(mlp[1], mlp[2], 1)

        init_params([self.conv1[0], self.conv2[0], self.conv3[0]], 'kaiming_normal')
        init_params([self.conv1[1], self.conv2[1], self.conv3[1]], 1)

    def forward(self, pc, feat, new_pc=None):
        batch_size = pc.size(0)

        npoint = new_pc.shape[2]
        k = self.nsample

        indices, num = self.query_depth_point(pc, new_pc)  # b*npoint*nsample

        assert indices.data.max() < pc.shape[2] and indices.data.min() >= 0
        grouped_pc = None
        grouped_feature = None

        if self.use_xyz:
            grouped_pc = torch.gather(
                pc, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, 3, -1)
            ).view(batch_size, 3, npoint, k)

            grouped_pc = grouped_pc - new_pc.unsqueeze(3)

        if self.use_feature:
            grouped_feature = torch.gather(
                feat, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, feat.size(1), -1)
            ).view(batch_size, feat.size(1), npoint, k)

            # grouped_feature = torch.cat([new_feat.unsqueeze(3), grouped_feature], -1)

        if self.use_feature and self.use_xyz:
            grouped_feature = torch.cat([grouped_pc, grouped_feature], 1)
        elif self.use_xyz:
            grouped_feature = grouped_pc.contiguous()

        grouped_feature = self.conv1(grouped_feature)
        grouped_feature = self.conv2(grouped_feature)
        grouped_feature = self.conv3(grouped_feature)
        # output, _ = torch.max(grouped_feature, -1)

        valid = (num > 0).view(batch_size, 1, -1, 1)
        grouped_feature = grouped_feature * valid.float()

        return grouped_feature
    
# multi-scale PointNet module
class PointNetFeat(nn.Module):
    def __init__(self, input_channel=3, num_vec=0):
        super(PointNetFeat, self).__init__()

        self.num_vec = num_vec
        u = cfg.DATA.HEIGHT_HALF
        assert len(u) == 4
        self.pointnet1 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[0], 32, use_xyz=True, use_feature=True)

        self.pointnet2 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[1], 64, use_xyz=True, use_feature=True)

        self.pointnet3 = PointNetModule(
            input_channel - 3, [128, 128, 256], u[2], 64, use_xyz=True, use_feature=True)

        self.pointnet4 = PointNetModule(
            input_channel - 3, [256, 256, 512], u[3], 128, use_xyz=True, use_feature=True)
        depth_multiplier = 4
        self.xconv1 = XConv(1, 128, depth_multiplier=1, with_X_transformation=False)
        self.xconv2 = XConv(1, 128, depth_multiplier=1, with_X_transformation=False)
        self.xconv3 = XConv(1, 256, depth_multiplier=1, with_X_transformation=False)
        self.xconv4 = XConv(1, 512, depth_multiplier=1, with_X_transformation=False)

    def forward(self, point_cloud, sample_pc, feat=None, one_hot_vec=None):
        pc = point_cloud
        pc1 = sample_pc[0] # [32, 3, 280]
        pc2 = sample_pc[1]
        pc3 = sample_pc[2]
        pc4 = sample_pc[3]

        feat1 = self.pointnet1(pc, feat, pc1) # [32, 128, 280, 32]
        feat1 = self.xconv1(pc1, feat1)
        feat1, _ = torch.max(feat1, -1)

        feat2 = self.pointnet2(pc, feat, pc2)
        feat2 = self.xconv2(pc2, feat2)
        feat2, _ = torch.max(feat2, -1)

        feat3 = self.pointnet3(pc, feat, pc3)
        feat3 = self.xconv3(pc3, feat3)
        feat3, _ = torch.max(feat3, -1)

        feat4 = self.pointnet4(pc, feat, pc4)
        # feat4 = self.xconv4(pc4, feat4)
        feat4, _ = torch.max(feat4, -1)

        if one_hot_vec is not None:
            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat1.shape[-1])
            # print(feat1.shape, one_hot.shape)
            feat1 = torch.cat([feat1, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat2.shape[-1])
            feat2 = torch.cat([feat2, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat3.shape[-1])
            feat3 = torch.cat([feat3, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat4.shape[-1])
            feat4 = torch.cat([feat4, one_hot], 1)

        return feat1, feat2, feat3, feat4


# FCN
class ConvFeatNet(nn.Module):
    def __init__(self, i_c=128, num_vec=3):
        super(ConvFeatNet, self).__init__()

        self.block1_conv1 = Conv1d(i_c + num_vec, 128, 3, 1, 1)

        self.block2_conv1 = Conv1d(128, 128, 3, 2, 1)
        self.block2_conv2 = Conv1d(128, 128, 3, 1, 1)
        self.block2_merge = Conv1d(128 + 128 + num_vec, 128, 1, 1)

        self.block3_conv1 = Conv1d(128, 256, 3, 2, 1)
        self.block3_conv2 = Conv1d(256, 256, 3, 1, 1)
        self.block3_merge = Conv1d(256 + 256 + num_vec, 256, 1, 1)

        self.block4_conv1 = Conv1d(256, 512, 3, 2, 1)
        self.block4_conv2 = Conv1d(512, 512, 3, 1, 1)
        self.block4_merge = Conv1d(512 + 512 + num_vec, 512, 1, 1)

        self.block2_deconv = DeConv1d(128, 256, 1, 1, 0)
        self.block3_deconv = DeConv1d(256, 256, 2, 2, 0)
        self.block4_deconv = DeConv1d(512, 256, 4, 4, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):

        x = self.block1_conv1(x1)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = torch.cat([x, x2], 1)
        x = self.block2_merge(x)
        xx1 = x

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = torch.cat([x, x3], 1)
        x = self.block3_merge(x)
        xx2 = x

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = torch.cat([x, x4], 1)
        x = self.block4_merge(x)
        xx3 = x

        xx1 = self.block2_deconv(xx1)
        xx2 = self.block3_deconv(xx2)
        xx3 = self.block4_deconv(xx3)

        x = torch.cat([xx1, xx2[:, :, :xx1.shape[-1]], xx3[:, :, :xx1.shape[-1]]], 1)

        return x


# the whole pipeline
class PointNetDet(nn.Module):
    def __init__(self, input_channel=3, num_vec=0, num_classes=2):
        super(PointNetDet, self).__init__()

        self.feat_net = PointNetFeat(input_channel, 0)
        # self.feat_net = PCNNModule()
        self.conv_net = ConvFeatNet()

        self.num_classes = num_classes

        num_bins = cfg.DATA.NUM_HEADING_BIN
        self.num_bins = num_bins

        output_size = 3 + num_bins * 2 + NUM_SIZE_CLUSTER * 4

        self.reg_out = nn.Conv1d(768, output_size, 1)
        self.cls_out = nn.Conv1d(768, 2, 1)
        self.relu = nn.ReLU(True)

        nn.init.kaiming_uniform_(self.cls_out.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.reg_out.weight, mode='fan_in')

        self.cls_out.bias.data.zero_()
        self.reg_out.bias.data.zero_()

    def _slice_output(self, output):

        batch_size = output.shape[0]

        num_bins = self.num_bins

        center = output[:, 0:3].contiguous()

        heading_scores = output[:, 3:3 + num_bins].contiguous()

        heading_res_norm = output[:, 3 + num_bins:3 + num_bins * 2].contiguous()

        size_scores = output[:, 3 + num_bins * 2:3 + num_bins * 2 + NUM_SIZE_CLUSTER].contiguous()

        size_res_norm = output[:, 3 + num_bins * 2 + NUM_SIZE_CLUSTER:].contiguous()
        size_res_norm = size_res_norm.view(batch_size, NUM_SIZE_CLUSTER, 3)

        return center, heading_scores, heading_res_norm, size_scores, size_res_norm

    def get_center_loss(self, pred_offsets, gt_offsets):

        center_dist = torch.norm(gt_offsets - pred_offsets, 2, dim=-1)
        center_loss = huber_loss(center_dist, delta=3.0)

        return center_loss

    def get_heading_loss(self, heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label):

        heading_class_loss = F.cross_entropy(heading_scores, heading_class_label)

        # b, NUM_HEADING_BIN -> b, 1
        heading_res_norm_select = torch.gather(heading_res_norm, 1, heading_class_label.view(-1, 1))

        heading_res_norm_loss = huber_loss(
            heading_res_norm_select.squeeze(1) - heading_res_norm_label, delta=1.0)

        return heading_class_loss, heading_res_norm_loss

    def get_size_loss(self, size_scores, size_res_norm, size_class_label, size_res_label_norm):
        batch_size = size_scores.shape[0]
        size_class_loss = F.cross_entropy(size_scores, size_class_label)

        # b, NUM_SIZE_CLUSTER, 3 -> b, 1, 3
        size_res_norm_select = torch.gather(size_res_norm, 1,
                                            size_class_label.view(batch_size, 1, 1).expand(
                                                batch_size, 1, 3))

        size_norm_dist = torch.norm(
            size_res_label_norm - size_res_norm_select.squeeze(1), 2, dim=-1)

        size_res_norm_loss = huber_loss(size_norm_dist, delta=1.0)

        return size_class_loss, size_res_norm_loss

    def get_corner_loss(self, preds, gts):

        center_label, heading_label, size_label = gts
        center_preds, heading_preds, size_preds = preds

        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)

        corners_3d_pred = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

        # N, 8, 3
        corners_dist = torch.min(
            torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1).mean(-1),
            torch.norm(corners_3d_pred - corners_3d_gt_flip, 2, dim=-1).mean(-1))
        # corners_dist = torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1)
        corners_loss = huber_loss(corners_dist, delta=1.0)

        return corners_loss, corners_3d_gt

    def forward(self,
                data_dicts):
        point_cloud = data_dicts.get('point_cloud')
        one_hot_vec = data_dicts.get('one_hot')
        cls_label = data_dicts.get('label')
        size_class_label = data_dicts.get('size_class')
        center_label = data_dicts.get('box3d_center')
        heading_label = data_dicts.get('box3d_heading')
        size_label = data_dicts.get('box3d_size')

        center_ref1 = data_dicts.get('center_ref1')
        center_ref2 = data_dicts.get('center_ref2')
        center_ref3 = data_dicts.get('center_ref3')
        center_ref4 = data_dicts.get('center_ref4')

        batch_size = point_cloud.shape[0]

        object_point_cloud_xyz = point_cloud[:, :3, :].contiguous()
        if point_cloud.shape[1] > 3:
            object_point_cloud_i = point_cloud[:, [3], :].contiguous()
        else:
            object_point_cloud_i = None

        mean_size_array = torch.from_numpy(MEAN_SIZE_ARRAY).type_as(point_cloud)

        # print("Point cloud")
        # print(point_cloud.shape)
        # print("one_hot_vec")
        # print(one_hot_vec.shape)
        # print("cls_label")
        # print(cls_label.shape)
        # print("size_class_label")
        # print(size_class_label.shape)
        # print("center_label")
        # print(center_label.shape)
        # print("heading_label")
        # print(heading_label.shape)
        # print("size_label")
        # print(size_label.shape)
        # print("center_ref1")
        # print(center_ref1.shape)
        # print("center_ref2")
        # print(center_ref2.shape)
        # print("center_ref3")
        # print(center_ref3.shape)
        # print("center_ref4")
        # print(center_ref4.shape)
        # print("Class labels")
        # torch.set_printoptions(profile="full")
        # torch.set_printoptions(profile="default")
        
        # print(center_ref1.shape, center_ref2.shape, center_ref3.shape)
        feat1, feat2, feat3, feat4 = self.feat_net(
            object_point_cloud_xyz,
            [center_ref1, center_ref2, center_ref3, center_ref4],
            object_point_cloud_i,
            one_hot_vec)
        # print(feat1.shape, feat2.shape, feat3.shape)

        x = self.conv_net(feat1, feat2, feat3, feat4)

        cls_scores = self.cls_out(x)
        outputs = self.reg_out(x)

        num_out = outputs.shape[2]
        output_size = outputs.shape[1]
        # b, c, n -> b, n, c
        cls_scores = cls_scores.permute(0, 2, 1).contiguous().view(-1, 2)
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, output_size)

        center_ref2 = center_ref2.permute(0, 2, 1).contiguous().view(-1, 3)

        cls_probs = F.softmax(cls_scores, -1)

        if center_label is None:
            assert not self.training, 'Please provide labels for training.'

            det_outputs = self._slice_output(outputs)

            center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = det_outputs

            # decode
            heading_probs = F.softmax(heading_scores, -1)
            size_probs = F.softmax(size_scores, -1)

            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            center_preds = center_boxnet + center_ref2

            heading_preds = angle_decode(heading_res_norm, heading_pred_label)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            # corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

            cls_probs = cls_probs.view(batch_size, -1, 2)
            center_preds = center_preds.view(batch_size, -1, 3)

            size_preds = size_preds.view(batch_size, -1, 3)
            heading_preds = heading_preds.view(batch_size, -1)

            outputs = (cls_probs, center_preds, heading_preds, size_preds)
            return outputs

        fg_idx = (cls_label.view(-1) == 1).nonzero().view(-1)

        assert fg_idx.numel() != 0

        outputs = outputs[fg_idx, :]
        center_ref2 = center_ref2[fg_idx]

        det_outputs = self._slice_output(outputs)

        center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = det_outputs

        heading_probs = F.softmax(heading_scores, -1)
        size_probs = F.softmax(size_scores, -1)

        # cls_loss = F.cross_entropy(cls_scores, mask_label, ignore_index=-1)
        cls_loss = softmax_focal_loss_ignore(cls_probs, cls_label.view(-1), ignore_idx=-1)

        # prepare label
        center_label = center_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        heading_label = heading_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]
        size_label = size_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        size_class_label = size_class_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]

        # encode regression targets
        center_gt_offsets = center_encode(center_label, center_ref2)
        heading_class_label, heading_res_norm_label = angle_encode(heading_label)
        size_res_label_norm = size_encode(size_label, mean_size_array, size_class_label)

        # loss calculation

        # center_loss
        center_loss = self.get_center_loss(center_boxnet, center_gt_offsets)


        # heading loss
        heading_class_loss, heading_res_norm_loss = self.get_heading_loss(
            heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label)

        # size loss
        size_class_loss, size_res_norm_loss = self.get_size_loss(
            size_scores, size_res_norm, size_class_label, size_res_label_norm)

        # corner loss regulation
        center_preds = center_decode(center_ref2, center_boxnet)
        heading = angle_decode(heading_res_norm, heading_class_label)
        size = size_decode(size_res_norm, mean_size_array, size_class_label)

        corners_loss, corner_gts = self.get_corner_loss(
            (center_preds, heading, size),
            (center_label, heading_label, size_label)
        )

        BOX_LOSS_WEIGHT = cfg.LOSS.BOX_LOSS_WEIGHT
        CORNER_LOSS_WEIGHT = cfg.LOSS.CORNER_LOSS_WEIGHT
        HEAD_REG_WEIGHT = cfg.LOSS.HEAD_REG_WEIGHT
        SIZE_REG_WEIGHT = cfg.LOSS.SIZE_REG_WEIGHT

        # Weighted sum of all losses
        loss = cls_loss + \
            BOX_LOSS_WEIGHT * (center_loss +
                               heading_class_loss + size_class_loss +
                               HEAD_REG_WEIGHT * heading_res_norm_loss +
                               SIZE_REG_WEIGHT * size_res_norm_loss +
                               CORNER_LOSS_WEIGHT * corners_loss)

        # some metrics to monitor training status

        with torch.no_grad():

            # accuracy
            cls_prec = get_accuracy(cls_probs, cls_label.view(-1))
            heading_prec = get_accuracy(heading_probs, heading_class_label.view(-1))
            size_prec = get_accuracy(size_probs, size_class_label.view(-1))

            # iou metrics
            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            heading_preds = angle_decode(heading_res_norm, heading_pred_label)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)
            overlap = rbbox_iou_3d_pair(corner_preds.detach().cpu().numpy(), corner_gts.detach().cpu().numpy())

            iou2ds, iou3ds = overlap[:, 0], overlap[:, 1]
            iou2d_mean = iou2ds.mean()
            iou3d_mean = iou3ds.mean()
            iou3d_gt_mean = (iou3ds >= cfg.IOU_THRESH).mean()
            iou2d_mean = torch.tensor(iou2d_mean).type_as(cls_prec)
            iou3d_mean = torch.tensor(iou3d_mean).type_as(cls_prec)
            iou3d_gt_mean = torch.tensor(iou3d_gt_mean).type_as(cls_prec)

        losses = {
            'total_loss': loss,
            'cls_loss': cls_loss,
            'center_loss': center_loss,
            'head_cls_loss': heading_class_loss,
            'head_res_loss': heading_res_norm_loss,
            'size_cls_loss': size_class_loss,
            'size_res_loss': size_res_norm_loss,
            'corners_loss': corners_loss
        }

        metrics = {
            'cls_acc': cls_prec,
            'head_acc': heading_prec,
            'size_acc': size_prec,
            'IoU_2D': iou2d_mean,
            'IoU_3D': iou3d_mean,
            'IoU_' + str(cfg.IOU_THRESH): iou3d_gt_mean
        }

        return losses, metrics

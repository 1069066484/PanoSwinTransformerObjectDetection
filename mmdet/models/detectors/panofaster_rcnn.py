from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch.nn as nn
import torch


@DETECTORS.register_module()
class PanoFasterRCNN(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(PanoFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        # print(backbone)# {'type': 'PanoSwinTransformer', 'embed_dim': 16, 'in_chans': 11, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24], 'window_size': 7,
        # print(neck) # {'type': 'FPN', 'in_channels': [96, 192, 384, 768], 'out_channels': 256, 'num_outs': 5}
        # print(rpn_head) # {'type': 'PanoRPNHead', 'in_channels': 256, 'feat_channels': 256, 'anchor_generator': {'ty
        # print(roi_head) # {'type': 'StandardRoIHead', 'bbox_roi_extractor': {'type': 'SingleRoIExtractor', 'roi_layer':
        # print(train_cfg) # {'rpn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.7, 'neg_iou_thr': 0.3, 'mi
        # print(test_cfg) # {'rpn': {'nms_pre': 1000, 'max_per_img': 1000, 'nms': {'type': 'nms', 'iou_threshold': 0.7}


    def extract_feat(self, img, pano_ratio_v):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img, pano_ratio_v)
        # LZX notes
        # x, a 4-len list, feature pyramid
        if self.with_neck:
            x = self.neck(x) # LZX notes: call FPN
        return x

    def forward_dummy(self, img, pano_ratio_v=None):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img, pano_ratio_v)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      pano_ratio_v=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

            pano_ratio_v : (N, 2) pano ratio
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # LZX notes
        # img: 1x3x576x512
        x = self.extract_feat(img, pano_ratio_v)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            """
            print(self.rpn_head)
            RPNHead(
              (loss_cls): CrossEntropyLoss()
              (loss_bbox): L1Loss()
              (rpn_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (rpn_cls): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
              (rpn_reg): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
            )
            """
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,  # (mat1[2,256,168,336], mat2[2,256,84,168], mat3[2,256,42,84], mat4[2,256,21,42], mat5[2,256,11,21])
                img_metas,
                # img_metas:
                # {'filename': 'E:/ori_disks/D/fduStudy/labZXD/repos/datasets/OmnidirectionalStreetViewDataset/equirectangular/JPEGImages/000090.jpg',
                # 'ori_filename': '000090.jpg',
                # 'ori_shape': (1000, 2000, 3),
                # 'img_shape': (667, 1333, 11),
                # 'pad_shape': (672, 1344, 11),
                # 'scale_factor': array([     0.6665,       0.667,      0.6665,       0.667], dtype=float32),
                # 'flip': True, 'flip_direction': 'horizontal', 'img_norm_cfg': {'mean': array([     123.68,      116.28,      103.53], dtype=float32),
                # 'std': array([     58.395,       57.12,      57.375], dtype=float32), 'to_rgb': True}}
                gt_bboxes,  # mat1[10,4], mat2[11,4]
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)  # {'nms_pre': 2000, 'max_per_img': 1000, 'nms': {'type': 'nms', 'iou_threshold': 0.7}, 'min_bbox_size': 0}
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        # LZX notes: roi_head ~ standardROIhead
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False,
                                pano_ratio_v=None
                                ):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img, pano_ratio_v)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, pano_ratio_v=None):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img, pano_ratio_v)

        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False, pano_ratio_vs=None):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs, pano_ratio_vs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def extract_feats(self, imgs, pano_ratio_vs=None):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        assert isinstance(pano_ratio_vs, list)
        # return [self.extract_feat(img) for img in imgs]

        if pano_ratio_vs is None:
            return [self.extract_feat(img) for img in imgs]
        else:
            return [self.extract_feat(img, pano_ratio_v) for img, pano_ratio_v in zip(imgs, pano_ratio_vs)]
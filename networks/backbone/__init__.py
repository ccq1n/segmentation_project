from networks.backbone.resnet import ResNet101


def build_backbone(backbone, output_stride, BatchNorm, pretrained=True, out_all_features=False, g_in_dim=3):
    if backbone == 'resnet':
        return ResNet101(output_stride, BatchNorm, pretrained=pretrained, out_all_features=out_all_features,
                         g_in_dim=g_in_dim)
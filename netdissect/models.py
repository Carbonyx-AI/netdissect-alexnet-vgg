"""Models, layer names, feature extraction and receptive-field geometry."""
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tvm

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_small(variant, num_classes=10):
    """AlexNet variants that we train ourselves on Imagenette (paper Sec. 3.5 / 3.7)."""
    if variant in ("baseline", "repeat1", "repeat2", "nodropout"):
        return tvm.alexnet(weights=None, num_classes=num_classes, dropout=0.0 if variant == "nodropout" else 0.5)
    m = tvm.alexnet(weights=None, num_classes=num_classes)
    if variant == "batchnorm":            # BN after every conv
        layers = []
        for mod in m.features:
            layers.append(mod)
            if isinstance(mod, nn.Conv2d):
                layers.append(nn.BatchNorm2d(mod.out_channels))
        m.features = nn.Sequential(*layers)
    elif variant == "gapwide" or variant.startswith("gap"):   # paper Sec 3.7: FC layers -> GAP + linear,
        width = 768 if variant == "gapwide" else int(variant[3:])   # conv5 widened (gap256 = no widening, control)
        f = list(m.features)
        f[10] = nn.Conv2d(256, width, 3, padding=1)
        m.features = nn.Sequential(*f[:12])
        m.avgpool = nn.AdaptiveAvgPool2d(1)
        m.classifier = nn.Linear(width, num_classes)
    else:
        raise ValueError(variant)
    return m


def load_model(name, weights_dir=None, seed=0):
    """alexnet-imagenet | alexnet-places365 | alexnet-random | vgg16-imagenet | vgg16bn-imagenet
    | small-<variant>@<checkpoint>"""
    if name == "alexnet-imagenet":
        m = tvm.alexnet(weights="IMAGENET1K_V1")
    elif name == "alexnet-random":
        torch.manual_seed(seed)
        m = tvm.alexnet(weights=None)
    elif name == "alexnet-places365":     # official CSAIL PyTorch checkpoint
        m = tvm.alexnet(weights=None, num_classes=365)
        ckpt = torch.load(os.path.join(weights_dir or ".", "alexnet_places365.pth.tar"), map_location="cpu",
                          weights_only=False)
        m.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.get("state_dict", ckpt).items()})
    elif name == "vgg16-imagenet":
        m = tvm.vgg16(weights="IMAGENET1K_V1")
    elif name == "vgg16bn-imagenet":
        m = tvm.vgg16_bn(weights="IMAGENET1K_V1")
    elif name == "vgg16-random":
        torch.manual_seed(seed)
        m = tvm.vgg16(weights=None)
    elif name.startswith("caffe-"):      # the paper's original Caffe models, see netdissect/caffe.py
        from netdissect.caffe import load_caffe, mean_from_binaryproto
        proto, model_file, mean_file = CAFFE[name]
        d = weights_dir or "."
        m = load_caffe(os.path.join(d, proto), os.path.join(d, model_file), mean_from_binaryproto(os.path.join(d, mean_file)))
    elif name.startswith("small-"):
        variant, ckpt = name[len("small-"):].split("@", 1)
        m = build_small(variant)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        raise ValueError(name)
    for mod in m.modules():               # in-place ReLU would overwrite the tensors we capture
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    return m.eval()


def arch_of(name):
    return "alexnet" if ("alexnet" in name or name.startswith("small")) else "vgg16"


# name -> (prototxt, caffemodel, mean image), all in the weights directory
CAFFE = {
    "caffe-alexnet-imagenet": ("proto_bvlc_alexnet.prototxt", "bvlc_alexnet.caffemodel", "ilsvrc12/imagenet_mean.binaryproto"),
    "caffe-alexnet-places205": ("places205CNN_deploy_upgraded.prototxt", "places205CNN_iter_300000_upgraded.caffemodel",
                                "places205CNN_mean.binaryproto"),
    "caffe-alexnet-hybrid": ("hybridCNN_deploy_upgraded.prototxt", "hybridCNN_iter_700000_upgraded.caffemodel",
                             "hybridCNN_mean.binaryproto"),
    "caffe-alexnet-places365": ("proto_alexnet_places365.prototxt", "alexnet_places365.caffemodel", "places365CNN_mean.binaryproto"),
    "caffe-vgg16-places365": ("proto_vgg16_places365.prototxt", "vgg16_places365.caffemodel", "places365CNN_mean.binaryproto"),
    "caffe-vgg16-hybrid1365": ("proto_vgg16_hybrid1365.prototxt", "vgg16_hybrid1365.caffemodel", "places365CNN_mean.binaryproto"),
}


def preprocess(model, x01):
    """RGB images in [0,1] -> network input (Caffe models: BGR 0-255 minus mean; others: ImageNet mean/std)."""
    if hasattr(model, "preprocess"):
        return model.preprocess(x01)
    return (x01 - torch.tensor(MEAN, device=x01.device).view(1, 3, 1, 1)) / torch.tensor(STD, device=x01.device).view(1, 3, 1, 1)


@dataclass
class Layer:
    name: str        # conv5 (AlexNet) or conv5_3 (VGG)
    conv_idx: int    # index of the Conv2d in model.features
    relu_idx: int    # index of the ReLU after it
    channels: int


def conv_layers(model, arch):
    out, block, sub = [], 1, 0
    for i, mod in enumerate(model.features):
        if isinstance(mod, nn.Conv2d):
            sub += 1
            j = i + 1
            while not isinstance(model.features[j], nn.ReLU):
                j += 1
            name = f"conv{len(out) + 1}" if arch == "alexnet" else f"conv{block}_{sub}"
            out.append(Layer(name, i, j, mod.out_channels))
        elif isinstance(mod, nn.MaxPool2d):
            block, sub = block + 1, 0
    return out


def get_layer(model, arch, name):
    return next(l for l in conv_layers(model, arch) if l.name == name)


def rf_geometry(model, upto_idx):
    """(start, jump): unit i of features[upto_idx] is centred at input pixel start + i*jump."""
    start, jump = 0.0, 1.0
    for mod in list(model.features)[: upto_idx + 1]:
        if isinstance(mod, (nn.Conv2d, nn.MaxPool2d)):
            k, s, p, d = (v if isinstance(v, int) else v[0] for v in (mod.kernel_size, mod.stride, mod.padding, mod.dilation))
            start += ((d * (k - 1) + 1 - 1) / 2.0 - p) * jump
            jump *= s
    return start, jump


class FeatureExtractor:
    """Runs model.features up to the deepest asked layer, returns {layer: activation}.
    pre_relu=True gives the conv output before ReLU (needed for negative dissection)."""

    def __init__(self, model, arch, layers, pre_relu=False):
        self.model, self.arch, self.pre_relu = model, arch, pre_relu
        self.want = {}
        for n in layers:
            l = get_layer(model, arch, n)
            self.want[l.conv_idx if pre_relu else l.relu_idx] = n
        self.mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(STD).view(1, 3, 1, 1)

    def to(self, dev):
        self.model.to(dev)
        self.mean, self.std = self.mean.to(dev), self.std.to(dev)
        return self

    def normalize(self, x_uint8_nhwc):
        x = x_uint8_nhwc.permute(0, 3, 1, 2).float().div_(255.0)
        if hasattr(self.model, "preprocess"):
            return self.model.preprocess(x)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def __call__(self, x):
        out = {}
        for i, mod in enumerate(self.model.features):
            x = mod(x)
            if i in self.want:
                out[self.want[i]] = x.clone()
            if i >= max(self.want):
                break
        return out

    def geometry(self, name):
        l = get_layer(self.model, self.arch, name)
        return rf_geometry(self.model, l.conv_idx if self.pre_relu else l.relu_idx)

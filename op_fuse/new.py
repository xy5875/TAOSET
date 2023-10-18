import torch
import torchvision
import time
import torch.nn as nn
from torch import autocast as autocast

def fuse(conv, bn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    ).to(conv.weight.device)  # Ensure fused layer is on the same device as conv

    # setting weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1).to(device)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var))).to(device)
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))

    # setting bias
    if conv.bias is not None:
        b_conv = conv.bias.to(device)
    else:
        b_conv = torch.zeros(conv.weight.size(0)).to(device)
    b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fused.bias.copy_(b_conv + b_bn)

    return fused

def merge_conv_bn(net):
    """
    replace CNN+BN with CNN to speed up
    """
    previous = None
    has_seen_cnn = False
    conv_replace_queue = []
    bn_replace_queue = []
    for s in net.modules():
        if has_seen_cnn and isinstance(s, nn.BatchNorm2d):
            conv_replace_queue.append(previous)
            bn_replace_queue += [s]
        if isinstance(s, nn.Conv2d):
            has_seen_cnn = True
        else:
            has_seen_cnn = False
        previous = s

    if len(conv_replace_queue):
        print('here')
        if isinstance(net, nn.Sequential):
            for i, sub in enumerate(net):
                if isinstance(sub, nn.Conv2d) and sub in conv_replace_queue:
                    idx = conv_replace_queue.index(sub)
                    bn = bn_replace_queue[idx]
                    new_conv = fuse(sub, bn)
                    net[i] = new_conv
                    net[i + 1] = nn.Identity()
        else:
            for n in dir(net):
                sub = getattr(net, n)
                if isinstance(sub, nn.Conv2d) and sub in conv_replace_queue:
                    idx = conv_replace_queue.index(sub)
                    bn = bn_replace_queue[idx]
                    new_conv = fuse(sub, bn)
                    setattr(net, n, new_conv)
            for n in dir(net):
                sub = getattr(net, n)
                if isinstance(sub, nn.BatchNorm2d) and sub in bn_replace_queue:
                    setattr(net, n, nn.Identity())
        
        return net
    
def merge_conv_bn_new(net):
    """
    replace CNN+BN with CNN to speed up
    """
    previous = None
    has_seen_cnn = False
    conv_replace_queue = []
    bn_replace_queue = []
    for s in net.modules():
        if has_seen_cnn and isinstance(s, nn.BatchNorm2d):
            conv_replace_queue.append(previous)
            bn_replace_queue += [s]
        if isinstance(s, nn.Conv2d):
            has_seen_cnn = True
        else:
            has_seen_cnn = False
        previous = s

    if len(conv_replace_queue):
        print('here')
        if isinstance(net, nn.Sequential):
            for i, sub in enumerate(net):
                if isinstance(sub, nn.Conv2d) and sub in conv_replace_queue:
                    idx = conv_replace_queue.index(sub)
                    bn = bn_replace_queue[idx]
                    new_conv = fuse(sub, bn)
                    net[i] = new_conv
                    net[i + 1] = nn.Identity()
        else:
            for n in dir(net):
                subnet = getattr(net, n)
                if isinstance(subnet, nn.Sequential):
                    for i, sub in enumerate(subnet):
                        flag_to_change = 0
                        for name, module in sub.named_children():
                            if isinstance(module, nn.Conv2d) and module in conv_replace_queue:
                                if flag_to_change == 0:
                                    idx = conv_replace_queue.index(module)
                                    bn = bn_replace_queue[idx]
                                    new_conv = fuse(module, bn)
                                    setattr(sub, name, new_conv)
                                    flag_to_change = 1
                                elif flag_to_change == 1:
                                    setattr(sub, name, nn.Identity())
                                    flag_to_change = 0

        for n in dir(net):
            sub = getattr(net, n)
            if isinstance(sub, nn.Conv2d) and sub in conv_replace_queue:
                idx = conv_replace_queue.index(sub)
                bn = bn_replace_queue[idx]
                new_conv = fuse(sub, bn)
                setattr(net, n, new_conv)
        for n in dir(net):
            sub = getattr(net, n)
            if isinstance(sub, nn.BatchNorm2d) and sub in bn_replace_queue:
                setattr(net, n, nn.Identity())
        
        return net

# Testing
torch.set_grad_enabled(False)
device = 'cuda'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_iter = 100

x = torch.randn(16, 3, 256, 256).to(device)
resnet18 = torchvision.models.resnet18(pretrained=False)
resnet18 = resnet18.to(device)

#alexnet = AlexNet(num_classes=1000)

# removing all learning variables, etc
resnet18.eval()
model = torch.nn.Sequential(
    resnet18.conv1,
    resnet18.bn1,
).to(device)

f1 = resnet18.forward(x)
a = time.time()
for i in range(run_iter):
    with autocast(device_type=device, dtype=torch.float32):
        f1 = resnet18.forward(x)
xx = time.time() - a
print(f"initial model time is {time.time() - a}")

#fused = fuse(model[0], model[1]).to(device)
print('model fusing...')
fused = merge_conv_bn_new(resnet18).to(device)
print('model has fused')

f2 = fused.forward(x)
a = time.time()
for i in range(run_iter):
    with autocast(device_type=device, dtype=torch.float32):
        f2 = fused.forward(x)
yy = time.time() - a
print(f"fused model execution time is {time.time() - a}")

with autocast(device_type=device, dtype=torch.float16):
    f3 = fused.forward(x)
a = time.time()
with autocast(device_type=device, dtype=torch.float16):
    for i in range(run_iter):
        f3 = fused.forward(x)
zz = time.time() - a
print(f"quantized fused model execution time is {time.time() - a}")

d = (f1 - f2).mean().item()
print("error:", d)
print(f"Optimization rate: {100 * (xx - yy) / xx}%")

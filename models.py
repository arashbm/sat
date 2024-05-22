import torch

def init_params(m: torch.nn.Module, gain: float = 1.0):
    if m.weight.dim() < 2:
        return
    fan_in = 1.0
    for dim in m.weight.shape[1:]:
        fan_in *= dim
    bound = gain * (3.0 / fan_in)**0.5
    with torch.no_grad():
        m.weight.uniform_(-bound, bound)
        if m.bias is not None:
            m.bias.uniform_(-bound, bound)


class SimpleMNIST(torch.nn.Module):
    def __init__(self, gain=1.0):
        super(SimpleMNIST, self).__init__()

        self.input_size = 28*28
        self.output_size = 10

        h1 = 512
        h2 = 256
        h3 = 128
        self.fc1 = torch.nn.Linear(self.input_size, h1)
        bounds = gain*(6/self.input_size)**0.5
        torch.nn.init.uniform_(self.fc1.weight, -bounds, bounds)
        self.fc2 = torch.nn.Linear(h1, h2)
        bounds = gain*(6/h1)**0.5
        torch.nn.init.uniform_(self.fc2.weight, -bounds, bounds)
        self.fc3 = torch.nn.Linear(h2, h3)
        bounds = gain*(6/h2)**0.5
        torch.nn.init.uniform_(self.fc3.weight, -bounds, bounds)
        self.fc4 = torch.nn.Linear(h3, self.output_size)
        bounds = gain*(6/h3)**0.5
        torch.nn.init.uniform_(self.fc4.weight, -bounds, bounds)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(self.relu(x))
        return x

    def param_sample(self, indecies: dict[str, torch.Tensor]):
        return {
            name: tensor.ravel()[indecies[name]]
            for name, tensor in self.state_dict().items()}


class SimpleSo2Sat(torch.nn.Module):
    def __init__(self, gain=1.0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=32,
                                     kernel_size=3, padding=1)
        init_params(self.conv1, gain=gain*2**0.5)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=3, padding=1)
        init_params(self.conv2, gain=gain*2**0.5)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3, padding=1)
        init_params(self.conv3, gain=gain*2**0.5)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64*32*32, 128)
        init_params(self.fc1, gain=gain*2**0.5)
        self.fc2 = torch.nn.Linear(128, 64)
        init_params(self.fc2, gain=gain*2**0.5)
        self.fc3 = torch.nn.Linear(64, 17)
        init_params(self.fc3, gain=gain*2**0.5)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def param_sample(self, indecies: dict[str, torch.Tensor]):
        return {
            name: tensor.ravel()[indecies[name]]
            for name, tensor in self.state_dict().items()}


class ResNetBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, gain=1.0):
        super(BasicBlock, self).__init__()
        # 3x3 convolution with padding to keep dimensions
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False)
        init_params(self.conv1, gain=gain*2**0.5)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        # Second 3x3 convolution to increase depth without changing dimension
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False)
        init_params(self.conv2, gain=gain*2**0.5)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            c2d = torch.nn.Conv2d(
                    in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False)
            init_params(c2d, gain=gain*2**0.5)
            self.shortcut = torch.nn.Sequential(
                c2d, torch.nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNetCIFAR10(torch.nn.Module):
    def __init__(self, gain=1.0):
        block = ResNetBasicBlock
        num_blocks =  [2, 2, 2, 2]
        num_classes=10

        super().__init__()
        self.in_channels = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0],
                                       stride=1, gain=gain)
        self.layer2 = self._make_layer(block, 128, num_blocks[1],
                                       stride=2, gain=gain)
        self.layer3 = self._make_layer(block, 256, num_blocks[2],
                                       stride=2, gain=gain)
        self.layer4 = self._make_layer(block, 512, num_blocks[3],
                                       stride=2, gain=gain)
        self.linear = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, gain):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class VGGConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, gain=1.0):
        super(VGGConvBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            conv = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, padding=1)
            init_params(conv, gain*2**0.5)
            layers.append(conv)
            layers.append(torch.nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class VGGNet16CIFAR10(torch.nn.Module):
    def __init__(self, gain=1.0):
        super(VGGNet16CIFAR10, self).__init__()
        num_classes = 10
        self.features = torch.nn.Sequential(
            # First ConvBlock: 64 filters, 2 convolutional layers
            VGGConvBlock(3, 64, 2, gain=gain),
            # Second ConvBlock: 128 filters, 2 convolutional layers
            VGGConvBlock(64, 128, 2, gain=gain),
            # Third ConvBlock: 256 filters, 3 convolutional layers
            VGGConvBlock(128, 256, 3, gain=gain),
            # Fourth ConvBlock: 512 filters, 3 convolutional layers
            VGGConvBlock(256, 512, 3, gain=gain),
            # Fifth ConvBlock: 512 filters, 3 convolutional layers
            VGGConvBlock(512, 512, 3, gain=gain)
        )

        l1 = torch.nn.Linear(512*1*1, 4096)
        init_params(l1, gain=gain*2**0.5)
        l2 = torch.nn.Linear(4096, 4096)
        init_params(l2, gain=gain*2**0.5)
        l3 = torch.nn.Linear(4096, num_classes)
        init_params(l3, gain=gain*2**0.5)
        self.classifier = torch.nn.Sequential(
            l1, torch.nn.ReLU(inplace=True), torch.nn.Dropout(),
            l2, torch.nn.ReLU(inplace=True), torch.nn.Dropout(),
            l3
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

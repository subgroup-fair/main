import numpy as np, torch, torch.nn as nn, torch.optim as optim

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    
# ----- 모델들 -----
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class Linear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)
    def forward(self, x):
        return self.net(x).squeeze(-1)

class SmallConvNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        ch=[32,64,128,128]
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        self.features = nn.Sequential(
            block(3, ch[0]),
            block(ch[0], ch[1]),
            block(ch[1], ch[2]),
            block(ch[2], ch[3]),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch[3], n_classes)
        )
    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)
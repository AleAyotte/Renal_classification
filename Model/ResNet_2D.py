import torch
import torch.nn as nn
from torchvision import models


class ResNet2D(nn.Module):
    def __init__(self,
                 nb_clinical_data: int = 10,
                 drop_rate: float = 0.5):
        super().__init__()
        assert nb_clinical_data > 0, "You should use at least on clinical features."
        self._nb_clin_features = nb_clinical_data

        # ResNet backend
        self.t1_net = models.resnet50(pretrained=True, progress=False)
        self.t2_net = models.resnet50(pretrained=True, progress=False)

        # We remove the last layer of the ResNet model.
        self.t1_net.fc = nn.Identity()
        self.t2_net.fc = nn.Identity()

        self.fc_images = nn.Sequential(nn.Linear(4096, 1024),
                                       nn.Linear(1024, 24))
        self.fc_out = nn.Sequential(nn.Dropout(p=drop_rate),
                                    nn.Linear(24+self._nb_clin_features, 2))

    def forward(self, images_t1, images_t2, clin_features):
        # Forward pass in the backend
        t1_features = self.t1_net(images_t1)
        t2_features = self.t2_net(images_t2)

        imgs_out = self.fc_images(torch.cat((t1_features, t2_features), -1))
        out = self.fc_out(torch.cat((imgs_out, clin_features), -1))

        return out

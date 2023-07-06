import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image

from tqdm import tqdm
from PIL import Image
from typing import Optional
import os


def compute_gram_matrix(input_tensor):
    """
    Computes the Gram matrix for an input tensor.
    The Gram matrix is used to capture style information from the image.
    """
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    gram_matrix = torch.mm(features, features.t())
    return gram_matrix.div(a * b * c * d)


class StyleLoss(nn.Module):
    """
    Computes the style loss between the target and input images.
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = compute_gram_matrix(target_feature).detach()

    def forward(self, x):
        gram_matrix = compute_gram_matrix(x)
        self.loss = nn.functional.mse_loss(gram_matrix, self.target)
        return x


class ContentLoss(nn.Module):
    """
    Computes the content loss between the target and input images.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


class ImageNormalization(nn.Module):
    """
    Normalizes an image using a given mean and standard deviation.
    """
    def __init__(self, mean, std):
        super(ImageNormalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class VGGStyleTransfer(nn.Module):
    """
    Uses a VGG model to perform style transfer between two images.
    """
    def __init__(self, device: Optional[str] = None):
        super().__init__()

        if device is None:
            self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device: str = device

        self.image_size: int = 512 if self.device == 'cuda' else 128
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        self.vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features.to(self.device).eval()

    def create_style_transfer_model(self, content_image, style_image):
        """
        Builds the style transfer model.
        """
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        content_losses = []
        style_losses = []

        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        normalization = ImageNormalization(cnn_normalization_mean, cnn_normalization_std).to(self.device)

        model = nn.Sequential(normalization)

        i = 0  
        for layer in self.vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def perform_style_transfer(self, content_image,
                               style_image,
                               output_image_path: str = None,
                               num_steps=100,
                               style_weight=100000,
                               content_weight=1):
        """
        Performs the style transfer and saves the resulting image.
        """
        model, style_losses, content_losses = self.create_style_transfer_model(content_image, style_image)

        content_image.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = optim.LBFGS([content_image])

        for _ in tqdm(range(num_steps)):

            def closure():
                with torch.no_grad():
                    content_image.clamp_(0, 1)

                optimizer.zero_grad()
                model(content_image)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            content_image.clamp_(0, 1)

        if output_image_path:
            save_image(content_image[0], output_image_path)
        return content_image

    def load_and_preprocess_image(self, image_path):
        """
        Loads an image from a file and performs necessary preprocessing.
        """
        image = Image.open(image_path)
        image = self.image_transforms(image).unsqueeze(0)
        return image.to(self.device, torch.float)

import torch
from ngdsac_horizon.ngdsac import NGDSAC
from ngdsac_horizon.loss import Loss
from ngdsac_horizon.model import Model
from pathlib import Path
from torchvision import transforms
from importlib.resources import files

class HorizonEstimator:
  def __init__(
    self,
    capacity=4,
    imagesize=256,
    inlierthreshold=0.05,
    inlierbeta=100.0,
    inlieralpha=0.1,
    hypotheses=16,
    model=None,
    uniform=False,
    device="cpu",
  ):
    self.ngdsac = NGDSAC(hypotheses, inlierthreshold, inlierbeta, inlieralpha, Loss(imagesize), 1)
    self.nn = Model(capacity)

    if model is None:
      model = files("ngdsac_horizon.models").joinpath("weights_ngdsac_pretrained.net")

    self.nn.load_state_dict(torch.load(model, map_location=device))
    self.nn.eval()
    self.nn = self.nn.to(device)
    self.device = device
    self.uniform = uniform

  def process_image(self, image):
    NEW_IMSIZE = 256  # (256 is the default image size for the model)
    w, h = image.shape[2], image.shape[1]
    image_scale = NEW_IMSIZE / max(w, h)

    # convert image to RGB
    if image.shape[0] > 3:
      image = image[:3, :, :]  # remove alpha channel
    elif len(image.shape) < 3:
      image = image.repeat(3, 1, 1)  # convert gray scale to RGB

    src_h = int(h * image_scale)
    src_w = int(w * image_scale)

    # resize and to gray scale
    image = transforms.functional.resize(image, (src_h, src_w))
    image = transforms.functional.adjust_saturation(image, 0)

    padding_left = int((NEW_IMSIZE - image.size(2)) / 2)
    padding_right = NEW_IMSIZE - image.size(2) - padding_left
    padding_top = int((NEW_IMSIZE - image.size(1)) / 2)
    padding_bottom = NEW_IMSIZE - image.size(1) - padding_top

    padding = torch.nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
    image = padding(image)

    image_src = image.clone().unsqueeze(0)

    # normalize image (mean and variance), values estimated offline from HLW training set
    img_mask = image.sum(0) > 0
    image[:, img_mask] -= 0.45
    image[:, img_mask] /= 0.25
    image = image.unsqueeze(0).to(self.device)

    with torch.no_grad():
      # predict data points and neural guidance
      points, log_probs = self.nn(image)

      if self.uniform:
        # overwrite neural guidance with uniform sampling probabilities
        log_probs.fill_(1 / log_probs.size(1))
        log_probs = torch.log(log_probs)

      # fit line with NG-DSAC, providing dummy ground truth labels
      self.ngdsac(points, log_probs, torch.zeros((1, 2)), torch.zeros((1)), torch.ones((1)), torch.ones((1)))

    score = self.ngdsac.batch_inliers[0].sum() / points.shape[2]

    info = {
      "img_src": image_src,
      "points": points,
      "log_probs": log_probs,
      "padding": (padding_left, padding_right, padding_top, padding_bottom),
      "batch_inliers": self.ngdsac.batch_inliers,
    }

    return self.ngdsac.est_parameters, score, info

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .mask_decoder_draft import MaskDecoderDraft
from .prompt_encoder import PromptEncoder
from contextlib import nullcontext
import math


def thresholding(x, alpha=0.0):
    if alpha == 0:
      return x
    mask = x > alpha / x.shape[-1] 
    x_masked = x * mask.float()
    row_sums = torch.sum(x_masked, dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1
    x_normalized = x_masked / row_sums
    return x_normalized

def extract_relation(x, alpha=0.0):
    x_normalized = x / (x.norm(dim=-1, keepdim=True) + 1e-16)
    similarity = x_normalized @ x_normalized.transpose(-2, -1)

    mask = torch.eye(similarity.size(1), device=x.device)[None, :, :]
    mask = mask.expand_as(similarity)
    mask = mask.to(torch.float) * float(-1e16)

    masked_similarity = similarity + mask
    output = torch.softmax(masked_similarity, dim=-1)

    output = thresholding(output, alpha=alpha)

    return output

class SamDraft(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        draft_decoder: MaskDecoderDraft,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        args = None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.draft_decoder = draft_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.vatt_alpha = args.vatt_alpha
        self.vatt_mask_detach = args.vatt_mask_detach == 1

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        ctx = torch.no_grad if not self.training else nullcontext

        with ctx():
          input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
          
          image_embeddings, interm_embeddings = self.image_encoder(input_images)

          outputs = []
          for image_record, curr_embedding in zip(batched_input, image_embeddings):
              if "point_coords" in image_record:
                  points = (image_record["point_coords"], image_record["point_labels"])
              else:
                  points = None
              sparse_embeddings, dense_embeddings_draft = self.prompt_encoder(
                  points=points,
                  boxes=image_record.get("boxes", None),
                  masks=image_record.get("mask_inputs", None),
              )
              
              visual_sim = extract_relation(curr_embedding.unsqueeze(0).flatten(2).permute(0, 2, 1), alpha=self.vatt_alpha)
              
              low_res_masks_1, iou_predictions = self.mask_decoder(
                  image_embeddings=curr_embedding.unsqueeze(0),
                  image_pe=self.prompt_encoder.get_dense_pe(),
                  sparse_prompt_embeddings=sparse_embeddings,
                  dense_prompt_embeddings=dense_embeddings_draft,
                  multimask_output=multimask_output,
                  visual_sim=visual_sim.detach() if self.vatt_mask_detach else visual_sim
              )
              
              masks = self.postprocess_masks(
                  low_res_masks_1,
                  input_size=image_record["image"].shape[-2:],
                  original_size=image_record["original_size"],
              )
              masks = masks > self.mask_threshold

              draft_masks = (low_res_masks_1.detach()).float()
              #thresholding is not good
              #detach is good for vit-b. vit-l still needs further tuning.
              #no detach is also ok for vit-b, but achieves slightly ~0.2% IOU.
              #nodetach is ok for vit-l, achieving comparable performance with dora.
              #sigmoid with detach is good for vit-l.
              #but sigmoid with detach is bad for vit-b.

              _, dense_embeddings = self.prompt_encoder(
                  points=None,
                  boxes=None,
                  masks=draft_masks, 
              )
              assert dense_embeddings.shape[-1] == dense_embeddings_draft.shape[-1]

              low_res_masks_2, _ = self.draft_decoder(
                  image_embeddings=curr_embedding.unsqueeze(0),
                  image_pe=self.prompt_encoder.get_dense_pe(),
                  sparse_prompt_embeddings=sparse_embeddings,
                  dense_prompt_embeddings=dense_embeddings,
                  multimask_output=multimask_output,
                  visual_sim=visual_sim
              )

              outputs.append(
                  {
                      "masks": masks,
                      "iou_predictions": iou_predictions,
                      "low_res_logits_draft": low_res_masks_1,
                      "low_res_logits": low_res_masks_2,
                      "encoder_embedding": curr_embedding.unsqueeze(0),
                      "image_pe": self.prompt_encoder.get_dense_pe(),
                      "sparse_embeddings":sparse_embeddings,
                      "dense_embeddings":dense_embeddings,
                  }
              )

          return outputs, interm_embeddings

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

import torch
from TrailBlazer.BBox import BoundingBox
import math 

def keyframed_bbox(bundle, frames):

    keyframe = bundle["keyframe"]
    bbox_per_frame = []
    f = lambda start, end, index: (1 - index) * start + index * end
    n = len(keyframe)
    for i in range(n - 1):
        if i == 0:
            # start_fr = keyframe[i]["frame"]
            start_fr = 0
        else:
            # start_fr = keyframe[i]["frame"] + 1
            start_fr = 0 + 1

        # end_fr = keyframe[i + 1]["frame"]
        end_fr = frames
        start_bbox = keyframe[i]["bbox_ratios"]
        end_bbox = keyframe[i + 1]["bbox_ratios"]
        clip_length = end_fr - start_fr + 1
        for fr in range(clip_length):
            index = float(fr) / (clip_length - 1)
            bbox = []
            for j in range(4):
                bbox.append(f(start_bbox[j], end_bbox[j], index))
            bbox_per_frame.append(bbox)

    return bbox_per_frame

def dd_core(attention_probs: torch.Tensor, dim_x, dim_y, num_frames, bundle, bbox_per_frame, strengthen_scale, weaken_scale, ):
        attention_probs_copied = attention_probs.detach().clone()

        # NOTE: Spatial cross attention editing
        if len(attention_probs.size()) == 4:    
            all_tokens_inds = list(range(1,21))
            """
            left top right down
            """
            # bbox_left=list()
            # for _ in range(num_frames):
            #     bbox_left.append([0.0, 0.3, 0.5, 0.7])

            # bbox_right=list()
            # for _ in range(num_frames):
            #     bbox_right.append([0.5, 0.3, 1.0, 0.7])

            bbox_top=list()
            for _ in range(num_frames):
                bbox_top.append([0.0, 0.0, 1.0, 0.4])

            bbox_down=list()
            for _ in range(num_frames):
                bbox_down.append([0.0, 0.4, 1.0, 1.0])

            # bbox_tl=list()
            # for _ in range(num_frames):
            #     bbox_tl.append([0.0, 0.0, 0.5, 0.4])

            # bbox_tr=list()
            # for _ in range(num_frames):
            #     bbox_tr.append([0.5, 0.0, 1.0, 0.4])

            # bbox_bl=list()
            # for _ in range(num_frames):
            #     bbox_bl.append([0.0, 0.4, 0.5, 1.0])

            # bbox_br=list()
            # for _ in range(num_frames):
            #     bbox_br.append([0.5, 0.4, 1.0, 1.0])

            #--- object 1
            strengthen_map1 = localized_weight_map(
                attention_probs_copied,
                token_inds=all_tokens_inds,
                bbox_per_frame=bbox_top,
                dim_x = dim_x,
                dim_y = dim_y
            )
            #--- object 2
            strengthen_map2 = localized_weight_map(
                attention_probs_copied,
                token_inds=all_tokens_inds,
                bbox_per_frame=bbox_down,
                dim_x = dim_x,
                dim_y = dim_y
            )            
            # #--- object 3
            # strengthen_map3 = localized_weight_map(
            #     attention_probs_copied,
            #     token_inds=all_tokens_inds,
            #     bbox_per_frame=bbox_bl,
            #     # bbox_per_frame=bbox_right,
            #     dim_x = dim_x,
            #     dim_y = dim_y
            # )
            # #--- object 4
            # strengthen_map4 = localized_weight_map(
            #     attention_probs_copied,
            #     token_inds=all_tokens_inds,
            #     bbox_per_frame=bbox_br,
            #     # bbox_per_frame=bbox_right,
            #     dim_x = dim_x,
            #     dim_y = dim_y
            # )
            
            strengthen_map = torch.zeros_like(strengthen_map1)
            
            ## Up & Down
            strengthen_map[..., 1:11] = strengthen_map1[..., 1:11] - strengthen_map2[..., 1:11]
            strengthen_map[..., 11:21] = strengthen_map2[..., 11:21] - strengthen_map1[..., 11:21]

            ## Up & Down_Left & Down_Right
            # strengthen_map[..., 1:11] = strengthen_map1[..., 1:11] - strengthen_map2[..., 1:11] - strengthen_map3[..., 1:11] - strengthen_map4[..., 1:11]
            # strengthen_map[..., 11:21] = strengthen_map2[..., 11:21] - strengthen_map3[..., 11:21] - strengthen_map4[..., 11:21] - strengthen_map1[..., 1:11]
            # strengthen_map[..., 21:31] = strengthen_map3[..., 21:31] - strengthen_map4[..., 21:31] - strengthen_map1[..., 21:31] - strengthen_map2[..., 21:31]
            # strengthen_map[..., 31:41] = strengthen_map4[..., 31:41] - strengthen_map1[..., 31:41] - strengthen_map2[..., 31:41] - strengthen_map3[..., 31:41]

            zero_indices = torch.where(strengthen_map==0)
            weaken_map = torch.ones_like(strengthen_map)
            weaken_map[zero_indices] = weaken_scale

            ## weakening
            attention_probs_copied[..., all_tokens_inds] *= weaken_map[..., all_tokens_inds]
            ## strengthen
            attention_probs_copied[..., all_tokens_inds] += (strengthen_scale * strengthen_map[..., all_tokens_inds])
            
        # NOTE: Temporal cross attention editing
        elif len(attention_probs.size()) == 5:
            strengthen_map = localized_temporal_weight_map(
                attention_probs_copied,
                bbox_per_frame=bbox_per_frame,
                dim_x = dim_x,
                dim_y = dim_y
            )
            weaken_map = torch.ones_like(strengthen_map)
            zero_indices = torch.where(strengthen_map == 0)
            weaken_map[zero_indices] = weaken_scale
            # weakening
            attention_probs_copied *= weaken_map
            # strengthen
            attention_probs_copied += strengthen_scale * strengthen_map

        return attention_probs_copied

KERNEL_DIVISION = 3.
INJECTION_SCALE = 1.0

def localized_weight_map(attention_probs_4d, token_inds, bbox_per_frame, dim_x, dim_y, scale=1):
        """Using guassian 2d distribution to generate weight map and return the
        array with the same size of the attention argument.
        """
        dim = int(attention_probs_4d.size()[1])
        max_val = attention_probs_4d.max()
        weight_map = torch.zeros_like(attention_probs_4d).half()
        frame_size = attention_probs_4d.shape[0] // len(bbox_per_frame)
        for i in range(len(bbox_per_frame)):
            bbox_ratios = bbox_per_frame[i]
            bbox = BoundingBox(dim_x, dim_y, bbox_ratios)
            # Generating the gaussian distribution map patch
            x = torch.linspace(0, bbox.height, bbox.height)
            y = torch.linspace(0, bbox.width, bbox.width)
            x, y = torch.meshgrid(x, y, indexing="ij")
            noise_patch = (
                gaussian_2d(
                    x,
                    y,
                    mx=int(bbox.height / 2),
                    my=int(bbox.width / 2),
                    sx=float(bbox.height / KERNEL_DIVISION),
                    sy=float(bbox.width / KERNEL_DIVISION),
                )
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(frame_size, 1, 1, len(token_inds))
                .to(attention_probs_4d.device)
            ).half()

            scale = attention_probs_4d.max() * INJECTION_SCALE
            noise_patch.mul_(scale)

            b_idx = frame_size * i
            e_idx = frame_size * (i + 1)
            bbox.sliced_tensor_in_bbox(weight_map)[
                b_idx:e_idx, ..., token_inds
            ] = noise_patch
        return weight_map

def localized_temporal_weight_map(attention_probs_5d, bbox_per_frame, dim_x, dim_y, scale=1):
        """Using guassian 2d distribution to generate weight map and return the
        array with the same size of the attention argument.
        """
        dim = int(attention_probs_5d.size()[1])
        f = attention_probs_5d.shape[-1]
        max_val = attention_probs_5d.max()
        weight_map = torch.zeros_like(attention_probs_5d).half()

        def get_patch(bbox_at_frame, i, j, bbox_per_frame):
            bbox = BoundingBox(dim_x, dim_y, bbox_at_frame)
            # Generating the gaussian distribution map patch
            x = torch.linspace(0, bbox.height, bbox.height)
            y = torch.linspace(0, bbox.width, bbox.width)
            x, y = torch.meshgrid(x, y, indexing="ij")
            noise_patch = (
                gaussian_2d(
                    x,
                    y,
                    mx=int(bbox.height / 2),
                    my=int(bbox.width / 2),
                    sx=float(bbox.height / KERNEL_DIVISION),
                    sy=float(bbox.width / KERNEL_DIVISION),
                )
                .unsqueeze(0)
                .repeat(attention_probs_5d.shape[0], 1, 1)
                .to(attention_probs_5d.device)
            ).half()
            scale = attention_probs_5d.max() * INJECTION_SCALE
            noise_patch.mul_(scale)
            inv_noise_patch = noise_patch - noise_patch.max()
            dist = (float(abs(j - i))) / len(bbox_per_frame)
            final_patch = inv_noise_patch * dist + noise_patch * (1. - dist)
            #final_patch = noise_patch * (1. - dist)
            #final_patch = inv_noise_patch * dist
            return final_patch, bbox


        for j in range(len(bbox_per_frame)):
            for i in range(len(bbox_per_frame)):
                patch_i, bbox_i = get_patch(bbox_per_frame[i], i, j, bbox_per_frame)
                patch_j, bbox_j = get_patch(bbox_per_frame[j], i, j, bbox_per_frame)
                bbox_i.sliced_tensor_in_bbox(weight_map)[..., i, j] = patch_i
                bbox_j.sliced_tensor_in_bbox(weight_map)[..., i, j] = patch_j

        return weight_map

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """ 2d Gaussian weight function
    """
    gaussian_map = (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )
    gaussian_map.div_(gaussian_map.max())
    return gaussian_map
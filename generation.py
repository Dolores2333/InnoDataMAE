# -*- coding: utf-8 _*_
# @Time : 5/1/2022 10:34 am
# @Author: ZHA Mengyue
# @FileName: generation.py
# @Blog: https://github.com/Dolores2333


from components import *

"""All generation codes are adaptet for embed_loss here"""


def generate_pseudo_masks(args, num_samples):
    # xxxx
    # xxxx
    # xxxx
    masks = np.zeros((num_samples, args.ts_size), dtype=bool)
    return masks


def generate_random_masks(args, num_samples):
    # xxxo
    # oxxx
    # xxox
    num_patches = int(args.ts_size // args.mask_size)

    def single_sample_mask():
        idx = np.random.permutation(num_patches)[:args.num_masks]
        mask = np.zeros(args.ts_size, dtype=bool)
        for j in idx:
            mask[j * args.mask_size:(j + 1) * args.mask_size] = 1
        return mask

    masks_list = [single_sample_mask() for _ in range(num_samples)]
    masks = np.stack(masks_list, axis=0)  # (num_samples, ts_size)
    return masks


def generate_cross_masks(args, num_samples, idx):
    # oxxx
    # oxxx
    # oxxx
    masks = np.zeros((num_samples, args.ts_size), dtype=bool)
    masks[:, idx * args.mask_size:(idx + 1) * args.mask_size] = 1  # masks(num_samples, ts_size)
    return masks


def full_generation(args, model, ori_data):
    masks = generate_pseudo_masks(args, len(ori_data))  # (len(ori_data), seq_len)
    x_enc, art_data, masks = model(ori_data, masks, 'random_generation')
    np.save(args.masks_dir, masks)
    return art_data


def random_generation(args, model, ori_data):
    masks = generate_random_masks(args, len(ori_data))  # (len(ori_data), seq_len)
    x_enc, art_data, masks = model(ori_data, masks, 'random_generation')
    np.save(args.masks_dir, masks)
    return art_data


def cross_generation(args, model, ori_data):
    def generate_one_cross(idx):
        cross_masks = generate_cross_masks(args, len(ori_data), idx)
        x_enc, cross_data, cross_masks = model(ori_data, cross_masks, 'cross_generation')  # (len(ori_data), ts_size, z_dim)
        cross_data = cross_data[:, idx * args.mask_size:(idx + 1) * args.mask_size, :]
        # cross_data (len(ori_data), args.mask_size, z_dim)
        return cross_data
    cross_mosaics = [generate_one_cross(i).cpu().detach().numpy()
                     for i in range(0, (args.ts_size-args.mask_size), args.mask_size)]
    art_data = np.concatenate(cross_mosaics, axis=1)
    return art_data

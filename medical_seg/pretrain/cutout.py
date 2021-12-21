
import torch 
import numpy as np 

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, state_dict: np.random.RandomState, n_holes, length):
        self.n_holes = n_holes
        self.length = length
        self.state_dict = state_dict

    def __call__(self, img):
        
        z = img.shape[1]
        h = img.shape[2]
        w = img.shape[3]

        mask = np.ones((z, h, w), np.float32)

        for n in range(self.n_holes):
            z = self.state_dict.randint(z)
            y = self.state_dict.randint(h)
            x = self.state_dict.randint(w)

            z1 = np.clip(z - self.length // 2, 0, z)
            z2 = np.clip(z + self.length // 2, 0, z)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[z1: z2, y1: y2, x1: x2] = 0.

        # mask = torch.from_numpy(mask)
        # mask = mask.expand_as(img)
        mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img

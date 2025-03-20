import os
import glob
import warnings
import torch
import rasterio as rio
from torch.utils.data import Dataset
from torchvision.transforms import ElasticTransform, RandomHorizontalFlip, Compose
from torchvision.transforms.functional import InterpolationMode

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


class RadargramDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        seq_len,
        patch_width,
        stride,
        seq_stride=None,
        data_aug=False,
        first_only=False,
    ):
        """
        The elements of this dataset are sequences of patches.
        An item of the dataset is a 2-tuple with items with dimension (THW,THW) where:
        - T is the length of the sequence
        - H is the height of the patch
        - W is the width of the patch
        Sequences are extracted from radargram-segmentation_map pairs saved in .tiff format
        within the dataset_path_folder.
        Users should assert that there are no other tiff in the folder and the width
        of the pairs matches.
        The stride parameter regulates the overlapping between patches while the seq_stride
        parameter regulates the overlapping between sequences (in term of number of patches).
        """
        super().__init__()
        torch.manual_seed(0)
        self.seq_len = seq_len
        self.patch_width = patch_width
        self.seq_stride = seq_stride if seq_stride is not None else seq_len
        self.data_aug = data_aug
        self.first_only = first_only
        self.transform = Compose(
            [
                ElasticTransform(
                    alpha=float(patch_width) * 20,
                    sigma=float(patch_width) / 2,
                    interpolation=InterpolationMode.NEAREST,
                ),
                RandomHorizontalFlip(p=0.5),
            ]
        )

        # List files
        all_tiff = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))
        rg_path = []
        sg_path = []
        for file in all_tiff:
            if "map" in os.path.basename(file):
                sg_path.append(file)
            else:
                rg_path.append(file)

        rg_path = sorted(rg_path)
        sg_path = sorted(sg_path)

        # Get items
        rgs = []
        sgs = []
        for rgf, sgf in zip(rg_path, sg_path):
            with rio.open(rgf) as rg, rio.open(sgf) as sg:
                rg, sg = torch.tensor(rg.read(1)), torch.tensor(sg.read(1))
                assert rg.shape[1] == sg.shape[1], "Image and map shape does not match!"
                rgs.append(
                    torch.permute(
                        rg.unfold(1, patch_width, stride).unfold(
                            1, seq_len, self.seq_stride
                        ),
                        [1, 3, 0, 2],
                    )
                )
                sgs.append(
                    torch.permute(
                        sg.unfold(1, patch_width, stride).unfold(
                            1, seq_len, self.seq_stride
                        ),
                        [1, 3, 0, 2],
                    )
                )
        # Both rgs and sgs items have dimensions NTHW where N=num_of_sequences,T=seq_len,H=image_height,W=image_width
        self.rgs = torch.cat(rgs, dim=0).float()
        self.sgs = torch.cat(sgs, dim=0).float()

    def __len__(self):
        return self.rgs.shape[0]

    def __getitem__(self, index):
        rg = self.rgs[index]  # THW
        sg = self.sgs[index]
        if self.data_aug:
            rg = torch.cat(list(rg), dim=1)  # THW -> H x TW
            sg = torch.cat(list(sg), dim=1)
            composite = self.transform(torch.stack([rg, sg], dim=0))
            rg = composite[0]
            sg = composite[1]
            rg = rg.unfold(1, self.patch_width, self.patch_width).permute(1, 0, 2)
            sg = sg.unfold(1, self.patch_width, self.patch_width).permute(1, 0, 2)

        if self.first_only:
            rg = rg[0].unsqueeze(0)
            sg = sg[0].unsqueeze(0)
        return rg, sg

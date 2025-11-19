"""
Dataset for the Reconstruction model. Based on pytorch
"""
from typing import List, Optional
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt



class ClassificationDataset(Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        means: Optional[List[np.float32]] = None,
        stds: Optional[List[np.float32]] = None
    ):
        """Convert the files from the simulation to simple lists.

        Args:
            input_df (pd.DataFrame): Must contain as first level columns:
                ["Parameters", "Inputs", "Targets", "Context"], the names of the further dimensions
                are ignored.

        Returns:
            torch.DataSet
        """
        self.df = input_df
        #self.df = self.filter_infs_and_nans(self.df)
        print ('params')
        self.parameters = self.df["Parameters"].to_numpy("float32")
        self.targets = self.df["Classes"].to_numpy("float32")
        if "Context" in self.df.columns:
            self.context = self.df["Context"].to_numpy("float32")
        else:
            self.context = torch.empty(self.targets.shape[0],0)
        print ('done')

        #self.inputs = self.df["Inputs"].to_numpy("float32").reshape(-1,3,8)
        #self.inputs = [self.inputs[:,0],self.inputs[:,1],self.inputs[:,2]]

        # Crop params #
        crop_size = 32

        # Get layers #
        print ('inputs')
        N_sensors = len([col for col in self.df['Inputs'] if 'sensor_layer' in col])
        layer_indices = np.unique(self.df['Inputs'][[col for col in self.df['Inputs'] if 'sensor_layer' in col]])

        ls  = self.df['Inputs'][[f'sensor_layer_{i}' for i in range(N_sensors)]].to_numpy("float32")
        #xs  = self.df['Inputs'][[f'sensor_x_{i}' for i in range(N_sensors)]].to_numpy("float32")
        #ys  = self.df['Inputs'][[f'sensor_y_{i}' for i in range(N_sensors)]].to_numpy("float32")
        #zs  = self.df['Inputs'][[f'sensor_z_{i}' for i in range(N_sensors)]].to_numpy("float32")
        #dxs = self.df['Inputs'][[f'sensor_dx_{i}' for i in range(N_sensors)]].to_numpy("float32")
        #dys = self.df['Inputs'][[f'sensor_dy_{i}' for i in range(N_sensors)]].to_numpy("float32")
        #dzs = self.df['Inputs'][[f'sensor_dz_{i}' for i in range(N_sensors)]].to_numpy("float32")
        Es  = self.df['Inputs'][[f'sensor_energy_{i}' for i in range(N_sensors)]].to_numpy("float32")
        #Es = np.where(Es>0,np.log(Es),0)
        print ('done')

        self.inputs = []
        for idx in layer_indices:
            mask = ls==idx
            assert np.all(mask == mask[0])
            mask = mask[0]
            #x  = xs[:,mask]
            #y  = ys[:,mask]
            #z  = zs[:,mask]
            #dx = dxs[:,mask]
            #dy = dys[:,mask]
            #dz = dzs[:,mask]
            E  = Es[:,mask]

            if E.shape[-1] > 1:
                # granular image #
                width = int(math.sqrt(E.shape[-1]))
                assert width*width == E.shape[-1]
                #x = x.reshape(-1,1,width,width)
                #y = y.reshape(-1,1,width,width)
                #z = z.reshape(-1,1,width,width)
                #dz = dz.reshape(-1,1,width,width)
                E = E.reshape(-1,1,width,width)

                # crop and center
                if crop_size is not None:
                    cropped_Es = np.zeros((E.shape[0],1,crop_size,crop_size),dtype=np.float32)
                    losses = np.zeros(E.shape[0])
                    print (f'Cropping images of layer {idx}')
                    for i in range(E.shape[0]):
                        img = E[i,0]
                        center = self.shower_centroid(img)
                        if center is not None:
                            cropped_img = self.crop_around_center(img, center, size=crop_size)
                        else:
                            cropped_img = np.zeros((crop_size,crop_size))
                        if cropped_img.sum() > 0:
                            losses[i] = (img.sum()-cropped_img.sum())/img.sum()
                        cropped_Es[i,0,:,:] = cropped_img
                    print (f'Layer {idx}: cropped {E.shape[2]}x{E.shape[3]} -> {crop_size}x{crop_size}: losses = {losses.mean()*100:5.3f}% +/- {losses.std()*100:5.3f}%')
                    E = cropped_Es

            # Preprocess energy #
            eps = 1e-6
            E = E / (E.sum(axis=(2,3),keepdims=True)+eps)
            E = np.log(E+eps)

            self.inputs.append(
                np.concatenate(
                    [
                        E,
                    ],
                    axis = 1,
                ),
            )


        if means is None:
            self.means = [
                self.parameters.mean(axis=0),
                [
                    (inputs.mean(axis=(0,2,3)) if inputs.ndim==4 else inputs.mean(axis=0)).reshape(1,-1)
                    for inputs in self.inputs
                ],
                self.context.mean(axis=0),
            ]
        else:
            self.means = means
        if stds is None:
            self.stds = [
                self.parameters.std(axis=0) + 1e-10,
                [
                    (inputs.std(axis=(0,2,3)) if inputs.ndim==4 else inputs.std(axis=0)).reshape(1,-1) + 1e-10
                    for inputs in self.inputs
                ],
                self.context.std(axis=0) + 1e-10,
            ]
        else:
            self.stds = stds


        #self.parameters = self.normalize(self.parameters,self.means[0],self.stds[0])
        self.inputs = [
            self.normalize(inputs, self.means[1][i], self.stds[1][i])
            for i,inputs in enumerate(self.inputs)
        ]
        self.context = self.normalize(self.context,self.means[2],self.stds[2])

        self.shape = (
            self.parameters.shape[1],
            [inputs.shape[1:] for inputs in self.inputs],
            self.targets.shape[1],
            self.context.shape[1]
        )

#        dev = "cuda" if torch.cuda.is_available() else "cpu"
#        self.c_means = [torch.tensor(a).to(dev) for a in self.means]
#        self.c_stds = [torch.tensor(a).to(dev) for a in self.stds]

    def normalize(self,tensor,means,stds):
        if tensor.ndim == 2:
            out = (tensor - means.reshape(1,-1)) / stds.reshape(1,-1)
        elif tensor.ndim == 4:
            out = (tensor - means.reshape(1,tensor.shape[1],1,1)) / stds.reshape(1,tensor.shape[1],1,1)
        else:
            raise NotImplementedError
        out[:,(stds<=1e-10).ravel()] = 0.
        return out

    @staticmethod
    def bounding_box(img):
        rows = np.any(img != 0, axis=1)
        cols = np.any(img != 0, axis=0)

        if not rows.any() or not cols.any():
            return 0 # image is entirely zero

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return max(rmax-rmin,cmax-cmin)

    @staticmethod
    def shower_centroid(img):
        """
        Computes the energy-weighted centroid (x_cog, y_cog) of a 2D calorimeter image.
        Assumes img is a 2D numpy array.
        """
        H, W = img.shape
        ys, xs = np.indices((H, W))

        total = img.sum()
        if total == 0:
            return None  # no hits

        x_cog = (img * xs).sum() / total
        y_cog = (img * ys).sum() / total

        return y_cog, x_cog  # (row, col)

    @staticmethod
    def crop_around_center(img, center, size):
        """
        Crops a square of side `size` centered on (y, x).
        Automatically pads if the crop goes out of bounds.
        img: 2D numpy array
        center: (y, x)
        size: crop size (e.g. 32 or 64)
        """
        H, W = img.shape
        y, x = center
        y = int(round(y))
        x = int(round(x))

        half = size // 2

        # Compute desired crop boundaries
        ymin = y - half
        ymax = y + half
        xmin = x - half
        xmax = x + half

        # Create an empty (zero-padded) crop
        crop = np.zeros((size, size), dtype=img.dtype)

        # Find overlapping region between crop and image
        y0_img = max(ymin, 0)
        y1_img = min(ymax, H)
        x0_img = max(xmin, 0)
        x1_img = min(xmax, W)

        # Corresponding coordinates in the crop
        y0_crop = y0_img - ymin
        y1_crop = y0_crop + (y1_img - y0_img)
        x0_crop = x0_img - xmin
        x1_crop = x0_crop + (x1_img - x0_img)

        # Copy actual pixels into padded crop
        crop[y0_crop:y1_crop, x0_crop:x1_crop] = img[y0_img:y1_img, x0_img:x1_img]

        return crop

    def filter_infs_and_nans(self, df: pd.DataFrame):
        '''
        Removes all events that contain infs or nans.
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0, ignore_index=True)
        return df

    def filter_empty_events(self, df: pd.DataFrame):
        df = df[df["Inputs"]["sensor_energy_0"] > 0.0]
        df = df.dropna(axis=0, ignore_index=True)
        return df

#    def unnormalize_detector(self, detector: torch.Tensor):
#        return detector * self.c_stds[1] + self.c_means[1]
#
#    def normalize_detector(self, detector: torch.Tensor):
#        return (detector - self.c_means[1]) / self.c_stds[1]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.parameters[idx], [inputs[idx] for inputs in self.inputs], self.context[idx], self.targets[idx]

    def plot(self, idx, size=None):
        if isinstance(idx,int):
            idx = [idx]
        N_sensors = len([col for col in self.df['Inputs'] if 'sensor_layer' in col])
        ls  = self.df['Inputs'].iloc[idx][[f'sensor_layer_{i}' for i in range(N_sensors)]].to_numpy("float32")[0]
        xs  = self.df['Inputs'].iloc[idx][[f'sensor_x_{i}' for i in range(N_sensors)]].to_numpy("float32")[0]
        ys  = self.df['Inputs'].iloc[idx][[f'sensor_y_{i}' for i in range(N_sensors)]].to_numpy("float32")[0]
        dxs = self.df['Inputs'].iloc[idx][[f'sensor_dx_{i}' for i in range(N_sensors)]].to_numpy("float32")[0]
        dys = self.df['Inputs'].iloc[idx][[f'sensor_dy_{i}' for i in range(N_sensors)]].to_numpy("float32")[0]
        Es  = self.df['Inputs'].iloc[idx][[f'sensor_energy_{i}' for i in range(N_sensors)]].to_numpy("float32").sum(axis=0)
        if Es.sum() == 0:
            print ('No measured energy')
            return
        E_min = Es[Es>0].min()
        E_max = Es[Es>0].max()

        layers = np.unique(ls)
        N = len(layers)
        fig,axs = plt.subplots(ncols=N,figsize=(6*N,5))
        plt.suptitle(f'Target = {self.targets[idx]}')
        if not isinstance(axs,np.ndarray):
            axs = np.array([axs])
        for i,j in enumerate(layers):
            x = xs[ls==j]
            y = ys[ls==j]
            dx = dxs[ls==j]
            dy = dys[ls==j]
            E = Es[ls==j]
            width = int(math.sqrt(len(x)))
            assert width*width == len(x)
            x = x.reshape(width,width)
            y = y.reshape(width,width)
            E = E.reshape(width,width)
            if E.ndim == 1:
                E = E[:,np.newaxis,np.newaxis]
            if size is not None:
                center = self.shower_centroid(E)
                if center is not None:
                    E = self.crop_around_center(E, center, size=size)
                    x = self.crop_around_center(x, center, size=size)
                    y = self.crop_around_center(y, center, size=size)
            im = axs[i].imshow(
                E,
                extent = [x.min()-dx.min(),x.max()+dx.max(),y.min()-dy.min(),y.max()+dy.max()],
                norm = matplotlib.colors.LogNorm(vmin=E_min,vmax=E_max),
                origin='lower',
            )
            fig.colorbar(im, ax=axs[i])
            axs[i].set_title(f'Layer {i}')
        plt.show()


if __name__ == '__main__':
    import sys
    dataset = ClassificationDataset(pd.read_parquet(sys.argv[1]))

    fig,axs = plt.subplots(ncols=len(dataset.inputs),figsize=(5*len(dataset.inputs),4))
    for i in range(len(dataset.inputs)):
        imgs = dataset.inputs[i]
        e_imgs = imgs[dataset.targets==1]
        g_imgs = imgs[dataset.targets==0]
        e_frac = (e_imgs>0).sum(axis=(1,2)) / (e_imgs.shape[1]*e_imgs.shape[2])
        g_frac = (g_imgs>0).sum(axis=(1,2)) / (g_imgs.shape[1]*g_imgs.shape[2])
        bins = np.linspace(0,max([e_frac.max(),g_frac.max()]),100)
        axs[i].hist(
            e_frac,
            bins = bins,
            color = 'blue',
            histtype = 'step',
            label = 'e'
        )
        axs[i].hist(
            g_frac,
            bins = bins,
            color = 'red',
            histtype = 'step',
            label = r'$\gamma$'
        )
        axs[i].set_yscale('log')
        axs[i].set_xlabel('Fraction of non-empty cells')
        axs[i].set_title(f'Layer {i}')
        axs[i].legend()
    plt.show()


    from IPython import embed; embed()

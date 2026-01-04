import sys
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

def concat_dataset(datasets):
    return CaloGraphDataset.from_data_list(
        [
            data
            for dataset in datasets
            for data in dataset
        ]
    )

class CaloGraphDataset(InMemoryDataset):
    def __init__(self,data,slices):
        super().__init__()
        self._data = data
        self.slices = slices

    @classmethod
    def from_data_list_and_parameters(cls, data_list, parameters):
        parameters = parameters.unsqueeze(0).to(torch.float32)
        for data in data_list:
            data['parameters'] = parameters.repeat_interleave(data.pos.shape[0],dim=0)
        return cls.from_data_list(data_list)

    @classmethod
    def from_data_list(cls, data_list):
        data,slices = cls.collate(data_list)
        return cls(data,slices)

    def save(self, path):
        """Save only tensors needed to reconstruct the dataset."""
        torch.save(
            {
                "data": self._data,
                "slices": self.slices,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        """Load dataset saved via `save`."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        return cls(ckpt["data"],ckpt["slices"])

    def get_shapes(self):
        return {
            key : value.shape[-1]
            for key,value in self[0].items()
        }

    def get_input_mean(self,values):
        if values.dim() == 1:
            values = values.reshape(-1,1)
        elif values.dim() == 2:
            pass
        else:
            raise NotImplementedError
        means = torch.zeros((values.shape[1],),dtype=values.dtype)
        if not torch.is_floating_point(values):
            return means
        for i in range(values.shape[1]):
            if not ((values[:,i] == 0) | (values[:,i] == 1)).all():
                means[i] = values[:,i].mean()
        return means

    def get_input_std(self,values):
        if values.dim() == 1:
            values = values.reshape(-1,1)
        elif values.dim() == 2:
            pass
        else:
            raise NotImplementedError
        stds = torch.ones((values.shape[1],),dtype=values.dtype)
        if not torch.is_floating_point(values):
            return stds
        for i in range(values.shape[1]):
            if not ((values[:,i] == 0) | (values[:,i] == 1)).all():
                stds[i] = values[:,i].std() + 1e-10
        return stds

    def get_means(self):
        return {
            key : self.get_input_mean(values)
            for key,values in self._data.items()
        }

    def get_stds(self):
        return {
            key : self.get_input_std(values)
            for key,values in self._data.items()
        }

    def plot(self,idx):
        data = self[idx]

        types = torch.unique(data.particle_type)
        z_pos = torch.unique(data.pos[:,2])
        cmap_qual = plt.cm.get_cmap('Set1', len(types))
        label_colors = {int(typ): cmap_qual(i)[:3] for i,typ in enumerate(types)}

        label_dict = {
            11      : r'$e^{-}$',
            -11     : r'$e^{+}$',
            22      : r'$\gamma$',
            111     : r'$\pi^{0}$',
            211     : r'$\pi^{+}$',
            -211    : r'$\pi^{-}$',
            2212    : r'$p$',
            2112    : r'$n$',
        }
        for typ in label_colors.keys():
            assert typ in label_dict.keys(), f'Missing {typ} in label_dict'

        N = len(z_pos)
        fig,axs = plt.subplots(ncols=N+1,nrows=2,figsize=(N*6,9))
        plt.subplots_adjust(wspace=0.4,hspace=0.4)
        if axs.ndim == 1:
            axs = axs.reshape(-1,1)

        def make_bins(center_left,center_right,width):
            n_bins = int(round((center_right - center_left)/width)) + 1
            centers = center_left + np.arange(n_bins) * width
            edges = np.concatenate([[centers[0] - width/2], centers + width/2])
            return edges

        def make_label_cmap(color, n=256):
            cdict = {'red':   [(0, 1.0, 1.0), (1, color[0], color[0])],
                     'green': [(0, 1.0, 1.0), (1, color[1], color[1])],
                     'blue':  [(0, 1.0, 1.0), (1, color[2], color[2])]}
            return LinearSegmentedColormap('label_cmap', cdict, N=n)

        for i in range(N):
            # Select data at appropriate z layer
            mask = data.pos[:,2] == z_pos[i]
            pos = data.pos[mask]
            cell = data.cell[mask][0] # assume same granularity per layer
            labels = data.labels[mask]
            E_dep = data.node_attr[mask,0]

            # Get axes #
            assert cell[0] == cell[1]
            bins = make_bins(
                float(pos[:,:2].min()),
                float(pos[:,:2].max()),
                float(cell[0]),
            )

            # Fill with label #
            img = np.ones((len(bins)-1,len(bins)-1)) * -1
            for (x,y,z),l in zip(pos,labels):
                ix = np.digitize(x,bins) - 1
                iy = np.digitize(y,bins) - 1
                img[ix,iy] = l
            masked_img = np.ma.masked_equal(img, -1)

            # Plot per label #
            cmap = plt.cm.Set1
            cmap.set_bad(color='white')
            im = axs[0,i].imshow(
                masked_img,
                cmap = cmap,
                interpolation = 'nearest',
                extent = [bins[0],bins[-1],bins[0],bins[-1]],
                vmin = 0,
            )
            cbar = fig.colorbar(im, ax=axs[0,i], ticks=np.arange(img.max() + 1))
            cbar.set_label('Label')

            # Plot per type, with energy deposit #
            img_type = np.ones((len(bins)-1,len(bins)-1)) * -1
            img_E = np.zeros((len(bins)-1,len(bins)-1))
            for (x,y,z),l,E in zip(pos,labels,E_dep):
                ix = np.digitize(x,bins) - 1
                iy = np.digitize(y,bins) - 1
                img_type[ix,iy] = data.particle_type[l]
                img_E[ix,iy] = E

            # Make rgba array, following energy and type #
            rgba = np.ones((*img_type.shape, 4))
            for typ in types:
                mask = img_type == typ
                rgba[mask, :3] = label_colors[int(typ)][:3]
                typ_values = img_E[mask]
                if typ_values.max() > 0:
                    rgba[mask, 3] = (np.log10(typ_values+1e-10) - np.log10(typ_values.min()+1e-10)) / (np.log10(typ_values.max()+1e-10) - np.log10(typ_values.min()+1e-10))
                else:
                    rgba[mask, 3] = 0.0  # if all zeros
            # -1 pixels to white
            rgba[img_type == -1, :3] = 1.0
            rgba[img_type == -1, 3] = 1.0

            im = axs[1,i].imshow(
                rgba,
                interpolation = 'nearest',
                extent = [bins[0],bins[-1],bins[0],bins[-1]],
            )

            # Axis labels #
            axs[0,i].set_xlabel('x [mm]')
            axs[1,i].set_xlabel('x [mm]')
            axs[0,i].set_ylabel('y [mm]')
            axs[1,i].set_ylabel('y [mm]')
            axs[0,i].set_title(f'Layer {i} (z = {z_pos[i]:.3f} mm)')
            axs[1,i].set_title(f'Layer {i} (z = {z_pos[i]:.3f} mm)')

        # Now add colorbar to the last axis #
        axs[0,-1].cla()
        axs[0,-1].set_axis_off()
        axs[1,-1].cla()
        axs[1,-1].set_axis_off()
        width_fraction = 0.25 / len(types)
        x0 = 0
        for i, typ in enumerate(types):
            max_val = data.node_attr[:,0].max()
            min_val = data.node_attr[data.node_attr[:,0]>0,0].min()
            norm = LogNorm(vmin=min_val, vmax=max_val)
            cmap = make_label_cmap(label_colors[int(typ)])
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            inset_ax = axs[1,-1].inset_axes([x0, 0, width_fraction, 1.0])
            cbar = plt.colorbar(sm, cax=inset_ax, orientation='vertical')
            cbar.set_label(f'{label_dict[int(typ)]} energy [GeV]')
            x0 += width_fraction * 4

        fig.savefig('test.png',bbox_inches='tight')

if __name__ == '__main__':
    dataset = torch.load(sys.argv[1],weights_only=False)
    dataset.plot(int(sys.argv[2]))
    from IPython import embed; embed()

import sys
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            data.global_params = parameters
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
        shapes = {}
        data = self[0]
        for node_type in data.node_types:
            shapes[node_type] = {}
            for key,val in data[node_type].items():
                if torch.is_tensor(val):
                    shapes[node_type][key] = val.shape[-1]
        if hasattr(data,'global_params'):
            shapes['global_params'] = data.global_params.shape[-1]
        return shapes

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

    def get_means(self):
        means = {}
        data = self._data
        for node_type in data.node_types:
            means[node_type] = {}
            for key,val in data[node_type].items():
                if torch.is_tensor(val):
                    means[node_type][key] = self.get_input_mean(val)
        if hasattr(data,'global_params'):
            means['global_params'] = self.get_input_mean(data.global_params)
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

    def get_stds(self):
        stds = {}
        data = self._data
        for node_type in data.node_types:
            stds[node_type] = {}
            for key,val in data[node_type].items():
                if torch.is_tensor(val):
                    stds[node_type][key] = self.get_input_std(val)
        if hasattr(data,'global_params'):
            stds['global_params'] = self.get_input_std(data.global_params)
        return stds

    def plot(self,idx,savepath=None,show=False):
        data = self[idx]

        types = torch.arange(data['particles'].id.shape[1])
        z_pos = torch.unique(data['hits'].pos[:,2])
        cmap_qual = plt.get_cmap('Set1', len(types))
        type_colors = {int(typ): cmap_qual(i)[:3] for i,typ in enumerate(types)}
        rng = np.random.default_rng(42)
        label_colors = rng.choice(plt.cm.tab20b.colors, size=data['particles'].num_nodes, replace=False)
        label_dict = {
            int(typ): f'Part #{typ}' for typ in types
        }

        #label_dict = {
        #    11      : r'$e^{-}$',
        #    -11     : r'$e^{+}$',
        #    22      : r'$\gamma$',
        #    111     : r'$\pi^{0}$',
        #    211     : r'$\pi^{+}$',
        #    -211    : r'$\pi^{-}$',
        #    2212    : r'$p$',
        #    2112    : r'$n$',
        #}
        for typ in type_colors.keys():
            assert typ in label_dict.keys(), f'Missing {typ} in label_dict'

        N = len(z_pos)
        ncols = N+1
        nrows = 2
        if 'predictions' in data.node_types:
            nrows += 1
        fig,axs = plt.subplots(ncols=ncols,nrows=nrows,figsize=(ncols*5,nrows*4))
        plt.subplots_adjust(wspace=0.4,hspace=0.4)

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
            mask = data['hits'].pos[:,2] == z_pos[i]
            pos = data['hits'].pos[mask]
            cell = data['hits'].cell[mask][0] # assume same granularity per layer
            labels = data['hits'].labels[mask]
            E_dep = data['hits'].E
            if 'predictions' in data.node_types:
                beta = data['predictions'].beta[mask]
            else:
                beta = None
            if 'vertices' in data.node_types:
                vert_pos = data['hits'].pos[data['vertices'].idx]
                vert_pos = vert_pos[vert_pos[:,2] == z_pos[i]]
            else:
                vert_pos = None

            # Get axes #
            assert cell[0] == cell[1]
            bins = make_bins(
                min(
                    float(pos[:,:2].min()),
                    - float(pos[:,:2].max()),
                ),
                max(
                    - float(pos[:,:2].min()),
                    + float(pos[:,:2].max()),
                ),
                float(cell[0]),
            )

            # Expand if true particles outside the range or other layers go wider #
            min_part_pos = min(
                data['particles'].pos.min(),
                (data['hits'].pos[:,:2].min() - data['hits'].cell[:,:2]/2).min(),
            )
            max_part_pos = max(
                data['particles'].pos.max(),
                (data['hits'].pos[:,:2].max() + data['hits'].cell[:,:2]/2).max(),
            )
            if min_part_pos < bins[0] or max_part_pos > bins[-1]:
                add_dist = max(
                    max_part_pos - bins[-1],
                    bins[0] - min_part_pos,
                )
                add_bins = round(float(add_dist) / float(cell[0]))
                bins = np.concatenate(
                    [
                        bins[0] - float(cell[0]) * np.arange(add_bins, 0, -1),
                        bins,
                        bins[-1] + float(cell[0]) * np.arange(1, add_bins + 1),
                    ]
                )

            # Fill with label #
            img = np.ones((len(bins)-1,len(bins)-1)) * -1
            for (x,y,z),l in zip(pos,labels):
                ix = max(0,min(len(bins)-1,np.digitize(x,bins))) - 1
                iy = max(0,min(len(bins)-1,np.digitize(y,bins))) - 1
                img[iy,ix] = l
            masked_img = np.ma.masked_equal(img, -1)

            # Plot per label #
            cmap = ListedColormap(label_colors)
            cmap.set_bad(color='white')
            im = axs[0,i].imshow(
                masked_img,
                cmap = cmap,
                origin = 'lower',
                extent = [bins[0],bins[-1],bins[0],bins[-1]],
                vmin = 0,
            )

            # Add true particle locations #
            colors = cmap(np.arange(data['particles'].num_nodes))
            for j in range(data['particles'].num_nodes):
                axs[0,i].scatter(
                    data['particles'].pos[j,0],
                    data['particles'].pos[j,1],
                    color = colors[j],
                    edgecolor = 'black',
                    linewidths=1,
                    marker = 'X',
                    s = 100,
                )
            if 'vertices' in data.node_types:
                if 'pos' in data['vertices'].keys():
                    axs[0,i].scatter(
                        data['vertices'].pos[:,0],
                        data['vertices'].pos[:,1],
                        facecolor = 'none',
                        edgecolor = 'black',
                        linewidths = 1,
                        marker = 'o',
                        s = 50,
                    )
                axs[0,i].scatter(
                    vert_pos[:,0],
                    vert_pos[:,1],
                    facecolor = 'none',
                    edgecolor = 'black',
                    linewidths = 1,
                    marker = 'D',
                    s = 100,
                )

            # Plot per type, with energy deposit #
            img_type = np.ones((len(bins)-1,len(bins)-1)) * -1
            img_E = np.zeros((len(bins)-1,len(bins)-1))
            for (x,y,z),l,E in zip(pos,labels,E_dep):
                ix = max(0,min(len(bins)-1,np.digitize(x,bins))) - 1
                iy = max(0,min(len(bins)-1,np.digitize(y,bins))) - 1
                img_type[iy,ix] = data['particles'].id[l].argmax(dim=-1)
                img_E[iy,ix] = E

            # Make rgba array, following energy and type #
            rgba = np.ones((*img_type.shape, 4))
            for typ in types:
                mask = img_type == typ
                if mask.sum() == 0:
                    continue
                rgba[mask, :3] = type_colors[int(typ)][:3]
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
                origin = 'lower',
                extent = [bins[0],bins[-1],bins[0],bins[-1]],
            )
            for typ in types:
                axs[1,i].scatter(
                    data['particles'].pos[data['particles'].id.argmax(dim=-1)==typ,0],
                    data['particles'].pos[data['particles'].id.argmax(dim=-1)==typ,1],
                    color = type_colors[int(typ)][:3],
                    edgecolor = 'black',
                    linewidths=1,
                    marker = 'X',
                    s = 100,
                )
            if 'vertices' in data.node_types:
                if 'pos' in data['vertices'].keys() and 'id' in data['vertices'].keys():
                    for typ in types:
                        axs[1,i].scatter(
                            data['vertices'].pos[data['vertices'].id.argmax(dim=-1)==typ,0],
                            data['vertices'].pos[data['vertices'].id.argmax(dim=-1)==typ,1],
                            facecolor = 'none',
                            edgecolor = type_colors[int(typ)][:3],
                            linewidths = 3,
                            marker = 'o',
                            s = 50,
                        )
                axs[1,i].scatter(
                    vert_pos[:,0],
                    vert_pos[:,1],
                    facecolor = 'none',
                    edgecolor = 'black',
                    linewidths = 1,
                    marker = 'D',
                    s = 100,
                )

            # Add beta plot #
            if 'predictions' in data.node_types:
                img_beta = np.ones((len(bins)-1,len(bins)-1)) * -1
                for (x,y,z),b in zip(pos,beta):
                    ix = max(0,min(len(bins)-1,np.digitize(x,bins))) - 1
                    iy = max(0,min(len(bins)-1,np.digitize(y,bins))) - 1
                    img_beta[iy,ix] = b
                im = axs[2,i].imshow(
                    img_beta,
                    interpolation = 'nearest',
                    origin = 'lower',
                    cmap = 'Blues',
                    extent = [bins[0],bins[-1],bins[0],bins[-1]],
                    vmin = 0.,
                    vmax = 1.,
                )
                if 'vertices' in data.node_types:
                    axs[2,i].scatter(
                        vert_pos[:,0],
                        vert_pos[:,1],
                        facecolor = 'none',
                        edgecolor = 'black',
                        linewidths = 1,
                        marker = 'D',
                        s = 100,
                    )

            # Axis esthetics #
            axs[0,i].set_xlabel('x [cm]')
            axs[1,i].set_xlabel('x [cm]')
            axs[0,i].set_ylabel('y [cm]')
            axs[1,i].set_ylabel('y [cm]')
            axs[0,i].set_title(f'Layer {i} (z = {z_pos[i]:.3f} cm)')
            axs[1,i].set_title(f'Layer {i} (z = {z_pos[i]:.3f} cm)')
            axs[0,i].set_aspect('equal', adjustable='box')
            axs[1,i].set_aspect('equal', adjustable='box')


        # Add markers #
        axs[0,-1].cla()
        axs[0,-1].set_axis_off()
        axs[0,-1].scatter(
            [],[],
            color = 'white',
            edgecolor = 'black',
            linewidths=1,
            marker = 'X',
            s = 100,
            label = 'True particle',
        )
        if 'vertices' in data.node_types:
            axs[0,-1].scatter(
                [],[],
                facecolor = 'white',
                edgecolor = 'black',
                linewidths = 1,
                marker = 'o',
                s = 50,
                label = 'Reco particle',
            )
            axs[0,-1].scatter(
                [],[],
                facecolor = 'white',
                edgecolor = 'black',
                linewidths = 1,
                marker = 'D',
                s = 100,
                label = 'Condensation point',
            )
        axs[0,-1].legend(loc='center left',fontsize=16,frameon=False)

        # Now add colorbar to the last axis #
        axs[1,-1].cla()
        axs[1,-1].set_axis_off()
        width_fraction = 0.25 / len(types)
        x0 = 0
        max_val = data['hits'].E.max()
        min_val = data['hits'].E[data['hits'].E>0].min()
        for i, typ in enumerate(types):
            norm = LogNorm(vmin=min_val, vmax=max_val)
            cmap = make_label_cmap(type_colors[int(typ)])
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            inset_ax = axs[1,-1].inset_axes([x0, 0, width_fraction, 1.0])
            cbar = plt.colorbar(sm, cax=inset_ax, orientation='vertical')
            cbar.set_label(f'{label_dict[int(typ)]} energy [GeV]')
            x0 += width_fraction * 4

        # Add beta colorbar #
        if 'predictions' in data.node_types:
            axs[2,-1].cla()
            axs[2,-1].set_axis_off()
            cmap = plt.cm.Blues
            norm = matplotlib.colors.Normalize(vmin=0,vmax=1)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            divider = make_axes_locatable(axs[2,-1])
            cax = divider.append_axes("left", size="10%", pad=0.1)
            cb = fig.colorbar(sm, cax=cax)
            cb.set_label(r"$\beta$ (condensation)")

        if savepath is not None:
            fig.savefig(savepath,bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Event plotting")

    parser.add_argument(
        "--dataset",
        type = str,
        help = "Path to dataset",
        required = True,
    )
    parser.add_argument(
        "--event",
        type = int,
        help = "Event number (integer)",
        required = False,
    )
    parser.add_argument(
        "--output",
        type = str,
        help = "Output plot path",
        required = False,
        default = None,
    )
    parser.add_argument(
        "--embed",
        action = 'store_true',
        help = "Interactive environment",
        required = False,
        default = False,
    )

    args = parser.parse_args()

    print ('Loading dataset')
    dataset = CaloGraphDataset.load(args.dataset)
    print ('... done')
    if args.event is not None and args.output is not None:
        dataset.plot(
            idx = args.event,
            savepath = args.output,
        )
    if args.embed:
        from IPython import embed; embed()

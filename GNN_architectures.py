#Package Imports
import os
import os.path as osp
import sys
import shutil

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error as mae
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.nn import  CGConv
from matplotlib.offsetbox import AnchoredText
import torch.nn.functional as F
sys.path.append("./")
from pathlib import Path
import pickle
import pandas as pd

from help_function import *  

#NNConv model architecture
class NNConv_Model(torch.nn.Module):
    def __init__(self,dim_node_attr,dim_edge_attr,
                 node_embedding_dim,
                 edge_embedding_dim, aggr='mean', processing_steps=3, num_layers=1, num_iter=3):
        super().__init__()
        self.num_iter=num_iter
        self.lin0 = torch.nn.Linear(dim_node_attr, node_embedding_dim)

        nn = Sequential(Linear(dim_edge_attr, edge_embedding_dim),
                        ReLU(),
                        Linear(edge_embedding_dim, node_embedding_dim * node_embedding_dim))
        self.conv = NNConv(node_embedding_dim, node_embedding_dim, nn, aggr=aggr)
        self.gru = GRU(node_embedding_dim, node_embedding_dim)

        self.set2set = Set2Set(node_embedding_dim, processing_steps=processing_steps, num_layers=num_layers)
        self.lin1 = torch.nn.Linear(2 * node_embedding_dim, node_embedding_dim)
        self.lin2 = torch.nn.Linear(node_embedding_dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

#CGCNN model architecture  
class CGCNN_Model(torch.nn.Module):
    """
        Initialize CrystalGraphConvNet.
        Parameters
        ----------
        dim_node_attr: int
          Number of atom features in the input.
        dim_edge_attr: int
          Number of edge features in the input.
        edges_embedding_size: int
          Number of bond features.
        atom_embedding_size: int
          Number of hidden atom features in the convolutional layers
        num_graph_conv_layers: int
          Number of convolutional layers
        num_fc_layers: int
          Number of hidden layers after pooling
        fc_feat_size (int, optional):
            Size of fully connected layers.
    """
    def __init__(self,
                 dim_node_attr,
                 dim_edge_attr,
                 atom_embedding_size=64,
                 edges_embedding_size=128,
                 fc_feat_size=128,
                 num_graph_conv_layers=6,
                 num_fc_layers=4,
                 batch_norm=False,
                 bias=True):
        super(CGCNN_Model, self).__init__()

        self.embedding_nodes = torch.nn.Linear(dim_node_attr, atom_embedding_size)

        #  Embedding edges features

        self.embedding_edges = torch.nn.Linear(dim_edge_attr, edges_embedding_size)
        self.embedding_fc = torch.nn.Linear(dim_node_attr, atom_embedding_size)
        self.convs = nn.ModuleList(
            [
                CGCNNConv(
                    node_dim=atom_embedding_size,
                    edge_dim=dim_edge_attr,
                )
                for _ in range(num_graph_conv_layers)
            ]
        )

        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_embedding_size, fc_feat_size), nn.Softplus()
        )

        if num_fc_layers > 1:
            layers = []
            for _ in range(num_fc_layers - 1):
                layers.append(nn.Linear(fc_feat_size, fc_feat_size))
                layers.append(nn.Softplus())
            self.fcs = nn.Sequential(*layers)
        self.fc_out = nn.Linear(fc_feat_size, 1)
        
    #Forward pass through the network
    def _forward(self, data):
        mol_feats = self._convolve(data)
        mol_feats = self.conv_to_fc(mol_feats)
        if hasattr(self, "fcs"):
            mol_feats = self.fcs(mol_feats)
        out = self.fc_out(mol_feats)
        return  out.view(-1)

    def forward(self, data):
        return self._forward(data)

    def _convolve(self, data):
        """
        Returns the output of the convolution layers before they are passed
        into the dense layers.
        """
        node_feats = self.embedding_nodes(data.x)
        edge_attr = data.edge_attr

        for f in self.convs:
            node_feats = f(node_feats, data.edge_index, edge_attr )
        mol_feats = global_mean_pool(node_feats, data.batch)
        return mol_feats

class CGCNNConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(self, node_dim, edge_dim,  **kwargs):
        super(CGCNNConv, self).__init__(aggr="add")
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim

        self.lin1 = nn.Linear(
            2 * self.node_feat_size + self.edge_feat_size,
            2 * self.node_feat_size,
        )
        self.bn1 = nn.BatchNorm1d(2 * self.node_feat_size)
        self.ln1 = nn.LayerNorm(self.node_feat_size)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        self.bn1.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        out = nn.Softplus()(self.ln1(out) + x)
        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]
        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = self.lin1(torch.cat([x_i, x_j, edge_attr], dim=1))
        z = self.bn1(z)
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2
    
#Class to handle GNN architectures    
class Model_Handler():
    def __init__(self,config_file,camp_name):
        '''
        config_file (dictionnary): GNN specifications
        camp_name (str): save name

        '''
        if config_file['model'] == 'cgcnn':
            print('Importing CGCNN Configuration')
            self.config = config_file
            self.campaign_name = camp_name
            self.a=config_file["aggr"]
            self.s=config_file["num_graph_conv_layers"]
            self.l=config_file["atom_embedding_size"]
            self.h=config_file["num_fc_layers"]
            self.b=config_file["fc_feat_size"]
        if config_file['model'] == 'nnconv':
            print('Importing NNConv Configuration')
            self.config = config_file
            self.campaign_name = camp_name
            self.n=config_file["node_embedding_dim"]
            self.e=config_file["edge_embedding_dim"]
            self.a=config_file["aggr"]
            self.s=config_file["processing_steps"]
            self.l=config_file["num_layers"]
            self.m=config_file["num_iter"]
        
    #Plot:
    def plot_train(self,df, name='train_history', title='Parity Plot', x='Target Value', y='Predicted Value', color='b',  hue='adsorbate'):
        '''
        df (pandas df): training history data
        title (str): save title
        x/y (str): axis naming
        color (str): plotting color
        hue (str): self explanatory
        '''
        sns.set_style("whitegrid", {'axes.grid' : False})

        actual = df[x].to_numpy()
        pred = df[y].to_numpy()
        RMSD = np.sqrt( sum( (pred - actual)**2 ) / len(pred) )
        MAE = np.mean(np.abs(pred - actual))
        fig, ax = plt.subplots()

        sns.scatterplot(data=df, x=x, y=y, color=color, hue=hue, s=40,  legend='full', linewidth=0, alpha = 0.7)
        sns.lineplot(x=np.linspace(np.amin(actual), np.amax(actual), 100),
                 y=np.linspace(np.amin(actual), np.amax(actual), 100), color='red')

        at = AnchoredText(
            '{:<8s} = {:.3f} eV \n {:<8s} = {:.3f} eV'.format('RMSD', RMSD, 'MAE', MAE), prop=dict(size=12), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        ax.legend( title='adsorbate', loc='lower right')
        plt.xlabel('$\Delta E_{DFT}$ [eV]', fontsize=15)
        plt.ylabel('$\Delta E_{Predicted}$ [eV]', fontsize=15)
        plt.title(title, fontsize=15)
        fig.savefig(name+'.png', dpi=300, transparent=True, bbox_inches='tight')
        
    def _train(self,model, loader, epoch,normalizer, optimizer, device='cpu'):
        '''
        model (torch geometric model): model object
        loader (torch dataloader): dataloader for train dataset
        epoch (int): number of epochs
        normalizer (torch.normalizer object): self explanatory
        optimizer (torch.optimizer object): self explanatory
        device (str): cpu or gpu training
        '''
        model.train()
        loss_all = 0
        losses = AverageMeter()
        for i, data in enumerate(loader):
            target_normed = normalizer.norm(data.y)
            target_normed = target_normed.to(device)
            data = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(data), target_normed)
            loss.backward()
            losses.update(loss.data.cpu(), data.y.size(0))
            optimizer.step()
        return losses.avg.item()
    
    #Evaluate predictions on loader
    def _test(self,model, loader, normalizer, device='cpu', test=False):
        '''
        model (torch geometric model): model object
        loader (torch dataloader): dataloader for test dataset
        normalizer (torch.normalizer object): self explanatory
        device (str): cpu or gpu training
        test (bool): add extra detail on testing results
        '''
        model.eval()
        error = 0
        mae_errors = AverageMeter()
        test_preds = []
        test_targets = []
        test_idx =[]
        test_adsorbate = []

        for i, data in enumerate(loader):
            data = data.to(device)
            pred = model(data)

            pred_denormed = normalizer.denorm(pred.data.cpu())
            mae_error = mae(pred_denormed, data.y)
            mae_errors.update(mae_error, data.y.size(0))
            test_preds.extend(pred_denormed.tolist())
            test_targets.extend(data.y.tolist())
            try:
                test_idx.extend(data.idx.tolist())
            except:
                test_idx.extend(data.idx)
            try:
                test_adsorbate.extend(data.adsorbate)
            except:
                test_adsorbate.extend(data.adsorbate)
        if test:
            my_dict = {"idx": test_idx,
                   "adsorbate": test_adsorbate,
                   'Target Value': test_targets,
                   'Predicted Value': test_preds}
            df = pd.DataFrame(my_dict)
            return mae_errors.avg , df
        else:
            return mae_errors.avg.item()
    
    #Save model:
    def save_checkpoint(self,state, is_best, filename, checkpoint_dir='.'):
        '''
        state (torch object): model state
        is_best (bool): best model or not
        filename (str): save name
        checkpoint_dir (str): directory to save results
        '''
        path = os.path.join(checkpoint_dir, f"{self.config['model']}_checkpoint")
        torch.save(state, path)
        if is_best:
            bestfile = f"{self.config['model']}_best_model"
            shutil.copyfile(path, os.path.join(checkpoint_dir,bestfile))
            return bestfile 
        
    #Training
    def train_function(self, train_dataset, val_dataset,normalizer, device = "cpu", checkpoint_dir=None):
        '''
        train_dataset (list): training set
        val_dataset (list): val set
        normalizer (torch.normalizer object): self explanatory
        device (str): cpu or gpu training
        checkpoint_dir (str): directory to save results
        '''
        best_model = None
        best_val_error = 10e15

        epoch_history =[]
        val_error_history =[]
        loss_history =[]

        #Create dataloaders
        val_loader = DataLoader(val_dataset, batch_size=int(self.config["batch_size"]), shuffle=True, exclude_keys=['symbol','atomic_number' ,'distance', 'EN' ])
        train_loader = DataLoader(train_dataset, batch_size=int(self.config["batch_size"]), shuffle=True, exclude_keys=['symbol','atomic_number' ,'distance', 'EN' ])

        dim_node_attr = train_dataset[0].x.shape[-1]
        dim_edge_attr= train_dataset [0].edge_attr.shape[-1]
        #generate insstance of GNN for training
        if self.config['model']=='cgcnn':
            net = CGCNN_Model(dim_node_attr=dim_node_attr,
                         dim_edge_attr=dim_edge_attr,
                         atom_embedding_size=self.config['atom_embedding_size'],
                         num_graph_conv_layers=self.config['num_graph_conv_layers'],
                         fc_feat_size=self.config["fc_feat_size"],
                         num_fc_layers=self.config['num_fc_layers'],
                         batch_norm=self.config['batch_norm'],
                         bias=self.config['bias'])
        if self.config['model'] == 'nnconv':
            net = NNConv_Model(dim_node_attr=dim_node_attr,
                          dim_edge_attr=dim_edge_attr,
                          node_embedding_dim=self.config["node_embedding_dim"],
                          edge_embedding_dim=self.config["edge_embedding_dim"],
                          aggr=self.config["aggr"],
                         processing_steps=self.config["processing_steps"],
                          num_layers=self.config["num_layers"],
                         num_iter=self.config["num_iter"])            
        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)
        #Commence training loop
        for epoch in range(self.config['Nepoch']):  
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = self._train(net, train_loader, epoch, normalizer, optimizer, device=device)
            val_error = self._test(net, val_loader, normalizer)
            scheduler.step(val_error)
            is_best = val_error < best_val_error
            best_val_error = min(val_error , best_val_error)
            if self.config['model']=='cgcnn':
                model_state = {
                        "epoch": epoch + 1,
                        "state_dict": net.state_dict(),
                        "best_val_mae": best_val_error,
                        "optimizer": optimizer.state_dict(),
                        "normalizer": normalizer.state_dict(),
                        "criterion": "MSELoss",
                        "dim_node_attr":dim_node_attr,
                        "dim_edge_attr" : dim_edge_attr,
                        "atom_embedding_size": self.config['atom_embedding_size'],
                        "num_fc_layers" :self.config['num_fc_layers'],
                        "num_graph_conv_layers": self.config['num_graph_conv_layers'],
                        "fc_feat_size":self.config["fc_feat_size"]
                }
            if self.config['model']=='nnconv':
                model_state = {
                         "epoch": epoch + 1,
                         "state_dict": net.state_dict(),
                         "best_val_mae": best_val_error,
                         "optimizer": optimizer.state_dict(),
                         "normalizer": normalizer.state_dict(),
                         "criterion": "MSELoss",
                         "dim_node_attr": dim_node_attr,
                         "dim_edge_attr": dim_edge_attr,
                         "processing_steps": self.config["processing_steps"],
                         "num_layers":self.config["num_layers"],
                         "node_embedding_dim": self.config["node_embedding_dim"],
                         "edge_embedding_dim": self.config["edge_embedding_dim"],
                         "num_iter":self.config["num_iter"],
                         "aggr":self.config["aggr"]
                }

            self.save_checkpoint(model_state, is_best, f"{self.config['model']}_checkpoint", checkpoint_dir)
            epoch_history.append(epoch)
            val_error_history.append(val_error)
            loss_history.append(loss)
            if epoch%50 ==0:
                print(f'Epoch: {epoch:04d}, LR: {lr:7f}, Loss: {loss:.7f}, '
                      f'Val MAE: {val_error:.7f}')

            if is_best:
                best_model = model_state

        my_dict = {"epoch":epoch_history,
                       "loss":loss_history ,
                       "val_error":val_error_history}

        df = pd.DataFrame(my_dict)    
        if self.config['model'] == 'cgcnn':
            df.to_csv(f"{checkpoint_dir}/{self.config['model']}_{self.config['emb']}_train_history_{self.a}_{self.s}_{self.l}_{self.h}_{self.b}.csv", index=False)
        
        if self.config['model'] == 'nnconv':
            df.to_csv(f"{checkpoint_dir}/{self.config['model']}_{self.config['emb']}_train_history_{self.n}_{self.e}_{self.a}_{self.s}_{self.l}_{self.m}.csv", index=False)            
        return best_model
    
    #Test best model:
    def test_best_model(self,
                        test_dataset=None, testfile=None, checkpoint_dir=None, best_model=None,
                        batch_size=8, device="cpu",save_df=True):
        '''
        test_dataset (list): test set
        testfile (str): test file name
        checkpoint_dir (str): checkpoint folder path
        best_model (torch geometric model): self explanatory
        batch_size (int): self explanatory
        device (str): cpu or gpu implementation
        save_df (bool): self explanatory
        '''

        if testfile is not None:
            with open(testfile, 'rb') as handle:
                    test_dataset= pickle.load(handle)

        #Extract attributes from best model
        if self.config['emb'] != 'cgcnn92':
            atom_features =  get_atom_embedding(self.config['emb'])

            for idata, data in enumerate(test_dataset):
                node_attr = np.vstack([np.array(atom_features[s]).astype(np.float32) for s in  data.symbol])
                test_dataset[idata].x=torch.from_numpy(node_attr)

        dim_node_attr =  test_dataset[0].x.shape[-1]
        dim_edge_attr= test_dataset[0].edge_attr.shape[-1]

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 exclude_keys=['symbol','atomic_number' ,'distance', 'EN' ])

        sample_target = torch.vstack([ data.y for data in test_dataset])
        normalizer = Normalizer(sample_target)

        if best_model:
            model=best_model

        else:

            best_checkpoint_path = os.path.join(checkpoint_dir, f"{self.config['model']}_best_model")
            model = torch.load(best_checkpoint_path)

        #Create best model parameters from extract attributes
        if self.config['model']=='cgcnn':
            best_trained_model = CGCNN_Model(dim_node_attr=dim_node_attr,
                         dim_edge_attr=dim_edge_attr,
                         atom_embedding_size=self.config['atom_embedding_size'],
                         num_graph_conv_layers=self.config['num_graph_conv_layers'],
                         fc_feat_size=self.config["fc_feat_size"],
                         num_fc_layers=self.config['num_fc_layers'],
                         batch_norm=self.config['batch_norm'],
                         bias=self.config['bias'])
        if self.config['model']=='nnconv':
            best_trained_model = NNConv_Model(dim_node_attr=dim_node_attr,
                                         dim_edge_attr=dim_edge_attr,
                                         node_embedding_dim=model['node_embedding_dim'],
                                         edge_embedding_dim=model['edge_embedding_dim'],
                                         aggr=model['aggr'],
                                         processing_steps=model["processing_steps"],
                                         num_layers=model["num_layers"],
                                        num_iter=model["num_iter"])

        best_trained_model.to(device)
        normalizer.load_state_dict(model['normalizer'])
        best_trained_model.load_state_dict(model['state_dict'])

        test_error, test_df  = self._test(best_trained_model, test_loader, normalizer,device=device,test=True)
        #save results
        if save_df:
            if self.config['model']=='cgcnn':
                test_df.to_csv(f"{self.campaign_name}/{self.config['model']}_{self.campaign_name}_{self.config['emb']}_{self.a}_{self.s}_{self.l}_{self.h}_{self.b}.csv", index=False)
            if self.config['model'] == 'nnconv':
                test_df.to_csv(f'{self.campaign_name}/results_{self.campaign_name}_{self.n}_{self.e}_{self.a}_{self.s}_{self.l}_{self.m}.csv', index=False)      
        return  test_error, test_df    

    #Graph functions:
    def get_atom_embedding(emb="cgcnn92"):
        '''
        emb (str): feature embedding
        '''
        
        elem_emb = join(os.path.dirname(os.path.realpath("__file__")), f"element/{emb}.json")
        with open(elem_emb) as f:
                atom_features = json.load(f)
        return atom_features

    #Update edge of generate graph with gaussian distance embedding
    def update_edges(dataset):
        '''
        dataset (list): dataset to update edges for
        '''
        for idata, data in enumerate(dataset):
            #dataset[idata].edge_attr = torch.stack([1/data.distance[:,0],1/data.EN[:,0]], dim=1).to(torch.float)
            dataset[idata].edge_attr = torch.stack([1/data.distances[:],1/data.EN[:]], dim=1).to(torch.float)
        return dataset    
   
    #Main function:
    def main(self, datafile, data_size=None, train_ratio=None, val_ratio=0.1, test_ratio=0.1):
        #prep directory to save results and prep data for testing
        try:
            Path(self.campaign_name).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"Folder [{self.campaign_name}] is already there")
        else:
            print(f"Folder [{self.campaign_name}] was created")

        with open(datafile, 'rb') as handle:
            dataset = pickle.load(handle)

        fails=[]
        for i in range(len(dataset)):
            try:
                check_for_y = dataset[i].y
            except:
                fails.append(i)
                print('fail')
        for i in fails:
            dataset.remove(dataset[i])

        if data_size is not None:

            dataset = random.sample(dataset,  int(len(dataset)*data_size))

        if self.config['emb'] != 'cgcnn92':
            atom_features =  get_atom_embedding(self.config['emb'])

            for idata, data in enumerate(dataset):
                node_attr = np.vstack([np.array(atom_features[s]).astype(np.float32) for s in  data.symbol])
                dataset[idata].x=torch.from_numpy(node_attr)

        train_sampler, val_sampler, test_sampler = get_train_val_test_indices(len(dataset),
                                                                              train_ratio=train_ratio,
                                                                              val_ratio=val_ratio,
                                                                              test_ratio=test_ratio)
        #prep training loop data
        dim_node_attr = dataset[0].x.shape[-1]
        dim_edge_attr=dataset[0].edge_attr.shape[-1]

        sample_target = torch.vstack([ data.y for data in dataset])
        normalizer = Normalizer(sample_target)

        train_dataset  = [ dataset[idx] for idx in train_sampler]
        val_dataset = [ dataset[idx] for idx in val_sampler]
        test_dataset   = [ dataset[idx] for idx in test_sampler]

        train_dataset += test_dataset
        print('train_sampler len:', len(train_dataset))
        print('val_sampler len:', len(val_dataset))

        best_model = self.train_function(train_dataset =train_dataset, val_dataset =val_dataset, normalizer=normalizer,checkpoint_dir=self.campaign_name)

        return best_model, test_dataset
    
    def get_train_history(self,):
        #read in saved results and plot them
        if self.config['model'] =='cgcnn':
            train_hist = pd.read_csv(f"{self.campaign_name}/{self.config['model']}_{self.config['emb']}_train_history_{self.a}_{self.s}_{self.l}_{self.h}_{self.b}.csv")
        if self.config['model']=='nnconv':
            train_hist = pd.read_csv(f"{self.campaign_name}/{self.config['model']}_{self.config['emb']}_train_history_{self.n}_{self.e}_{self.a}_{self.s}_{self.l}_{self.m}.csv")
            
        dfm = train_hist.melt('epoch', 
                                 var_name='error_type', 
                                 value_name='error')

        sns.lineplot(data=dfm, x='epoch', y='error',hue='error_type' )
        return train_hist      
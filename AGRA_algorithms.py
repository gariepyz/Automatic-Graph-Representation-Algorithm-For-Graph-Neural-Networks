#Package Imports
import json
import multiprocessing as mp
import os
import random
import glob
import copy
import math
from os.path import abspath, dirname, exists, join
import sys
sys.path.append("./")
from pathlib import Path
from sklearn.model_selection import train_test_split

import ase
from ase import Atoms, Atom
from ase.visualize import view
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.visualize.plot import plot_atoms
from ase.constraints import dict2constraint
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.io import extxyz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import networkx as nx
import mendeleev
import itertools
import pickle
import shutil
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial import distance_matrix

class GaussianDistance():
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """

    def __init__(self, dmin=0.0, dmax=6, step=0.2, var=None):
        """
        Args:
            dmin (float): Minimum interatomic distance
            dmax (float): Maximum interatomic distance
            step (float): Step size for the Gaussian filter
            var (float, optional): Variance of Gaussian basis. Defaults to step if not given
        """
        assert dmin < dmax
        assert dmax - dmin > step

        self.filter = np.arange(dmin, dmax + step, step)
        self.embedding_size = len(self.filter)

        if var is None:
            var = step

        self.var = var

    def expand(self, distances):
        """Apply Gaussian distance filter to a numpy distance array
        Args:
            distances (ArrayLike): A distance matrix of any shape
        Returns:
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        distances = np.array(distances)

        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var ** 2)

    def get_layer(self,slb, thresh = 1):
        """
        * slb (atoms object): The system, WITHOUT adsorbate attached
        * thresh (float): The threshold in Angstroms of the comparison between z heights
        * layer_remapped (list of floats): An ordered list of the layer heights identified in the system
        * num_layers (int): # of layers in system
        * num_atoms (int): # of atoms in system
        * layer_ID_Indices (list of lists): A list of lists of atom ID #s corresponding to the atoms placed in each layer classification.
          NOTE: layer_ID_Indices[0] contains the ID for atoms in the ASE system that were identified as belonging to the height in layer_remapped[0]

        """

        layer_ID_Indices = []
        similar_idx = []
        num_atoms = len(slb)
        slab_z_list= slb.positions[:,2].tolist()
        unique_z = np.unique(slab_z_list).tolist()

        # For the unique heights in the system...
        # loop, and see if any other entries are close
        # If they are, save the indices of these redundant heights so we can eliminate them
        for h1 in unique_z:
            for h2 in unique_z:
                if h1 != h2 and unique_z.index(h2) > unique_z.index(h1) and math.isclose(h1, h2, abs_tol = thresh):
                    similar_idx.append(h2)

        # This gets all the distinct z values together by removing the entries in the similar list
        layer_z = [unique_z[i] for i in range(len(unique_z)) if unique_z[i] not in similar_idx]
        # This is how many layers we think we have
        num_layers = len(layer_z)
        # This is the set of layer heights we found
        layer_heights = layer_z

        # Find the indices present in each layer we have found
        slotted_indices = []
        for i in range(num_layers):
            layer_ID_Indices.append([])
            for j in range(num_atoms):
                if math.isclose(slab_z_list[j], layer_heights[i], abs_tol = 1) == True and j not in slotted_indices:
                    layer_ID_Indices[i].append(j)
                    slotted_indices.append(j)
        # This next section organizes the indices so that we have each list of indices for a layer ordered in terms of layer ascending/descending instead of randomly
        layer_remapped = copy.deepcopy(layer_heights)
        layer_remapped.sort()
        ordering = [layer_heights.index(layers) for layers in layer_remapped]
        #print("Ordering", ordering)
        layer_ID_Indices = [layer_ID_Indices[i] for i in ordering]

        return layer_remapped, ordering, layer_ID_Indices

#This class handle model/data imports
class ASE_To_Graph():
    def __init__(self,db_file,embedding):
        '''
        db_file (str): filename/path of the ASE db file
        embedding (str): name of the embedding you want to implement
        
        '''
        
        self.db_filename = db_file
        self.embedding = embedding
        
    def import_db(self):
        with connect(self.db_filename) as db:
            ase_db = db
        self.db = ase_db    
        return ase_db
    
    #Not used in algorithm, just helps with data inspection 
    def bond_symbol(self,atoms, a1, a2):
        return "{}{}".format(*sorted((atoms[a1].symbol, atoms[a2].symbol)))
    
    #Get electronegativity (EN)
    def get_EN(self,atoms, a1, a2):
        try:
            avg = (mendeleev.element(atoms[a1].symbol).en_pauling + mendeleev.element(atoms[a2].symbol).en_pauling)/2
        except:
            avg = 0.0
        try:
            dif = abs(mendeleev.element(atoms[a1].symbol).en_pauling - mendeleev.element(atoms[a2].symbol).en_pauling)
        except:
            dif = 0
        return np.array([avg, dif])

    def node_symbol(self,atom):
        return "".format(atom.symbol)


    def add_atoms_node(self,graph, atoms, a1):
        graph.add_node(self.node_symbol(atoms[a1]), index=a1)


    def add_atoms_edge(self,graph, atoms, a1, a2):
        EN = self.get_EN(atoms, a1, a2)
        distance = atoms.get_distances(a1,a2)
        graph.add_edge(a1,
                       a2,
                       EN= EN,
                       distance = distance)   

    #Main surface analysis automation function
    def analyse_surface(self,atoms, adsorbate,radii_multiplier=1.1, skin = 0.25, **kwargs):
        
        #Extract nearest neighbor indexes 
        nl = NeighborList(natural_cutoffs(atoms, radii_multiplier), self_interaction=False,
                              bothways=True, skin=skin)

        nl.update(atoms)
        #Extract index of adsorbate atoms 
        adsorbate_atoms = [atom.index for atom in atoms if atom.symbol in adsorbate]

        # Add all atoms to graph
        distances = atoms.get_all_distances(mic=True)

        binding_atoms = []
        bonds = []
        for index, atom in enumerate(atoms):
            neighbors, offsets = nl.get_neighbors(index)
            for neighbor, offset in zip(neighbors, offsets):
                if np.all(offset==np.zeros(3)):
                    if distances[index][neighbor] > 2.5 and (bool(index in adsorbate_atoms) ^ bool(neighbor in adsorbate_atoms)):

                        continue
                    if (bool(index not in adsorbate_atoms) ^ bool(neighbor not in adsorbate_atoms)) and (index not in adsorbate_atoms):
                        binding_atoms.append(index)

                bonds.append((index, neighbor))

        return binding_atoms, adsorbate_atoms, bonds        
    
    def extract_ads_geom(self, atms, adsorbate, elem_features, plot=False):
        '''
        atms (ASE Atoms): slab to analyze
        adsorbate (str): adsorbate to analyze
        elem_features (str): embedding features initialized with the class
        plot (bool): plot the exactracted structure
        '''
        #Extract neighbors
        binding_atoms, adsorbate_atoms, bonds = self.analyse_surface(atms, adsorbate=adsorbate)
        nl = NeighborList(natural_cutoffs(atms, 1.1), self_interaction=False,
                                  bothways=True, skin=0.25)
        nl.update(atms)
        poss = [atms.positions[i] for i in binding_atoms]
        symbols = [atms[i].symbol for i in binding_atoms]
        distances =atms.get_all_distances(mic=True)
        #extract coordiantes of atoms
        for atm in  binding_atoms:
            neighbors, offsets = nl.get_neighbors(atm)
            for i, offset in zip(neighbors, offsets):
                poss.append(atms.positions[i] + np.dot(offset, atms.get_cell()))
                symbols.append(atms[i].symbol)
        #extract ads atoms coords
        for atm in  adsorbate_atoms:
            poss.append(atms.positions[atm])
            symbols.append(atms[atm].symbol)

        coords = np.vstack(poss)

        #generate extracted neighbors structure
        surf_n = Atoms(symbols,positions=coords )
        ase.geometry.get_duplicate_atoms(surf_n, cutoff=0.1, delete=True)
        if plot:
            fig, ax = plt.subplots(1,2)
            plot_atoms(surf_n,ax[0],radii=0.9, rotation=('0x,0y,0z'))
            plot_atoms(surf_n,ax[1],radii=0.9, rotation=('-90x,45y,0z'))

        params = {"binding_atoms":binding_atoms,
                "adsorbate_atoms":adsorbate_atoms,
                "bonds":bonds}


        adsorbate_atms = [atom.index for atom in surf_n if atom.symbol in adsorbate]

        #Generate Graph

        full = nx.Graph()

        full.add_nodes_from(range(len(surf_n)))
        distances = surf_n.get_all_distances(mic=True)

        numbers = surf_n.get_atomic_numbers().tolist()
        symbols = surf_n.get_chemical_symbols()


        for i, feat_dict in full.nodes(data=True):
            feat_dict.update({'symbol':  symbols[i]})
            feat_dict.update({'atomic_number':  numbers[i]})
            feat_dict.update({'node_attr':  np.array(elem_features[symbols[i]]).astype(np.float32)})
        comb = list(itertools.combinations([i for i in range(len(surf_n))],2))

        for atm1, atm2 in comb:
            if atm1==atm2:
                continue
            if distances[atm1][atm2] > 2.8 :
                continue

            if distances[atm1][atm2] > 2.3 and (bool(atm1 in adsorbate_atms) ^ bool(atm2 in adsorbate_atms)):
                continue
            if distances[atm1][atm2] > 2.3 and (bool(atm1 in adsorbate_atms) and bool(atm2 in adsorbate_atms)):
                continue
            self.add_atoms_edge(full, surf_n,  atm1, atm2)
        return surf_n,  full #params 
    
    #convert a ASE atoms object from an ASE db into a graph   
    def get_data(self,row, atom_features,map_warning=False):
        '''
        row (ASE rows): row to convert into a torch graph
        atom_features (str): embedding features 
        map_warning (bool): print warnings about failed mappings
        '''
        gdf = GaussianDistance()
        adsorbate=row.Adsorbate
        atm = row.toatoms()
        slabId = row.Hea_ID
        ads_energy = row.Energy
        #embed graph object with features
        try:
            atms, G = self.extract_ads_geom(atms=atm, adsorbate=list(set(list(adsorbate))), elem_features=atom_features)
            data =from_networkx(G)

            data.x = data.node_attr
            edge_attr = data.EN[:,0]*data.distance[:,0]
            edge_attr.unsqueeze(-1)
            edge_attr = gdf.expand(edge_attr)
            edge_attr = torch.tensor(edge_attr)
            data.edge_attr = edge_attr
            data.__setattr__('idx',slabId)
            data.__setattr__('adsorbate',adsorbate)
            data.__setattr__('postions',torch.from_numpy(atms.positions))
            data.y = torch.tensor([ads_energy])

            try:
                atm1 = copy.deepcopy(atms)
                del(atm1[-len(list(adsorbate)):])
                atm1.center()
                _,_, layers = gdf.get_layer( atm1, thresh = 1)
                surface = layers[-1]
                subsurface = layers[-2]
                data.__setattr__('surface',[ atm1.symbols[i] for i in surface])
                data.__setattr__('subsurface',[ atm1.symbols[i] for i in subsurface])
            #Throw error 
            except Exception as er1:
                if map_warning:
                    print(f" Warnn ing!!! surface and subsurface atoms not found in ID[{slabId}]")
                data.__setattr__('surface',[])
                data.__setattr__('subsurface',[])
            return data
        except Exception as err:
            print(f"{err}")
            return None
        
    def get_data_from_asedb(self, db,  adsorbate, emb="cgcnn92", ncpus=1):
        '''
        db (ASE db): db to fully extract
        adsorbate (str): adsorbate to analyze/extract
        emb (str): embedding features
        ncpus (int): core cpus 
        '''
        
        gdf = GaussianDistance()
        rows = [row for row in db.select('Energy', Adsorbate=adsorbate) ]

        elem_emb = join(os.path.dirname(os.path.realpath("__file__")), f"element/{emb}.json")
        with open(elem_emb) as f:
                atom_features = json.load(f)
        
        if len(rows) > 100:
            print(f'Warning, {len(rows)} ASE atoms will likely take {len(rows)/10}+ min to complete')
        
        results = []
        for i in range(len(rows)):
            if i==0:
                results.append(self.get_data( rows[i],atom_features))
            elif i%20==0:
                print(f'{i}/{len(rows)} Graphs generated')
            else:
                results.append(self.get_data( rows[i],atom_features))

        datas = results
        

        with open(f"data_{adsorbate}_{emb}.pickle", 'wb') as handle:
            pickle.dump(datas, handle)
            
        self.adsorbate = adsorbate
        self.emb = emb
        self.pickle_name = f"data_{adsorbate}_{emb}.pickle"
    
    #Provide a sample visual of AGRA extraction   
    def sample_visual(self):
        with open('element/cgcnn92.json') as f:
            atom_features = json.load(f)

        slab_sample_visual = self.db.get_atoms(1)
        extraction_sample_visual, g = self.extract_ads_geom(slab_sample_visual,
                                                           adsorbate=['O','C'],
                                                           elem_features=atom_features)
        
        ocp_a2g = OCP_AtomsToGraphs(
            max_neigh=50,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=True,
            r_edges=True,
            r_fixed=True,)

        ocp_slab = ocp_a2g.convert(slab_sample_visual)
        ocp_graph = to_networkx(ocp_slab)
        #nx.draw_networkx(ocp,with_labels=True)
        
        
        fig, ax = plt.subplots(2,2,figsize=(8,6),subplot_kw={'aspect': 'equal'})
       
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        ax[0, 0] = plt.subplot(gs[0, 0])
        ax[0, 1] = plt.subplot(gs[0, 1])
        ax[1, 0] = plt.subplot(gs[1, 0])
        ax[1, 1] = plt.subplot(gs[1, 1])
        
        plot_atoms(slab_sample_visual,ax[0,0],radii=0.9, rotation=('0x,0y,0z'))
        ax[0,0].set_title("ASE db Extracted Structure")
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])

        plot_atoms(extraction_sample_visual,ax[0,1],radii=0.9, rotation=('0x,0y,0z'))
        ax[0,1].set_title("AGRA Refined Structure")
        ax[0,1].set_xticks([])
        ax[0,1].set_yticks([])

        nx.draw_networkx(g,with_labels=True,ax=ax[1,1])
        ax[1,1].set_title("Resulting AGRA Graph")  
        
        nx.draw_networkx(ocp_graph,with_labels=True,ax=ax[1,0])
        ax[1,0].set_title("Resulting OCP Graph")
        
        plt.tight_layout()
        return None
    #split generated graph data into train/test data
    def split_data(self,split=0.1):
        with open(self.pickle_name, 'rb') as handle:
            dataset= pickle.load(handle)   
            
        train,test,ytrain,ytest = train_test_split(dataset,dataset,test_size=split)
        
        with open(f"data_{self.adsorbate}_Train_{self.emb}.pickle", 'wb') as handle:
            pickle.dump(train, handle)
        with open(f"data_{self.adsorbate}_Test_{self.emb}.pickle", 'wb') as handle:
            pickle.dump(test, handle)
        return train,test

#This entire class is copied over from the OCP github solely for the purpose of graph generation comparison (https://github.com/Open-Catalyst-Project/ocp)
class OCP_AtomsToGraphs:
    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=True,
        r_fixed=True,
        r_pbc=False,
    ):
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_distances = r_distances
        self.r_fixed = r_fixed
        self.r_edges = r_edges
        self.r_pbc = r_pbc

    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets

    def _reshape_features(self, c_index, n_index, n_distance, offsets):
        """Stack center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # remove distances smaller than a tolerance ~ 0. The small tolerance is
        # needed to correct for pymatgen's neighbor_list returning self atoms
        # in a few edge cases.
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets

    def convert(
        self,
        atoms,
    ):
        """Convert a single atomic stucture to a graph.
        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.
        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with positions, atomic_numbers, tags,
            and optionally, energy, forces, distances, edges, and periodic boundary conditions.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(atoms.get_cell()).view(1, 3, 3)
        natoms = positions.shape[0]
        # initialized to torch.zeros(natoms) if tags missing.
        # https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_tags
        tags = torch.Tensor(atoms.get_tags())

        # put the minimum data in torch geometric data object
        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags,
        )

        # optionally include other properties
        if self.r_edges:
            # run internal functions to get padded indices and distances
            split_idx_dist = self._get_neighbors_pymatgen(atoms)
            edge_index, edge_distances, cell_offsets = self._reshape_features(
                *split_idx_dist
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.y = energy
        if self.r_forces:
            forces = torch.Tensor(atoms.get_forces(apply_constraint=False))
            data.force = forces
        if self.r_distances and self.r_edges:
            data.distances = edge_distances
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms

                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx
        if self.r_pbc:
            data.pbc = torch.tensor(atoms.pbc)

        return data

    def convert_all(
        self,
        atoms_collection,
        processed_file_path=None,
        collate_and_save=False,
        disable_tqdm=False,
    ):
        """Convert all atoms objects in a list or in an ase.db to graphs.
        Args:
            atoms_collection (list of ase.atoms.Atoms or ase.db.sqlite.SQLite3Database):
            Either a list of ASE atoms objects or an ASE database.
            processed_file_path (str):
            A string of the path to where the processed file will be written. Default is None.
            collate_and_save (bool): A boolean to collate and save or not. Default is False, so will not write a file.
        Returns:
            data_list (list of torch_geometric.data.Data):
            A list of torch geometric data objects containing molecular graph info and properties.
        """

        # list for all data
        data_list = []
        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
            
        elif isinstance(atoms_collection, ase.db.sqlite.SQLite3Database):
            atoms_iter = atoms_collection.select()
            
        elif isinstance(
            atoms_collection, ase.io.trajectory.SlicedTrajectory
        ) or isinstance(atoms_collection, ase.io.trajectory.TrajectoryReader):
            atoms_iter = atoms_collection
        else:
            raise NotImplementedError

        for atoms in tqdm(
            atoms_iter,
            desc="converting ASE atoms collection to graphs",
            total=len(atoms_collection),
            unit=" systems",
            disable=disable_tqdm,
        ):
            # check if atoms is an ASE Atoms object this for the ase.db case
            if not isinstance(atoms, ase.atoms.Atoms):
                atoms = atoms.toatoms()
            data = self.convert(atoms)
            data_list.append(data)

        if collate_and_save:
            data, slices = collate(data_list)
            torch.save((data, slices), processed_file_path)

        return data_list        
        
        
        
        
        
        
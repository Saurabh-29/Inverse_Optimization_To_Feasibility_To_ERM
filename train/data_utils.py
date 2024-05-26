import numpy as np
import torch
import pickle

class loadData():
    def __init__(self, data_path, cfg):
        super(loadData, self).__init__()
        """
        args:
            data_path: path to the data file
            cfg: configuration dict for model_params
        

        
        This function loads the data from the data_path and stores it in the class.
        """
        with open(data_path, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.loaded_dict = loaded_dict
        self.data_path = data_path

        self.train_data = loaded_dict["train"]
        self.valid_data = loaded_dict["valid"]
        self.test_data = loaded_dict["test"]

        if cfg["dataset"]["gen_binv"]:
            self.train_data["A_b"], self.train_data["A_n"], self.train_data["A_b_inv"], self.train_data["A_inv_grad"], self.train_data["b_inv_grad"] = self.get_A_b_n(self.train_data)
        
        if cfg["normalize_data"]:
            self.normalize_data()

    def normalize_data(self):
        if len(self.train_data["z"].shape)<4:
            return 
        mean = self.train_data["z"].mean(axis=(0,1,2), keepdims=True)
        std = self.train_data["z"].std(axis=(0,1,2), keepdims=True)
        self.train_data["z"] = (self.train_data["z"] - mean)/std
        self.valid_data["z"] = (self.valid_data["z"] - mean)/std
        self.test_data["z"] = (self.test_data["z"] - mean)/std
            
        
   
    def get_A_b_n(self, data_split):
        """
        args:
            data_split: data split (train, valid, test)

        This function takes the data split and returns the A_b, A_n, A_b_inv, A_inv_grad, b_inv_grad. 
        This was useful for the reduced cost gradient method but is not used for KKT formulation.
        
        returns A_b, A_n, A_b_inv, A_inv_grad, b_inv_grad
        """
        A = data_split["A"]
        A = torch.from_numpy(A).float()
        b = data_split["b"]
        basis = data_split["basis"]
        not_basis = data_split["not_basis"]

        a_basis_3d = np.expand_dims(basis, axis=1).repeat(A.shape[1], axis=1)
        a_basis_3d = torch.from_numpy(a_basis_3d).long()
        A_b = torch.gather(A, 2, a_basis_3d)

        a_non_basis_3d = np.expand_dims(not_basis, axis=1).repeat(A.shape[1], axis=1)
        a_non_basis_3d = torch.from_numpy(a_non_basis_3d).long()
        A_n = torch.gather(A, 2, a_non_basis_3d)

        A_b_inv = torch.inverse(A_b)

        A_inv_grad = torch.zeros(A.shape[0], not_basis.shape[1], A.shape[2]+1)
        b_inv_grad = torch.zeros(A.shape[0], not_basis.shape[1])
        for i in range(A.shape[0]):
            A_inv_grad[i, :, basis[i]] = torch.mm(torch.inverse(A_b[i]), A_n[i]).T
            for j, index in enumerate(not_basis[i]):
                A_inv_grad[i, j, index] = -1
        A_inv_grad[:, :, -1] = 1
        return A_b, A_n, A_b_inv, A_inv_grad, b_inv_grad

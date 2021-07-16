import json
import pdb

import numpy as np
import glob
import os
import ase
from dscribe import descriptors
import cfg
from tqdm import tqdm


class PepData:
    def __init__(self):
        self.names = []
        self.peptides = []
        self.eigen_CMs = []
        self.test_data_by_name = {}
        self.test_data_by_pep = {}

    def load_data(self, name, peptide, eigen_CM):
        self.names.append(name)
        self.peptides.append(peptide)
        self.eigen_CMs.append(eigen_CM)

    def get_len(self):
        return len(self.names)

    def split_train_test(self):
        train_indexes = []
        for i in tqdm(range(self.get_len())):
            flag = False
            for j in range(self.get_len()):
                if i == j:
                    continue
                if self.peptides[i] == self.peptides[j]:
                    flag = True
                    if str(self.peptides[i]) not in self.test_data_by_pep:
                        self.test_data_by_pep[str(self.peptides[i])] = {}
                        self.test_data_by_pep[str(self.peptides[i])]['peptide'] = self.peptides[i]
                        self.test_data_by_pep[str(self.peptides[i])]['eigen_CM'] = []
                        self.test_data_by_pep[str(self.peptides[i])]['name'] = []
                    self.test_data_by_pep[str(self.peptides[i])]['eigen_CM'].append(self.eigen_CMs[i])
                    self.test_data_by_pep[str(self.peptides[i])]['name'].append(self.names[i])
                if self.names[i][:4] == self.names[j][:4]:
                    flag = True
                    if self.names[i][:4] not in self.test_data_by_name:
                        self.test_data_by_name[self.names[i][:4]] = {}
                        self.test_data_by_name[self.names[i][:4]]['peptide'] = []
                        self.test_data_by_name[self.names[i][:4]]['eigen_CM'] = []
                        self.test_data_by_name[self.names[i][:4]]['name'] = []
                    self.test_data_by_name[self.names[i][:4]]['peptide'].append(self.peptides[i])
                    self.test_data_by_name[self.names[i][:4]]['eigen_CM'].append(self.eigen_CMs[i])
                    self.test_data_by_name[self.names[i][:4]]['name'].append(self.names[i])
            if not flag:
                train_indexes.append(i)

        self.names = np.array(self.names)[train_indexes].tolist()
        self.peptides = np.array(self.peptides)[train_indexes].tolist()
        self.eigen_CMs = np.array(self.eigen_CMs)[train_indexes].tolist()

        print(f'train_data length :{self.get_len()}')
        print(f'test_data_by_pep length :{len(self.test_data_by_pep)}')
        print(f'test_data_by_name length :{len(self.test_data_by_name)}')

    def save(self):
        self.split_train_test()
        data = {'names': self.names, 'peptides': self.peptides, 'eigen_CMs': self.eigen_CMs,
                'test_data_by_name': self.test_data_by_name, 'test_data_by_pep': self.test_data_by_pep}
        json.dump(data, open(cfg.pepbdb_processed,'w'))



def get_peptide_info(pdb_lines):
    """
    get peptide sequence adn atoms from pdb files
    :param pdb_lines: line list from peptide pdb file
    :return:
    peptide_info
        ['atoms'] atom list in peptide, to be used for receptor pocket retriever
        ['sequence'] peptide sequence, represented in their vocab code, padded and added <start>&<eos>
    """
    peptide_info = {}
    vector = []
    peptide_info['atoms'] = []
    sq_index = -9999999
    for line in pdb_lines:
        if len(line) >= 4 and line[:4] == 'ATOM':
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            peptide_info['atoms'].append([x, y, z])

            if line[13:15]=='CA':
                if sq_index != -9999999 and abs(int(line[22:26]) - sq_index) != 1:
                    raise Exception("peptide not continuous")
                sq_index = int(line[22:26])
                AA_str = line[17:20]
                if AA_str in cfg.AA_dict:
                    AA_type = cfg.AA_dict[AA_str]
                else:
                    AA_type = 0
                vector.append(AA_type)
        elif len(line) >= 3 and line[:3] == 'TER':
            break
    if len(vector) > cfg.pep_max_length_pepbdb-2:
        raise Exception("peptide too long")
    if len(vector) < 1:
        raise Exception("empty peptide")
    peptide_info['sequence'] = [1 for _ in range(cfg.pep_max_length_pepbdb)]
    peptide_info['sequence'][0] = 2
    peptide_info['sequence'][len(vector) + 1] = 3
    for i in range(len(vector)):
        peptide_info['sequence'][i + 1] = vector[i]

    return peptide_info


def get_pocket_info(peptide_atoms, pdb_lines):
    """
    get the pocket info when given peptide atom list and receptor pdb lines
    :param peptide_atoms: atom list from peptide
    :param pdb_lines: line list from receptor pdb file
    :return:
    pocket_info
        ['AA_types']: amino acid type list of receptor pocket
        ['atom_types']: atom type list of receptor pocket
        ['atoms']: atom list of receptor pocket

    """
    receptor_info = {}
    atom_list=[]
    receptor_info['AA_atoms'] = []
    receptor_info['CAs'] = []
    sq_index = -9999999
    for line in pdb_lines:
        if len(line) >= 4 and line[:4] == 'ATOM':
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if sq_index != -9999999 and int(line[22:26]) != sq_index:
                receptor_info['AA_atoms'].append(atom_list)
                atom_list = []
            sq_index = int(line[22:26])
            AA_str = line[17:20]
            if AA_str in cfg.AA_dict:
                AA_type = cfg.AA_dict[AA_str]
            else:
                AA_type = 0
            atom_type = line[13:17].strip()
            atom_list.append((AA_type, atom_type, [x,y,z]))
            if line[13:15]=='CA':
                receptor_info['CAs'].append([x,y,z])
        elif len(line) >= 3 and line[:3] == 'TER':
            if atom_list:
                receptor_info['AA_atoms'].append(atom_list)
                atom_list = []
            sq_index = -9999999
    if atom_list:
        receptor_info['AA_atoms'].append(atom_list)
    if len(receptor_info['AA_atoms']) != len(receptor_info['CAs']):
        raise Exception("receptor read error!")
    peptide_atoms = np.array(peptide_atoms)
    receptor_CAs = np.array(receptor_info['CAs'])
    distance_matrix = calc_dis(peptide_atoms, receptor_CAs)

    pocket_info = {}
    pocket_info['AA_types'] = []
    pocket_info['atom_types'] = []
    pocket_info['atoms'] = []
    for i in range(receptor_CAs.shape[0]):
        if np.min(distance_matrix[:, i]) < 6.5:
            for j in range(len(receptor_info['AA_atoms'][i])):
                pocket_info['AA_types'].append(receptor_info['AA_atoms'][i][j][0])
                pocket_info['atom_types'].append(receptor_info['AA_atoms'][i][j][1])
                pocket_info['atoms'].append(receptor_info['AA_atoms'][i][j][2])
    return pocket_info


def calc_dis(point_list1, point_list2):
    """
    calculate the distance between every point pair of point_list1 and point_list2
    :param point_list1: the first point list
    :param point_list2: the second point list
    :return:
    distance_matrix: the distance matrix whose element represent pairwise distance between point_list1 and point_list2
    """
    y = []
    for i in range(point_list1.__len__()):
        y.append(point_list2)
    y = np.array(y)
    x=[]
    for i in range(point_list2.__len__()):
        x.append(point_list1)
    x = np.array(x)
    x = x.transpose(1, 0, 2)
    distance_matrix = np.linalg.norm(np.array(x) - np.array(y), axis=2)
    return distance_matrix


def data_preprocess_pepbdb(save_path):
    """
    get peptide info, retrieve receptor pocket, and save them to save_path
    :param save_path: path to save peptide info and receptor pocket info
    :return:
    """
    print('get peptide info, retrieve receptor pocket, and save them to save_path')
    # data_by_name = {}
    # data_by_seq = {}
    pepdata = PepData()
    for j_file in tqdm(glob.glob(os.path.join(cfg.pepbdb_source, "*"))):
        name = j_file.split('/')[-1]
        try:
            with open(os.path.join(j_file, "peptide.pdb")) as peptide_file:
                peptide_info = get_peptide_info(peptide_file.readlines())
            with open(os.path.join(j_file, "receptor.pdb")) as receptor_file:
                pocket_info = get_pocket_info(peptide_info['atoms'], receptor_file.readlines())

            atoms = np.array(pocket_info['atoms'])
            symbols = []
            positions = []
            for i in range(len(atoms)):
                symbols.append(pocket_info['atom_types'][i][0])
                positions.append(pocket_info['atoms'][i])
            cm = descriptors.coulombmatrix.CoulombMatrix(cfg.EGCM_max_length, flatten=False)
            eigen_CM = np.linalg.eig(cm.create(ase.Atoms(symbols, positions)))[0].tolist()
            pepdata.load_data(name, peptide_info['sequence'], eigen_CM)


            # if str(peptide_info['sequence']) not in data:
            #     data[str(peptide_info['sequence'])] = {}
            #     data[str(peptide_info['sequence'])]['peptide'] = []
            #     data[str(peptide_info['sequence'])]['name'] = []
            #     data[str(peptide_info['sequence'])]['eigen_CM'] = []
            # data[str(peptide_info['sequence'])]['peptide'].append(peptide_info['sequence'])
            # data[str(peptide_info['sequence'])]['name'].append(name)
            # data[str(peptide_info['sequence'])]['eigen_CM'].append(eigen_CM)
            tqdm.write(f'{name}...success!')

            # TODO calculate adj_matrix, save pocket_info and adj_matrix to save_path for GCN training
            # adj_matrix = calc_dis(atoms, atoms)
            # adj_matrix = (adj_matrix < 1) + adj_matrix * (adj_matrix >= 1)
            # adj_matrix = 1 / adj_matrix
            # adj_matrix = adj_matrix.tolist()
            # data[name]['pocket'] = pocket_info
            # data[name]['adj_matrix'] = adj_matrix
        except Exception as e:
            tqdm.write(f'{name}...{str(e)}!')
    # print(f'data length:{len(data)}')
    # file = open(save_path, 'w')
    # json.dump(data, file)
    # file.close()
    pepdata.save()


def get_data_from_uniprot(pep_list, source_path):
    """
    get peptide list from uniprot
    :param pep_list: peptide list
    :param source_path: uniprot file source path
    :return:
    """
    peptide = ''
    for line in open(source_path).readlines():
        if line.startswith('>'):
            if 1 <= len(peptide) <= cfg.pep_max_length_uniprot -2:
                peptide_code = []
                for ch in peptide:
                    if ch in cfg.AA_abb_dict:
                        peptide_code.append(cfg.AA_abb_dict[ch])
                    else:
                        peptide_code.append(cfg.AA_abb_dict["<unk>"])
                padded_peptide = [1 for _ in range(cfg.pep_max_length_uniprot)]
                padded_peptide[0] = 2
                for i in range(len(peptide_code)):
                    padded_peptide[i+1] = peptide_code[i]
                padded_peptide[len(peptide) + 1] = 3
                pep_list.append(padded_peptide)
            peptide = ""
        else:
            peptide += line.strip()


def data_preprocess_uniprot(save_path):
    """
    get peptide list from uniprot, and save them to save_path
    :param save_path: path to save
    :return:
    """
    print('get peptide list from uniprot...')
    pep_list = []
    print('getting from review-yes...')
    get_data_from_uniprot(pep_list, cfg.uniprot_yes_source)
    print(f'total length:{len(pep_list)}')
    print('getting from review-no...')
    get_data_from_uniprot(pep_list, cfg.uniprot_no_source)
    print(f'total length:{len(pep_list)}')
    with open(save_path, 'w') as save_file:
        data = json.dumps(pep_list)
        save_file.write(data)


if __name__ == '__main__':
    if cfg.process_pepbdb:
        data_preprocess_pepbdb(cfg.pepbdb_processed)
    if cfg.process_uniprot:
        data_preprocess_uniprot(cfg.uniprot_processed)
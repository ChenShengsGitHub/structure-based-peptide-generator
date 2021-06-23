import numpy as np
import glob
import os

AA_dict={"ALA":1,"CYS":2,"ASP":3,"GLU":4,"PHE":5,"GLY":6,"HIS":7,"ILE":8,"LYS":9,"LEU":10,"MET":11,
         "ASN":12,"PRO":13,"GLN":14,"ARG":15,"SER":16,"THR":17,"VAL":18,"TRP":19,"TYR":20}

def get_info_from_pdb(pdb_lines):
    gt_info=dict()
    gt_info['CA_sites']=[]
    gt_info['atom_sites'] = []
    gt_info['AA_types'] = []
    gt_info['sq_index'] = []
    for line in pdb_lines:
        if len(line)>=4 and line[:4]=='ATOM':
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            gt_info['atom_sites'].append([x, y, z])
            if line[13:15]=='CA':
                gt_info['CA_sites'].append([x,y,z])
                AA_str = line[17:20]
                if AA_str in AA_dict:
                    AA_type = AA_dict[AA_str] - 1
                else:
                    AA_type = 20
                gt_info['AA_types'].append(AA_type)
                sq_index=int(line[22:26])
                gt_info['sq_index'].append(sq_index)


    gt_info['CA_sites'] = np.array(gt_info['CA_sites'])
    return gt_info

def calc_dis(distanceList1,distanceList2):
    y = []
    for i in range(distanceList1.__len__()):
        y.append(distanceList2)
    y = np.array(y)
    x=[]
    for i in range(distanceList2.__len__()):
        x.append(distanceList1)
    x = np.array(x)
    x = x.transpose(1, 0, 2)
    a = np.linalg.norm(np.array(x) - np.array(y), axis=2)
    return a

for j_file in glob.glob("/home/chens/data/pepbdb/pepbdb/*"):
    with open(os.path.join(j_file, "peptide.pdb")) as peptide_file:
        peptide_info = get_info_from_pdb(peptide_file.readlines())
    with open(os.path.join(j_file, "receptor.pdb")) as receptor_file:
        receptor_info = get_info_from_pdb(receptor_file.readlines())
    peptide_atoms = np.array(peptide_info['atom_sites'])
    receptor_CAs = np.array(peptide_info['CA_sites'])

    distance_matrix = calc_dis(peptide_atoms, receptor_CAs)
    print(peptide_atoms.shape, receptor_CAs.shape, distance_matrix.shape,)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cython: language_level=3

import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import *
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from NNCalculator import NNCalculator
from datetime import datetime
import json

import os
# 设置 OMP_NUM_THREADS 环境变量为 20
# os.environ['OMP_NUM_THREADS'] = '20'

def initialize_potential():
    # An example for atoms object, while it is used for initialization as well.

    # You should first verify the atom list Z of the molecule.
    Z = np.array([8, 6, 1, 1, 1, 6, 1, 1, 1, 8], dtype=np.int64)

    # Cartesian coordinates R are list following the atom list Z.
    # !!!!Be carfull! The example used is in angstrom, but the rpmd used bohr.
    R = np.array(
           [[      0.000000000000,    0.000000000000,    0.000000000000],
            [      1.411940000000,    0.000000000000,    0.000000000000],
            [      1.737578199947,    1.036437646381,    0.000000000000],
            [      1.806201858568,   -0.503036734286,   -0.889219791299],
            [      1.806201323636,   -0.503035904741,    0.889220497754],
            [     -0.499380786980,   -1.320679140895,   -0.000000000000],
            [     -1.584011456538,   -1.258701155658,   -0.000000000000],
            [     -0.168303202145,   -1.867373527968,    0.889220493670],
            [     -0.197047734893,   -1.819907131629,   -0.812013632203],
            [      1.163450809891,   -4.066433508339,   -4.466074790474]])


    # Construct the atoms object
    atoms = Atoms(numbers = Z, positions = R)

    # Initialize the NNCalculator
    global calc
    calc = NNCalculator(
        checkpoint=os.environ["PHYSNET_DIFF_CHECKPOINT"],
        atoms=atoms,
        charge=0,
        F=64,
        K=32,
        num_blocks=3,
        num_residual_atomic=2,
        num_residual_interaction=3,
        num_residual_output=1,
        sr_cut=6.0,
        use_electrostatic=True,
        use_dispersion=True,
        s6=1.0000,                    #s6 coefficient for d3 dispersion, by default is learned
        s8=2.3550,                    #s8 coefficient for d3 dispersion, by default is learned
        a1=0.5238,                    #a1 coefficient for d3 dispersion, by default is learned
        a2=3.5016)                   #a2 coefficient for d3 dispersion, by default is learned)

def get_potential(q, call_count):
#    global call_count
#    call_count += 1
#    log_call_count(call_count, "count_complete")

    Natoms = q.shape[1]
    Nbeads = q.shape[2]

    V = np.zeros(Nbeads)
    dVdq = np.zeros((Nbeads, Natoms, 3))

    # Transpose q(3, Natoms, Nbeads)(Bohr) into R(3, Natoms, Nbeads)(Angstrom).
    R = np.transpose(q, (2, 1, 0)) * 0.529177208
    Z = np.array([8, 6, 1, 1, 1, 6, 1, 1, 1, 8], dtype=np.int32)

    def calculate_angle(P1, P2, P3, P4):

        # 计算向量P1P2
        vector_P1P2 = P2 - P1

        # 计算向量P1P3和P1P4
        vector_P1P3 = P3 - P1
        vector_P1P4 = P4 - P1

        # 计算法线向量
        normal_vector = np.cross(vector_P1P3, vector_P1P4)

        # 计算向量P1P2与法线向量之间的夹角
        dot_product = np.dot(vector_P1P2, normal_vector)
        angle = np.arccos(abs(dot_product) / (np.linalg.norm(vector_P1P2) * np.linalg.norm(normal_vector)))

        # 转换为度
        angle_degrees = np.degrees(angle)
        angle_with_plane  = 90 - angle_degrees

        return angle_with_plane

    def constrain_ch(P1, P2):
        distance = np.linalg.norm(P1 - P2)

        # Check if the distance is greater than 2
        if distance > 2 or distance < 0.8:
            return 1
        else:
            return 0

    atoms_list = [Atoms(numbers=Z, positions=R[i]) for i in range(Nbeads)]
    energies, forces = calc.get_energy_forces_batch(atoms_list, need_forces=True)
    energies = np.array(energies, ndmin=1)
    forces = np.array(forces)
    if forces.ndim == 2:
        forces = forces[np.newaxis, ...]

    # !!!!Be careful! energy in PES uses kcal/mol, while rpmd requires hartree.
    for i in range(Nbeads):
        M = R[i]
        angle = calculate_angle(M[1], M[2], M[3], M[4])
        energy = energies[i]
        V[i] = energy/627.5094740631
        bond_info = [
            {"vector": M[1] - M[2], "start_atom": 1, "end_atom": 2, "eq_length": 1.09},
            {"vector": M[1] - M[3], "start_atom": 1, "end_atom": 3, "eq_length": 1.09},
            {"vector": M[1] - M[4], "start_atom": 1, "end_atom": 4, "eq_length": 1.09},
            {"vector": M[5] - M[6], "start_atom": 5, "end_atom": 6, "eq_length": 1.09},
            {"vector": M[5] - M[7], "start_atom": 5, "end_atom": 7, "eq_length": 1.09},
            {"vector": M[5] - M[8], "start_atom": 5, "end_atom": 8, "eq_length": 1.09},
        ]

        # 初始化梯度矩阵
        num_atoms = 10
        num_dimensions = 3
        constraint_potential = 0.0
        gradient_matrix = np.zeros((num_atoms, num_dimensions))
        # 计算键长、约束势能和势能梯度
        for bond_info_item in bond_info:
            bond_vector = bond_info_item["vector"]
            start_atom = bond_info_item["start_atom"]
            end_atom = bond_info_item["end_atom"]
            eq_length = bond_info_item["eq_length"]

            # 计算键长
            bond_length = np.linalg.norm(bond_vector)

            if constrain_ch(M[start_atom], M[end_atom]) == 1:

                # 计算约束势能
                constraint_potential += 100 * (bond_length - eq_length) ** 2

                # 计算势能梯度
                #gradient = 100 * 2 * (bond_length - eq_length) * (bond_vector / bond_length)

                # 将梯度加入梯度矩阵的起点和终点位置
                #gradient_matrix[start_atom] += gradient
                #gradient_matrix[end_atom] -= gradient

        #V[i] = (calc.get_potential_energy(atoms = atoms) + constraint_potential)/627.5094740631
        V[i] += constraint_potential/627.5094740631
        if angle < 10:
            V[i] += 100/627.5094740631
        E_final = V[i] * 627.5094740631

#        if E_final < -0.1:
#            with open("diff_E_check", "a") as log_file:
#                log_file.write(f"energy: {E_final} {angle} {constraint_potential}\n")
#                log_file.write(f"{M}\n")
#                log_file.write(f"{dVdq}\n")

        dVdq[i] = ((forces[i])*(-1) + gradient_matrix) * 0.529177208/627.5094740631

#        if call_count==0 or call_count % 100 == 0:
#            with open("diff_log", "a") as log_file:
#                log_file.write(f"energy: {E_final}   {call_count} \n")
#                log_file.write(f"{M}\n")
#                log_file.write(f"{dVdq}\n")


    # Transpose dVdq(3, Natoms, Nbeads) into dVdq(3, Natoms, Nbeads).
    dVdq = np.transpose(dVdq, (2, 1, 0))
    return V, dVdq

def log_call_count(count, count_logfile):
    # 将调用次数写入日志文件
    with open(count_logfile, "a") as log_file:
        log_file.write(f"{count}\n")

call_count = 0

for line in sys.stdin:
    data = json.loads(line.strip())
    command = data["command"]

    if command == 'initialize_potential':
        initialize_potential()
    elif command == 'get_potential':
        q = np.array(data["data"])
        V, dVdq = get_potential(q, call_count)
        call_count += 1
        result = json.dumps({"V": V.tolist(), "dVdq": dVdq.tolist()})
        sys.stdout.write(result + '\n')
        sys.stdout.flush()
    else:
        sys.stdout.write(json.dumps({"error": "Unknown command"}) + '\n')
        sys.stdout.flush()

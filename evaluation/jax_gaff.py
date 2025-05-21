import jax
import jax.numpy as jnp
import pickle as pkl
from functools import partial
from tqdm import tqdm

import os
import sys
import numpy as np

# 29 28 54 159 87
NMAX_ATOMS = 29
NMAX_BONDS = 28
NMAX_ANGLES = 54
NMAX_DIHEDRALS = 159
NMAX_NB14S = 87

FF_TARGET_SHAPE = {
    'C': (NMAX_ATOMS,),
    'S': (NMAX_ATOMS,),
    'E': (NMAX_ATOMS,),
    'Ba': (NMAX_BONDS,),
    'Bb': (NMAX_BONDS,),
    'Bk': (NMAX_BONDS,),
    'Br': (NMAX_BONDS,),
    'Aa': (NMAX_ANGLES,),
    'Ab': (NMAX_ANGLES,),
    'Ac': (NMAX_ANGLES,),
    'Ak': (NMAX_ANGLES,),
    'At': (NMAX_ANGLES,),
    'Da': (NMAX_DIHEDRALS,),
    'Db': (NMAX_DIHEDRALS,),
    'Dc': (NMAX_DIHEDRALS,),
    'Dd': (NMAX_DIHEDRALS,),
    'Dk': (NMAX_DIHEDRALS,),
    'Dp': (NMAX_DIHEDRALS,),
    'Dn': (NMAX_DIHEDRALS,),
    'Na': (NMAX_NB14S,),
    'Nb': (NMAX_NB14S,),
    'Nlf': (NMAX_NB14S,),
    'Nqf': (NMAX_NB14S,),
    'Ex': (NMAX_ATOMS, NMAX_ATOMS)
}

class GAFFForceFieldEnergy:
    """
    Vectorized force-field energy evaluator.
    Args
    ----
    n_atom : int
        Number of atoms in the system (per batch element).
    Notes
    -----
    • Expected shapes follow MindSpore’s code:
        crds            : (B, N, 3)
        lj_epsilon      : (B, N)
        lj_sigma        : (B, N)
        exclude         : (B, N, N)  – 1.0 for “count”, 0.0 for “ignore”
        bond_*          : (B, N_bond)
        angle_*         : (B, N_angle)
        dihedral_*      : (B, N_dihe)
    """
    def __init__(self, n_atom: int, eps=1e-12):
        self.n_atom = n_atom
        self.eps = eps
        # (1, N, N) mask with zeros on the diagonal, ones elsewhere
        self.temp_mask = (1.0 - jnp.eye(n_atom, dtype=jnp.float32))[None, :, :]

    # ---------------- core helpers ---------------- #
    @staticmethod
    def _scatter_add(vec, idx, upd):
        # idx : 1-D int array, same length as upd
        # returns a NEW array with values added at idx
        return vec.at[idx].add(upd)

    @staticmethod
    def _clip_unit(x):
        """Numerical safety for acos/atan2."""
        return jnp.clip(x, -0.999999, 0.999999)

    # ---------------- public API ------------------ #
    def __call__(
        self,
        crds, charge,
        lj_epsilon, lj_sigma, exclude,
        bond_a, bond_b, bond_k, bond_r0,
        angle_a, angle_b, angle_c, angle_k, angle_theta0,
        dih_a, dih_b, dih_c, dih_d, dih_n, dih_k, dih_phi0,
        nb14_a, nb14_b, nb14_lj_f, nb14_q_f, 
    ):
        B, N, _ = crds.shape
        f32 = crds.dtype

        # ---------- pair-wise geometry ---------- #
        disp = crds[:, :, None, :] - crds[:, None, :, :]        # (B,N,N,3)
        sqd  = jnp.sum(disp * disp, axis=-1)                    # (B,N,N)
        sqd  = jnp.where(sqd < self.eps, jnp.asarray(1e3, f32), sqd)
        inv_sqd = self.temp_mask / sqd + 1e-6
        dist = jnp.sqrt(sqd)                                    # (B,N,N)

        # ---------- Lennard-Jones --------------- #
        eps_ij    = jnp.sqrt(lj_epsilon[:, :, None] * lj_epsilon[:, None, :])
        sigma     = (lj_sigma[:, :, None] + lj_sigma[:, None, :]) * 0.5
        sig6      = sigma ** 6
        sig12     = sig6 * sig6
        r6        = inv_sqd ** 3
        r12       = r6 * r6
        pair_e_lj = 4.0 * eps_ij * (sig12 * r12 - sig6 * r6)
        pair_e    = jnp.where(exclude > 1e-12, pair_e_lj, 0.0)
        energy    = 0.5 * jnp.sum(pair_e, axis=(1, 2))                   # (B,)
        
        # ---------- Coulumbic --------------- #
        charge_ij = charge[:, :, None] * charge[:, None, :]
        pair_e_q  = charge_ij / dist
        pair_e    = jnp.where(exclude > 1e-12, pair_e_q, 0.0)
        energy    = energy + 0.5 * jnp.sum(pair_e, axis=(1, 2))

        # ---------- Bond energy ----------------- #
        # Gather distances once; then scatter-add
        bond_batch = jnp.repeat(jnp.arange(B), bond_k.shape[1])
        b_a = bond_a.reshape(-1)
        b_b = bond_b.reshape(-1)
        b_k = bond_k.reshape(-1)
        b_r0 = bond_r0.reshape(-1)
        bond_len = dist[bond_batch, b_a, b_b] - b_r0
        energy = self._scatter_add(energy, bond_batch,
                                   b_k * bond_len ** 2)
        

        # ---------- Angle energy ---------------- #
        ang_batch = jnp.repeat(jnp.arange(B), angle_k.shape[1])
        a_a = angle_a.reshape(-1)
        a_b = angle_b.reshape(-1)
        a_c = angle_c.reshape(-1)
        a_k = angle_k.reshape(-1)
        a_t0 = angle_theta0.reshape(-1)

        ab = disp[ang_batch, a_a, a_b]
        bc = disp[ang_batch, a_b, a_c]
        ab_dot_bc = -jnp.sum(ab * bc, axis=-1)
        ab_bc_norm = (dist[ang_batch, a_a, a_b] *
                      dist[ang_batch, a_b, a_c])
        cos_th = self._clip_unit(ab_dot_bc / ab_bc_norm)
        dtheta = jnp.arccos(cos_th) - a_t0
        energy = self._scatter_add(energy, ang_batch,
                                   a_k * dtheta ** 2)

        # ---------- Dihedral energy ------------- #
        dih_batch = jnp.repeat(jnp.arange(B), dih_k.shape[1])
        d_a = dih_a.reshape(-1)
        d_b = dih_b.reshape(-1)
        d_c = dih_c.reshape(-1)
        d_d = dih_d.reshape(-1)
        d_n = dih_n.reshape(-1)
        d_k = dih_k.reshape(-1)
        d_phi0 = dih_phi0.reshape(-1)

        # unit vectors along the three bonds
        v1 = disp[dih_batch, d_a, d_b] / dist[dih_batch, d_a, d_b][..., None]
        v2 = disp[dih_batch, d_b, d_c] / dist[dih_batch, d_b, d_c][..., None]
        v3 = disp[dih_batch, d_c, d_d] / dist[dih_batch, d_c, d_d][..., None]

        a_norm = jnp.cross(v2, v1)
        b_norm = jnp.cross(v3, v2)
        cross_ab = jnp.cross(a_norm, b_norm)

        sin_phi = jnp.sum(cross_ab * v2, axis=-1)
        cos_phi = jnp.sum(a_norm * b_norm, axis=-1)
        sin_phi = jnp.where(d_k > 1e-9, self._clip_unit(sin_phi), 1.0)
        cos_phi = jnp.where(d_k > 1e-9, self._clip_unit(cos_phi), 0.0)

        phi = jnp.arctan2(-sin_phi, cos_phi)
        dphi = d_n * phi - d_phi0
        energy = self._scatter_add(energy, dih_batch,
                                   d_k * (1.0 + jnp.cos(dphi)))
        
        # ---------- nb14 energy ------------- #
        # Gather distances once; then scatter-add
        nb14_batch = jnp.repeat(jnp.arange(B), nb14_lj_f.shape[1])
        n_a = nb14_a.reshape(-1)
        n_b = nb14_b.reshape(-1)
        n_lj_f = nb14_lj_f.reshape(-1)
        n_q_f = nb14_q_f.reshape(-1)
        energy = self._scatter_add(energy, nb14_batch, n_lj_f * pair_e_lj[nb14_batch, n_a, n_b] + n_q_f * pair_e_q[nb14_batch, n_a, n_b])

        return energy
    
def gaff_ene(crds, ff_inputs):
    return GAFFForceFieldEnergy(n_atom=NMAX_ATOMS)(crds, *ff_inputs)[0]

def gaff_ene_frc(crds, ff_inputs):    
    ene, grad = jax.value_and_grad(gaff_ene)(crds, ff_inputs)
    return ene, -grad

def parse_num_file(file):
    with open(file, 'r') as f:
        return [[float(x.strip()) for x in l.split()] for l in f.readlines() if l.strip() != '']
    
def convert_sponge_input_to_dict(prefix):    
    ff_param_dict = {}
    
    ### charge
    charge = np.loadtxt(prefix + '_charge.txt')
    
    ### charge
    charge = np.loadtxt(prefix + '_charge.txt')[1:]
    n_atoms = len(charge)
    ff_param_dict['n_atoms'] = n_atoms
    ff_param_dict['charge'] = charge

    ### exclude
    ff_param_dict['exclude'] = np.ones((n_atoms, n_atoms), dtype=np.int32) - np.identity(n_atoms, dtype=np.int32)
    exclude_items = parse_num_file(prefix + '_exclude.txt')
    for atom_i in range(n_atoms):
        for atom_j in exclude_items[1+atom_i][1:]:
            atom_j = int(atom_j)
            ff_param_dict['exclude'][atom_i][atom_j] = ff_param_dict['exclude'][atom_j][atom_i] = 0
            
    ### LJ 
    LJ_items = parse_num_file(prefix + '_LJ.txt')
    n_LJ_types = int(LJ_items[0][1])
    LJ_items = LJ_items[1:]
    C12, C6 = np.zeros([n_LJ_types,n_LJ_types ]), np.zeros([n_LJ_types,n_LJ_types ])
    for type_i in range(n_LJ_types):
        for type_j in range(type_i + 1):
            C12[type_i][type_j] = C12[type_j][type_i] = LJ_items[type_i][type_j]
    LJ_items = LJ_items[n_LJ_types:]
    for type_i in range(n_LJ_types):
        for type_j in range(type_i + 1):
            C6[type_i][type_j] = C6[type_j][type_i] = LJ_items[type_i][type_j]
    LJ_items = LJ_items[n_LJ_types:]
    diag_C12 = np.diag(C12)
    diag_C6 = np.diag(C6)
    lj_sigma = (diag_C12/ np.maximum(diag_C6, 1e-12)) ** (1.0 / 6.0)
    lj_epsilon = (diag_C6) / (4.0 * np.maximum(lj_sigma, 1e-12) ** 6)
    lj_sigma = lj_sigma[[int(x[0]) for x in LJ_items]]
    lj_epsilon = lj_epsilon[[int(x[0]) for x in LJ_items]]
    ff_param_dict['lj_epsilon'] = lj_epsilon
    ff_param_dict['lj_sigma'] = lj_sigma

    ### bond
    bond_items = np.array(parse_num_file(prefix + '_bond.txt')[1:])
    bond_a, bond_b, bond_k, bond_r0 = bond_items[:, 0].astype(np.int32), bond_items[:, 1].astype(np.int32), bond_items[:, 2].astype(np.float32), bond_items[:, 3].astype(np.float32)
    ff_param_dict['bond'] = {
        'a': bond_a, 'b': bond_b, 'k': bond_k, 'r0': bond_r0
    }

    ### angle
    angle_items = np.array(parse_num_file(prefix + '_angle.txt')[1:])
    angle_a, angle_b, angle_c, angle_k, angle_t0 = angle_items[:, 0].astype(np.int32), angle_items[:, 1].astype(np.int32), angle_items[:, 2].astype(np.int32), angle_items[:, 3].astype(np.float32), angle_items[:, 4].astype(np.float32)
    ff_param_dict['angle'] = {
        'a': angle_a, 'b': angle_b, 'c': angle_c, 'k': angle_k, 't0': angle_t0,
    }

    ### dihedral
    if os.path.exists(prefix + '_dihedral.txt'):
        dihedral_terms = np.array(parse_num_file(prefix + '_dihedral.txt')[1:])
        dihedral_a, dihedral_b, dihedral_c, dihedral_d, dihedral_p, dihedral_k, dihedral_p0 = dihedral_terms[:, 0].astype(np.int32),dihedral_terms[:, 1].astype(np.int32), dihedral_terms[:, 2].astype(np.int32), dihedral_terms[:, 3].astype(np.int32), dihedral_terms[:, 4].astype(np.int32), dihedral_terms[:, 5].astype(np.float32), dihedral_terms[:, 6].astype(np.float32)
        ff_param_dict['dihedral'] = {
            'a': dihedral_a, 'b': dihedral_b, 'c': dihedral_c, 'd': dihedral_d, 'k': dihedral_k, 'p0': dihedral_p0, 'p': dihedral_p,
        }
    else:
        ff_param_dict['dihedral'] = {
            'a': np.zeros(0, dtype=np.int32), 'b': np.zeros(0, dtype=np.int32), 'c': np.zeros(0, dtype=np.int32), 'd': np.zeros(0, dtype=np.int32), 'k': np.zeros(0, dtype=np.float32), 'p0': np.zeros(0, dtype=np.float32), 'p': np.zeros(0, dtype=np.int32),
        }
        
    ### nb14
    if os.path.exists(prefix + '_nb14.txt'):
        nb14_terms = np.array(parse_num_file(prefix + '_nb14.txt')[1:])
        nb14_a, nb14_b, nb14_lj_f, nb14_q_f = nb14_terms[:, 0].astype(np.int32), nb14_terms[:, 1].astype(np.int32), nb14_terms[:, 2].astype(np.float32), nb14_terms[:, 3].astype(np.float32)
        ff_param_dict['nb14'] = {
            'a': nb14_a, 'b': nb14_b, 'lj_f': nb14_lj_f, 'q_f': nb14_q_f,
        }
    else:
        ff_param_dict['nb14'] = {
            'a': np.zeros(0, dtype=np.int32), 'b': np.zeros(0, dtype=np.int32), 'lj_f': np.zeros(0, dtype=np.float32), 'q_f': np.zeros(0, dtype=np.float32)
        }
        
    ff_inputs = {
            'C': ff_param_dict['charge'], 
            'S': ff_param_dict['lj_sigma'],
            'E': ff_param_dict['lj_epsilon'],
            'Ba': ff_param_dict['bond']['a'],
            'Bb': ff_param_dict['bond']['b'],
            'Bk': ff_param_dict['bond']['k'],
            'Br': ff_param_dict['bond']['r0'],
            'Aa': ff_param_dict['angle']['a'],
            'Ab': ff_param_dict['angle']['b'],
            'Ac': ff_param_dict['angle']['c'],
            'Ak': ff_param_dict['angle']['k'],
            'At': ff_param_dict['angle']['t0'],
            'Da': ff_param_dict['dihedral']['a'],
            'Db': ff_param_dict['dihedral']['b'],
            'Dc': ff_param_dict['dihedral']['c'],
            'Dd': ff_param_dict['dihedral']['d'],
            'Dk': ff_param_dict['dihedral']['k'],
            'Dp': ff_param_dict['dihedral']['p0'],
            'Dn': ff_param_dict['dihedral']['p'],
            'Na': ff_param_dict['nb14']['a'], 
            'Nb': ff_param_dict['nb14']['b'],
            'Nlf': ff_param_dict['nb14']['lj_f'], 
            'Nqf': ff_param_dict['nb14']['q_f'], 
            'Ex': ff_param_dict['exclude']
        }
    
    return ff_inputs
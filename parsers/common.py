# Metric names for Kepler and Volta
import math
import os
import re

import pandas as pd

METRIC_NAMES = {
    # Double arithmetic
    "flop_count_dp_fma": "DFMA", "flop_count_dp_mul": "DMUL", "flop_count_dp_add": "DADD",
    # Single arithmetic
    "flop_count_sp_add": "FADD", "flop_count_sp_mul": "FMUL", "flop_count_sp_fma": "FFMA",
    "flop_count_sp_special": "FSPC",
    # # Half arithmetic
    "flop_count_hp_add": "HADD", "flop_count_hp_mul": "HMUL", "flop_count_hp_fma": "HFMA",
    # LD/ST, IF, and INT
    "inst_compute_ld_st": "LDST", "inst_control": "IF",
    # Communication, MISC, and conversion
    "inst_inter_thread_communication": "THCOMM", "inst_misc": "MISC", "inst_bit_convert": "BITCONV",
    # Atomic functions
    "atomic_transactions": "ATOMIC",
    # GPR
    "GPR": "GPR",
    # Tensor
    "tensor_count": "MMA",
    # Int
    "inst_integer": "INT"
}

RENAME_FLAGS = {
    '-Xptxas -O0 -Xcompiler -O0': 'O0',
    '-Xptxas -O1 -Xcompiler -O1': 'O1',
    '-Xptxas -O3 -Xcompiler -O3': 'O3',
    #  Control single-precision denormals support.
    '--ftz=true': 'FTZ-ON',
    '--ftz=false': 'FTZ-OFF',
    #  This option controls single-precision floating-point division and reciprocals.
    '--prec-div=true': 'PrecDiv-ON',
    '--prec-div=false': 'PrecDiv-OFF',
    #  This option controls single-precision floating-point square root.
    '--prec-sqrt=true': 'PrecSqrt-ON',
    '--prec-sqrt=false': 'PrecSqrt-OFF',
    #  This option enables (disables) the contraction of floating-point multiplies and adds/subtracts into
    #  floating-point multiply-add operations (FMAD, FFMA, or DFMA).
    '--fmad=true': 'FMAD-ON',
    '--fmad=false': 'FMAD-OFF',
    # Fast math --use_fast_math implies --ftz=true --prec-div=false --prec-sqrt=false --fmad=true.
    '--use_fast_math': 'FAST-MATH',
    # This option enables more aggressive device code vectorization.
    '--extra-device-vectorization': 'DevVec',
    # Max registers per thread
    '--maxrregcount=16': 'MinRF',
    '--maxrregcount=32': 'MinRF',
    '-Xptxas --allow-expensive-optimizations=true -Xcompiler --expensive-optimizations': 'ExpensiveOPT-ON',
    '-Xptxas --allow-expensive-optimizations=false -Xcompiler --no-expensive-optimizations': 'ExpensiveOPT-OFF',
}

BOARD_NAMES = {'NVIDIA Tesla K40c': "kepler", 'NVIDIA TITAN V': "volta", 'NVIDIA GeForce RTX 3060 Ti': "ampere"}

PARSED_FLAGS = {re.sub("-*=*[ ]*\"*", "", k): v for k, v in RENAME_FLAGS.items()}
INSTRUCTION_GROUP_ID = ["FP64", "FP32", "LD", "PR", "NODEST", "OTHERS", "GPPR", "GP"]

FP64_INSTRUCTIONS = [v for k, v in METRIC_NAMES.items() if "flop_count_dp" in k]
FP32_INSTRUCTIONS = [v for k, v in METRIC_NAMES.items() if "flop_count_sp" in k]
FP16_INSTRUCTIONS = [v for k, v in METRIC_NAMES.items() if "flop_count_hp" in k]
GP_INSTRUCTIONS = ["ATOMIC", "BITCONV", "IF", "INT", "THCOMM", "MISC"]
LD_INSTRUCTIONS = ["LDST"]

# Default sea flux
DEFAULT_SEA_FLUX = 13e9

"""
FOR DRAWING
"""
# Marker for the striplot
MARKER_OPTIONS = {
    'MinRF': {'marker': 'v', 'color': '#DBB40C',
              'rgba': (0.8588235294117647, 0.7058823529411765, 0.047058823529411764, 1.0)},
    'O0': {'marker': '^', 'color': 'olivedrab',
           'rgba': (0.4196078431372549, 0.5568627450980392, 0.13725490196078433, 1.0)},
    'O1': {'marker': '>', 'color': 'rebeccapurple',
           'rgba': (0.4, 0.2, 0.6, 1.0)},
    'O3': {'marker': 'X', 'color': 'blue',
           'rgba': (0.0, 0.0, 1.0, 1.0)},
    'FAST-MATH': {'marker': '<', 'color': 'aqua',
                  'rgba': (0.0, 1.0, 1.0, 1.0)},
    'FMAD-OFF': {'marker': 'p', 'color': 'lightslategray',
                 'rgba': (0.4666666666666667, 0.5333333333333333, 0.6, 1.0)},
    'FTZ-ON': {'marker': 'P', 'color': 'black',
               'rgba': (0.0, 0.0, 0.0, 1.0)},
    'PrecDiv-OFF': {'marker': 'D', 'color': 'darkorange',
                    'rgba': (1.0, 0.5490196078431373, 0.0, 1.0)},

    'PrecSqrt-OFF': {'marker': '8', 'color': 'darkviolet',
                     'rgba': (1.0, 0.5490196078431373, 0.0, 1.0)}
}

MARKERS_PLOTLY = {
    'MinRF': "triangle-down",
    'O0': "triangle-up",
    'O1': "triangle-right",
    'O3': "x",
    'FAST-MATH': "triangle-left",
    'FMAD-OFF': "pentagon",
    'FTZ-ON': "cross",
    'PrecDiv-OFF': "diamond",
    'PrecSqrt-OFF': "octagon"
}

REDUCED_APP_NAMES = {'ACCL': 'CCL', 'BFS': 'BFS', 'FCFD': 'CFD', 'FGAUSSIAN': 'GSS', 'FHOTSPOT': 'HST',
                     'FLAVA': 'LVA', 'FLUD': 'LUD', 'FMXM': 'GEMM', 'MERGESORT': 'MST', 'NW': 'NW'}

# MAP the instructions from the opcode to
# Floating point instructions
fadd_inst = {'FADD', 'FADD32I', 'FSWZADD', }
ffma_inst = {'FFMA', 'FFMA32I', }
fmul_inst = {'FMUL', 'FMUL32I', }

float_instructions = {'FCHK', 'FCMP', 'FMNMX', 'FSEL', 'FSET',
                      'FSETP', 'IPA', 'MUFU', 'RRO'}

# Integer Instructions
iadd_inst = {'IADD', 'IADD3', 'IADD32I', 'ISCADD', 'ISCADD32I', }
imul_inst = {'IMUL', 'IMUL32I', }
imad_inst = {'IMAD', 'IMAD32I', 'IMADSP', 'XMAD', 'IMNMX'}

integer_instructions = {'IDP', 'IDP4A', 'BFE', 'BFI', 'BMSK', 'BREV', 'FLO', 'ICMP', 'ISET', 'ISETP', 'LEA',
                        'LOP', 'LOP3', 'LOP32I', 'PLOP3', 'POPC', 'SHF', 'SHL', 'SHR'}

# MMA Instructions
mma_instructions = {'IMMA', 'HMMA'}
# Load/Store Instructions
mem_instructions = {'LD', 'LDC', 'LDG', 'LDL', 'LDS', 'ST', 'STG', 'STL', 'STS', 'MATCH', 'QSPC', 'ATOM', 'ATOMS',
                    'RED', 'CCTL', 'CCTLL', 'ERRBAR', 'MEMBAR', 'CCTLT', 'SUATOM', 'SULD', 'SURED', 'SUST', }

# # Miscellaneous Instructions
MISC_INSTRUCTIONS = ['NOP', 'CS2R', 'S2R', 'LEPC', 'B2R', 'BAR', 'R2B', 'VOTE', 'DEPBAR', 'GETCRSPTR', 'GETLMEMBASE',
                     'SETCRSPTR', 'SETLMEMBASE', 'PMTRIG', 'SETCTAID']

# Max errors to calc the error model
ERROR_UPPER_LIMIT = 40

# Default directories
PARSER_DIRECTORY = os.path.abspath(os.path.dirname("./"))
DATA_PATH = os.path.join(PARSER_DIRECTORY, "data")
KEPLER_PROFILE_DATABASE = os.path.join(DATA_PATH, "profile_arch_Kepler.csv")
VOLTA_PROFILE_DATABASE = os.path.join(DATA_PATH, "profile_arch_Volta.csv")
CROSS_SECTION_KEPLER = os.path.join(DATA_PATH, "cross_section_kepler.csv")
CROSS_SECTION_KEPLER_2021 = os.path.join(DATA_PATH, "ChiIR_06_2021_carolK401_cross_section.csv")
CROSS_SECTION_VOLTA = os.path.join(DATA_PATH, "cross_section_volta.csv")
FINAL_NVBITFI_DATABASE = os.path.join(DATA_PATH, "final_nvbitfi_processed.csv")
FINAL_PROFILE_DATABASE = os.path.join(DATA_PATH, "final_profile_processed.csv")
FINAL_SASS_COUNT_DATABASE = os.path.join(DATA_PATH, "final_sass_count.csv")


def load_profile_database(profile_database_path):
    profile_df = pd.read_csv(profile_database_path)
    profile_df["flag"] = profile_df["flag"].apply(lambda r: RENAME_FLAGS[r.replace('"', "")])
    profile_df = profile_df.rename(columns={"nvcc_version": "nvcc"})
    profile_df["nvcc"] = profile_df["nvcc"].astype(str)
    profile_df = profile_df.set_index(["board", "app", "nvcc", "flag"])
    profile_df = profile_df.dropna(axis="rows", how='all')
    profile_df = profile_df.rename(columns=METRIC_NAMES)
    return profile_df.fillna(0)


def calc_err_sdc(row):
    numerator = (1.96 * row["SDC"])
    if row["#SDC"] >= ERROR_UPPER_LIMIT:
        return numerator / (math.sqrt(row["#SDC"]))
    return numerator / row["#SDC"]


def calc_err_due(row):
    numerator = (1.96 * row["DUE"])
    if row["#DUE"] >= ERROR_UPPER_LIMIT:
        return numerator / (math.sqrt(row["#DUE"]))
    return numerator / row["#DUE"]

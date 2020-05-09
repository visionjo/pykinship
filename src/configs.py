from addict import Dict
from pathlib import Path

CONFIGS = None
f_config = "logs/configs.json"
config = Dict()
config.settings.abbr_gender = ["f", "m"]
config.settings.abbr_gender.sort()
config.settings.n_folds = 5
config.settings.rand_seed = 123
config.settings.types = [
    "bb",
    "ss",
    "sibs",
    "fd",
    "fs",
    "md",
    "ms",
    "gfgd",
    "gfgs",
    "gmgd",
    "gmgs",
]

config.path.d_db = "../../data/v0.1.3/"
config.path.fid = f"{config.path.d_db}FIDs/"

config.path.name.non_mid = "unrelated_and_nonfaces/"
config.path.data_table = f"{config.path.d_db}datatable.pkl"

config.path.features = f"{config.path.d_db}features/"
config.path.features_unrelated = f"{config.path.features}fiwunrelated-feats/"

config.path.images = f"{config.path.d_db}Images/"

config.path.features = f"{config.path.features}features.pkl"

config.path.lists = f"{config.path.d_db}lists/"
config.path.pairs = f"{config.path.lists}pairs/"
config.path.logs = f"{config.path.d_db}logs/"
config.path.f_log = f"{config.path.logs}fiw.log"

config.path.master_pairs_list = f"{config.path.lists}merged_pairs.csv"
config.path.master_pairs_list_pkl = f"{config.path.d_db}pairs_scored.pkl"
config.path.subject_lut = f"{config.path.lists}subject_lut.csv"
config.path.subject_lut_pkl = f"{config.path.lists}subject_lut.pkl"
config.path.image_lut_pkl = f"{config.path.lists}image_lut.pkl"
config.path.image_lut = f"{config.path.lists}image_lut.csv"

config.path.f_rid = f"{config.path.d_db}FIW_RIDs.csv"  # file path for relationship type look-up (i.e., RID table)
config.path.f_fid = f"{config.path.d_db}FIW_FIDs.csv"  # file path FID LUT
config.path.f_pid = f"{config.path.d_db}FIW_PIDs.csv"  # file path PID LUT

config.path.fn_mid = "mid.csv"  # filename: file storing FID labels for each families
config.path.fn_log = "fiwdb.log"  # filename: output log

config.path.outputs.cropped_faces = f"{config.path.db}faces-cropped-aligned/"
config.path.outputs.sdm = f"{config.path.output}signal_detection_models.pdf"

config.image.ext = ".jpg"

PATH_ROOT = Path("../data/fiw-mm").resolve()
PATH_DATA = Path(f"{PATH_ROOT}/FIDs-MM")
PATH_IMAGE = Path(f"{PATH_DATA}/visual/image")
PATH_VIDEO = Path(f"{PATH_DATA}/visual/video")
PATH_OUT = Path(f"{PATH_DATA}/visual/video-frames")

CONFIGS = Dict(config)

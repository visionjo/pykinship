from addict import Dict

CONFIGS = None
f_config = 'logs/configs.json'
config = Dict()
config.settings.abbr_gender = ['f', 'm']
config.settings.abbr_gender.sort()
config.settings.n_folds = 5
config.settings.rand_seed = 123
config.settings.types = ['bb', 'ss', 'sibs', 'fd', 'fs', 'md', 'ms', 'gfgd', 'gfgs', 'gmgd', 'gmgs']

config.path.d_db = '../data/v0.1.3/'
config.path.fid = config.path.db + 'FIDs/'

config.path.name.non_mid = 'unrelated_and_nonfaces/'
config.path.data_table = config.path.db + 'datatable.pkl'

config.path.features = config.path.db + 'features/'
config.path.features_unrelated = config.path.features + 'fiwunrelated-feats/'

config.path.images = config.path.db + 'Images/'

config.path.features = config.path.features + 'features.pkl'

config.path.lists = config.path.db + 'lists/'
config.path.pairs = config.path.lists + 'pairs/'
config.path.logs = config.path.db + 'logs/'
config.path.f_log = config.path.logs + 'fiw.log'

config.path.master_pairs_list = config.path.lists + 'merged_pairs.csv'
config.path.master_pairs_list_pkl = config.path.db + 'pairs_scored.pkl'
config.path.subject_lut = config.path.lists + 'subject_lut.csv'
config.path.subject_lut_pkl = config.path.lists + 'subject_lut.pkl'
config.path.image_lut_pkl = config.path.lists + 'image_lut.pkl'
config.path.image_lut = config.path.lists + 'image_lut.csv'

config.path.fn_mid = 'mid.csv'  # file storing FID labels for each families
config.path.fn_pid = 'PIDs.csv'  # master PID database file
config.path.fn_log = 'fiwdb.log'  # output log filename
config.path.fn_rid = 'FIW_RIDs.csv'  # file name for relationship type look-up (i.e., RID table)
config.path.fn_fid = 'FIW_FIDs.csv'  # master FID database file

config.path.output = config.path.db + 'faces-cropped-aligned/'

config.path.outputs.sdm = config.path.output + 'signal_detection_models.pdf'

config.image.ext = '.jpg'

CONFIGS = Dict(config)
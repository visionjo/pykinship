"""
Module of utilities for handling FIW DB.

Methods to download PIDs using URL, load Anns and LUTs, along with metadata (e.g., gender, mids, pair lists).

# TODO urllib.request to handle thrown exceptions <p>Error: HTTP Error 403: Forbidden</p>

"""
from __future__ import print_function

import csv
import glob
import operator
from pathlib import Path

import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# import pyfiw.helpers as helpers
# import utils.log as log
# from pyfiw import io
from src.configs import CONFIGS
from tools import log
# from tools import image as image_tools

# from utils.pairwise import nchoosek

logger = log.setup_custom_logger(
    __name__, f_log=CONFIGS.path.dbroot + "fiw-db.log", level=log.INFO
)


def parse_relationship_matrix(data):
    """
    :param
    """


def parse_siblings(data, my_log=logger):
    """
    Parse sibling pairs by referencing member ID LUT and relationship matrix.

    Siblings RID is 2 and sister, of course, are Females. Thus, these are the factors we use to identify pairs.

    :param kind:
    :param data:
    :param my_log:
    :return:
    """
    # family directories
    my_log.info("{} families are being processed".format(len(data)))
    # Load MID LUT for all FIDs.

    # # Load relationship matrices for all FIDs.
    # df_relationships = load_relationship_matrices(dirs_fid, f_csv=f_rel_matrix)

    if kind == "brothers" or kind == "b":
        gender = "m"
    elif kind == "sisters":
        gender = "f"
    elif kind == "siblings":
        gender = ["m", "f"]
    else:
        my_log.error("Not a type of sibling")
        return

    # dfs = {f: df for f, df in df_families.items() if np.any(df.Gender==genders)}

    data = {
        f: df for f, df in data.items() if len(np.where(df.iloc[:, :-2].values == 2)[0])
    }
    siblings = []
    brothers = []
    sisters = []
    for fid, df in data.items():
        # ids = [i for i, s in enumerate(genders) if 'Male' in s]
        # rel_mat = FiwDB.parse_relationship_matrices(df)
        rel_mat = df.iloc[:, :-2].values
        genders = list(df.Gender)
        success, genders = helpers.check_gender_label(genders)
        if not success:
            my_log.error("Gender notation incorrect for {}".format(fid))
        # zero out female subjects
        # rel_mat = FiwDB.specify_gender(rel_mat, genders, gender)

        ids = np.where(rel_mat == 2)
        ids = [
            (id[0], id[1]) if id[0] < id[1] else (id[1], id[0])
            for id in zip(list(ids[0]), list(ids[1]))
        ]
        ids = list({id: id for id in ids}.values())
        for id in ids:
            if genders[id[0]] == "m" and genders[id[1]] == "m":
                brothers.append(
                    Pair(mid_pair=[df.index[i] for i in id], fid=fid, kind="brother")
                )
            elif genders[id[0]] == "f" and genders[id[1]] == "f":
                sisters.append(
                    Pair(mid_pair=[df.index[i] for i in id], fid=fid, kind="sister")
                )
            else:
                siblings.append(
                    Pair(mid_pair=[df.index[i] for i in id], fid=fid, kind="sibling")
                )

    return siblings


def gen_read_files(csv_files):
    """generator to read a file at a time"""
    for csvfile in csv_files:
        if io.is_file(csvfile):
            data = pd.read_csv(csvfile)
            yield data
        else:
            print("Unable to locate {}\n Skipping...".format(csvfile))
            continue


class Subject(object):
    # Class members
    dir_root = None
    name = None
    family = None
    image_paths = []
    n_images = 0

    def __init__(self, dir_db):
        self.dir_root = dir_db
        self.name = io.file_base(dir_db)
        self.family = io.file_base(io.parent_dir(dir_db))
        self.image_paths = glob.glob(dir_db + "/*.jpg")
        self.n_images = len(self.image_paths)

    def __len__(self):
        return self.n_images

    def __repr__(self):
        return "{} ({}) has {} images".format(self.name, self.family, self.n_images)


class Pair(object):
    def __init__(self, mid_piar, fid, kind=""):
        self.mid_piar = mid_piar
        self.fid = fid
        self.type = kind

    def __str__(self):
        return "FID: {} ; MIDS: ({}, {}) ; Type: {}".format(
            self.fid, self.mid_piar[0], self.mid_piar[1], self.type
        )

    def __key(self):
        return self.mid_piar[0], self.mid_piar[1], self.fid, self.type

    # def __eq__(self, other):
    #     return self.fid == other.fid and self.mids[0] == other.mids[0] and self.mids[1] == other.mids[1]

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    def __lt__(self, other):
        return np.uint(self.fid[1::]) < np.uint(other.fid[1::])


class FiwDB(object):
    """
    Class for FIW DB.

    Methods to download data, load and parse family labels, along with split by relationship types.

    Additionally, methods to generate summaries of the state of data (i.e., FIDs, MIDs, No. faces, etc.,)
    """

    fn_mid = CONFIGS.path.fn_mid  # file storing FID labels for each families
    f_pid = CONFIGS.path.fn_pid  # master PID database file
    fn_log = CONFIGS.path.fn_log  # output log filename
    f_rid = CONFIGS.path.f_rid  # file name for relationship look-up (RID table)
    f_fid = CONFIGS.path.fn_fid  # master FID database file

    def __init__(self, dir_db, dir_fids, dir_pid):
        self.dir_db = dir_db
        self.dir_fid = dir_fids
        self.dir_pid = dir_pid
        self.file_pid_master = glob.glob(dir_pid + "/*.csv")
        self.logger = log.setup_custom_logger(
            __name__, f_log=self.fn_log, level=log.INFO
        )
        self.logger.info("FIW-DB")

    def load_pid_set(self):
        """ load all pid labels in Ann directory"""
        self.file_pid_master = glob.glob(self.dir_pid + "/*.csv")
        return {
            f.replace(self.dir_pid, "")[:-4]: pd.read_csv(f)
            for f in self.file_pid_master
        }

    def get_pid(self):
        files_pid = glob.glob(self.dir_pid + "P*")
        for f in files_pid:
            yield pd.read_csv(f), f.replace(self.dir_pid, "")[:-4]

    def load_all_fids(self):
        """ load all FID labels in Ann directory"""
        files_fid = glob.glob(self.dir_fid + "F*")
        return {f.replace(self.dir_fid, "")[:-4]: pd.read_csv(f) for f in files_fid}

    def get_fid(self):
        files_fid = glob.glob(self.dir_fid + "F*")
        for f in files_fid:
            yield pd.read_csv(f)

    def load_pid_lut(self):
        """  Load PID LUT and return as pd.DataFrame() """
        return pd.read_csv(self.file_pid_master, delimiter="\t")

    def download_images(self, dir_out="fiw-raw-images/"):
        """
        Download FIW database by referencing PID LUT. Each PID is listed with corresponding URL. URL is downloaded and
        saved as <FID>/PID.jpg
        :type dir_out: object
        """

        self.logger.info(
            "FIW-DB-- Download_images!\n Source: {}\n Destination: {}".format(
                self.file_pid_master, dir_out
            )
        )
        # load urls (image location), pids (image name), and fids (output subfolder)
        df_pid = self.load_pid_lut()

        df_io = df_pid[["FIDs", "PIDs", "URL"]]

        self.logger.info("{} photos to download".format(int(df_io.count().mean())))

        for i, img_url in enumerate(df_io["URL"]):
            try:
                f_out = (
                    str(dir_out) + df_io["FIDs"][i] + "/" + df_io["PIDs"][i] + ".jpg"
                )
                img = image_tools.url_to_image(img_url)
                self.logger.info(
                    "Downloading {}\n{}\n".format(df_io["PIDs"][i], img_url)
                )
                image_tools.write(f_out, img)
            except Exception as e0:
                self.logger.error(
                    "Error with {}\n{}\n".format(df_io["PIDs"][i], img_url)
                )
                error_message = "<p>Error: %s</p>\n" % str(e0)
                self.logger.error(error_message)

    @staticmethod
    def get_unique_pairs(ids_in):
        """ Return list of pairs without repeating instances. """
        ids = [
            (p1, p2) if p1 < p2 else (p2, p1)
            for p1, p2 in zip(list(ids_in[0]), list(ids_in[1]))
        ]
        return list(set(ids))

    def get_rid_lut(self):
        rids = pd.read_csv(self.f_rid, delimiter=",")
        return [
            (rids.RID[1], rids.RID[1]),  # siblings
            (rids.RID[0], rids.RID[3]),  # parent-child
            (rids.RID[2], rids.RID[5]),  # grandparent-grandchild
            (rids.RID[4], rids.RID[4]),  # spouses
            (rids.RID[6], rids.RID[7]),
        ]  # great-grandparent-great-grandchild

    def get_fid_lut(self):
        """ Load FIW_FIDs.csv-- FID- Surname LUT """
        return pd.read_csv(self.f_fid, delimiter="\t")

    def get_fid_list(self):
        """
        Function loads fid directories / labels.
        :return: (list, list):  (fid file paths and fid labels of these)
        """
        dirs = glob.glob(self.dir_fid + "/F????/")

        return [(str(Path(d).parent.name)) for d in dirs]

    def load_mids(self):
        """
        Load CSV file containing member information, i.e., {MID : ID, Name, Gender}
        """
        files_fid = glob.glob(self.dir_fid + "/F*/" + self.fn_mid)
        return [pd.read_csv(f) for f in files_fid]

    def load_relationship_matrices(self):
        """
        Load CSV file containing member information, i.e., {MID : ID, Name, Gender}
        :return:
        """
        files_fid = glob.glob(self.dir_fid + "/F*/" + self.fn_mid)

        iter_df_fids = gen_read_files(files_fid)
        # df_relationships = [pd.read_csv(f) for f in files_fid]
        list_df_relationships = []
        for i, df in enumerate(iter_df_fids):
            # df_relationships = [content.ix[:, 1:len(content) + 1] for content in df_rel_contents]

            df.index = range(1, len(df) + 1)
            df = df.ix[:, 1 : len(df) + 1]
            list_df_relationships.append(df)
        return list_df_relationships

    def get_relationship_dictionaries(self):
        """
        Load CSV file containing member information
        :return: {MID : ID, Name, Gender}
        """
        fids = self.get_fid_list()
        dict_relationships = {}
        for i, fid in enumerate(fids):
            df_relationships = pd.read_csv(self.dir_fid + "/" + fid + "/" + self.fn_mid)
            df_relationships.index = range(1, len(df_relationships) + 1)
            df_relationships = df_relationships.ix[:, 1 : len(df_relationships) + 1]
            dict_relationships[fid] = df_relationships
        return dict_relationships

    def get_names_dictionaries(self):
        """ Load CSV file containing member information
            :return: {MID : ID, Name, Gender}
        """
        fids = self.get_fid_list()
        dict_names = {}
        for i, fid in enumerate(fids):
            df_relationships = pd.read_csv(self.dir_fid + "/" + fid + "/" + self.fn_mid)
            dict_names[fid] = list(df_relationships["Name"])

        return dict_names

    def set_pairs(self, ids_in, kind, fid):
        """
        Adds items to list of unique pairs.
        :param ids_in:
        :param kind:
        :param fid:
        :return:
        """
        ids = self.get_unique_pairs(ids_in)
        indices = []
        for i in enumerate(ids):
            print(i)
            indices = list(np.array(i[1]) + 1)
            # del indices
        return Pair(indices, fid, kind)

    @staticmethod
    def specify_gender(rel_mat, genders, gender):
        """
        :param rel_mat:
        :param genders: list of genders
        :param gender:  gender to search for {'Male' or Female}
        :type gender:   str
        :return:
        """
        ids_not = [j for j, s in enumerate(genders) if gender not in s]
        rel_mat[ids_not, :] = 0
        rel_mat[:, ids_not] = 0

        return rel_mat

    @staticmethod
    def folds_to_sets(
        d_csv="journal_data/Pairs/folds_5splits/", dir_save="journal_data/Pairs/sets/"
    ):
        """ Method used to merge 5 fold splits into 3 sets for RFIW (train, val, and test)"""

        f_in = glob.glob(d_csv + "*-folds.csv")

        for file in f_in:
            # each list of pairs <FOLD, LABEL, PAIR_1, PAIR_2>
            f_name = Path(file).name[:-4]
            print("\nProcessing {}\n".format(f_name))

            df_pairs = pd.read_csv(file)

            # merge to form train set
            df_train = df_pairs[(df_pairs["fold"] == 1) | (df_pairs["fold"] == 5)]
            df_train.to_csv(
                dir_save + "train/" + f_name.replace("-folds", "-train") + ".csv"
            )

            # merge to form val set
            df_val = df_pairs[(df_pairs["fold"] == 2) | (df_pairs["fold"] == 4)]
            df_val.to_csv(dir_save + "val/" + f_name.replace("-folds", "-val") + ".csv")

            # merge to form test set
            df_test = df_pairs[(df_pairs["fold"] == 3)]
            df_test.to_csv(
                dir_save + "test/" + f_name.replace("-folds", "-test") + ".csv"
            )

            # print stats
            print(
                "{} Training;\t {} Val;\t{} Test".format(
                    df_train["fold"].count(),
                    df_val["fold"].count(),
                    df_test["fold"].count(),
                )
            )

    @staticmethod
    def parse_relationship_matrices(mids):
        """ Parses out relationship matrix from MID dataframe.  """
        # df_relationships = [content.ix[:, 1:len(content) + 1] for content in df_rel_contents]
        if isinstance(mids, type(pd.DataFrame())):
            df_relationships = mids
        elif Path(mids).is_file():
            df_relationships = pd.read_csv(mids)
        else:
            print(
                "ERROR: pass in mids.csv loaded as pd.DataFrame or as filepath to CSV file."
            )
            return None

        df_relationships.index = range(1, len(df_relationships) + 1)

        df_relationships = df_relationships.ix[:, 1: len(df_relationships) + 1]

        return df_relationships.values

    def parsing_families(self):
        import pdb

        pdb.set_trace()
        fid_list = self.get_fid_list()
        fid_list.sort()
        tup_mid = [(d[-6:-1], pd.read_csv(d + "/" + self.fn_mid)) for d in fid_list]

        n_members = [int(mid[1].count().mean()) for mid in tup_mid]

        # ids = np.array(n_members).__lt__(3)
        df_mid = [(mid[1], mid[0]) for mid in tup_mid]

        fam_list = []
        fam_list2 = []
        tr_mids = []
        for i, df in enumerate(df_mid):
            vals = df[0].values[:, 1:-2]
            votes = np.zeros_like(vals)
            votes[(vals == 4) | (vals == 1)] += 3
            votes[(vals == 2)] += 2
            votes[(vals == 6) | (vals == 3)] += 1
            mid_list = np.linspace(1, vals.shape[0], num=vals.shape[0])

            max_index, max_value = max(
                enumerate(vals.sum(axis=1)), key=operator.itemgetter(1)
            )
            vals[max_index] = 0
            mid_list[max_index] = 0
            max_index2, max_value2 = max(
                enumerate(vals.sum(axis=1)), key=operator.itemgetter(1)
            )
            mid_list[max_index2] = 0
            nrelationships = np.size(np.nonzero(votes[max_index, :]))

            fam_list.append((zip(*df), max_index, max_value, nrelationships))

            mlist = ("MID" + str(int(mid)) for mid in mid_list if mid > 0)

            if nrelationships >= 3:
                fam_list2.append(
                    [
                        df[1],
                        "MID" + str(1 + max_index),
                        max_value,
                        nrelationships,
                        "MID" + str(1 + max_index2),
                        max_value2,
                        np.size(np.nonzero(votes[max_index2, :])),
                        zip(*mlist),
                    ]
                )
                tr_mids.append((df[1], mlist))

        ofile = open("test.csv", "w")

        writer = csv.writer(ofile, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)

        for row in fam_list2:
            writer.writerow(row)
        ofile.close()

        for fl in fam_list2:
            print(fl)

        # df_fams = pd.DataFrame(fam_list, columns=['rel', 'MID', 'val', 'nrel'])
        te_list = []
        tr_list = []
        val_list = []
        for tt in fam_list2:
            te_list.append(
                glob.glob(CONFIGS.path.dfid + tt[0] + "/" + tt[1] + "/*.jpg")
            )

        for tt in fam_list2:
            val_list.append(
                glob.glob(CONFIGS.path.dfid + tt[0] + "/" + tt[4] + "/*.jpg")
            )

        for tt in fam_list2:
            for ttt in list(tt[7:]):
                tr_list.append(
                    glob.glob(CONFIGS.path.dfid + tt[0] + "/" + ttt + "/*.jpg")
                )

        with open("test_no_labels.list", "w") as f:
            for _list in te_list:
                if len(_list) == 0:
                    continue
                fid = _list[0][69:74]
                # for _string in _list:
                for token in _list:
                    f.write(str(token) + " " + fid + "\n")

        with open("val_no_labels.list", "w") as f:
            for _list in val_list:
                if len(_list) == 0:
                    continue
                fid = _list[0][69:74]
                # for _string in _list:
                for token in _list:
                    f.write(str(token) + " " + fid + "\n")

        with open("train.list", "w") as f:
            for _list in tr_list:
                if len(_list) == 0:
                    continue
                fid = _list[0][69:74]
                # for _string in _list:
                for token in _list:
                    f.write(str(token) + " " + fid + "\n")

def write_meta_lists():
    """
    Create lists of images, fids, and subjects.
    :param do_save: Save list files in CONFIGS.path.dlists
    :return:
    """
    df_master = pd.DataFrame(data=None, columns=['id', 'ref', 'mid', 'family'])
    fiw_handle = FiwDB(CONFIGS.path.d_db, CONFIGS.path.fid, CONFIGS.path.f_pid)

    mid_paths = [str(path) for path in Path(CONFIGS.path.fid).glob('F????/MID*')]
    n_subjects = len(mid_paths)
    df_master.loc[:, 'id'] = np.arange(n_subjects)

    df_master.loc[:, 'ref'] = mid_paths
    df_master['ref']=df_master['ref'].str.replace(CONFIGS.path.fid, '')

    df_master['mid'] = df_master['ref'].apply(lambda x: x.split('/')[1].lower())
    df_master['family'] = df_master['ref'].apply(lambda x: x.split('/')[0].lower())

    # df_master.loc[:, 'family'] = [s.split('/')[0].lower() for s in subjects]
    # df_master.loc[:, 'gender'] = [s.split('/')[0].split('_')[1][0] for s in subjects]

    if Path(CONFIGS.lists).joinpath('name_list.csv').is_file():
        shutil.move(Path(CONFIGS.lists).joinpath('name_list.csv'),Path(CONFIGS.lists).joinpath('name_list_old.csv'))
    df_master.to_csv(Path(CONFIGS.lists).joinpath('name_list.csv'), index=False)


    image_list = samples.ref
    ids = []
    for image_instance in image_list:
        id = df_master.loc[df_master.ref == io.parent_dir(image_instance), 'id'].values
        ids.append(id[0])

    df_image_list = pd.DataFrame(data=None, columns=['id', 'impath', 'family'])
    df_image_list.loc[:, 'id'] = ids

    df_image_list.loc[:, 'impath'] = image_list
    df_image_list.loc[:, 'family'] = samples.family
    df_image_list.loc[:, 'mid'] = ids

    # df_image_list.loc[:, 'gender'] = [s.split('/')[0].split('_')[1][0] for s in df_image_list['impath']]
    if Path(CONFIGS.lists).joinpath('image_list.csv').is_file():
        shutil.move(Path(CONFIGS.lists).joinpath('image_list.csv'),Path(CONFIGS.lists).joinpath('image_list_old.csv'))
    df_image_list.to_csv(Path(CONFIGS.lists).joinpath('image_list.csv'), index=False)



    dfids = [d for d in io.dir_list(CONFIGS.path.dfid) if d.split('/')[-1][0] == 'F']
    dfids.sort()

    df_fids_list = pd.DataFrame(data=None, columns=['fid', 'nmids', 'nfaces'])
    df_fids_list.fid = [d.split('/')[-1] for d in dfids]
    df_fids_list.nmids = df_fids_list.fid.apply(lambda x: len(glob.glob(CONFIGS.path.dfid + x + '/M*')))
    df_fids_list.nfaces = df_fids_list.fid.apply(lambda x: len(glob.glob(CONFIGS.path.dfid + x + '/M*/*.jpg')))

    if Path(CONFIGS.lists).joinpath('fid_list.csv').is_file():
        shutil.move(Path(CONFIGS.lists).joinpath('fid_list.csv'),Path(CONFIGS.lists).joinpath('fid_list_old.csv'))
    df_fids_list.to_csv(Path(CONFIGS.lists).joinpath('fid_list.csv'), index=False)

    return df_master, df_image_list, df_fids_list


if __name__ == "__main__":
    do_lists = False
    do_splits = False
    do_merge = False
    do_add_negatives = False

    dir_data = "/home/jrobby/Documents/pykinship/data/v0.1.3/"
    dir_fid = f"{dir_data}FIDs/"
    f_fid_list = f"{dir_data}lists/fid_list.csv"
    write_meta_lists()
    # ml = DatabaseHandle(dir_fid, f_fid_list)
    # if do_lists:
    #     df_master, df_image_list, df_fid_list =
    # if do_splits:
    #     ml.split_fids_in_kfolds()
    # if do_merge:
    #     df_mid, df_images = ml.merge_pair_files()
    # elif do_add_negatives:
    #     df = ml.add_negative_pairs_to_folds()

    mid_lut = {
        f: pd.read_csv(f + "mid.csv")
        for f in glob.glob(dir_fid + "F????/")
        if Path(f"{f}mid.csv").is_file()
    }
    siblings = parse_siblings(mid_lut)

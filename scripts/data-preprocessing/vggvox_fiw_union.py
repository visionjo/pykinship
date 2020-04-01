import pandas as pd

f_fiw_list = '../../data/family_members.csv'
f_vggvox = '../../data/vox1_meta.csv'
f_db_union = '../../data/vggface_fiw_name_overlap.csv'
df_fiw = pd.read_csv(f_fiw_list)

# del df_fiw[df_fiw.columns[-1]]
# del df_fiw[df_fiw.columns[-1]]

df_fiw['lastname'] = df_fiw['surname'].apply(lambda x: x.split('.')[0])
df_fiw['mid'] = df_fiw['mid'].astype(str)
df_fiw['fid.mid'] = df_fiw['fid'] + '.' + df_fiw['mid']
df_fiw['overlap'] = 0
df_fiw['vgg_id'] = -1

df_vgg = pd.read_csv(f_vggvox, sep='\t')
df_vgg['lastname'] = df_vgg['VGGFace1 ID'].apply(lambda x: x.split('_')[-1]).str.lower()
df_vgg['firstname'] = df_vgg['VGGFace1 ID'].apply(lambda x: x.split('_')[0]).str.lower()


def merge_lists(row, df):
    df_fiw.loc[df_fiw['fid.mid'] == row['fid.mid'], 'overlap'] = 1
    df_fiw.loc[df_fiw['fid.mid'] == row['fid.mid'], 'vgg_id'] = row['db2.id']


df_union = pd.read_csv(f_db_union)
df_union.apply(lambda x: merge_lists(x, df_fiw), axis=1)

df_overlapped = df_fiw.loc[df_fiw.overlap > 0]

pd.concat([df_overlapped, df4.reindex(df_overlapped.index)], axis=1)

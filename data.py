import pandas as pd
import numpy as np

df = pd.read_csv('dataAll.csv',dtype = {'ZIP':str}, low_memory=False)
df = df[df.ZIP.notna()]



### Features

BINFEAT = []
CONTFEAT = []

df['LOS'].describe()

df = df[df.LOS.notna()]

#Lenght of Stay
df['DischDt'] = pd.to_datetime(df['DischDt'], format = '%d%b%Y')
df['AdmitDt'] = pd.to_datetime(df['AdmitDt'], format = '%d%b%Y')
df['los'] = df['DischDt'] - df['AdmitDt']

print(sum(df['los'] > pd.Timedelta('3 days'))/df.shape[0])

print(sum(df['los'] > pd.Timedelta('7 days'))/df.shape[0])



#Age  cont

AGE = ['AGEi']
df.AGEi.isna().sum()
#df.AGEi.hist()

CONTFEAT+=AGE

#AgeGroup
df['AHA_AGEgroup'].value_counts()

#Gender
#print(df['GENDERi'].isna().sum())
#df['GENDERi'].value_counts()
GENDER = ['GENDER_M','GENDER_F','GENDER_U']
df[GENDER] = pd.get_dummies(df['GENDERi'], dummy_na = True)
df[GENDER].sum()

BINFEAT+=GENDER

#insurnce Others, M'd, M'e, Unknown/Missing

INSURANCE = ['INSURANCE_O','INSURANCE_Md', 'INSURANCE_Me', 'INSURANCE_U']
df[INSURANCE] = pd.get_dummies(df['INSURANCEi'].fillna(4.0))#.sum()
df[INSURANCE].sum()

BINFEAT+=INSURANCE

#race
RACE = ['RACE_W','RACE_B','RACE_H','RACE_A','RACE_O']
df[RACE] = pd.get_dummies(df['race2i'].fillna(5.0))
df[RACE].sum()

BINFEAT+=RACE

df.shape

# Social Deprivation Index (SDI) cont
#0.0 is filled to 0
sdi = pd.read_csv('/mnt/workspace/uploaded_files/researcher_data/sdi.csv', dtype = {'zcta': str},low_memory=False)
sdi.zcta = sdi.zcta.apply(lambda z: '0'*(5-len(z))+z)
df['ZIP5'] = df['ZIP'].apply(lambda z: z[:5] if pd.notnull(z) else z)
df['SDI'] = df.merge(sdi[['zcta','sdi_score']], how = 'left', left_on = 'ZIP5', right_on = 'zcta')['sdi_score']
#df['SDI'] = df['SDI'].fillna(0.0)

CONTFEAT+=['SDI']

df['ZIP5'] = df['ZIP'].apply(lambda z: z[:5] if pd.notnull(z) else z)

df.ZIP5.nunique()

df[df.ZIP5.str.startswith(('60','61','62'))].ZIP5.nunique()

df = df[df.SDI.notna()]

df.shape


sum(df['SDI'].notna())

df.SDI.hist()

#Medical History
MEDHIST = [c for c in df.columns if c.startswith('DYN_MEDHIST_')]
MEDHIST += [c for c in df.columns if c.startswith(('SLEEP_TYPE_','SLEEP_EQUIP_'))]
MEDHIST += [c for c in df.columns if c.startswith('HXINFECTOPT_')]
MEDHIST.remove('DYN_MEDHIST_25'),MEDHIST.remove('HXINFECTOPT_1'),MEDHIST.remove('HXINFECTOPT_Missing')  #all zero

#Fill nan with 0
df[MEDHIST] = df[MEDHIST].fillna(0.0)

#Diabete Type
DBTYPES = pd.get_dummies(df['DMTYPE'].fillna(3.0).astype(int),prefix='DMTYPE')
df[DBTYPES.columns.tolist()] = DBTYPES
MEDHIST+=DBTYPES.columns.tolist()

BINFEAT+=MEDHIST

#Diabete durations cont
df['DMDURATION'] = df['DMDURATION'].map({5:np.nan,4:4,3:3,2:2,1:1})

MEDHIST += ['DMDURATION']

CONTFEAT+=['DMDURATION']

#SMOKING
SMOKING = ['JC_HXSMOKING']
df[SMOKING] = df[SMOKING].fillna(0.0)

BINFEAT+=SMOKING

#HF History
HFHIST = ['OH_ISCHEMIC']
HFHIST += [c for c in df.columns if c.startswith('OH_NONISCHEMIC_ETIOLOGY_')]
HFHIST += ['OH_TRANSPLANT']
df[HFHIST] = df[HFHIST].fillna(0.0)

df['OH_HFHOSPADM'] = df['OH_HFHOSPADM'].map({5:np.nan,4:4,3:3,2:2,1:1})

BINFEAT += HFHIST
CONTFEAT += ['OH_HFHOSPADM']

# DX
DX = []
DX+=['DYN_ATRIALFIB','DYN_ATRIALFIB_NEW']
DX+=['HF_ATRIALFLUTTER','HF_ATRIALFLUTTER_NEW']
DX += [c for c in df.columns if c.startswith(('DYN_ADMSYMPTOMS_', 'DYN_OTHERCONDITION_'))]
DX += ['ACTIVEINFEC_1','ACTIVEINFEC_2','ACTIVEINFEC_3']
DX += ['ACTIVEINFECOPT_1','ACTIVEINFECOPT_2','ACTIVEINFECOPT_3','ACTIVEINFECOPT_4']
# ACTIVEINFECOPT_2 is covid 19, #=222 LOS>7:75

df[DX] = df[DX].fillna(0.0)
df['AHA_DIAGDM'] = df['AHA_DIAGDM'].fillna(0.0).map({88:0.0}).fillna(df['AHA_DIAGDM']).fillna(0.0)
DX += ['AHA_DIAGDM']

BINFEAT += DX



MEDS = pd.read_csv('coding.csv',sep=',')
MEDS.columns = list('abcdefgh')
MEDS = MEDS[MEDS[MEDS.columns[-1]].fillna('nan').str.startswith('Meds Prior to Admission:')]
MEDS = MEDS[~MEDS.h.str.contains('Type|other|Missing')]

MEDS = MEDS['c'].tolist()

df[MEDS] = df[MEDS].fillna(0.0)

BINFEAT += MEDS


#LABS

LABSSTR, LABSCONT = [], []
LABSCONT += ['BMIi', 'OH_HEARTRATE', 'AHA_DIASTOLIC', 'AHA_SYSTOLIC', 'OH_RESPRATE'] # VITAL cont, missing
LABSCONT += ['OH_JVD_CM']

RALES = pd.get_dummies(df['OH_RALES_LOCATION'], dummy_na =False, prefix = 'RALES')
df[RALES.columns.tolist()] = RALES
LABSSTR += RALES.columns.tolist()

EDEMAS = pd.get_dummies(df['OH_LOWEREXTREMITY_EDEMA_DEG'], dummy_na =False, prefix = 'EDEMAS')
df[EDEMAS.columns.tolist()] = EDEMAS
LABSSTR += EDEMAS.columns.tolist()

LIPIDS = ['CHOL200i','HDL40i','LDL100i']
df[LIPIDS] = df[LIPIDS].fillna(0.0)
LABSSTR+=LIPIDS

LABSCONT += ['SODIUMi_admit','BNPi_admit', 'POTASSIUMi_admit', 'HGBi_admit', 
'HFS_ALBUMIN','NBNPi_admit', 'SCRi_admit', 'BUNi_admit', 'TROPNi_admit', 'OH_FERRITIN',
'OH_HBA1C', 'AHA_FASTINGBLOOD', 'AHA_EKG']

#df[LABSCONT] = df[LABSCONT].fillna(-1.0)


df['AHA_EKG_MOR'] = df['AHA_EKG_MOR'].fillna(5.0)
EKG_MOR = pd.get_dummies(df['AHA_EKG_MOR'], dummy_na =False, prefix = 'EKG_MOR')
df[EKG_MOR.columns.tolist()] = EKG_MOR
LABSSTR += EKG_MOR.columns.tolist()

BINFEAT+=LABSSTR
CONTFEAT+=LABSCONT

#ADMISSION
AD = ['JC_TRANSOTHED']
df[AD] = df[AD].fillna(0.0)

df['ADMITSOURCEi'] = df['ADMITSOURCEi'].map({8:8, 6:6, 3:3, 2:2, 7:7, 10:np.nan, 14:14})
ADMITSOURCE = pd.get_dummies(df['AHA_EKG_MOR'], dummy_na =False, prefix = 'ADMITSOURCE')
df[ADMITSOURCE.columns.tolist()] = ADMITSOURCE
BINFEAT += AD
BINFEAT += ADMITSOURCE.columns.tolist()

#BINFEAT+=LABSSTR
#CONTFEAT+=LABSCONT

df.PATIENT_ID.nunique()

df = df[['PATIENT_ID']+BINFEAT+CONTFEAT]

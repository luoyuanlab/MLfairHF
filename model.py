from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv', index_col = 0)

bool_cols = [col for col in df if df[col].dropna().value_counts().index.isin([0,1]).all()]
df[bool_cols] = df[bool_cols].fillna(0)


SDI = pd.read_csv('sdi.csv',dtype={'zcta':str}, index_col=0)
SDI.zcta = SDI.zcta.apply(lambda z: '0'*(5-len(z))+z)
df.ZIP = df.ZIP.apply(lambda z: '0'*(5-len(z))+z)

admission =  pd.read_csv('dataAll.csv',usecols=['ADMITSOURCEi'])
admission['ADMITSOURCEi'] = admission['ADMITSOURCEi'].map({8:8, 6:6, 3:3, 2:2, 7:7, 10:np.nan, 14:14})
ADMITSOURCE = pd.get_dummies(admission['ADMITSOURCEi'], dummy_na =False, prefix = 'ADMITSOURCE')
df[[c for c in df.columns if c.startswith('ADMITSOURCE')]] = ADMITSOURCE[ADMITSOURCE.index.isin(df.index.tolist())].values


feats = df.columns[2:].tolist()

mortality = pd.read_csv('dataAll.csv',usecols=['DISPOSITIONi'])
mortality = (mortality == 6).astype('int')
mortality.columns = ['Mortality']

df = df.merge(mortality, how = 'left', left_index=True, right_index=True)

df = pd.merge(df,SDI,left_on='ZIP',right_on='zcta',how = 'inner')
df = df[df.sdi_score.notna()]

train_ind, test_ind = train_test_split(list(range(df.shape[0])), test_size = 0.3, random_state = 20220312)#20210824)
y = [int(l) for l in ((df.LOS >7) | (df.Mortality)).tolist()]
y_train = [y[i] for i in train_ind]
y_test = [y[i] for i in test_ind]

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
scaler = MinMaxScaler()
clf = RandomForestClassifier(class_weight='balanced')
# rus = RandomUnderSampler(random_state=20200902)
rus = RandomUnderSampler(random_state=20220332)

sd_feat = ['NA','percnt_ltfpl100', 'percnt_singlparntfly', 'percnt_black',
'percnt_dropout', 'percnt_hhnocar', 'percnt_rentoccup',
'percnt_crowding', 'percnt_nonemp', 'percnt_unemp', 'percnt_highneeds',
'percnt_hispanic', 'percnt_frgnborn', 'percnt_lingisol', 'ADI_NATRANK',
'ADI_STATERNK']



param_grid = {
    'clf__n_estimators': np.linspace(20,620,6,dtype=int),
    'clf__max_depth': [3,7,11,31,61]
}




pipe = Pipeline(steps=[('imputer', imputer), ('scaler', scaler), ('clf', clf)])
search = GridSearchCV(pipe, param_grid, n_jobs=24, scoring='roc_auc', verbose = 2, cv = 5)
res = pd.DataFrame(columns=sd_feat)


for s in tqdm(sd_feat):
    if(s == 'NA'):
        X = df[feats].to_numpy()
    else:
        X = df[feats+[s]].to_numpy()
    X_train = np.array([X[i] for i in train_ind])
    X_test = np.array([X[i] for i in test_ind])
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    search.fit(X_rus, y_rus)
    res[s] = search.predict_proba(X_test)[:,1]

res.to_csv('results_revised_fillna.csv')

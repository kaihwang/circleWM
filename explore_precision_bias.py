# explore precision and bias results, using in person pilot data collected in November.
# rnage 2, 300 ms desgin

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import ttest_1samp, ttest_rel
plt.ion()
sns.set_context('talk', font_scale=1)
sns.set_style('white')
sns.set_palette("colorblind")

#script fro XT.
working_list = glob.glob('/mnt/cifs/rdss/rdss_kahwang/PrecisionWM_data/range2/300ms&hann/*.csv')

df_dict={'prolific_id':[],
'sub_numbers':[],
'precision':[],
'cue':[],
'similarity_range':[],
'similarity':[],
'distance':[],
'bias':[],
'bias_standard':[]}

figure1,axs1=plt.subplots(ncols=5,nrows=5,sharex=True,sharey=True,figsize=(12,15))
axs1=axs1.flatten()

figure2,axs2=plt.subplots(ncols=5,nrows=5,sharex=True,sharey=True,figsize=(12,15))
axs2=axs2.flatten()

sub=1
df = pd.DataFrame()
for csv,ax1,ax2 in zip(working_list,axs1,axs2):

    #main session contains 300 trials
    df_sub=pd.read_csv(csv).iloc[-300:]
    df_sub['SUBJECT_ID'] = sub
    sub+=1

    # cue type
    df_sub.loc[df_sub.cue == 'left','Condition'] = 'valid'
    df_sub.loc[df_sub.cue == 'right','Condition'] = 'valid'
    df_sub.loc[df_sub.cue == 'neutral','Condition'] = 'neutral'
    #df_sub['similarity'] = abs(df_sub['similarity'])

    # get all raw orientation on the same scale of 0 to 180
    df_sub['left_distractor_ori'] = (df_sub['left_distractor_ori'] +180)%180
    df_sub['right_distractor_ori'] = (df_sub['right_distractor_ori'] +180)%180

    df_sub.loc[df_sub['probe_x']<0,'distractor_orientation']=df_sub['left_distractor_ori']
    df_sub.loc[df_sub['probe_x']>0,'distractor_orientation']=df_sub['right_distractor_ori']

    df_sub['correct_response']
    df_sub['response_ori'] = (df_sub['response_ori']+180)%180

    # recalculate similarity
    #df_sub['similarity'] = df_sub['distractor_orientation'] - df_sub['correct_response']
    #df_sub.loc[df_sub['similarity']>=90, 'similarity'] = df_sub.loc[df_sub['similarity']>=90, 'similarity']-180
    #df_sub.loc[df_sub['similarity']<=-90, 'similarity'] = df_sub.loc[df_sub['similarity']<=-90, 'similarity']-(-180)
    ranges = np.arange(-36,37,18)

    #### here I am recalcuating the similarty range, negative values mean the diff between target and distractor is more towards clock wise, positve means it is more towards counter-clock direction (or is it the opposite!? you know what I mean)
    for n in np.arange(0,len(ranges)-1):
        if n <=1:
            df_sub.loc[(df_sub['similarity']>=ranges[n]) & (df_sub['similarity']<=ranges[n+1]), 'similarity_range'] = n - 2
        elif n>=2:
            df_sub.loc[(df_sub['similarity']>=ranges[n]) & (df_sub['similarity']<=ranges[n+1]), 'similarity_range'] = n - 1

    # calcuate precision (distance from target)
    #df_sub['similarity'] = ((df_sub['correct_response'] - df_sub['distractor_orientation'])+180) %180
    df_sub['precision'] = df_sub['response_ori'] - df_sub['correct_response']
    df_sub.loc[df_sub['precision']>=90, 'precision'] = df_sub.loc[df_sub['precision']>=90, 'precision']-180
    df_sub.loc[df_sub['precision']<=-90, 'precision'] = df_sub.loc[df_sub['precision']<=-90, 'precision']-(-180)
    df_sub['precision_normalized'] = df_sub['precision'] / abs(df_sub['similarity'])

    # distance from distractor, response closer or further away from distractor? 0 means closer to distractor
    df_sub['response_distance_from_distractor'] = df_sub['response_ori'] - df_sub['distractor_orientation']
    df_sub.loc[df_sub['response_distance_from_distractor']>=90, 'response_distance_from_distractor'] = df_sub.loc[df_sub['response_distance_from_distractor']>=90, 'response_distance_from_distractor']-180
    df_sub.loc[df_sub['response_distance_from_distractor']<=-90, 'response_distance_from_distractor'] = df_sub.loc[df_sub['response_distance_from_distractor']<=-90, 'response_distance_from_distractor']-(-180)
    #df_sub['response_distance_from_distractor_normalized'] = df_sub['response_distance_from_distractor'] / abs(df_sub['similarity'])
    df_sub['response_distance_from_distractor_normalized'] = df_sub['response_distance_from_distractor']

    #bias, closer to distractor or target? positve indicates closer towards target, negative means closer towards distractor
    df_sub['bias'] =  (abs(df_sub['response_distance_from_distractor']) - abs(df_sub['precision'])) / abs(df_sub['similarity'])
    #filter out the responses without pressing space. Concat across subjects
    df = df.append(df_sub.loc[df_sub['probe_resp.keys']=='space'])

    '''
    h1=sns.histplot(df_sub,x=df_sub['response_ori'].to_numpy(),hue='Condition',multiple="layer",ax=ax1)
    handles1=h1.legend_.legendHandles
    labels1=[t.get_text() for t in h1.legend_.get_texts()]
    ax1.get_legend().remove()
    '''
    h1=sns.scatterplot(x='similarity', y='precision', data = df_sub.loc[df_sub.Condition=='valid'], hue='Condition',ax=ax1)
    handles1=h1.legend_.legendHandles
    labels1=[t.get_text() for t in h1.legend_.get_texts()]
    ax1.get_legend().remove()

    h2=sns.scatterplot(x='similarity', y='precision', data = df_sub.loc[df_sub.Condition=='neutral'], hue='Condition',ax=ax2)
    handles2=h2.legend_.legendHandles
    labels2=[t.get_text() for t in h2.legend_.get_texts()]
    ax2.get_legend().remove()



### Plots
## precision effect
sns.kdeplot(x='precision', hue='Condition', data=df, common_norm=False)

# similarity
a1=sns.lmplot(x="similarity", y="precision", col="Condition", hue="Condition", data=df,
           col_wrap=2, ci=False,palette="muted")
a1.set(ylim=(-90,90))
a2 = sns.lmplot(x="similarity", y="precision", col="Condition", hue="Condition", data=df,
           col_wrap=2, palette="muted", scatter= False)
a2.set(ylim=(-90,90))
gdf = df.groupby(['SUBJECT_ID','similarity_range', 'Condition']).mean().reset_index()
sns.pointplot(x='similarity_range', y='precision', data = gdf, hue='Condition')

import statsmodels.formula.api as smf
smf.mixedlm(formula = 'precision ~ 1 + similarity*Condition', groups = df['SUBJECT_ID'], re_formula = '1', data = df).fit().summary()

sns.pointplot(x='similarity_range', y='bias', data = gdf, hue='Condition')
smf.mixedlm(formula = 'bias ~ 1 + similarity_range*Condition', groups = df['SUBJECT_ID'], re_formula = '1', data = df).fit().summary()

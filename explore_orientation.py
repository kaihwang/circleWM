import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import ttest_1samp, ttest_rel

working_list = glob.glob('prolific_data_0714/*.csv')
df = pd.DataFrame()
for s, csv in enumerate(working_list):

    #main session contains 300 trials
    df_sub=pd.read_csv(csv).iloc[-300:]
    df_sub['SUBJECT_ID'] = s + 1

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
    df_sub['similarity'] = df_sub['distractor_orientation'] - df_sub['correct_response']
    df_sub.loc[df_sub['similarity']>=90, 'similarity'] = 180-df_sub.loc[df_sub['similarity']>=90, 'similarity']
    df_sub.loc[df_sub['similarity']<=-90, 'similarity'] = -180-df_sub.loc[df_sub['similarity']<=-90, 'similarity']
    ranges = np.arange(-90,91,18)

    #### here I am recalcuating the similarty range, negative values mean the diff between target and distractor is more towards clock wise, positve means it is more towards counter-clock direction (or is it the opposite!? you know what I mean)
    for n in np.arange(0,len(ranges)-1):
        df_sub.loc[(df_sub['similarity']>=ranges[n]) & (df_sub['similarity']<=ranges[n+1]), 'similarity_range'] = n - 5

    # calcuate precision (distance from target)
    #df_sub['similarity'] = ((df_sub['correct_response'] - df_sub['distractor_orientation'])+180) %180
    df_sub['precision'] = df_sub['response_ori'] - df_sub['correct_response']
    df_sub.loc[df_sub['precision']>=90, 'precision'] = 180-df_sub.loc[df_sub['precision']>=90, 'precision']
    df_sub.loc[df_sub['precision']<=-90, 'precision'] = -180-df_sub.loc[df_sub['precision']<=-90, 'precision']
    df_sub['precision_normalized'] = df_sub['precision'] / abs(df_sub['similarity'])

    # distance from distractor, response closer or further away from distractor? 0 means closer to distractor
    df_sub['response_distance_from_distractor'] = df_sub['response_ori'] - df_sub['distractor_orientation']
    df_sub.loc[df_sub['response_distance_from_distractor']>=90, 'response_distance_from_distractor'] = 180-df_sub.loc[df_sub['response_distance_from_distractor']>=90, 'response_distance_from_distractor']
    df_sub.loc[df_sub['response_distance_from_distractor']<=-90, 'response_distance_from_distractor'] = -180-df_sub.loc[df_sub['response_distance_from_distractor']<=-90, 'response_distance_from_distractor']
    df_sub['response_distance_from_distractor_normalized'] = df_sub['response_distance_from_distractor'] / abs(df_sub['similarity'])

    #bias, closer to distractor or target? positve indicates closer towards target, negative means closer towards distractor
    df_sub['bias'] =  (abs(df_sub['response_distance_from_distractor']) - abs(df_sub['precision'])) / abs(df_sub['similarity'])

    #filter out the responses without pressing space. Concat across subjects
    df = df.append(df_sub.loc[df_sub['probe_resp.keys']=='space'])


#### I think the pattern here does show that in general when the distractor is clock or counter clockwise responses do get pull towards that side.
sns.scatterplot(x='similarity', y='precision', data = df, hue='Condition')
plt.show()

sns.scatterplot(x='similarity', y='precision', data = df.loc[df.Condition=='neutral'], hue='Condition')
plt.show()

sns.scatterplot(x='similarity', y='precision', data = df.loc[df.Condition=='valid'], hue='Condition')
plt.show()

# for plotting with point estimates, make sure the dof is at the level of subject not individual trials. So calcuate mean within each subject before plotting
gdf = df.groupby(['SUBJECT_ID','similarity_range', 'Condition']).mean().reset_index()

sns.scatterplot(x='similarity_range', y='precision', data = gdf, hue='Condition')
plt.show()

sns.pointplot(x='similarity_range', y='precision_normalized', data = gdf, hue='Condition')
plt.show()

sns.pointplot(x='similarity_range', y='response_distance_from_distractor_normalized', data = gdf, hue='Condition')
plt.show()

sns.pointplot(x='similarity_range', y='response_distance_from_distractor', data = gdf, hue='Condition')
plt.show()

sns.pointplot(x='similarity_range', y='bias', data = gdf, hue='Condition')
plt.show()

sns.scatterplot(x='distractor_orientation', y='response_distance_from_distractor', data = df.loc[df.Condition=='neutral'], hue='Condition')
plt.show()

# distribution
sns.kdeplot(x='precision', data = gdf, hue='Condition')
plt.show()





#end of line

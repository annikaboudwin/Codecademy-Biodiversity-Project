#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# In[2]:


# Read in CSV files
observations = pd.read_csv('observations.csv')
species = pd.read_csv('species_info.csv')


# In[3]:


# Ensure files are readable
print(observations.head())
print(species.head())


# In[4]:


# List park names, species categories and various levels of conservation status
print(observations['park_name'].unique())
# Result = 'Great Smoky Mountains National Park', 'Yosemite National Park', 'Bryce National Park', 'Yellowstone National Park'
print(species['category'].unique())
# Result = 'Mammal', 'Bird', 'Reptile', 'Amphibian', 'Fish', 'Vascular Plant', 'Nonvascular Plant'
print(species['conservation_status'].unique())
# Result = 'NaN', 'Species of Concern', 'Endangered', 'Threatened', 'In Recovery'


# In[5]:


# How many items per list?
print(len(observations))
print(len(species))

print('There are 23,296 items of observation as opposed to only 5,824 species listed. However, there exactly four times the number of observations are there are the number of species listed and there are four different parks where observations are made.')


# In[6]:


pivoted = pd.pivot_table(observations, values='observations', index=['scientific_name'], columns='park_name').reset_index()

display(pivoted.head())


# In[7]:


# Combine dataframes
species = species.fillna('')
combined = pivoted.merge(species, on = 'scientific_name', how = 'inner')
print(len(combined))
display(combined.head())


# In[8]:


# How many species are listed under each type of conservation status?
conservation_number = combined.groupby('conservation_status').scientific_name.count().reset_index()
conservation_number = conservation_number.rename(columns={'conservation_status': 'Conservation Status'})
conservation_number = conservation_number.rename(columns={'scientific_name': 'Number of Species'})


print(conservation_number)
print('There are 161 species of concern, 10 threatened species, 16 endangered species and 4 species listed as being in recovery.')


# In[9]:


# Given there seem to be extra entries for the gray wolf, how many entries are there total?

wolf_observations = observations[(observations.scientific_name == 'Canis lupus')]
wolf_species = species[(species.scientific_name == 'Canis lupus')]
canis_lupus = pivoted[(pivoted.scientific_name == 'Canis lupus')]
print(wolf_observations)
print(len(wolf_observations))
print(wolf_species)
print(len(wolf_species))
#print(canis_lupus)
#print(len(canis_lupus))
print('The number of items has increased. On reviewing the observations table, it is evident that some species are observed multiple times within the same park. For example, canis lupus is observed a total of 36 times across all four parks. Review of canis lupus shows the data is duplicated three times, with each entry listing \'Gray Wolf - endangered\', \'Gray Wolf, Wolf - In Recovery\' and \'Gray Wolf, Wolf - Endangered\'.')


# In[10]:


# How many species are observed multiple times within the same park?
extra_observations = observations.scientific_name.value_counts().loc[lambda x: x > 11].reset_index()['index']
extra_species = species.scientific_name.value_counts().loc[lambda x: x > 2].reset_index()['index']
print(extra_observations)
print('These are the species that show up twelve times within the observations list.')
print(extra_species)
print('These are the species that show up three times within the species list. Both lists have multiple entries for the same set of species.')


# In[11]:


# What is the range of observations?

yosemite_min_obs = combined['Yosemite National Park'].min()
print(yosemite_min_obs)
yosemite_max_obs = combined['Yosemite National Park'].max()
print(yosemite_max_obs)
print('For any given species within Yosemite National Park, the lowest number of observations is 31 and the highest number of observations is 223.')

yosemite_fewest = combined[(combined['Yosemite National Park']) == 31]
yosemite_most = combined[(combined['Yosemite National Park']) == 223]

print(yosemite_fewest)
print(yosemite_most)

print('In Yosemite, the fewest observations recorded were of the fish, noturus baileyi (\'Smoky Madtom\'), and the most observations were of the vascular plant ivesia shockleyi.')


# In[12]:


bryce_min_obs = combined['Bryce National Park'].min()
print(bryce_min_obs)
bryce_max_obs = combined['Bryce National Park'].max()
print(bryce_max_obs)
print('For any given species within Bryce National Park, the lowest number of observations is 9 and the highest number of observations is 176.')

bryce_fewest = combined[(combined['Bryce National Park']) == 9]
bryce_most = combined[(combined['Bryce National Park']) == 176]

print(bryce_fewest)
print(bryce_most)

print('In Bryce National Park, the fewest observations recorded were of the vascular plant, corydalis aurea (\'Golden Corydalis, Scrambled Eggs\'), and the most observations were of the vascular plant valerianella radiata (\'Beaked Corn-Salad, Corn Salad\').')


# In[13]:


smoky_min_obs = combined['Great Smoky Mountains National Park'].min()
print(smoky_min_obs)
smoky_max_obs = combined['Great Smoky Mountains National Park'].max()
print(smoky_max_obs)
print('For any given species within Great Smoky Mountains National Park, the lowest number of observations is 10 and the highest number of observations is 147.')

smoky_fewest = combined[(combined['Great Smoky Mountains National Park']) == 10]
smoky_most = combined[(combined['Great Smoky Mountains National Park']) == 147]

print(smoky_fewest)
print(smoky_most)

print('In Great Smoky Mountains National Park, the fewest observations recorded were of the vascular plant, collomia tinctoria (\'Staining Collomia, Yellowstain Collomia\'), and the most observations were of the vascular plant sonchus asper (\'Spiny Sowthistle\').')


# In[14]:


yellowstone_min_obs = combined['Yellowstone National Park'].min()
print(yellowstone_min_obs)
yellowstone_max_obs = combined['Yellowstone National Park'].max()
print(yellowstone_max_obs)
print('For any given species within Yellowstone National Park, the lowest number of observations is 10 and the highest number of observations is 147.')

yellowstone_fewest = combined[(combined['Yellowstone National Park']) == 57]
yellowstone_most = combined[(combined['Yellowstone National Park']) == 321]

print(yellowstone_fewest)
print(yellowstone_most)

print('In Yellowstone National Park, the fewest observations recorded were of the bird, grus americana (\'Whooping Crane\'), and the most observations were of the vascular plant lycopodium tristachyum (\'Deep-Root Clubmoss, Ground Cedar\').')


# In[15]:


# What are the number of observations per park?

park_observations = observations.groupby('park_name')['observations'].sum()
print(park_observations)

# Break the data down into individual dataframes per park
#yellowstone = observations[(observations['park_name']) == 'Yellowstone National Park']
#smoky_mountains = observations[(observations['park_name']) == 'Great Smoky Mountains National Park']
#bryce = observations[(observations['park_name']) == 'Bryce National Park']
#yosemite = observations[(observations['park_name']) == 'Yosemite National Park']

# Total the number of observations per park
#yellowstone_observations = yellowstone.observations
#smoky_mountains_observations = smoky_mountains.observations
#bryce_observations = bryce.observations
#yosemite_observations = yosemite.observations

# Combine the park observations into a list that can be graphed
#all_park_observations = [yellowstone_observations, smoky_mountains_observations, bryce_observations, yosemite_observations]

ax = sns.boxplot(x = 'park_name', y = 'observations', data = observations)
plt.ylabel('Observations per Species')
plt.xlabel('National Parks')
plt.xticks(rotation=15)
plt.title('Observations in National Parks')
ax.set_xticklabels(['Great Smoky Mountains', 'Yosemite', 'Bryce', 'Yellowstone'])
plt.show()
plt.clf()

print('Yellowstone National Park has the greatest number of observations with a total of 1,443,562. Second is Yosemite National Park with a total of 863,332 observations. Third is Bryce National Park with 576,025 observations. Finally, Great Smoky Mountains National Park has a total of 431,820 observations.')


# In[16]:


endangered = species.loc[species['conservation_status'] == 'Endangered']
threatened = species.loc[species['conservation_status'] == 'Threatened']
species_of_concern = species.loc[species['conservation_status'] == 'Species of Concern']
in_recovery = species.loc[species['conservation_status'] == 'In Recovery']


# In[17]:


endangered_count = len(endangered)
print(endangered_count)
display(endangered)
print('Most endangered species are mammals with a few fish and birds.')


# In[18]:


threatened_count = len(threatened)
print(threatened_count)
display(threatened)
print('Threatened species appear to be evenly distributed across several categories.')


# In[19]:


concern_count = len(species_of_concern)
print(concern_count)
display(species_of_concern)
print(species_of_concern['category'].value_counts())
print('Most species listed as \'Species of Concern\' are birds with 72 species, followed by vascular plants with 43 species and mammals with 28 species.')


# In[20]:


recovery_count = len(in_recovery)
print(recovery_count)

display(in_recovery)


# In[21]:


total_conservation = [endangered_count, threatened_count, concern_count, recovery_count]

plt.bar(range(len(total_conservation)), total_conservation)
ax = plt.subplot()
plt.ylabel('Number of Species')
plt.xlabel('Conservation Status of Species')
ax.set_xticks(range(4))
ax.set_xticklabels(['Endangered', 'Threatened', 'Species of Concern', 'In Recovery'])
plt.show()
plt.clf()


# In[22]:


# Set NaN to blank spaces
not_nan = combined[(combined.conservation_status != '')]
not_nan['Bryce National Park'] = not_nan['Bryce National Park'].astype(int)
not_nan['Great Smoky Mountains National Park'] = not_nan['Great Smoky Mountains National Park'].astype(int)
not_nan['Yellowstone National Park'] = not_nan['Yellowstone National Park'].astype(int)
not_nan['Yosemite National Park'] = not_nan['Yosemite National Park'].astype(int)
display(not_nan)


# In[23]:


combined_endangered = not_nan.loc[not_nan['conservation_status'] == 'Endangered']
combined_threatened = not_nan.loc[not_nan['conservation_status'] == 'Threatened']
combined_concern = not_nan.loc[not_nan['conservation_status'] == 'Species of Concern']
combined_recovery = not_nan.loc[not_nan['conservation_status'] == 'In Recovery']
combined_endangered = combined_endangered.rename(columns={'scientific_name': 'Scientific Name', 'category': 'Category', 'common_names': 'Common Names', 'conservation_status': 'Conservation Status'})
#combined = combined.rename(columns={'common_names': 'Common Names'})
display(combined_endangered)


# In[24]:


combined_threatened = combined_threatened.rename(columns={'scientific_name': 'Scientific Name', 'category': 'Category', 'common_names': 'Common Names', 'conservation_status': 'Conservation Status'})
#combined = combined.rename(columns={'common_names': 'Common Names'})
display(combined_threatened)


# In[25]:


combined_concern = combined_concern.rename(columns={'scientific_name': 'Scientific Name', 'category': 'Category', 'common_names': 'Common Names', 'conservation_status': 'Conservation Status'})
#combined = combined.rename(columns={'common_names': 'Common Names'})
display(combined_concern)


# In[26]:


combined_recovery = combined_recovery.rename(columns={'scientific_name': 'Scientific Name', 'category': 'Category', 'common_names': 'Common Names', 'conservation_status': 'Conservation Status'})
display(combined_recovery)


# In[27]:


# Make a list of park names
park_names = ['Bryce National Park', 'Great Smoky Mountains National Park', 'Yellowstone National Park', 'Yosemite National Park']

bryce_endangered_total = combined_endangered['Bryce National Park'].sum()
bryce_threatened_total = combined_threatened['Bryce National Park'].sum()
bryce_concern_total = combined_concern['Bryce National Park'].sum()
bryce_recovery_total = combined_recovery['Bryce National Park'].sum()

all_bryce = [bryce_endangered_total, bryce_threatened_total, bryce_concern_total, bryce_recovery_total]

smoky_endangered_total = combined_endangered['Great Smoky Mountains National Park'].sum()
smoky_threatened_total = combined_threatened['Great Smoky Mountains National Park'].sum()
smoky_concern_total = combined_concern['Great Smoky Mountains National Park'].sum()
smoky_recovery_total = combined_recovery['Great Smoky Mountains National Park'].sum()

all_smoky = [smoky_endangered_total, smoky_threatened_total, smoky_concern_total, smoky_recovery_total]

yellowstone_endangered_total = combined_endangered['Yellowstone National Park'].sum()
yellowstone_threatened_total = combined_threatened['Yellowstone National Park'].sum()
yellowstone_concern_total = combined_concern['Yellowstone National Park'].sum()
yellowstone_recovery_total = combined_recovery['Yellowstone National Park'].sum()

all_yellowstone = [yellowstone_endangered_total, yellowstone_threatened_total, yellowstone_concern_total, yellowstone_recovery_total]

yosemite_endangered_total = combined_endangered['Yosemite National Park'].sum()
yosemite_threatened_total = combined_threatened['Yosemite National Park'].sum()
yosemite_concern_total = combined_concern['Yosemite National Park'].sum()
yosemite_recovery_total = combined_recovery['Yosemite National Park'].sum()

all_yosemite = [yosemite_endangered_total, yosemite_threatened_total, yosemite_concern_total, yosemite_recovery_total]

#print(bryce_endangered_total)
#print(gsm_endangered_total)
#print(yellowstone_endangered_total)
#print(yosemite_endangered_total)

conservation_list = ['Endangered', 'Threatened', 'Species of Concern', 'In Recovery']

# Bryce values
n = 1
t = 4
d = 4
w = 0.8
park1_x = [t*element + w*n for element in range(d)]

plt.bar(park1_x, all_bryce)

# Great Smoky Mountains values
n = 2
t = 4
d = 4
w = 0.8
park2_x = [t*element + w*n for element in range(d)]

plt.bar(park2_x, all_smoky)

# Yellowstone values
n = 3
t = 4
d = 4
w = 0.8
park3_x = [t*element + w*n for element in range(d)]

plt.bar(park3_x, all_yellowstone)

# Yosemite values
n = 4
t = 4
d = 4
w = 0.8
park4_x = [t*element + w*n for element in range(d)]

plt.bar(park4_x, all_yosemite)

ax = plt.subplot()
plt.xticks(np.arange(2, 16, 4))
ax.set_xticklabels(['Endangered', 'Threatened', 'Species of Concern', 'In Recovery'], rotation = 15)
plt.legend(['Bryce National Park', 'Great Smoky Mountains National Park', 'Yellowstone National Park', 'Yosemite National Park'], loc=2)
plt.xlabel('Conservation Status of Species within each park')
plt.ylabel('Number of Observations')
plt.title('Total number of observations based on species\' conservation status.')
plt.show()
plt.clf()

#all_endangered_total = [bryce_endangered_total, gsm_endangered_total, yellowstone_endangered_total, yosemite_endangered_total]


# In[28]:


all_bryce_minus_concern = [bryce_endangered_total, bryce_threatened_total, bryce_recovery_total]
all_smoky_minus_concern = [smoky_endangered_total, smoky_threatened_total, smoky_recovery_total]
all_yellowstone_minus_concern = [yellowstone_endangered_total, yellowstone_threatened_total, yellowstone_recovery_total]
all_yosemite_minus_concern = [yosemite_endangered_total, yosemite_threatened_total, yosemite_recovery_total]

# Bryce values
n = 1
t = 4
d = 3
w = 0.8
park1_x = [t*element + w*n for element in range(d)]

plt.bar(park1_x, all_bryce_minus_concern)

# Great Smoky Mountains values
n = 2
t = 4
d = 3
w = 0.8
park2_x = [t*element + w*n for element in range(d)]

plt.bar(park2_x, all_smoky_minus_concern)

# Yellowstone values
n = 3
t = 4
d = 3
w = 0.8
park3_x = [t*element + w*n for element in range(d)]

plt.bar(park3_x, all_yellowstone_minus_concern)

# Yosemite values
n = 4
t = 4
d = 3
w = 0.8
park4_x = [t*element + w*n for element in range(d)]

plt.bar(park4_x, all_yosemite_minus_concern)

ax = plt.subplot()
plt.xticks(np.arange(2, 12, 4))
ax.set_xticklabels(['Endangered', 'Threatened', 'In Recovery'])
plt.legend(['Bryce', 'Great Smoky Mntns', 'Yellowstone', 'Yosemite'])
plt.xlabel('Conservation Status of Species within each park')
plt.ylabel('Number of Observations')
plt.title('Number of observations based on species\' conservation status.')
plt.show()
plt.clf()


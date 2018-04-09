# Morphology of SDSS galaxies

The goal of this exercice is to develop a classical machine learning algorithm to build a sample of nearby early-type spirals (Sa-b) in the SDSS. 
- The training set is made of the detailed visual classifications of [Nair&Abraham2010](http://adsabs.harvard.edu/abs/2010ApJS..186..427N) of ~14.000 SDSS galaxies. 
Use the TType column.
- All measurements included in the [catalog provided](https://github.com/mhuertascompany/deeplearning4astronomy/blob/master/morphology/Nair_Abraham_cat.fit) can be used (or any other of your choice!).
- We will give priority to solutions that boost purity, although completeness is also important.
- The jpeg images can be donwnloaded [here](https://drive.google.com/drive/folders/1ufj6ATroZ3emBbSQfQhcL_6W87EPgTaS?usp=sharing). The name is  matched to the ID in the catalog.
- A very simple [jupyter notebook](https://github.com/mhuertascompany/deeplearning4astronomy/blob/master/morphology/shallow/morph_classical_ML.ipynb) using a RF classifier is provided.

# Information extraction on drug dosage

Information extraction system that returns quantity and measuring unit given a natural language description of the dosage for a farmaceutical drug. The model is based on python's CRF implementation, crfsuite. Part-ofspeech tags were obtained with nltk.

For more details, read the pdf in the Report folder.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need python3 to run this application. The following modules need to be installed:
```
nltk
```
```
sklearn
```
```
sklearn-crfsuite
```


### Installing

Assuming you have python3 and pip3 already installed, run the following commands to obtain the needed dependencies:

```
pip3 install nltk
```

```
pip3 install sklearn
 ```

```
pip3 install sklearn-crfsuite
 ```
To run the application, open a terminal session and issue the following commands:

```
git clone https://github.com/costimasca/farmaceuticalNLP.git
```
```
cd farmaceuticalNLP
```
```
chmod +x dosage.py
```
```
./dosage.py "The recommended dosage for X is 42 grams."
```
You can specify any sentence using different quantities (1, two, 3 to 5, 0.5 etc.) and different measuring units (drops, ml, miligrams, tablespoons etc.) Feel free to choose one from the corpus.

In the above example, the output is the following touple: (['42'],['grams'])

### Performance

Using 10-fold cross validation:

```

 	precision	recall	f1-measure
WHO 	0.934		0.871	0.897
UNIT 	0.983		0.969	0.976
DOS 	0.980		0.970	0.975
avg 	0.964		0.936	0.949
```
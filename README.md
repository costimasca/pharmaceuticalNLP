# Information extraction on drug dosage

Information extraction system that labels the following named entities given a natural language description of the dosage for a farmaceutical drug:
 - quantity (DOS)
 - measuring unit (UNIT) 
 - to whom the treatement is directed (WHO)
 - frequency (FREQ)
 - period (PER)
  
The model is based on python's CRF implementation, crfsuite. Part-ofspeech tags were obtained with nltk. For more details, read the pdf in the Report folder.


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
./model.py "The recommended oral dosage for adults is 300 mg once daily at bedtime."
```
You can specify any sentence using different quantities (1, two, 3 to 5, 0.5 etc.) and different measuring units (drops, ml, miligrams, tablespoons etc.) Feel free to choose one from the corpus.

In the above example, the output is the following label list:
 (['O', 'O', 'O', 'O', 'O', 'WHO', 'O', 'DOS', 'UNIT', 'FREQ', 'FREQ', 'O', 'O', 'O'])

### Performance

Using 10-fold cross validation:

```

 	precision	recall	f1-measure
PER 	0.950		0.918	0.933
WHO 	0.944		0.934	0.938
UNIT 	0.983		0.956	0.969
DOS 	0.981		0.962	0.971
FREQ 	0.987		0.982	0.985
avg 	0.969		0.942	0.955
```

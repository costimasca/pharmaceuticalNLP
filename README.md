# Information extraction on drug dosage

Information extraction system that returns quantity and measuring unit given a natural language description of the dosage for a farmaceutical drug. For more details, read the pdf in the Report folder.




## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need python3 to run this application. The following modules need to be installed:
```
nltk
```
```
sklearn
```


### Installing

Assuming you have python3 and pip3 already installed, run the following commands to obtain the needed dependencies:

```
pip3 install nltk
```

```
pip3 install sklearn
 ```

To run the application, open a terminal session and issue the following commands:

```
git pull https://github.com/costimasca/farmaceuticalNLP.git
```
```
chmod +x dosage.py
```
```
./dosage.py "The recommended dosage for X is 42 grams."
```
You can specify any sentence using different quantities (1, two, 3 to 5, 0.5 etc.) and different measuring units (drops, ml, miligrams, tablespoons etc.) Feel free to choose one from the corpus.

In the above example, the output is the following touple: (['42'],['grams'])

##Performance

Using 10-fold cross validation:

```

		precision	recall	f1-measure
UNIT  	0.905		0.856	0.879
DOS 	0.908		0.875	0.891
avg		0.906		0.865	0.885
```
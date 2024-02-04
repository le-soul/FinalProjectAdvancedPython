# Machine Learning for houses in Madrid
This project aims to predict housing prices in Madrid using machine learning techniques. The dataset used contains various features such as square meters built, number of rooms, number of bathrooms, district ID, presence of parking, and presence of air conditioning, among others.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Data Visualization](#data-visualization)
4. [Predictions](#predictions)
5. [Tests](#tests)
6. [Coverage](#coverage)
7. [Pylint and Flake8](#pylint-and-flake8)

## Installation
It would be recommended to use the python version 3.11.7 and use an environment.
### Clone repository
First download FinalProjectAdvancedPython from the github repository:

    git clone https://github.com/le-soul/FinalProjectAdvancedPython.git

### Navigate to the repository
    cd FinalProjectAdvancedPython

### Install dependencies
    pip install -r requirements.txt

## Usage
This section highlights the main functionalities of the project and the commands to run them. Follow the command-line prompts to view data, train models, and make predictions. If at any moment you need to go to outputs click here.

[Link to Outputs Folder](outputs/)

`Foundation:` The first command will be used to access the options for the data visualization of the database.
```
python scripts/main.py view-data
```
The second command will be used to access the options for the predictions using machine learning techniques.
```
python scripts/main.py training
```

`Options that are on both:` -i is used to input the database and -o is to save the output in a set folder, or on a new one if it doesn't exist. Here is an example:
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs 
```


### Data Visualization

`Option for seeing null and duplicate values`
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -dp
```

`Option 1:` Shows a box to see the correlation between variables in the database. Output will be in the folder called outputs, named correlation_matrix.png
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "correlation"
```

`Option 2:` Shows a graph where the skewness of the variable buy_price can be seen. Output will be in the folder called outputs, named price_skewness.png
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "price skewness"
```

`Option 3:` Shows a bar graph where you can see the most expensive districts per average buying price. Output will be in the folder called outputs, named most_exp_districts.png
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "most exp districts"
```

`Option 4:` Shows a bar graph where you can see the districts with most average rooms. Output will be in the folder called outputs, named most_rooms_districts.png
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "most rooms districts"
```

`Option 5:` Shows a bar graph where you can see the districts with most average bathrooms. Output will be in the folder called outputs, named most_bathrooms_districts_matrix.png
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "most bathrooms districts"
```

`Option 6:` Show four scatter plots against price to see their visual correlation. Output will be in the folder called outputs, named scatter_price_x_variables.png
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "price x variables"
```

### Predictions

#### Predictions are separated into decision tree classification and linear regression.

#### Linear Regression

`Option 1:` Creates a multiple regression graph with buying price as the dependent variable. I would like to note that I removed number of bathrooms due to it's theoretical multicollinearity with square metres built, but not number of rooms because even though it is theoretical multicollinearity its VIF is not that high. Districts was turned into dummies and still gave multicollinearity to the model, so I removed it. Output will be in the folder called outputs, named Linear_Regression.png
```
python scripts/main.py training -i dataset/houses_Madrid.csv -o outputs -r -ln "regression"
```

`Option 2:` Creates a multiple regression graph with buying price as the dependent variable, but log has been used for buy_price and sq_mt_built. The purpose of this is to reduce homoscedasticity, normalize the distribution of the data and it makes sense to talk about % when talking about square metres built. I did the same as mentioned earlier with bathrooms and rooms. The R squared is also shown and it is higher than the one that doesn't have log. Output will be in the folder called outputs, named Log_Linear_Regression.png
```
python scripts/main.py training -i dataset/houses_Madrid.csv -o outputs -r -ln "log_regression"
```

`Option 3:` Shows information over the graphs such its R squared, their VIF values, p-values. 
```
python scripts/main.py training -i dataset/houses_Madrid.csv -o outputs -r -ln "multicollinearity+"
```

`Option 4:` Allows the user to input the independent variables to predict what the buying price would be. For example, when called in the command line it will ask you "Enter square meters built: ", and you would need to input a positive integer.
```
python scripts/main.py training -i dataset/houses_Madrid.csv -o outputs -r -ln "predict your buying price"
```

#### Decision Tree

`Option 1:` This command predicts air conditioning appearance using a decision tree classifier. The classifier utilizes one input features: "buy_price" to predict whether a property has air conditioning or not. Output will be in the folder called outputs, named Decision_Tree_Ac.png
```
python scripts/main.py training -i dataset/houses_Madrid.csv -o outputs -css -d "has ac"
```

`Option 2:`This command predicts parking availability using a decision tree classifier. The classifier utilizes two input features: "buy_price" and "district_id" to predict whether a property has parking or not. . Output will be in the folder called outputs, named Decision_Tree_Parking.png
```
python scripts/main.py training -i dataset/houses_Madrid.csv -o outputs -css -d "has parking"
```

### Tests

You can use pytest and 17 tests should be passed if all things go correctly.
```
pytest
```

You can also use unittest to check if they run, except for test_linearregression.py as it doesn't recognize it as I'm importing pytest for it. So they should ran 14 tests. Use these to check them all at once:
```
python -m unittest
```

Or individually:
```
python -m unittest tests/test_decisiontree.py
```
This one should ran 2 tests OK.
```
python -m unittest tests/test_graphs.py
```
This one should ran 6 tests OK.
```
python -m unittest tests/test_load.py
```
This one should ran 3 tests OK.
```
python -m unittest tests/test_mainfunctions.py
```
This one should ran 3 tests OK.


### Coverage

You can also view the coverage by doing this
```
coverage run -m pytest
```
```
coverage report
```
And ultimately view it in html
```
coverage html
```

### Pylint and Flake8

```
pylint scripts
```
```
pylint tests
```
```
flake8 scripts
```
```
flake8 tests
```
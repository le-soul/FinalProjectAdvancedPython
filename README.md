# Machine Learning for houses in Madrid
This project aims to predict housing prices in Madrid using machine learning techniques. The dataset used contains various features such as square meters built, number of rooms, number of bathrooms, district ID, presence of parking, and presence of air conditioning, among others.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Contributing](#contributing)
5. [License](#license)

## Installation
### Clone repository
First download FinalProjectAdvancedPython from the github repository:

    git clone https://github.com/le-soul/FinalProjectAdvancedPython.git

### Navigate to the repository

### Install dependencies
    pip install -r requirements.txt

## Usage
This section highlights the main functionalities of the project and the commands to run them. Follow the command-line prompts to view data, train models, and make predictions.

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

`Option 1:` Shows a box to see the correlation between variables in the database
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "correlation"

```

`Option 2:` Shows a graph where the skewness of the variable buy_price can be seen
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "price skewness"
```

`Option 3:` Shows a bar graph where you can see the most expensive districts per average buying price
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "most exp districts"
```

`Option 4:` Shows a bar graph where you can see the districts with most average rooms
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "most rooms districts"
```

`Option 5:` Shows a bar graph where you can see the districts with most average bathrooms.
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "most bathrooms districts"
```

`Option 6:` Show four scatter plots against price to see their visual correlation.
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -pl -gr "price x variables"
```

### Predictions

#### Predictions are separated into decision tree classification and linear regression.

Linear Regression

`Option 1:` Creates a multiple regression graph with buying price as the dependent variable
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -r -ln "regression"
```

`Option 2:` Shows information over the graphs such its R squared, their VIF values, p-values
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -r -ln "multicollinearity+"
```

`Option 3:` Shows a bar graph where you can see the most expensive districts per average buying price
```
python scripts/main.py view-data -i dataset/houses_Madrid.csv -o outputs -r -ln "predict your buying price"
```


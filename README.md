# Classification Disaster Related Messages

Libraries used: 
- Pandas 
- Scikit-Learn
- Sqlite3
- Flask

The motivation for this project is to classify text messages into one or more of 36 categories. The data is provided by Figure8. There are three separate components to this project: 

1. ETL with Pandas. `disaster_messages.csv` and `disaster_categories.csv` are cleaned, merged, and saved into a sqllite database. This process is encapsulated by `process_data.py`  

2. Machine learning with scikit-learn. Data from the database is loaded, transformed through a natural language processing pipeline, and fitted to a tuned `RandomForestClassifier`. The model is saved as a pickle. This process is encapsulated by `train_classifier.py`

3. Web dashboard with Flask. Flask loads the pickled classifier and generates predictions. The results are displayed in a plotly visualization. The relevant file is `run.py`. 

# Installation 
 
Use `pip` to install `pandas`, `scikit-learn`, and `flask` by typing something like: 

`pip install pandas sklearn flask`

in a Python command line. 

## Configuration and Running

The folder structure is as follows: 

- app
  - templates
    - go.html
    - master.html
  - run.py
- data
  - DisasterReponse.db
  - disaster_categories.csv
  - disaster_messages.csv
  - process_data.py
- models
  - classifier.pkl
  - train_classifier.py

(from Udacity's project workspace )
1. To run the ETL pipeline, type: 
    `python data/process_data.py data_disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run the ML pipeline that trains cnd saves a classifier, type: 
    `python models/train_classifier.py data/DisasterRepsonse.db models/classifier.pkl`
3. To run web app, type: 
   `python run.py`
4. Go to `http://0.0.0.0:3001/`

# Acknowledgements
All data was kindly provided by Figure8. The code templates ( as well as instructions) for the web app were provided by Udacity.  




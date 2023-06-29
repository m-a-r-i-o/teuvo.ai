# teuvo.ai
A simple flask app that fits a Self Organizing Map (SOM) to your data and outputs the SOM neurons as a set of data prototypes.

Usage:

```
pip install numpy pandas matplotlib flask minisom
python app.py
```

After the server is running connect to 127.0.0.1:5000.
You can upload your data in csv format as follows:

```
feature_1, feature_2, feature_3, ...
0, 3.1415926, 2.718281828
-3, -2, -1
...
```

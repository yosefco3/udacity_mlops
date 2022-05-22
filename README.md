# Predict Customer Churn
- The first project of ML DevOps Engineer Nanodegree Udacity program.

## Project Description
In this project, I will implement my learnings to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, logged, and tested). 
both files achived grade above 7 on pylint.

## Files and data description
### churn_library.py
this file includes functions with the flow of the project. from recieving the data until creating the models, and making predictions in one flow.
- The time to run the file is like 6 min. you could short it by pass to the function "train_models" the argument "load_models" with True. so if the models already created it would load them from the models directory. 
- if the models didnt created yet, it will throw an error with appropriate message. then you should pass True to the "load_models" argument.
    

### churn_script_logging_and_testing.py
a file which tests and logs all the flow of the project into a log file.
- first it creates a testing folders (if they aready exist it remove them and then create them) the purpose is to seperate the testing from the "production" code.
- the testing folders are:
   1. 'images/testing/' - this directory stores all the plots.     
   2. 'models/testing/' - this directory store the models.        
- then it runs all the flow in churn_library.py, includes the two opions: 
- with creating the models from begining, and with loading already created models.
- then it logs all to the file "churn_library.log"
- there is a function to cleanup the testing directories, and stay with the log file only. if you want to use it, uncomment the "cleanup" function.
the running time of the file with all tests is like 7 min.
    
    
## Running Files
### to run the churn_library.py: 
<code> python churn_library.py</code>
### to run the test and logs: 
<code> python churn_script_logging_and_testing.py </code>
### when you run the testing file, testing directoris created and a log file ("logs/churn_library.log") too, there you could see in details the results of the tests.
if there is any error in the process, the run wouldn't complete, it will throw an error and log it too.

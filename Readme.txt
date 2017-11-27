Navodila za namestitev:

This is the code for the conference scheduler described in the paper Automatic paper assignment to conference 
schedule using text mining and network analysis. Most of the code was developed for the paper Co-bidding graphs
for constrained clustering. To install the code, use the following instructions:

1.) Necessary software:
		-Python 3.4 or newer (https://www.python.org/downloads/)
		-All needed addons listed in requirements.txt
			
2.) Pull the code from github

3.) Confiture the settings
    The file diploma/settings_secret.py contains the database settings. An SQLite database is used
    by default. It also contains the secret key, which should be changed before deploying the application
    on a publicly available webserver.
    The file app/classes/dataset_manager.py contains the variable self.DATASET_PATH, which should be 
    changed to point to the folder containing papers in .txt format.

4.) Run Django migrations with python manage.py migrate and add an admin user with python manage.py createsuperuser

5.) Run the server with manage.py runserver
    
Upon running the server, you can log in. After that, create a conference and a schedule structure with the "schedule" tab, 
then access the clustering with the "import data" tab. Other tabs use the clustering methods presented in the older paper.
    
Most of the code described in the paper is located in:
   /app/classes/dataset_manager.py for loading files and extracting features
   /app/classes/clusterer_new.py for constrained clustering
   /app/views/views_import_page.py for the web application code
   

   
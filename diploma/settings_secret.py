__author__ = 'Tadej'
import os

class secret(object):
    SECRET_KEY = 'zp%=yxqq+rl$fep&!=ut7lmj_9#tx0)(d(b2=8x0mkfr33bvev'
    DATABASE = { # Changed to SQLite3 for easier development
        'ENGINE': 'django.db.backends.sqlite3',  # Add 'postgresql_psycopg2', 'mysql', 'sqlite3' or 'oracle'.
        'NAME': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'conf-sc.db'),
        'USER': '',  # Not used with sqlite3.
        'PASSWORD': '',  # Not used with sqlite3.
        'HOST': '',  # Set to empty string for localhost. Not used with sqlite3.
        'PORT': '',  # Set to empty string for default. Not used with sqlite3.
    }


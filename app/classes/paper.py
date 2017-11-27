__author__ = 'Tadej'
class paper(object):
    authors = []
    title = ""
    abstract = ""
    submission_id = 0
    def __init__(self, title, authors, abstract, submission_id):
        self.authors = authors
        self.title = title
        self.abstract = abstract
        self.submission_id = submission_id
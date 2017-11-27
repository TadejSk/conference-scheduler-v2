__author__ = 'Tadej'
import xlrd
import xlwt
import csv
from ..models import Author, Paper
from .paper import paper
from diploma.settings import MEDIA_ROOT
class raw_data(object):

    class PaperGraph(object):

        class Connection(object):
            paper1 = None
            paper2 = None
            weight = 0
            def __init__(self, paper1, paper2, weight):
                self.paper1 = paper1
                self.paper2 = paper2
                self.weight = weight


        connections = []
        nodes = []
        def __init__(self):
            #self.nodes = papers
            self.connections = []

        def add_node(self, paper):
            self.nodes.append(paper)

        def add_connection(self, paper1, paper2, weight):
            """
            Adds a new connection. If a connection already exists, then the weight is added to the existing connection's
            weight. If the weight is 0, the connection will not be created
            """
            if weight == 0:
                return
            for con in self.connections:
                if (con.paper1 == paper1 and con.paper2 == paper2) or (con.paper1 == paper2 and con.paper2 == paper1):
                    con.weight += weight
                    return
            connection = self.Connection(paper1, paper2, weight)
            self.connections.append(connection)

    accepted=None
    assigments=None
    accepted_papers_list = []
    graph = None
    YES_WEIGHT = 2
    MAYBE_WEIGHT = 1
    CONFILCT_WEIGHT = 0
    def __init__(self, accepted_path, assigments_path):
        if accepted_path is not None:
            self.accepted = xlrd.open_workbook(accepted_path)
        if assigments_path is not None:
            self.assigments = assigments_path


    def parse_accepted(self):
        """
        This function parses the xml file saved in self.accepted and generates a list of all papers contained in that
        file. This list is then saved into self.accepted_papers_list
        :return raw_data
        """
        sheet = self.accepted.sheet_by_index(0)
        for row in range(1,sheet.nrows):
            id = sheet.cell_value(rowx = row, colx = 0)
            title = sheet.cell_value(rowx = row, colx = 2)
            abstract = sheet.cell_value(rowx = row, colx = 3)
            authors = []
            for author in str(sheet.cell_value(rowx = row, colx = 1)).replace(" and ",", ").replace(".","").split(", "):
                if(author.endswith(' ')):
                    author = author[:-1]
                authors.append(author)
            p = paper(authors=authors, abstract=abstract, title=title, submission_id = id)
            self.accepted_papers_list.append(p)
        return self


    def write_accepted(self, papers_list):
        """
        This function writes an xml file from all papers currently imported
        :param papers_list: Paper
        :return: xlwt.Workbook()
        """
        wb = xlwt.Workbook()
        sheet = wb.add_sheet('Page 1')
        sheet.write(0,0,'ID')
        sheet.write(0,1,'Authors')
        sheet.write(0,2,'Title')
        sheet.write(0,3,'Abstract')
        for i, paper in enumerate(papers_list):
            sheet.write(i+1, 0, paper.submission_id)
            sheet.write(i+1, 1, '')
            sheet.write(i+1, 2, paper.title)
            sheet.write(i+1, 3, paper.abstract)
        return wb


    def parse_assignments(self):
        """
        Generates a graph of papers. In this graph, two papers are connected, if a reviewer expressed interset in reviewing
        both of them. The connections are weighted depending on the number of connections, and the interest expressed by
        the reviewer (yes or maybe). Returns a list describing the created graph.
        :return: list[list[int, int, int]]
        """
        self.graph = self.PaperGraph()
        with open(self.assigments, 'r') as file:
            reader = csv.reader(file, dialect='excel')
            matrix = []
            for rowi,row in enumerate(reader):
                if not row[0].isdigit():
                    continue
                id = -1
                row_list = []
                for coli,col in enumerate(row):
                    if coli == 0:
                       row_list.append(col)
                    else:
                        if col == 'maybe':
                            row_list.append(self.MAYBE_WEIGHT)
                        elif col == 'yes':
                           row_list.append(self.YES_WEIGHT)
                        else:
                            row_list.append(0)
                matrix.append(row_list)
            rows = len(matrix)
            cols = len(matrix[0])
            for col in range(cols):
                # The first column only contains ids and can be skipped
                if col == 0:
                    continue
                for row in range(rows):
                    id = matrix[row][0]
                    for row2 in range(row+1,rows):
                        id2 = matrix[row2][0]
                        self.graph.add_connection(id, id2, matrix[row][col] * matrix[row2][col])
                        #if(matrix[row][col] * matrix[row2][col] != 0):
                        #    print("Connected", id, id2, matrix[row][col] * matrix[row2][col])
            # This now contains a graph connecting the papers according to their reviewers
            # This now needs to be saved into the database
            # Serialize connections as a list of lists
            connection_list = []
            for connection in self.graph.connections:
                connection_list.append([int(connection.paper1), int(connection.paper2), connection.weight])
            return connection_list, rows, cols



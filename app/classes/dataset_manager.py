from collections import Counter
from sklearn import cluster
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from diploma.settings import BASE_DIR
from .common_adjectives import COMMON_ADJECTIVES
from .node2vec import *
from .node2vec_main import main as nv2_main
from sklearn.feature_selection import SelectKBest
from lxml import etree
import xml.etree.ElementTree
import operator
import os
import re
import gensim
import matplotlib.pyplot as plt
import nltk
import scipy

__author__ = 'Tadej'

##############################################
# Change self.DATASET_PATH to data directory #
##############################################
class DatasetManager(object):
    def __init__(self):
        self.raw_data = []
        self.raw_data_titles = []
        self.raw_data_classes = []
        self.paper_references = []

        self.DATASET_PATH = os.path.join(BASE_DIR, 'raw_data_without_ecml')
        #self.DATASET_PATH = os.path.join(BASE_DIR, 'raw_data_just_ecml')
        # self.DATASET_PATH = "./data/"
        self.tfidf = None
        self.tfidf_feature_names = None
        self.classifier = None
        self.pos_tagged_data = []
        self.term_candidates = []
        self.term_graph = nx.Graph()
        self.node2vec_embeddings = []
        self.file_names = []
        self.paper_terms = []

    def load_raw_data(self):
        """
        Loads data from the dataset folder and saves it into self.raw_data
        The classes are saved into self.raw_data_classes. They correspond to
        the subfolder of the textfile, since the dataset is structured by
        splitting different classes into different subfolders.
        """
        for path, sub_folder, files in os.walk(self.DATASET_PATH):
            for file in files:
                if file.endswith(".txt") and file != "w2_.txt":
                    print(file)
                    txt_path = str(os.path.join(path, file))
                    print(txt_path)
                    with open(txt_path, 'r') as f:
                        text = f.read()
                        # Remove before abstract to remove authors/publication details
                        if 'Abstract' in text:
                            text = text.split('Abstract', 1)[1]
                        # Remove references
                        # Split by 'references', discard the last part and join the rest
                        if 'References' in text:
                            before_references = 'References'.join(text.split('References')[:-1])
                            after_references = text.split('References')[-1]
                            # Extract references
                            references = self.parse_references(after_references)
                            text = before_references
                        text = text.encode('ascii', 'replace')
                        self.file_names.append(file)
                        self.raw_data.append(text)
                        self.raw_data_classes.append(os.path.basename(os.path.dirname(txt_path)))
                        self.paper_references.append(references)
                        self.raw_data_titles.append(file[:-4])


    def create_bidding_graph_xml(self, file):
        """
        Reads the biddings xml file and creates a bidding graph. The bidding graph can then be passed to the
        create_probability_matrix_xml function to create the PR probability matrix, or to the create_doc2vec_matrix
        for node2vec feature extraction
        :return:
        """
        parser = etree.XMLParser(recover=True)
        tree = xml.etree.ElementTree.parse(file, parser=parser).getroot()
        i = 0
        # A row starts with a Number, is then followed by 8 Strings and ends with a Number.
        # This number is the preference we want
        # The data is:
        # Paper Id, Number, interesting
        # Title, String, intersting for debug and visualization
        # Track, String, could be interesting, although it mostly seems to be research
        # Primary subject area, String, could be interesting
        # First Name, String
        # Last name, String
        # email, String
        # organization, String - These must all be anonymized - can just be removed
        # current bid, String, not interesting
        # current bid value, Number, interesting
        start = 0
        pos = 0
        entities = []
        entity = []
        try:
            for neighbor in tree.iter():
                i += 1
                data = neighbor.text
                if not 'Data' in neighbor.tag:
                    continue
                # ignore the first 11 - they just describe the text
                if start < 11:
                    start += 1
                    continue
                if pos > 9:
                    pos = 0
                    entities.append(entity[:])
                    # print('entity', entity) # Attempting to print entity throws unicode error
                    entity = []
                pos += 1
                entity.append(data)
                if i > 150:
                    pass
                    # break
            entities.append(entity)  # Add the last one
        except UnicodeEncodeError:
            print('broke on', i)
            print(len(entities))
        print('done')
        names_dict = {}
        papers_dict = {}
        paper_ids = set()  # Used as vertices when constructing graph
        for entity in entities:
            print('considering', entity[0])
            if entity[0] not in paper_ids:
                print('added', entity[0])
                paper_ids.add(entity[0])
            if entity[6].lower() not in names_dict.keys():
                names_dict[entity[6].lower()] = 0
                papers_dict[entity[6].lower()] = []
            else:
                names_dict[entity[6].lower()] += 1
                papers_dict[entity[6].lower()].append((entity[0], entity[9]))
        print(names_dict)
        print(len(entities))
        print(papers_dict)
        # papers_dict now contains:
        # {email : [(paper, bid_value), ....], ...}
        # This should be enough to build a graph
        # First add nodes
        bidding_graph = nx.Graph()
        bidding_graph.add_nodes_from(paper_ids)
        print(bidding_graph.nodes())
        # Then connect nodes if the same reviewer bidded for them
        for reviewer, bids in papers_dict.items():
            for i in range(len(bids)):
                for j in range(i, len(bids)):
                    bid1 = bids[i]
                    bid2 = bids[j]
                    # Add edge
                    if int(bid1[1]) * int(bid2[1]) > 0:
                        bidding_graph.add_edge(bid1[0], bid2[0], weight=min(int(bid1[1]), int(bid2[1])))
        # print(bidding_graph.edges())
        return bidding_graph

    def get_papers_xml(self, file):
        """
        Reads abstracts (full text to be added later if possible) from the papers xml file and returns them
        :return:
        """
        parser = etree.XMLParser(recover=True)
        tree = xml.etree.ElementTree.parse(file, parser=parser).getroot()
        i = 0
        # The data is:
        # Paper ID
        # Paper Title
        # Track Name
        # Abstract
        # Author Names
        # Author Emails
        # Subject Areas
        # Conflict Reasons
        # Files
        # Supplementary File
        # Reproducible research paper (RR)
        # Right to withdraw
        # Relevant columns are 1, 2, 4, possibly 7 (subject areas)
        start = 0
        pos = 0
        entities = []
        entity = []
        try:
            for neighbor in tree.iter():
                i += 1
                data = neighbor.text
                if not 'Data' in neighbor.tag:
                    continue
                # ignore the first 13 - they just describe the text
                if start < 13:
                    start += 1
                    continue
                # print(data)
                # print(data)
                if pos > 11:
                    pos = 0
                    entities.append(entity[:])
                    # print('entity paper', entity)
                    entity = []
                pos += 1
                entity.append(data)
                if i > 150:
                    pass
                    # break
            entities.append(entity)  # Add the last one
        except UnicodeEncodeError:
            print('broke at', data)
            print(len(entities))
        return entities

    def create_probability_matrix_xml(self, bidding_graph):
        """
        Creates a probability matrix from a bidding graph to be passed to the get_pr function
        In an unweighted graph, this is simply: M[i][j] = 1/OutboundConnections(j) if j links to i, or 0 otherwise
        In a weighted graph, it is therefore a*(1/OutCon(j)), where a is the weight of the connection from i to j
        Since the weight is the probability, it should be Weightij/Sum(weights of a node)
        :return:
        """
        prob_matrix = []
        for nodei in bidding_graph.nodes():
            probs = []
            for nodej in bidding_graph.nodes():
                if nodei == nodej:
                    probs.append(0)
                else:
                    if len(bidding_graph.neighbors(nodei)) == 0:  # Empty node
                        probs.append(0)  # Unconnected nodes would otherwise result in division by 0
                    else:
                        if nodej not in bidding_graph.neighbors(nodei):
                            probs.append(0)
                        else:
                            neighbors_dict = bidding_graph[nodei]
                            weights = [x[1]['weight'] for x in neighbors_dict.items()]
                            print(weights)
                            # probs.append(self.connection_weight(nodei, nodej)/self.total_weights[nodei])
                            probs.append(bidding_graph[nodei][nodej]['weight'] / sum(weights))
            prob_matrix.append(probs)
        return prob_matrix

    def get_pr(self, bidding_graph, start_node, prob_matrix, damping_factor, num_iterations):
        """
        The pagerank of all papers will be stored in a vector PR[]
            PR[i] = pagerank of i
        Initial PR[i] is defined as 1/N, where N is the number of nodes in a graph (in this case the number of papers)
        The matrix M is the adjencency matrix, and Mij is defined as probability of reaching i from j
        E is a vector of zeroes, except for the value matching the starting node, where it is one
        We can then calculate PR as:
            PR' = (1-d)*E + d*M*PR
        """
        # Probability matrix is stored in self.prob_matrix
        # Isolated nodes (nodes with no connections) need to be treated seperately, since by normal calculation
        #   their pagerank does not sum to 1
        PR = []
        E = []
        row = 0
        for index, node in enumerate(bidding_graph.nodes()):
            if node == start_node:
                E.append(1)
                PR.append(1)
                row = index
            else:
                E.append(0)
                PR.append(0)
        if sum(prob_matrix[row]) < 0.1:
            return np.array(PR)
        # for none in self.nodes:
        #    PR.append(1/len(self.nodes))
        PR = np.array(PR)
        E = np.array(E)
        for iteration in range(1, num_iterations):
            PR = E * (1 - damping_factor) + ((damping_factor) * np.transpose(prob_matrix).dot(PR))
        return PR

    def parse_xml(self):
        entities = self.get_papers_xml(os.path.join(self.DATASET_PATH,'ecmlpkdd-17-papers.xls'))
        print(entities[-1])
        bidding_graph = self.create_bidding_graph_xml(os.path.join(self.DATASET_PATH,'Bids.xls'))
        print('nodes', bidding_graph.nodes())
        prob_matrix = self.create_probability_matrix_xml(bidding_graph)
        print(prob_matrix)
        pr_dict = {}
        for node in bidding_graph.nodes():
            page_rank = self.get_pr(bidding_graph, node, prob_matrix, 0.85, 20)
            pr_dict[node] = page_rank
        papers = []
        for entity in entities:
            # Not all papers are accepted, so we have to skip those that aren't
            paper_id = entity[0]
            paper_title = entity[1]
            paper_abstract = entity[3]
            if paper_id not in pr_dict.keys():
                print('skipped', paper_id)
                continue
            paper_pr = pr_dict[paper_id]
            papers.append([paper_id, paper_title, paper_abstract, paper_pr])
        #for paper in papers:
        #    print('final', paper)
        #print('pr dict', pr_dict)
        return papers

    def parse_references(self, text):
        """
        Parses references from text. Text is assumed to be only the part of the
        paper containing references, ie. text.split('References')[-1]. Found
        references are returned as an array of titles.
        This seems to work very well at the moment, even if the approach is a
        bit basic. There is a problem where converting pdfs to text does not
        convert all the references.
        """
        # print(len(text))
        # print(text)
        # print('--------------------------')
        # Split parts by .
        parts = text.split('.')
        for i in range(len(parts)):
            parts[i] = parts[i].encode('ascii', 'replace')
            parts[i] = parts[i].decode('ascii')
            # remove leading spaces
            # print(parts[i])
            while len(parts[i]) > 1 and parts[i][0] == ' ':
                parts[i] = parts[i][1:]
        # Filter out obvious non titles
        filtered_parts = []
        for part in parts:
            # print(part)
            # in papers, \x0C signifies special formatting, like a header with the title of the paper on
            # the top of the page, which means this isn't a reference but a  header that is present when
            # references are split among multiple pages
            if re.search('\x0C', part):
                # print('filtered out', part)
                continue
            if re.search('[a-zA-Z]*, [A-Z]$', part):
                # print('filtered out', part)
                continue
            # Papers with titles shorter that 5 characters are extremely rare, and this helps eliminate
            # parsing errors
            if len(part) < 5:
                continue
            # Remove everything not starting with a capital letter. In the testing database, every paper
            # title started with a capital letter, so this only removes non titles
            if not re.search('^[A-Z]', part):
                continue
            # "In" at the start indicates that the part refers to a publication
            if re.search('^In', part) or re.search('^In:', part):
                continue
            # Letters at the end indicate year of publication
            if re.search('[0-9]$', part):
                continue
            # Remove single word parts, as single word titles are extremely uncommon
            if not re.search(' ', part):
                continue
            if re.search('and [A-Z]$', part):
                continue
            # A large comma to word ratio indicates that this is the list of authors and not a title
            comma_ratio = part.count(',') / len(part.split(' '))
            if comma_ratio > 0.3:
                continue
            filtered_parts.append(part)
        # Some common journal names
        filtered_parts = [item for item in filtered_parts if not 'ieee' in item.lower()]
        filtered_parts = [item for item in filtered_parts if not 'springer' in item.lower()]
        filtered_parts = [item for item in filtered_parts if not 'mit' in item.lower().split(' ')]
        filtered_parts = [item for item in filtered_parts if not 'mit.' in item.lower().split(' ')]
        filtered_parts = [item for item in filtered_parts if not 'arxiv' in item.lower()]
        filtered_parts = [item for item in filtered_parts if not 'press' in item.lower().split(' ')]
        filtered_parts = [item for item in filtered_parts if not 'press,' in item.lower().split(' ')]
        filtered_parts = [item for item in filtered_parts if not 'cambridge' in item.lower()]
        filtered_parts = [item for item in filtered_parts if not 'acm,' in item.lower().split(' ')]
        filtered_parts = [item for item in filtered_parts if not 'acm' in item.lower().split(' ')]
        filtered_parts = [item for item in filtered_parts if not 'journal' in item.lower()]
        filtered_parts = [item for item in filtered_parts if not 'technical report' in item.lower()]
        filtered_parts = [item for item in filtered_parts if not 'http:' in item.lower()]
        filtered_parts = [item for item in filtered_parts if not 'https:' in item.lower()]
        return filtered_parts

    def create_reference_graph(self):
        """
        Constructs a graph from references. Requires self.parse_references to be called
        first, which extracts references from papers and stores them in self.paper_references
        Nodes are papers. Two nodes share an edge if they cite the same paper.
        """
        reference_graph = nx.Graph()
        for i in range(len(self.raw_data_titles)):
            reference_graph.add_node(i)
        for i in range(len(self.paper_references)):
            for j in range(i + 1, len(self.paper_references)):
                intersection = self.paper_references[i].intersection(self.paper_references[j])
                if len(intersection) > 0:
                    reference_graph.add_edge(i, j, weight=len(intersection), inverse_weight=1 / len(intersection))
        # Remove edges where weight is too small
        to_remove = []
        bound = 1.0
        for edge in reference_graph.edges_iter(data='weight'):
            if edge[2] < bound:
                to_remove.append((edge[0], edge[1]))
        reference_graph.remove_edges_from(to_remove)
        self.reference_graph = reference_graph
        # Remove nodes with no edges
        degrees = reference_graph.degree()
        no_edges = [x for x in degrees if degrees[x] == 0]
        reference_graph.remove_nodes_from(no_edges)
        # nx.draw(self.reference_graph, node_color='c', edge_color='k', with_labels=True)
        # plt.draw()
        # plt.show()
        """
        # Evaluation code
        # Calculate distances
        classes_all = self.raw_data_classes
        first_class = classes_all[0]
        data_by_classes = []
        # lengths = nx.shortest_path_length(self.reference_graph, weight='inverse_weight')
        lengths = nx.shortest_path_length(self.reference_graph)
        # Split the data into classes
        random_classes = True
        data = []
        for i, classes in enumerate(classes_all):
            if not classes == first_class:
                first_class = classes
                data_by_classes.append(data[:])  # [:] copies the list
                data = []
            if i >= len(classes_all):
                break
            rand = random.randrange(0, len(classes_all))
            if random_classes:
                data.append(rand)  # Randomized for comparison
            else:
                data.append(i)
        avg_distances_by_class = []
        no_matches = 0
        for class_papers in data_by_classes:
            distances = []
            for i in range(len(class_papers)):
                for j in range(i + 1, len(class_papers)):
                    try:
                        distance = lengths[class_papers[i]][class_papers[j]]
                        distances.append(distance)
                    except KeyError:
                        pass
            if len(distances) == 0:
                # print('no matches')
                no_matches += 1
            else:
                # print("avg distance " + str(sum(distances)/len(distances)))
                avg_distances_by_class.append(sum(distances) / len(distances))
        print('bound', bound)
        print('random', random_classes)
        print('total avg', np.mean(avg_distances_by_class))
        print('total std', np.std(avg_distances_by_class))
        print('total no matches', no_matches)
        print('')
        """

    def preprocess_data(self):
        """
        Cleans up the raw data to remove unhelpful words
        """
        for i, text in enumerate(self.raw_data):
            # Remove words starting with a number. Also remove numbers. These
            # should hold no value for ing or classification
            if type(text) != str:
                text = text.decode('ascii')
            preprocessed_text = re.sub('\s[0-9].*?\s', ' ', text)
            preprocessed_text = re.sub('[0-9]', '', preprocessed_text)
            # Remove words containing _. These usually usually subscripts
            # appearing in formulas, which hold very little semantic
            # information
            preprocessed_text = re.sub('\w*_\w*', '', preprocessed_text)
            self.raw_data[i] = preprocessed_text

    def get_tfidf(self):
        """
        Constructs a tfidf matrix from self.raw_data and saves it into
        self.tfidf. Feature names (words for each row) are saved into
        self.tfidf_feature_names.
        """
        vectorizer = TfidfVectorizer(decode_error='replace', strip_accents='unicode', stop_words='english',
                                     ngram_range=(1, 1))
        print('fit transforming')
        tfidf = vectorizer.fit_transform(self.raw_data)
        print('done fit transforming')
        names = vectorizer.get_feature_names()
        self.tfidf = tfidf
        for i, name in enumerate(names):
            name_ascii = name.encode('ascii', 'replace')
            names[i] = name_ascii
        self.tfidf_feature_names = np.array(names)
        return tfidf

    def select_best_words(self):
        """
        Takes a tfidf matrix x and selects only the top 10000 most discriminative words
        Also returns a mask of the selected words so that they can be displayed
        """
        selector = SelectKBest(k=30)
        best_words = selector.fit_transform(self.tfidf, self.raw_data_classes)
        mask = selector.get_support()
        self.tfidf = best_words
        self.tfidf_feature_names = self.tfidf_feature_names[mask]

    def train_dataset(self, x, y):
        self.classifier = linear_model.LogisticRegression()
        # print(x)
        # print(y)
        # print(x.shape)
        # print(y.shape)
        self.classifier.fit(x, y)

    def predict(self, x):
        preds = self.classifier.predict(x)
        return preds

    def get_probs(self, x):
        probs = self.classifier.predict_proba(x)
        probs_log = self.classifier.predict_log_proba(x)
        return probs, probs_log

    def pos_tag_data(self, load=False):
        # self.raw_data = self.raw_data[:40]
        print("Starting PoS tagging")
        pos_tagged_data = []
        for i, data in enumerate(self.raw_data):
            print('data length was', len(data))
            tokenized = nltk.word_tokenize(data)
            pos_tagged = nltk.pos_tag(tokenized)
            pos_tagged_data.append(pos_tagged)
        self.pos_tagged_data = pos_tagged_data
        # print(pos_tagged_data[0])

    def find_candidate_terms(self, pos_tagged_data=None):
        """
        Finds candidates for term extraction.
        The candidates are all n-grams that match a certain form.
        Requires pos_tag_data to be ran first, as the form is determined from PoS tags.
        """
        if pos_tagged_data == None:
            pos_tagged_data = self.pos_tagged_data
        # candidate_forms = [['JJ', 'NN'], ['NN', 'NN'], ['JJ', 'JJ', 'NN'], ['JJ', 'NN', 'NN'], ['NN', 'NN', 'NN'], ['N']]
        candidate_forms = [['JJ', 'NN'], ['NN', 'NN'], ['JJ', 'JJ', 'NN'], ['JJ', 'NN', 'NN'], ['NN', 'NN', 'NN']]
        # Term candidates will be a 2d matrix, with each row belonging to a paper
        # This is needed, since term extraction uses information on which
        #   and how many papers a candidate appears in
        term_candidates = []
        for data in pos_tagged_data:
            candidates = []  # Candidates of an individual paper
            for i in range(len(data)):
                for candidate_form in candidate_forms:
                    ok = True
                    possible_candidate = ""
                    for j in range(len(candidate_form)):
                        if i + j >= len(data):
                            ok = False
                            break  # If the remaining data is too short to match this candidate, skip it
                        # possible_candidate.append(data[i+j][0])
                        possible_candidate += data[i + j][0] + " "
                        if not data[i + j][1].startswith(candidate_form[j]):
                            ok = False  # We discovered a mismatch, so the data cannot be in the proper form to be a candidate
                            # print("broke on" + data[i+j][1])
                            break
                    if ok == True:  # If no mismatch is found the form is good and we found a candidate
                        candidates.append(possible_candidate[:-1])
                        # print('candidate', possible_candidate[:-1])
                        # print("Found: " + str(possible_candidate))
                        break
            term_candidates.append(candidates)
        return term_candidates
        # print(candidates)

    def term_extraction_from_paper_candidates(self, terms, term_candidates):
        if term_candidates == None:
            term_candidates = self.term_candidates
        #print("term_candidates len 2", len(term_candidates))
        #print("term len", len(terms))
        for paper_candidates in term_candidates:
            valid_paper_terms = []
            paper_candidates_set = set(paper_candidates)
            #print("paper candidates len", len(paper_candidates))
            # Remove candidates that weren't actual terms
            valid_paper_terms = []
            for term in terms:
                if term[0] in paper_candidates_set:
                    valid_paper_terms.append(term)
            #print("Found valid terms:", len(valid_paper_terms))
            self.paper_terms.append(valid_paper_terms)

    def term_extraction_from_candidates(self, term_candidates=None, filter=True):
        """
        This method identifies probable terms from candidates.
        It requires the find_candidate_terme method to be ran first.
        """
        if term_candidates == None:
            term_candidates = self.term_candidates
        # Check how many papers the candidate appears in
        # If we want the terms to correspond to categories, the term should appear
        #   in more than one paper - at least 3 for a useable category
        # We also don't want terms that appear in to many papers. They might be
        #   valid terms, but they do not differentiate paper categories well.
        candidate_counts = {}
        for paper_candidates in term_candidates:
            # lowercase all items
            for i in range(len(paper_candidates)):
                paper_candidates[i] = paper_candidates[i].lower()
            # Remove duplicates by converting to set
            # print(paper_candidates)
            paper_candidates_set = set(paper_candidates)
            # Count how many papers a candidate appears in
            for candidate in paper_candidates_set:
                if candidate not in candidate_counts.keys():
                    candidate_counts[candidate] = 1
                else:
                    candidate_counts[candidate] += 1
        # Remove terms appearing in < 3 papers
        if filter:
            candidate_counts = {key: item for key, item in candidate_counts.items() if item >= 5}
        # Remove terms appearing in > 15% papers
        if filter:
            candidate_counts = {key: item for key, item in candidate_counts.items() if
                                item <= 0.15 * len(self.raw_data)}
        # candidate_counts = {key: item for key, item in candidate_counts.items() if item <= 20}
        # Remove terms containing mathematical notation - formulas aren't very interesting here. Also remove
        # dots, since they usually indicate surnames
        candidate_counts = {key: item for key, item in candidate_counts.items() if
                            not re.search(r'[.\+/*=()\[\]<>]', key)}
        # candidate_counts = {key: item for key, item in candidate_counts.items() if re.search(r'[=]', key)}
        # Remove terms with common adjectives
        # List obtained from http://www.talkenglish.com/vocabulary/top-500-adjectives.aspx
        # Removed "deep" and "big"
        common_adjectives = COMMON_ADJECTIVES
        for adjective in common_adjectives:
            candidate_counts = {key: item for key, item in candidate_counts.items() if not adjective in key}
        # Remove short (<= 5 letters) candidates. These are mostly caused by errors in parsing
        # (for example, x_i can parse into 'x i')
        candidate_counts = {key: item for key, item in candidate_counts.items() if len(key) > 5}
        # Remove candidates containing journal names, since that doesn't correspond to content
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'ieee' in key}
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'springer' in key}
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'mit' in key}
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'arxiv' in key}
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'press' in key}
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'cambridge' in key}
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'university' in key}
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'acm' in key}
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'journal' in key}
        # Remove appendix
        candidate_counts = {key: item for key, item in candidate_counts.items() if not 'appendix' in key}
        # Remove candidates where one word is a single letter
        # Those candidates are usually things like matrix A, solution W or graph G  - not very useful
        candidate_counts_new = {}
        for key, item in candidate_counts.items():
            words = key.split(' ')
            ok = True
            for word in words:
                if len(word) == 1:
                    ok = False
                    break
            if ok:
                candidate_counts_new[key] = item
        candidate_counts = candidate_counts_new
        # Also check how often (on average) an individual candidate appears in
        # a single paper. Needs to be weighted by paper length. Important terms
        # should appear much more often than unimportant terms.
        """
        # MEAN
        flattened_candidates = []
        for i, paper_candidates in enumerate(term_candidates):
            for candidate in paper_candidates:
                flattened_candidates.append((candidate, len(self.raw_data[i])))
        total_occurances = {}
        for candidate, paper_len in flattened_candidates:
            if candidate not in total_occurances.keys():
                total_occurances[candidate] = 0.0
            else:
                total_occurances[candidate] += 1/paper_len
        #total_occurances = dict(Counter(flattened_candidates))
        occurance_ratios = {}
        for key, item in candidate_counts.items():
            occurance_ratios[key] = total_occurances[key]/item
        """
        # MEDIAN - try with just sum now
        # Count terms in each paper
        paper_term_counts = []
        for i, paper_candidates in enumerate(term_candidates):
            counts = Counter(paper_candidates)
            for key, item in counts.items():
                counts[key] = item / len(self.raw_data[i])
            paper_term_counts.append(counts)
        # Count for term
        flattened_terms = []
        for i, paper_candidates in enumerate(term_candidates):
            for candidate in paper_candidates:
                flattened_terms.append(candidate)
        flattened_terms = set(flattened_terms)
        term_counts = {}
        for term in flattened_terms:
            term_counts[term] = []
            for counts in paper_term_counts:
                if term in counts:
                    term_counts[term].append(counts[term])
        # total_occurances = dict(Counter(flattened_candidates))
        occurance_medians = {}
        for key, item in term_counts.items():
            occurance_medians[key] = np.sum(item)   # CHANGE THIS TO NP.MEDIAN FOR MEDIANS
        sorted_by_occurances = sorted(candidate_counts.items(), key=operator.itemgetter(1), reverse=True)
        final_features = {}
        for x in sorted_by_occurances:
            final_features[x[0]] = [x[1], occurance_medians[x[0]]]
        # sorted_by_ratios = sorted(final_features.items(), key=lambda x: x[1][1], reverse=False)
        sorted_by_ratios = sorted(final_features.items(), key=lambda x: x[1][1], reverse=True)
        both = {}
        for x in candidate_counts.items():
            # print(candidate_counts[x[0]])
            # print(occurance_medians[x[0]])
            both[x[0]] = (candidate_counts[x[0]], occurance_medians[x[0]])
            print('x[0] was', x[0])
        # FIND TERMS IN GENERAL CORPUS
        general_occurances = {}
        scores = {}
        num_iters_done = 0
        for x in candidate_counts.items():
            with open(os.path.join(self.DATASET_PATH, 'w2_.txt'), 'r') as f:
                #if num_iters_done >= 1000:
                #    break
                num_iters_done += 1
                term = x[0]
                found = 0
                #print('searching term', term, words)
                pattern = re.compile(re.escape(term.replace(' ', '\t')))
                for line in f:
                    break   # COMMENT FOR DETAILED SEARCH
                    #print('line', line, words)
                    #print('s', word, line)
                    if re.search(pattern, line):
                        print('found', term, 'in', line)
                        found = int(line.split('\t')[0])
                if found == 0:
                    print('did not find', term)
                else:
                    print('found', term, found)
                general_occurances[term] = found + 1
                #print('general occurances', general_occurances[term])
                #print('candidate counts', candidate_counts[term])
                #print('occurance medians', occurance_medians[term])
                scores[term] = 1 - 1/(np.log2(2 + (occurance_medians[term] * candidate_counts[term] / general_occurances[term])))
                print(term, 'scored', scores[term], 'go', general_occurances[term], 'co', candidate_counts[term], 'om', occurance_medians[term])
        #for term in flattened_terms:
        #    scores[term] = 1 - 1/(np.log2(2+(occurance_medians[term]*candidate_counts[term]/general_occurances[term])))
        #    print(term, 'scored', scores[term])
        sorted_by_both = sorted(both.items(), key=lambda x: x[1][0] * x[1][1], reverse=True)
        sorted_by_score = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        for item in sorted_by_score:
            term = item[0]
            print(term, scores[term])
        # for x in range(10):
        #    print(sorted_by_both[x])
        # for x in sorted_by_ratios:
        #    print(x)
        # print(str(x) + ", " + str(occurance_ratios[x[0]]))
        # Extract single word terms, as done in "Simple but Powerful Automatic Term Extraction Method",
        # which extracts single word terms from multi word terms
        preceeding_words = {}
        succeeding_words = {}
        for item in sorted_by_score:
            # Extract words from term
            term = item[0]
            words = term.split(" ")
            for i in range(len(words)):
                # Find succeeding words
                if i + 1 < len(words):
                    current_word = words[i]
                    next_word = words[i + 1]
                    if current_word not in succeeding_words.keys():
                        succeeding_words[current_word] = set()
                    succeeding_words[current_word].add(next_word)
                # Find preceeding_words
                if i - 1 >= 0:
                    current_word = words[i]
                    previous_word = words[i - 1]
                    if current_word not in preceeding_words.keys():
                        preceeding_words[current_word] = set()
                    preceeding_words[current_word].add(next_word)
        # Count unique occurances
        preceeding_word_counts = {}
        succeeding_word_counts = {}
        for key in preceeding_words.keys():
            preceeding_word_counts[key] = len(preceeding_words[key])
        for key in succeeding_words.keys():
            succeeding_word_counts[key] = len(succeeding_words[key])
        # print(preceeding_word_counts)
        # print(succeeding_word_counts)
        multiplied_counts = {}
        for key in preceeding_word_counts.keys():
            if key in succeeding_word_counts.keys():
                multiplied_counts[key] = preceeding_word_counts[key] * succeeding_word_counts[key]
        sorted_multiplied_counts = sorted(multiplied_counts.items(), key=operator.itemgetter(1))
        # for c in sorted_multiplied_counts:
        # print(c, multiplied_counts[c[0]])
        # Add the found terms to candidates of individual papers
        # This is necessary since constructing a graph of term co-occurances needs to know
        # which papers contained which terms, which is extracted from term_candidates
        ok_one_word = []
        # self.raw_data = None
        # self.pos_tagged_data = None
        for term, _ in sorted_multiplied_counts:
            total = 0
            for i in range(len(term_candidates)):
                add_total = False
                to_append = 0
                for candidate in term_candidates[i]:
                    if term in candidate:
                        add_total = True
                        to_append += 1
                        # print(to_append)
                        # break
                for _ in range(to_append):
                    pass
                    # term_candidates[i].append(term)
                # print('candidates', term_candidates[i])
                if add_total:
                    total += 1
            # print(term, total)
            # print('added', term, 'to', str(total))
            if total < 40:
                ok_one_word.append((term, 0))
        self.term_candidates = term_candidates
        # return ok_one_word
        # return sorted_multiplied_counts
        # return sorted_by_ratios + ok_one_word
        # print(len(self.term_candidates), len(self.term_candidates[0]))
        return sorted_by_score + ok_one_word
        #return ok_one_word
        #return sorted_by_occurances + ok_one_word
        # print('sbr', sorted_by_ratios)
        # return sorted_by_ratios

    def create_term_matrix(self, terms, term_candidates=None):
        """
        Creates a term matrix that can be used for machine learning.
        It examines term candidates of each paper, taking only the
        actual terms present in "terms" and constructs a matrix out
        of these terms
        """
        #print("TERM MATRIX")
        #print(terms)
        terms = terms
        #raise TypeError
        #print(len(terms))
        #print(len(term_candidates), len(term_candidates[0]), len(term_candidates[1]))
        if term_candidates == None:
            term_candidates = self.term_candidates
        actual_terms = []
        for term in terms:
            word = term[0]
            actual_terms.append(word)
        term_matrix = []
        for i, paper_term_candidates in enumerate(term_candidates):
            term_vector = []
            for actual_term in actual_terms:
                if actual_term in paper_term_candidates:
                    # term_vector.append(100*paper_term_candidates.count(actual_term)/len(self.raw_data[i]))
                    #term_vector.append(paper_term_candidates.count(actual_term) / len(self.raw_data[i]))
                    term_vector.append(paper_term_candidates.count(actual_term)) # Seems to work much better if not divided by distance
                else:
                    term_vector.append(0)
            #print('term vector', np.array(term_vector).shape)
            term_matrix.append(term_vector)
        #print(np.array(term_matrix).shape)
        #for m in term_matrix:
        #    print(m)
        #raise TypeError
        return term_matrix


    def create_term_graph(self, terms, term_candidates=None):
        """
        Creates a term graph from a list of terms. The nodes of the graph
        will be the provided terms. Two terms will be connected if they
        appear in the same paper.
        Terms are in the form:
        (term_text, [# of appearances, median frequency of use])
        """
        if term_candidates == None:
            term_candidates = self.term_candidates
        # print("Creating graph")
        # Add nodes
        self.term_graph = nx.Graph()
        for term in terms:
            # print('added', term[0])
            self.term_graph.add_node(term[0])
        # print(nx.nodes(self.term_graph))
        # Find terms that occur together in papers
        occurances = {}
        occurs_in_papers = {}
        for i, paper_candidates in enumerate(term_candidates):
            # Go through the candidates of each paper and connect every pair
            # of terms
            valid_terms = []
            paper_candidates_set = set(paper_candidates)
            # Remove candidates that weren't actual terms
            for term in paper_candidates:
                if self.term_graph.has_node(term):
                    valid_terms.append(term)
            # occurances holds how often the two terms appear in the paper
            for i in range(len(valid_terms)):
                for j in range(i + 1, len(valid_terms)):
                    # Do not connect same terms
                    if (valid_terms[i] == valid_terms[j]):
                        continue
                    # Ensure that we do not get (a, b) and (b, a) combinations of terms
                    if (valid_terms[i] > valid_terms[j]):
                        t1 = valid_terms[j]
                        t2 = valid_terms[i]
                    else:
                        t1 = valid_terms[i]
                        t2 = valid_terms[j]
                    if not (t1, t2) in occurances.keys():
                        occurances[(t1, t2)] = 1 / len(valid_terms)
                    else:
                        occurances[(t1, t2)] += 1 / len(valid_terms)
                    if not (t1, t2) in occurs_in_papers.keys():
                        occurs_in_papers[(t1, t2)] = set()
                    occurs_in_papers[(t1, t2)].add(i)
                    # if not self.term_graph.has_edge(valid_terms[i], valid_terms[j]):
                    #    self.term_graph.add_edge(valid_terms[i], valid_terms[j], weight=1.0)
                    # else:
                    #    self.term_graph[valid_terms[i]][valid_terms[j]]['weight'] += 1.0
        # Add connections between terms that occur together often enough
        avg_value = sum(occurances.values()) / len(occurances.values())
        stddev = np.std(list(occurances.values()))
        for term_pair, value in occurances.items():
            t1 = term_pair[0]
            t2 = term_pair[1]
            # print(value, avg_value, stddev)
            if value > avg_value + 10 * stddev:
                self.term_graph.add_edge(t1, t2, weight=len(occurs_in_papers[term_pair]))
                # if not self.term_graph.has_edge(t1, t2):
                #    self.term_graph.add_edge(t1, t2, weight=1.0)
                # else:
                #    self.term_graph[t1][t2]['weight'] += 1.0
                print('added', t1, t2, value, avg_value, stddev)
        # print('occurances', occurances)
        for term_pair, value in occurances.items():
            # print(valid_terms[i], valid_terms[j])
            t1 = term_pair[0]
            t2 = term_pair[1]
            if self.term_graph.has_edge(t1, t2):
                self.term_graph[t1][t2]['inverse_weight'] = 1.0 / float(self.term_graph[t1][t2]['weight'])
                self.term_graph[t1][t2]['inverse_weight'] *= self.term_graph[t1][t2]['inverse_weight']
        # Remove edges with small weights
        to_remove = []
        for edge in self.term_graph.edges_iter(data='weight'):
            if edge[2] < 100.0:
                # print('removed', edge[0], edge[1], edge[2])
                to_remove.append((edge[0], edge[1]))
        self.term_graph.remove_edges_from(to_remove)
        # Remove nodes with no edges
        degrees = self.term_graph.degree()
        no_edges = [x for x in degrees if degrees[x] == 0]
        self.term_graph.remove_nodes_from(no_edges)
        # plot_graph
        #print(len(nx.nodes(self.term_graph)))
        nx.draw(self.term_graph, node_color='c', edge_color='k', with_labels=True)
        #plt.draw()
        #plt.show()
        #print("done")

    def compare_terms_to_graph(self, all_terms_sorted, random_classes=False):
        """
        Compares terms from a set of papers to the graph to determine
        if the papers with the same labels in the ground truth data have
        terms from similar clusters.
        """
        print("comparing terms to graph")
        classes_all = self.raw_data_classes
        data_all = self.pos_tagged_data
        first_class = classes_all[0]
        data = []
        data_by_classes = []
        # Trying with weighted lengths, although unweighted produces pretty good results by itself
        lengths = nx.shortest_path_length(self.term_graph, weight='inverse_weight')
        # lengths = nx.shortest_path_length(self.term_graph)
        # Split the data into classes
        for i, classes in enumerate(classes_all):
            # print(classes)
            if not classes == first_class:
                # print('classes end')
                # print(len(data))
                first_class = classes
                data_by_classes.append(data[:])  # [:] copies the list
                # print("data_by_classes1 " + str(data_by_classes))
                data = []
                # print('------------')
                # print("data_by_classes2 " + str(len(data_by_classes)))
            if i >= len(data_all):
                break
            rand = random.randrange(0, len(data_all))
            added = []
            # print('rand', rand, classes_all[rand], classes_all[i])
            if random_classes:
                data.append(data_all[rand])  # Randomized for comparison
                # print(random_classes, 'class', classes_all[rand])
            else:
                # print(random_classes, 'class', classes_all[i])
                data.append(data_all[i])
        all_distances = []
        all_passes = []
        for class_data in data_by_classes:
            class_term_candidates = self.find_candidate_terms(class_data)
            # print(class_term_candidates)
            class_terms_orig = self.term_extraction_from_candidates(class_term_candidates, filter=False)
            #print('found terms')
            #for x in range(20):
            #    print(class_terms_orig[x])
            # Compare only the top N (10) terms
            class_terms = [x[0] for x in class_terms_orig]
            # Terms actually encountered when constructing the graph
            valid_terms = []
            # print(class_terms)
            """
            for term in all_terms_sorted:
                #print("-----" + term[0])
                if term[0] in class_terms:
                    if term[0] in self.term_graph.nodes():
                        #print(term[0])
                        valid_terms.append(term[0])
                        if len(valid_terms) >= 20:
                            break
            """
            #print("------------")
            for term in class_terms:
                if term in self.term_graph.nodes():
                    valid_terms.append(term)
                    #print(term)
                    if len(valid_terms) >= 5:
                        break
            distances = []
            passes = 0
            for term1 in range(len(valid_terms)):
                for term2 in range(term1 + 1, len(valid_terms)):
                    try:
                        # distances.append(len(paths[valid_terms[term1]][valid_terms[term2]]))
                        distances.append(lengths[valid_terms[term1]][valid_terms[term2]])
                        #print(valid_terms[term1], valid_terms[term2], lengths[valid_terms[term1]][valid_terms[term2]])
                        # print(lengths[valid_terms[term1]][valid_terms[term2]])
                    except KeyError:
                        # Since the terms should all be in the graph, this means there is no path between the two nodes
                        # Treat this as distance 10
                        # distances.append(10)
                        passes += 1
                        print(valid_terms[term1], valid_terms[term2], 'passed')
                        continue
            # print("valid_terms " + str(len(valid_terms)))
            # print("class_terms" + str(len(class_terms)))
            # print(distances)
            if len(distances) > 0:
                #print("avg distance " + str(sum(distances) / len(distances)), passes)
                all_distances.append(sum(distances) / len(distances))
            all_passes.append(passes)
            # Draw graph
            # positions=nx.spring_layout(self.term_graph)
            # nx.draw(self.term_graph, pos=positions, node_color='c',edge_color='k', with_labels=True)
            # nx.draw_networkx_nodes(self.term_graph, positions, nodelist=valid_terms, node_color='r')
            # plt.draw()
            # plt.show()
        return all_distances, all_passes

    def write_graph_edgelist(self, graph, filename):
        """
        Takes a graph and writes its edgelist to filename. This is
        a utility function used to convert the graph to a format
        readable by node2vec.
        For writing weighted graphs, bigger weight means more closely
        connected, so data=['weight'] is correct (not data=['inverse_weight'])
        """

        #print('asdf')
        # nx.write_weighted_edgelist(graph, filename)
        nx.write_edgelist(graph, filename, data=['weight'])
        return


    def create_doc2vec_matrix(self, data):
        documents = []
        for i, d in enumerate(data):
            #words = d.split(' ')
            #bigrams = []
            #for i in range(len(sents) - 1):
            #    bigrams.append(sents[i] + " " + sents[i + 1])
            documents.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(d), [i]))
        model = gensim.models.Doc2Vec(size=128, window=8, min_count=5, workers=4, iter=20)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count)
        doc2vec_matrix = []
        for i in range(len(data)):
            inferred_vector = model.infer_vector(data[i])
            doc2vec_matrix.append(inferred_vector)
        #print(doc2vec_matrix[0])
        #print(doc2vec_matrix[1])
        #print(doc2vec_matrix[2])
        #raise TypeError
        return np.array(doc2vec_matrix)


    def test_word2vec(self, terms):
        terms_words = set([x[0] for x in terms])
        #print(terms_words)
        sentences = []
        for data in self.raw_data:
            #print(type(data))
            sents = data.split(' ')
            bigrams = []
            for i in range(len(sents) - 1):
                bigrams.append(sents[i] + " " + sents[i + 1])
            sentences.append(bigrams)
        # print(sentences[0])
        model = gensim.models.Word2Vec(sentences, size=1000, window=200, min_count=5, workers=4, max_vocab_size=1000000)
        wordVocab = [k for k in model.wv.vocab]
        printed = []
        for word in wordVocab:
            if word in terms_words:
                printed.append(word)
        #print('loaded model')
        #print(printed)
        # X = model[model.wv.vocab]
        X = model[printed]
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)
        #plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
        #for i, txt in enumerate(printed):
        #    plt.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))
        #plt.show()

    def create_node2vec_embeddings(self, graph):
        """
        Creates node2vec embeddings from the given graph and returns them.
        The embeddings are a matrix where each line represents a vector embedding
        of the paper in the same space of raw_data.
        p and q are node2vec parameters
        """
        # edgelist = nx.generate_edgelist(graph, data=['weight'])
        # Calls the implementation of node2vec
        # args: graph, directed, p, q, num_walks, walk_length, dimensions, window_size, workers, iter, output
        #print('emb1')
        embeddings = nv2_main(graph, directed=False, p=2, q=0.5)
        #print('emb2')
        # embeddings.wv.save_word2vec_format('out_embeddings.txt')
        embedding_mat = []
        #print(embeddings.wv.vocab.keys())
        for i in range(len(self.raw_data)):
            if str(i) in embeddings.wv.vocab.keys():
                embedding_mat.append(embeddings[str(i)])
            else:
                embedding_mat.append(np.zeros(len(embeddings[list(embeddings.wv.vocab.keys())[0]])))
        #print('emb3')
        return embedding_mat

    def load_node2vec_embeddings(self, filename):
        ids = []
        vectors = []
        vector_dict = {}
        with open(filename, 'r') as file:
            first = True
            for line in file.readlines():
                # Skip first line which just contains metadata
                if first:
                    first = False
                    continue
                elements = line.split(' ')
                ids.append(elements[0])
                embedding_elements = []
                for element in elements[1:]:
                    embedding_elements.append(float(element))
                vectors.append(embedding_elements)
                vector_dict[int(elements[0])] = embedding_elements
        self.node2vec_embeddings = []
        for i in range(len(self.raw_data)):
            # The graph does not contain papers that were not in the graph.
            # This means that some papers will not have node2vec embeddings,
            # which is problematic. For now those papers are set to all 0
            if i in vector_dict.keys():
                self.node2vec_embeddings.append(vector_dict[i])

    def test_node2vec_embeddings(self, vectors=None):
        if vectors == None:
            vectors = self.node2vec_embeddings
        vector_dict = {}
        for i, vector in enumerate(vectors):
            vector_dict[i] = vector
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(vectors)
        # Find average distance between random samples
        distances = []
        for i in range(1000):
            r1 = random.randrange(0, len(vectors))
            r2 = random.randrange(0, len(vectors))
            dist = scipy.spatial.distance.cosine(vectors[r1], vectors[r2])
            distances.append(dist)
        #print('avg rand dist', np.mean(distances))
        # Find average inter class distance
        means = []
        for i in set(self.raw_data_classes):
            distances = []
            locations = [a for a, b in enumerate(self.raw_data_classes) if b == i]
            for l1 in range(len(locations)):
                for l2 in range(l1 + 1, len(locations)):
                    # Since the graph removes unconnected nodes it might not contain all papers. This is checked here
                    if locations[l1] not in vector_dict.keys() or locations[l2] not in vector_dict.keys():
                        continue
                    dist = scipy.spatial.distance.cosine(vector_dict[locations[l1]], vector_dict[locations[l2]])
                    distances.append(dist)
            # print('avg dist', np.mean(distances))
            if not (np.isnan(np.mean(distances))):
                means.append(np.mean(distances))
        # print(means)
        #print('avg dist', np.mean(means))
        #print('-------------')
        """
        plt.figure()
        nx.draw(self.reference_graph, node_color='c',edge_color='k', with_labels=True)
        plt.draw()
        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
        for i, txt in enumerate(ids):
            plt.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))
        plt.show()
        """

    def clustering(self, X, use_term_mat=False):
        """
        Runs affinity propagation clustering on X. If use_term_mat is true the clustering will use a special
        metric suitable for clustering based on extracted terminology.
        """
        # clusterer = cluster.KMeans(n_clusters=39)
        # compute distances
        if use_term_mat:
            term_mat = X
            distances = []
            for l1 in term_mat:
                line = []
                for l2 in term_mat:
                    zipped = zip(l1, l2)
                    sum = 0
                    for a1, a2 in zipped:
                        if a1 > 0 and a2 > 0:
                            sum += a1 + a2
                    line.append(sum)
                distances.append(line)
            clusterer = cluster.AffinityPropagation(affinity='precomputed')
            preds = clusterer.fit_predict(distances)

        else:
            preds2 = []
            clusterer = cluster.AffinityPropagation(preference=-12)
            preds = clusterer.fit_predict(X)
            aff_mat = clusterer.affinity_matrix_
            similarity_dict = {}
            for i, line in enumerate(aff_mat):
                cl = preds[i]
                n = 0
                sum = 0
                for j in range(len(line)):
                    if preds[j] == cl:
                        n += 1
                        sum += line[j]
                if cl not in similarity_dict.keys():
                    similarity_dict[cl] = [sum / n]
                else:
                    similarity_dict[cl].append(sum / n)
                # print('similarity, ' + str(preds[i]) + ": " + str(sum/n))
                preds2.append((preds[i], sum / n))
            for key in similarity_dict.keys():
                similarity_dict[key] = np.sum(similarity_dict[key]) / len(similarity_dict[key])
                #print('avg similarity ', str(key), str(similarity_dict[key]))
            sorted_preds2 = sorted(preds2, key=lambda x: x[1])
            #for p in sorted_preds2:
                #print('similarity, ' + str(p[0]) + ": " + str(p[1]))
            #print(clusterer.get_params())
        return preds

    @staticmethod
    def combine_features(X_list):
        """
        Takes a list of feature matrices and returns a combined feature matrix. Matrices are converted to np.arrays,
        normalized and stacked
        :param X_list: list
        :return: np.ndarray
        """
        print('combining')
        total_X = None
        for X in X_list:
            if total_X is None:
                total_X = X
                if type(total_X) != np.ndarray:
                    total_X = np.array(total_X)
            else:
                if type(X) != np.ndarray:
                    X = np.array(X)

                #print('shapes at merging', total_X.shape, X.shape)
                print(total_X.shape, X.shape)
                total_X = np.hstack((total_X, X))
        #total_X = normalize(total_X, axis=0, norm='l1')
        return total_X

    def load_terms_from_file(self, filename, num):
        term_candidates = self.find_candidate_terms()
        f = open('term_list.txt', 'r')
        terms = []
        for line in f:
            term = line.split(' ')[0] + ' ' + line.split(' ')[1]
            terms.append((term, line.split(' ')[2]))
            num -= 1
            if num == 0:
                break
        return self.create_term_matrix(terms, term_candidates)
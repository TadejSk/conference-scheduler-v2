import ast
import random
import networkx as nx
import numpy as np
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from skrebate import ReliefF
from sklearn import preprocessing
from sympy.stats.crv_types import LogisticDistribution

from app.classes import Clusterer, schedule_manager_class
from ..classes.dataset_manager import DatasetManager
from ..models import Paper, Conference
from ast import literal_eval
import os
from ..classes.clusterer_new import ClustererNew
from diploma.settings import BASE_DIR
from sklearn.model_selection import cross_val_score, KFold


def setup_clustering(request, papers,  X):
    conf = request.session.get('conf', {})
    if conf == {}:
        return redirect('/app/index/')
    # initialize clustering
    #papers = Paper.objects.filter(user=request.user, is_locked=False, conference=request.session['conf'])
    print('import_page all papers1', papers)
    settings = Conference.objects.get(user=request.user, pk=request.session['conf']).settings_string
    settings = ast.literal_eval(settings)
    paper_schedule = Conference.objects.get(user=request.user, pk=request.session['conf']).schedule_string
    paper_schedule = ast.literal_eval(paper_schedule)
    schedule_db = Conference.objects.get(user=request.user, pk=request.session['conf'])
    schedule = ast.literal_eval(schedule_db.schedule_string)
    clusterer = ClustererNew(papers=papers, schedule=schedule, schedule_settings=settings, paper_schedule=paper_schedule,
                          func='aff', input_matrices=X)
    clusterer.num_clusters = 10
    clusterer.set_cluster_function('aff')
    clusterer.create_dataset()
    print('schedule was', clusterer.schedule)
    print('paper schedule was', clusterer.paper_schedule)
    #raise TypeError
    return clusterer


@login_required
def import_page(request):
    papers = Paper.objects.filter(uses_advanced_features=True, user=request.user,
                                  conference=request.session['conf'])
    paper_titles = [paper.title for paper in papers]
    paper_pos_tagged = [len(paper.pos_tagged_text_content) > 0 for paper in papers]
    paper_term_count = [len(literal_eval(paper.extracted_terms)) for paper in papers]
    paper_clusters = [paper.simple_cluster for paper in papers]
    return render(request, 'app/import_page.html', {'paper_titles': paper_titles,
                                                    'paper_pos_tagged': paper_pos_tagged,
                                                    'paper_term_count': paper_term_count,
                                                    'num_elements': len(paper_titles),
                                                    'paper_clusters': paper_clusters})


def load_raw_data(request):
    dataset_manager = DatasetManager()
    dataset_manager.load_raw_data()
    dataset_manager.preprocess_data()
    print('loading raw data')
    for filename, raw_data, ground_truth_class, references in zip(dataset_manager.file_names, dataset_manager.raw_data,
                                                dataset_manager.raw_data_classes, dataset_manager.paper_references):
        if not Paper.objects.filter(uses_advanced_features=True, title=filename.split('.')[0], user=request.user,
                                    conference=request.session['conf']).exists():
            paper = Paper()
            paper.title = filename.split('.')[0]
            paper.text_content = raw_data
            paper.abstract = raw_data
            paper.paper_references = references
            print(paper.paper_references)
            paper.user = request.user
            paper.conference = Conference.objects.get(pk=request.session['conf'])
            paper.submission_id = 0
            paper.uses_advanced_features = True
            paper.length = 20
            paper.ground_truth_class = ground_truth_class
            paper.save()
    return redirect('/app/import_page')


def pos_tag_data(request):
    dataset_manager = DatasetManager()
    papers = Paper.objects.filter(uses_advanced_features=True, user=request.user,
                                  conference=request.session['conf']).order_by('title')
    for paper in papers:
        dataset_manager.raw_data.append(paper.text_content)
        dataset_manager.file_names.append(paper.title)
    dataset_manager.preprocess_data()
    dataset_manager.pos_tag_data()
    for title, pos_tagged_data in zip(dataset_manager.file_names, dataset_manager.pos_tagged_data):
        paper = Paper.objects.get(uses_advanced_features=True, user=request.user,
                                  conference=request.session['conf'], title=title)
        #print('adding pos data to', title)
        #print('added', pos_tagged_data)
        paper.pos_tagged_text_content = pos_tagged_data
        paper.save()
    return redirect('/app/import_page')

def find_terms(request):
    print("finding terms")
    dataset_manager = DatasetManager()
    papers = Paper.objects.filter(uses_advanced_features=True, user=request.user,
                                  conference=request.session['conf']).order_by('title')
    for paper in papers:
        dataset_manager.raw_data.append(paper.text_content)
        dataset_manager.file_names.append(paper.title)
        txt_pos_tagged_data = paper.pos_tagged_text_content
        pos_tagged_data = literal_eval(txt_pos_tagged_data)
        dataset_manager.pos_tagged_data.append(pos_tagged_data)
    candidates = dataset_manager.find_candidate_terms()
    terms = dataset_manager.term_extraction_from_candidates(candidates, filter=True)
    dataset_manager.term_extraction_from_paper_candidates(terms, candidates)
    for title, terms in zip(dataset_manager.file_names, dataset_manager.paper_terms):
        paper = Paper.objects.get(uses_advanced_features=True, user=request.user,
                                  conference=request.session['conf'], title=title)
        paper.extracted_terms = terms
        paper.save()
    return redirect('/app/import_page')


def load_term_matrix(dataset_manager):
    print("loading term matrix")
    return dataset_manager.load_terms_from_file(os.path.join(BASE_DIR, 'term_list.txt'), 100)

def create_reference_graph(request):
    print("constructing reference graph")
    dataset_manager = DatasetManager()
    papers = Paper.objects.filter(uses_advanced_features=True, user=request.user,
                                  conference=request.session['conf']).order_by('title')
    for paper in papers:
        dataset_manager.raw_data.append(paper.text_content)
        dataset_manager.file_names.append(paper.title)
        print('references', paper.paper_references)
        dataset_manager.paper_references.append(set(literal_eval(paper.paper_references)))
    dataset_manager.create_reference_graph()
    reference_graph = dataset_manager.reference_graph
    reference_graph_edgelist = nx.generate_edgelist(reference_graph)
    conf = Conference.objects.get(id=request.session['conf'])
    conf.reference_graph_edgelist = reference_graph_edgelist
    conf.save()
    return redirect('/app/import_page')

def import_xml(saved_papers):
    dataset_manager = DatasetManager()
    papers = dataset_manager.parse_xml()
    saved_papers_dict = {p.title: p for p in saved_papers}
    xml_papers_dict = {}
    print(saved_papers_dict.keys())
    for paper_id, paper_title, paper_abstract, paper_pr in papers:
        if 'Paper ' + paper_id in saved_papers_dict.keys():
            print('found', 'Paper ' + paper_id)
            print(paper_pr)
            xml_papers_dict['Paper ' + paper_id] = paper_pr
            saved_papers_dict['Paper ' + paper_id].abstract = paper_abstract
            saved_papers_dict['Paper ' + paper_id].save()
            #repr(paper_pr)
        else:
            print('not found', 'Paper ' + paper_id)
    return xml_papers_dict

def run_clustering(request):
    dataset_manager = DatasetManager()
    # papers = Paper.objects.filter(uses_advanced_features=True, user=request.user,
    #                              conference=request.session['conf']).order_by('title')
    papers = Paper.objects.filter(uses_advanced_features=True, user=request.user,
                                  conference=request.session['conf'])

    xml_papers_dict = import_xml(papers)
    for paper in papers:
        dataset_manager.raw_data.append(paper.text_content)
        dataset_manager.file_names.append(paper.title)
        dataset_manager.raw_data_classes.append(paper.ground_truth_class)
        txt_pos_tagged_data = paper.pos_tagged_text_content
        #print(len(txt_pos_tagged_data))
        if txt_pos_tagged_data == "":
            print("Run POS tagging first")
            pos_tag_data(request)
            txt_pos_tagged_data = paper.pos_tagged_text_content
        pos_tagged_data = literal_eval(txt_pos_tagged_data)
        dataset_manager.pos_tagged_data.append(pos_tagged_data)
        #print('references', paper.paper_references)
        dataset_manager.paper_references.append(set(literal_eval(paper.paper_references)))
    """
    # BIDDINGS
    bidding_X = []
    for paper in papers:
        if paper.title in xml_papers_dict.keys():
            print('appended', len(xml_papers_dict[paper.title]))
            bidding_X.append(xml_papers_dict[paper.title])
        else:
            print('appended zeros', len(list(xml_papers_dict.values())[0]))
            bidding_X.append(np.zeros(len(list(xml_papers_dict.values())[0])))
    bidding_X = np.array(bidding_X)
    #print(bidding_X.shape)
    #print(bidding_X[0])
    #print(bidding_X[1])
    #print(bidding_X[2])
    #print(bidding_X[3])
    #print(bidding_X)
    """
    term_candidates = dataset_manager.find_candidate_terms()
    terms = dataset_manager.term_extraction_from_candidates(term_candidates)
    print('creating reference graph')
    dataset_manager.create_reference_graph()
    reference_graph = dataset_manager.reference_graph
    print('creating node2vec graph')
    reference_graph_X = dataset_manager.create_node2vec_embeddings(reference_graph)
    print('creating term graph')
    dataset_manager.create_term_graph(terms, term_candidates)
    #term_graph_X = dataset_manager.create_node2vec_embeddings(dataset_manager.term_graph)
    print('creating term mat')
    #term_matrix_X = dataset_manager.create_term_matrix(terms, term_candidates)
    term_matrix_X = np.array(load_term_matrix(dataset_manager))
    print('term mat shape', term_matrix_X.shape)
    print('creating tfidf')
    tfidf_X = dataset_manager.get_tfidf().todense()
    print('creating doc2vec')
    doc2vec_matrix_X = dataset_manager.create_doc2vec_matrix(dataset_manager.raw_data)
    #tfidf_X = dataset_manager.get_tfidf()
    print(np.array(reference_graph_X).shape)
    #print(np.array(term_graph_X).shape)
    print(np.array(term_matrix_X).shape)
    print(np.array(dataset_manager.raw_data_classes).shape)
    print(tfidf_X.shape)
    #X = dataset_manager.combine_features((reference_graph_X, term_graph_X, term_matrix_X, doc2vec_matrix_X, tfidf_X, bidding_X))
    X = dataset_manager.combine_features((reference_graph_X, term_matrix_X, doc2vec_matrix_X, tfidf_X))
    k_best_250 = SelectKBest(f_classif, 250)
    k_best_500 = SelectKBest(f_classif, 500)
    k_best_750 = SelectKBest(f_classif, 7500)
    k_best_1000 = SelectKBest(f_classif, 1000)
    k_best_1250 = SelectKBest(f_classif, 1250)
    k_best_1500 = SelectKBest(f_classif, 1500)
    k_best_2000 = SelectKBest(f_classif, 2000) # 1000 seems really good
    k_best_5000 = SelectKBest(f_classif, 5000) # 1000 seems really good
    k_best_10000 = SelectKBest(f_classif, 10000) # 1000 seems really good
    #k_best_200_relief = ReliefF(n_features_to_select=200, n_neighbors=10)
    k_best_1000_relief = ReliefF(n_features_to_select=1000, n_neighbors=10, n_jobs=-1)
    #k_best_2000_relief = ReliefF(n_features_to_select=2000, n_neighbors=50)
    #k_best_5000_relief = ReliefF(n_features_to_select=5000, n_neighbors=5)
    #k_best_10000_relief = ReliefF(n_features_to_select=10000, n_neighbors=5)
    print('fitting')
    k_best_250.fit(X, dataset_manager.raw_data_classes)
    k_best_1000.fit(X, dataset_manager.raw_data_classes)
    le = preprocessing.LabelEncoder()
    le.fit(dataset_manager.raw_data_classes)
    raw_data_classes_nums = le.transform(dataset_manager.raw_data_classes)
    #k_best_1000_relief.fit(X.astype(np.float), np.array(raw_data_classes_nums))
    #print('feature scores')
    #for i, feature_score in sorted(enumerate(k_best_1000_relief.feature_importances_), key=lambda x:x[1]):
    #    print(i, '\t', feature_score)
    #print('top features')
    #for x in k_best_1000_relief.top_features_:
    #    print(x)
    #raise TypeError
    best_features_1000 = k_best_1000.get_support(indices=True)
    best_features_250 = k_best_250.get_support(indices=True)
    print('feature sizes')
    print('reference graph', np.array(reference_graph_X).shape)
    print('term matrix', np.array(term_matrix_X).shape)
    print('doc2vec matrix', np.array(doc2vec_matrix_X).shape)
    print('tfidf', np.array(tfidf_X).shape)
    print('Best features 200')
    for f in best_features_250:
        print(f)
    print('Done best features 200')
    print('Best features 1000')
    for f in best_features_1000:
        print(f)
    print('Done best features 1000')
    #raise TypeError
    #k_best_500.fit(X, dataset_manager.raw_data_classes)
    #k_best_750.fit(X, dataset_manager.raw_data_classes)
    k_best_1250.fit(X, dataset_manager.raw_data_classes)
    #k_best_1500.fit(X, dataset_manager.raw_data_classes)
    #k_best_2000.fit(X, dataset_manager.raw_data_classes)
    #k_best_5000.fit(X, dataset_manager.raw_data_classes)
    #k_best_10000.fit(X, dataset_manager.raw_data_classes)
    print('X', X[0])
    print('classes', dataset_manager.raw_data_classes)
    #print('fit 1')
    #k_best_200_relief.fit(X.astype(np.float), np.array(raw_data_classes_nums))
    #print('fit 2')
    #k_best_1000_relief.fit(X.astype(np.float), np.array(raw_data_classes_nums))
    print('fit 3')
    #k_best_2000_relief.fit(X.astype(np.float), np.array(raw_data_classes_nums))
    #print('fit 4')
    #k_best_5000_relief.fit(X.astype(np.float), np.array(raw_data_classes_nums))
    #print('fit 5')
    #k_best_10000_relief.fit(X.astype(np.float), np.array(raw_data_classes_nums))
    # k_best_5000_relief_X = k_best_2000.transform(X.astype(np.float))
    #k_best_2000_relief_X = k_best_2000.transform(X.astype(np.float))
    #print('relief 2000')
    #classifier = SVC()
    #scores_kbest_2000_relief = cross_val_score(classifier, k_best_2000_relief_X, dataset_manager.raw_data_classes,
    #                                            cv=KFold(n_splits=10, shuffle=True))
    #print("Best 10000 relief: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_2000_relief.mean(), scores_kbest_2000_relief.std()))
    #print(scores_kbest_2000_relief)
    print('transforming')
    #k_best_250_X = k_best_250.transform(X)
    #k_best_500_X = k_best_500.transform(X)
    #k_best_750_X = k_best_750.transform(X)
    #k_best_1000_X = k_best_1000.transform(X)
    k_best_1250_X = k_best_1250.transform(X)
    #k_best_1500_X = k_best_1500.transform(X)
    #k_best_2000_X = k_best_2000.transform(X)
    #k_best_5000_X = k_best_2000.transform(X)
    #k_best_10000_X = k_best_2000.transform(X)
    #k_best_200_relief_X = k_best_200.transform(X.astype(np.float))
    #k_best_1000_relief_X = k_best_1000.transform(X.astype(np.float))
    #k_best_2000_relief_X = k_best_2000.transform(X.astype(np.float))
    #k_best_5000_relief_X = k_best_5000.transform(X.astype(np.float))

    #print('X shape', X.shape)
    #dataset_manager.get_tfidf()
    #X = dataset_manager.tfidf.todense()
    #print('got embeddings')
    # Old clustering, leaving here to cross check results
    """
    clusters = dataset_manager.clustering(X)
    for cluster, paper in zip(clusters, papers):
        paper.simple_cluster = cluster
        print(cluster, paper)
        paper.save()
    print("clustering results")
    for c in clusters:
        print(c)
    """
    print('NEW STUFF')
    # Select papers for clustering
    schedule = Conference.objects.get(user=request.user, pk=request.session['conf']).schedule_string
    schedule = ast.literal_eval(schedule)
    # Find papers already in the schedule so that new ones can later be displayed
    old_paper_ids_in_schedule = []
    # Before clustering, every non-locked paper must be removed from schedule, since the clustering algorithm currently
    # only works with empty slots.
    ids_to_remove = []
    for day in schedule:
        for row in day:
            for col in row:
                print("GOT COL: ", col)
                for id in col:
                    paper = Paper.objects.get(pk=id)
                    #print("ID ", id)
                    old_paper_ids_in_schedule.append(id)
                    if not paper.is_locked:
                        ids_to_remove.append(id)
                        #print("REMOVED ", id, "COL IS NOW ", col)
    schedule_db = Conference.objects.get(user=request.user, pk=request.session['conf'])
    schedule_manager = schedule_manager_class()
    schedule_manager.import_paper_schedule(schedule_db.schedule_string)
    for id in ids_to_remove:
        #print(schedule_manager.papers)
        schedule_manager.remove_paper(id)
    print(schedule_manager.papers)
    schedule_db.schedule_string = str(schedule_manager.papers)
    schedule_db.save()
    # Begin clustering, after reloading new schedule data
    schedule = Conference.objects.get(user=request.user, pk=request.session['conf']).schedule_string
    schedule = ast.literal_eval(schedule)
    print(schedule)
    # clusterer = Clusterer(papers=papers, schedule=schedule, schedule_settings=settings)
    # clusterer.add_graph(schedule_db.paper_graph_string)
    # First do classification for testing purposes
    shuffle_index = list(range(len(X)))
    random.shuffle(shuffle_index)
    shuffled_X = []
    shuffled_classes = []
    #selector = SelectKBest(k=100)
    for i in shuffle_index:
        shuffled_X.append(k_best_1250_X[i])
        shuffled_classes.append(dataset_manager.raw_data_classes[i])
    #features_chi = chi2(X, dataset_manager.raw_data_classes)
    #features_anova = f_classif(X, dataset_manager.raw_data_classes)
    #print("chi", features_chi)
    """
    print('anova f')
    for x in features_anova[0]:
        print(x)
    print('anova p')
    for x in features_anova[1]:
        print(x)
    """

    print('running cross validation')
    #print("anova", features_anova)
    print('shapes')
    classifier_svm = SVC()
    classifier_log = LogisticRegression()
    classifier_rf = RandomForestClassifier()
    #print(np.array(reference_graph_X).shape)
    #print(np.array(term_graph_X).shape)
    #print(np.array(term_matrix_X).shape)
    #print(np.array(X).shape)
    #print(np.array(dataset_manager.raw_data_classes).shape)

    # Missclassification matrix
    np.save('shuffled_x', shuffled_X)
    missclassif_rf = RandomForestClassifier()
    missclassif_rf.fit(shuffled_X[:int(len(shuffled_X) * 0.7)], shuffled_classes[:int(len(shuffled_classes) * 0.7)])
    misclassif_preds = missclassif_rf.predict(shuffled_X[int(len(shuffled_X) * 0.7):])
    misclassif_actual = shuffled_classes[int(len(shuffled_classes) * 0.7):]
    misclassif_confusion_matrix = confusion_matrix(misclassif_actual, misclassif_preds)
    print('Misclassification matrix')
    print(misclassif_confusion_matrix)
    for line in misclassif_confusion_matrix:
        print(line)
    raise TypeError
    binarizer = preprocessing.LabelBinarizer()
    binary_classes = binarizer.fit_transform(dataset_manager.raw_data_classes)
    y_classes = dataset_manager.raw_data_classes
    print('reference')
    scores_reference_graph_svm = cross_val_score(classifier_svm, reference_graph_X, y_classes,
                                                 cv=KFold(n_splits=10, shuffle=True))
    #print('term')
    #scores_term_graph_svm = cross_val_score(classifier, term_graph_X, dataset_manager.raw_data_classes, cv=KFold(n_splits=10, shuffle=True))
    print('term mat')
    scores_term_matrix_svm = cross_val_score(classifier_svm, term_matrix_X, y_classes,
                                             cv=KFold(n_splits=10, shuffle=True))
    print('doc2vec')
    scores_doc2vec_matrix_svm = cross_val_score(classifier_svm, doc2vec_matrix_X, y_classes,
                                               cv=KFold(n_splits=10, shuffle=True))
    print('tfidf')
    scores_tfidf_svm = cross_val_score(classifier_svm, tfidf_X, y_classes,
                                        cv=KFold(n_splits=10, shuffle=True))
    print('k200')
    scores_kbest_250_svm = cross_val_score(classifier_svm, k_best_250_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_500_svm = cross_val_score(classifier_svm, k_best_500_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_750_svm = cross_val_score(classifier_svm, k_best_750_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    print('k1000')
    scores_kbest_1000_svm = cross_val_score(classifier_svm, k_best_1000_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_1250_svm = cross_val_score(classifier_svm, k_best_1250_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_1500_svm = cross_val_score(classifier_svm, k_best_1500_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    print('k2000')
    scores_kbest_2000_svm = cross_val_score(classifier_svm, k_best_2000_X,y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_5000_svm = cross_val_score(classifier_svm, k_best_5000_X, y_classes,
                                             cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_10000_svm = cross_val_score(classifier_svm, k_best_10000_X, y_classes,
                                              cv=KFold(n_splits=10, shuffle=True))
    print('all')
    scores_all_svm = cross_val_score(classifier_svm, X, y_classes,
                                 cv=KFold(n_splits=10, shuffle=True))

    print('reference')
    scores_reference_graph_log = cross_val_score(classifier_log, reference_graph_X, y_classes,
                                                 cv=KFold(n_splits=10, shuffle=True))
    # print('term')
    # scores_term_graph_log = cross_val_score(classifier, term_graph_X, dataset_manager.raw_data_classes, cv=KFold(n_splits=10, shuffle=True))
    print('term mat')
    scores_term_matrix_log = cross_val_score(classifier_log, term_matrix_X, y_classes,
                                             cv=KFold(n_splits=10, shuffle=True))
    print('doc2vec')
    scores_doc2vec_matrix_log = cross_val_score(classifier_log, doc2vec_matrix_X, y_classes,
                                               cv=KFold(n_splits=10, shuffle=True))
    print('tfidf')
    scores_tfidf_log = cross_val_score(classifier_log, tfidf_X, y_classes,
                                       cv=KFold(n_splits=10, shuffle=True))
    print('k200')
    scores_kbest_250_log = cross_val_score(classifier_log, k_best_250_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_500_log = cross_val_score(classifier_log, k_best_500_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_750_log = cross_val_score(classifier_log, k_best_750_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    print('k1000')
    scores_kbest_1000_log = cross_val_score(classifier_log, k_best_1000_X, y_classes,
                                             cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_1250_log = cross_val_score(classifier_log, k_best_1250_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_1500_log = cross_val_score(classifier_log, k_best_1500_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    print('k2000')
    scores_kbest_2000_log = cross_val_score(classifier_log, k_best_2000_X, y_classes,
                                             cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_5000_log = cross_val_score(classifier_log, k_best_5000_X, y_classes,
                                             cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_10000_log = cross_val_score(classifier_log, k_best_10000_X, y_classes,
                                              cv=KFold(n_splits=10, shuffle=True))
    print('all')
    scores_all_log = cross_val_score(classifier_log, X, y_classes,
                                      cv=KFold(n_splits=10, shuffle=True))

    print('reference')
    scores_reference_graph_rf = cross_val_score(classifier_rf, reference_graph_X, y_classes,
                                                 cv=KFold(n_splits=10, shuffle=True))
    # print('term')
    # scores_term_graph_rf = cross_val_score(classifier, term_graph_X, dataset_manager.raw_data_classes, cv=KFold(n_splits=10, shuffle=True))
    print('term mat')
    scores_term_matrix_rf = cross_val_score(classifier_rf, term_matrix_X, y_classes,
                                             cv=KFold(n_splits=10, shuffle=True))
    print('doc2vec')
    scores_doc2vec_matrix_rf = cross_val_score(classifier_rf, doc2vec_matrix_X, y_classes,
                                                cv=KFold(n_splits=10, shuffle=True))
    print('tfidf')
    scores_tfidf_rf = cross_val_score(classifier_rf, tfidf_X, y_classes,
                                       cv=KFold(n_splits=10, shuffle=True))
    print('k200')
    scores_kbest_250_rf = cross_val_score(classifier_rf, k_best_250_X, y_classes,
                                           cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_500_rf = cross_val_score(classifier_rf, k_best_500_X, y_classes,
                                           cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_750_rf = cross_val_score(classifier_rf, k_best_750_X, y_classes,
                                           cv=KFold(n_splits=10, shuffle=True))
    print('k1000')
    scores_kbest_1000_rf = cross_val_score(classifier_rf, k_best_1000_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_1250_rf = cross_val_score(classifier_rf, k_best_1250_X, y_classes,
                                           cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_1500_rf = cross_val_score(classifier_rf, k_best_1500_X, y_classes,
                                           cv=KFold(n_splits=10, shuffle=True))
    print('k2000')
    scores_kbest_2000_rf = cross_val_score(classifier_rf, k_best_2000_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_5000_rf = cross_val_score(classifier_rf, k_best_5000_X, y_classes,
                                            cv=KFold(n_splits=10, shuffle=True))
    scores_kbest_10000_rf = cross_val_score(classifier_rf, k_best_10000_X, y_classes,
                                             cv=KFold(n_splits=10, shuffle=True))
    print('all')
    scores_all_rf = cross_val_score(classifier_rf, X, y_classes,
                                    cv=KFold(n_splits=10, shuffle=True))
    #print('relief 200')
    #scores_kbest_200_relief = cross_val_score(classifier, k_best_200_relief_X, dataset_manager.raw_data_classes,
    #                                   cv=KFold(n_splits=10, shuffle=True))
    #printy_classes
    #scores_kbest_1000_relief = cross_val_score(classifier, k_best_1000_relief_X, dataset_manager.raw_data_classes,
    #                                    cv=KFold(n_splits=10, shuffle=True))
    #print('relief 2000')
    #scores_kbest_2000_relief = cross_val_score(classifier, k_best_2000_relief_X, dataset_manager.raw_data_classes,
    #                                    cv=KFold(n_splits=10, shuffle=True))
    #print('relief 5000')
    #scores_kbest_5000_relief = cross_val_score(classifier, k_best_5000_relief_X, dataset_manager.raw_data_classes,
    #                                    cv=KFold(n_splits=10, shuffle=True))



    print('bidding')
    #scores_bidding = cross_val_score(classifier, bidding_X, dataset_manager.raw_data_classes, cv=KFold(n_splits=10, shuffle=True))
    print('Scores log:')
    print('------------------')
    print("tfidf: Accuracy: %0.5f,  stddev %0.5f)" % (scores_tfidf_log.mean(), scores_tfidf_log.std()))
    print(scores_tfidf_log)
    print("Reference graph: Accuracy: %0.5f,  stddev %0.5f)" % (scores_reference_graph_log.mean(), scores_reference_graph_log.std()))
    print(scores_reference_graph_log)
    #print("Term graph: Accuracy: %0.5f,  stddev %0.5f)" % (scores_term_graph.mean(), scores_term_graph.std()))
    #print(scores_term_graph)
    print("Term matrix: Accuracy: %0.5f,  stddev %0.5f)" % (scores_term_matrix_log.mean(), scores_term_matrix_log.std()))
    print(scores_term_matrix_log)
    print("Doc2Vec matrix: Accuracy: %0.5f,  stddev %0.5f)" % (scores_doc2vec_matrix_log.mean(), scores_doc2vec_matrix_log.std()))
    print(scores_doc2vec_matrix_log)
    print("All: Accuracy: %0.5f,  stddev %0.5f)" % (scores_all_log.mean(), scores_all_log.std()))
    print(scores_all_log)
    print("Best 250: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_250_log.mean(), scores_kbest_250_log.std()))
    print(scores_kbest_250_log)
    print("Best 500: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_500_log.mean(), scores_kbest_500_log.std()))
    print(scores_kbest_500_log)
    print("Best 750: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_750_log.mean(), scores_kbest_750_log.std()))
    print(scores_kbest_750_log)
    print("Best 1000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1000_log.mean(), scores_kbest_1000_log.std()))
    print(scores_kbest_1000_log)
    print("Best 1250: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1250_log.mean(), scores_kbest_1250_log.std()))
    print(scores_kbest_1250_log)
    print("Best 1500: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1500_log.mean(), scores_kbest_1500_log.std()))
    print(scores_kbest_1500_log)
    print("Best 2000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_2000_log.mean(), scores_kbest_2000_log.std()))
    print(scores_kbest_2000_log)
    print("Best 5000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_5000_log.mean(), scores_kbest_5000_log.std()))
    print(scores_kbest_5000_log)
    print("Best 10000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_10000_log.mean(), scores_kbest_10000_log.std()))
    print(scores_kbest_10000_log)

    print('Scores svm:')
    print('------------------')
    print("tfidf: Accuracy: %0.5f,  stddev %0.5f)" % (scores_tfidf_svm.mean(), scores_tfidf_svm.std()))
    print(scores_tfidf_svm)
    print("Reference graph: Accuracy: %0.5f,  stddev %0.5f)" % (scores_reference_graph_svm.mean(), scores_reference_graph_svm.std()))
    print(scores_reference_graph_svm)
    # print("Term graph: Accuracy: %0.5f,  stddev %0.5f)" % (scores_term_graph.mean(), scores_term_graph.std()))
    # print(scores_term_graph)
    print("Term matrix: Accuracy: %0.5f,  stddev %0.5f)" % (scores_term_matrix_svm.mean(), scores_term_matrix_svm.std()))
    print(scores_term_matrix_svm)
    print("Doc2Vec matrix: Accuracy: %0.5f,  stddev %0.5f)" % (scores_doc2vec_matrix_svm.mean(), scores_doc2vec_matrix_svm.std()))
    print(scores_doc2vec_matrix_svm)
    print("All: Accuracy: %0.5f,  stddev %0.5f)" % (scores_all_svm.mean(), scores_all_svm.std()))
    print(scores_all_svm)
    print("Best 250: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_250_svm.mean(), scores_kbest_250_svm.std()))
    print(scores_kbest_250_svm)
    print("Best 500: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_500_svm.mean(), scores_kbest_500_svm.std()))
    print(scores_kbest_500_svm)
    print("Best 750: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_750_svm.mean(), scores_kbest_750_svm.std()))
    print(scores_kbest_750_svm)
    print("Best 1000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1000_svm.mean(), scores_kbest_1000_svm.std()))
    print(scores_kbest_1000_svm)
    print("Best 1250: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1250_svm.mean(), scores_kbest_1250_svm.std()))
    print(scores_kbest_1250_svm)
    print("Best 1500: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1500_svm.mean(), scores_kbest_1500_svm.std()))
    print(scores_kbest_1500_svm)
    print("Best 2000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_2000_svm.mean(), scores_kbest_2000_svm.std()))
    print(scores_kbest_2000_svm)
    print("Best 5000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_5000_svm.mean(), scores_kbest_5000_svm.std()))
    print(scores_kbest_5000_svm)
    print("Best 10000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_10000_svm.mean(), scores_kbest_10000_svm.std()))
    print(scores_kbest_10000_svm)

    print('Scores rf:')
    print('------------------')
    print("tfidf: Accuracy: %0.5f,  stddev %0.5f)" % (scores_tfidf_rf.mean(), scores_tfidf_rf.std()))
    print(scores_tfidf_rf)
    print("Reference graph: Accuracy: %0.5f,  stddev %0.5f)" % (scores_reference_graph_rf.mean(), scores_reference_graph_rf.std()))
    print(scores_reference_graph_rf)
    # print("Term graph: Accuracy: %0.5f,  stddev %0.5f)" % (scores_term_graph.mean(), scores_term_graph.std()))
    # print(scores_term_graph)
    print("Term matrix: Accuracy: %0.5f,  stddev %0.5f)" % (scores_term_matrix_rf.mean(), scores_term_matrix_rf.std()))
    print(scores_term_matrix_rf)
    print("Doc2Vec matrix: Accuracy: %0.5f,  stddev %0.5f)" % (scores_doc2vec_matrix_rf.mean(), scores_doc2vec_matrix_rf.std()))
    print(scores_doc2vec_matrix_rf)
    print("All: Accuracy: %0.5f,  stddev %0.5f)" % (scores_all_rf.mean(), scores_all_rf.std()))
    print(scores_all_rf)
    print("Best 250: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_250_rf.mean(), scores_kbest_250_rf.std()))
    print(scores_kbest_250_rf)
    print("Best 500: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_500_rf.mean(), scores_kbest_500_rf.std()))
    print(scores_kbest_500_rf)
    print("Best 750: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_750_rf.mean(), scores_kbest_750_rf.std()))
    print(scores_kbest_750_rf)
    print("Best 1000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1000_rf.mean(), scores_kbest_1000_rf.std()))
    print(scores_kbest_1000_rf)
    print("Best 1250: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1250_rf.mean(), scores_kbest_1250_rf.std()))
    print(scores_kbest_1250_rf)
    print("Best 1500: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1500_rf.mean(), scores_kbest_1500_rf.std()))
    print(scores_kbest_1500_rf)
    print("Best 2000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_2000_rf.mean(), scores_kbest_2000_rf.std()))
    print(scores_kbest_2000_rf)
    print("Best 5000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_5000_rf.mean(), scores_kbest_5000_rf.std()))
    print(scores_kbest_5000_rf)
    print("Best 10000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_10000_rf.mean(), scores_kbest_10000_rf.std()))
    print(scores_kbest_10000_rf)
    print('RELIEF')
    #print("Best 200: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_200_relief.mean(), scores_kbest_200_relief.std()))
    #print(scores_kbest_200)
    #print("Best 1000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_1000_relief.mean(), scores_kbest_1000_relief.std()))
    #print(scores_kbest_1000)
    #print("Best 2000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_2000_relief.mean(), scores_kbest_2000_relief.std()))
    #print(scores_kbest_2000)
    #print("Best 5000: Accuracy: %0.5f,  stddev %0.5f)" % (scores_kbest_5000_relief.mean(), scores_kbest_5000_relief.std()))
    #print(scores_kbest_5000)


    #print("Best bidding: Accuracy: %0.5f,  stddev %0.5f)" % (scores_bidding.mean(), scores_bidding.std()))
    #print(scores_bidding)

    raise TypeError


    dataset_manager.train_dataset(X[136:], dataset_manager.raw_data_classes[136:])
    #selector.fit(shuffled_X[:-80], shuffled_classes[:-80])
    preds = dataset_manager.predict(X[:136])
    k_best = SelectKBest(f_classif, 5000)
    k_best.fit(X[136:], dataset_manager.raw_data_classes[136:])
    k_best_X = k_best.transform(X)
    dataset_manager.train_dataset(k_best_X[136:], dataset_manager.raw_data_classes[136:])
    preds_k_best = dataset_manager.predict(k_best_X[:136])
    for i in range(136):
        paper = papers[i]
        paper.abstract += preds_k_best[i]
        paper.save()
        print(paper.title, preds_k_best[i])
    print(dataset_manager.raw_data_classes[:136])
    probs, probs_log = dataset_manager.get_probs(k_best_X[:136])
    print('got probs', np.array(probs).shape)
    print(probs)
    # Count members of each class
    class_occurances = {}
    class_papers = {}
    class_probs = {}
    for i in range(len(preds)):
        #print(papers[i].title, papers[i].abstract[:200])
        print(preds[i], preds_k_best[i])
        if preds_k_best[i] not in class_occurances.keys():
            class_occurances[preds_k_best[i]] = 1
            class_papers[preds_k_best[i]] = [i]
            class_probs[preds_k_best[i]] = [(probs[i], probs_log[i])]
        else:
            class_occurances[preds_k_best[i]] += 1
            class_papers[preds_k_best[i]].append(i)
            class_probs[preds_k_best[i]].append((probs[i], probs_log[i]))
    print('preds', preds)
    print('preds term', preds_k_best)
    print('occurances', class_occurances)
    for k in class_papers.keys():
        print(k)
        #print(len(class_papers[k]))
        #print(class_papers[k])
        for paper_index, paper_prob in zip(class_papers[k], class_probs[k]):
            #print(papers[paper_index].abstract)
            print(paper_prob[0].max(), paper_prob[1].max())
            print('*****')
        print('--------------------------------------------------------------')
    #raise TypeError
    #classification_accuracy = np.sum(preds == shuffled_classes[-80:])/len(preds)
    #classification_accuracy_k_best = np.sum(preds_k_best == shuffled_classes[-80:])/len(preds_k_best)
    #print('accuracy', classification_accuracy)
    #print('accuracy_500', classification_accuracy_k_best)

    times = []
    # Reset papers
    clusterer = setup_clustering(request, papers, [k_best_1000_X])
    #clusterer.visualize_points(X[:136], preds_k_best)
    clusterer.reset_papers()
    clusterer.create_dataset()
    #X = selector.transform(X)
    cluster_assignments, avg_distances = clusterer.constrained_clustering(k_best_1000_X)
    # Run tests
    print('running tests')
    #rand_index = clusterer.evaluate_clustering_adjusted_rand_index(dataset_manager.raw_data_classes, cluster_assignments)
    #print('rand index', rand_index)
    silh_score = clusterer.evaluate_clustering_silhouette(X[:136], cluster_assignments)
    silh_score_rand = clusterer.evaluate_clustering_silhouette(X[:136], cluster_assignments, rand=True)
    silh_score_kmeans = clusterer.evaluate_clustering_silhouette(X[:136], cluster_assignments, km=True)
    print('silhouette score', silh_score)
    print('silhouette score rand', silh_score_rand)
    print('silhouette score kmeans', silh_score_kmeans)
    #raise TypeError
    dataset_manager.load_raw_data()
    classes = dataset_manager.raw_data_classes
    for key in cluster_assignments.keys():
        cluster_list = cluster_assignments[key]
        print('key', key)
        for e in cluster_list:
            print(e, classes[e])
        for distance in avg_distances[key]:
            print('dist', distance)
        if len(avg_distances[key]) > 0:
            print('average distance', sum(avg_distances[key])/len(avg_distances[key]))
        print('--------------')
    #clusterer.basic_clustering()
    #clusterer.fit_to_schedule2()
    # Add papers to schedule
    schedule = Conference.objects.get(user=request.user, pk=request.session['conf']).schedule_string
    settings = Conference.objects.get(user=request.user, pk=request.session['conf']).settings_string
    schedule_manager = schedule_manager_class()
    schedule_manager.import_paper_schedule(schedule)
    schedule_manager.set_settings(settings)
    print('import_page all papers2', papers)
    for paper in papers:
        print('views_import_page', paper.title, paper.add_to_day, paper.add_to_row, paper.add_to_col)
        if (paper.add_to_day != -1) and (paper.add_to_row != -1) and (paper.add_to_col != -1):
            schedule_manager.assign_paper(paper.pk, paper.add_to_day, paper.add_to_row, paper.add_to_col)
    schedule_settings = Conference.objects.get(user=request.user, pk=request.session['conf'])
    schedule_settings.schedule_string = schedule_manager.papers
    schedule_settings.save()
    # Find which papers were newly added by comparing the new schedule with papers in the old schedule
    schedule = ast.literal_eval(
        Conference.objects.get(user=request.user, pk=request.session['conf']).schedule_string)
    newly_added_paper_names = []
    newly_added_paper_slots = []
    for di, day in enumerate(schedule):
        for ri, row in enumerate(day):
            for ci, col in enumerate(row):
                for id in col:
                    paper = Paper.objects.get(pk=id)
                    if id not in old_paper_ids_in_schedule:
                        newly_added_paper_names.append(
                            paper.title + " (day " + str(di + 1) + ", row " + str(ri + 1) +
                            ", column " + str(ci + 1) + ")")
    """
    print('prediction debug')
    for i in range(len(preds)):
        print(preds[i], "|", shuffled_classes[-80:][i])
    """
    print('-------------')
    print('Final Scores:')
    print('-------------')
    print('Silhouette Score:', silh_score)
    #print('Adjusted Rand Index:', rand_index)
    #print('Classification Accuracy', classification_accuracy)
    #raise TypeError
    return render(request, 'app/clustering_results_overview.html',
                  {'newly_added_paper_names': newly_added_paper_names})
    # return redirect('/app/clustering/results/all')

    #return redirect('/app/import_page')

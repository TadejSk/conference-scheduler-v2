from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import Http404, HttpResponse
from django.shortcuts import render, redirect
from app.classes import *
import app.models
from app.models import Conference
from diploma.settings import MEDIA_ROOT
import os
from django.core.files import File

__author__ = 'Tadej'

@login_required
def index(request):
    float_msg = None
    if request.session.get('float_msg', None):
        float_msg = request.session.get('float_msg', None)
        request.session['float_msg'] = None
    if request.method == 'POST':
        request.session['conf'] = request.POST.get('conference',0)
    try:
        request.session['conf']
    except KeyError:
        return redirect('/app/conference/list')
    request.session['parallel_error']=""
    # Get the number of all authors and papers
    num_authors = Author.objects.filter(user=request.user).count()
    num_papers = Paper.objects.filter(user=request.user, conference=request.session['conf']).count()
    # Get the data necessary to display the schedule
    try:
        settings = Conference.objects.get(user=request.user, pk=request.session['conf'])
    except ObjectDoesNotExist:
        s = Conference()
        s.user = request.user
        s.num_days = 1
        s.slot_length = 60
        s.settings_string = "[[]]"
        s.schedule_string = "[]"
        s.save()
        settings = s
    num_days = None
    if settings is not None:
        num_days = settings.num_days
    day = int(request.GET.get('day',0))
    if day >= num_days and num_days >=1:
        raise Http404('The selected day is too high')
    settings_str = settings.settings_string
    settings_list = ast.literal_eval(settings.settings_string)[int(request.GET.get('day',0))]
    set = schedule_manager_class()
    if settings.schedule_string == "[]":
        set.create_empty_list(settings.settings_string)
    else:
        set.import_paper_schedule(settings.schedule_string)
    for ri, row in enumerate(settings_list):
        print("row: " + str(row))
        for si, slot in enumerate(row):
            print("slot: " + str(slot))
            set.set_settings(settings_str)
            free_time = set.get_slot_free_time(int(request.GET.get('day',0)), ri, si)
            slot = [slot, free_time]
            row[si] = slot
    # Create the form for uploading files
    papers_form = FileForm()
    assignments_form = AssignmentsFileForm()
    # Create a list of all the papers that the user has imported/created
    papers = Paper.objects.filter(user=request.user, conference=request.session['conf']).order_by('title')
    paper_titles = []
    paper_ids = []
    paper_lengths = []
    paper_locked = []
    paper_clusters = []
    for paper in papers:
        paper_titles.append(paper.title)
        paper_ids.append(paper.pk)
        paper_lengths.append(paper.length)
        paper_locked.append(paper.is_locked)
        paper_clusters.append(paper.cluster)
    paper_dict = dict(zip(paper_ids, paper_titles))
    paper_locked_dict = dict(zip(paper_ids, paper_locked))
    schedule = ast.literal_eval(settings.schedule_string)
    review_string = Conference.objects.get(user=request.user, pk=request.session['conf']).reviewer_biddings_string
    return render(request, 'app/index.html', {'num_authors':num_authors, 'num_papers':num_papers, 'num_days':num_days,
                                              'settings_list':settings_list,'paper_titles':paper_titles,
                                              'paper_ids':paper_ids, 'paper_dict':paper_dict, 'schedule':schedule,
                                              'day':day, 'paper_lengths':paper_lengths, 'paper_locked':paper_locked_dict,
                                              'paper_clusters': paper_clusters, 'papers_form':papers_form,
                                              'assignments_form':assignments_form, 'float_msg':float_msg,
                                              'review_string': review_string, 'conference_title':settings.title})


def import_demo_data(request):
    path = MEDIA_ROOT + "/accepted.xls"
    data = raw_data(path, None)
    p = data.parse_accepted()
    for paper in p.accepted_papers_list:
        if not Paper.objects.filter(title=paper.title, user=request.user, conference=request.session['conf']).exists():
            db_paper = Paper()
            db_paper.title = paper.title
            db_paper.abstract = paper.abstract
            db_paper.submission_id = paper.submission_id
            db_paper.user = request.user
            db_paper.conference = Conference.objects.get(pk=request.session['conf'])
            db_paper.save()
            for author in paper.authors:
                if not Author.objects.filter(name=author, user=request.user ).exists():
                    db_author = Author()
                    db_author.name=author
                    db_author.user = request.user
                    db_author.save()
                    db_author.papers.add(db_paper)
                else:
                    db_author = Author.objects.get(name=author, user=request.user)
                    db_author.papers.add(db_paper)
    request.session['float_msg'] = "Successfully loaded " + str(len(p.accepted_papers_list)) + " papers"
    return redirect('/app/index')


def import_data(request):
    file = request.FILES.get('file', None)
    if not file:
        return redirect('/app/index')
    file_model = UpoladedFile()
    file_model.file = file
    file_model.save()
    print(file_model.file.path)
    papers_path = file_model.file.path
    data = raw_data(papers_path,None)
    p = data.parse_accepted()
    for paper in p.accepted_papers_list:
        if not Paper.objects.filter(title=paper.title, user=request.user, conference=request.session['conf']).exists():
            db_paper = Paper()
            db_paper.title = paper.title
            db_paper.abstract = paper.abstract
            db_paper.submission_id = paper.submission_id
            db_paper.user = request.user
            db_paper.conference = Conference.objects.get(pk=request.session['conf'])
            db_paper.save()
            for author in paper.authors:
                if not Author.objects.filter(name=author, user=request.user ).exists():
                    db_author = Author()
                    db_author.name=author
                    db_author.user = request.user
                    db_author.save()
                    db_author.papers.add(db_paper)
                else:
                    db_author = Author.objects.get(name=author, user=request.user)
                    db_author.papers.add(db_paper)
    #request.session['accepted_path'] = accepted_path
    #request.session['assignments_path'] = assignments_path
    file_model.file.delete()
    request.session['float_msg'] = "Successfully loaded " + str(len(p.accepted_papers_list)) + " papers"
    return redirect('/app/index')


def import_assignments_data(request):
    papers = Paper.objects.filter(user=request.user, conference=request.session['conf']).order_by('submission_id')
    folder = MEDIA_ROOT + "/" + request.user.username + "/assignments/"
    path = folder + "assignments.csv"
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = request.FILES.get('file', None)
    if not file:
        return redirect('/app/index')
    file_model = UpoladedFile()
    file_model.file = file
    file_model.user = request.user
    file_model.save()
    print(file_model.file.path)
    path = file_model.file.path
    data = raw_data(None,path)
    #data.parse_accepted()
    graph_list, rows, cols = data.parse_assignments()
    settings = Conference.objects.get(user = request.user, pk=request.session['conf'])
    settings.paper_graph_string = str(graph_list)
    request.session['float_msg'] = "Successfully loaded reviewer biddings for " + str(rows) + " papers by " + str(cols) + " reviewers"
    # conf = Conference.objects.get(user=request.user, pk=request.session['conf'])
    settings.reviewer_biddings_string = "Reviewer biddings: " + str(rows) + " papers by " + str(cols) + " reviewers"
    # conf.save()
    settings.save()
    return redirect('/app/index')


def import_demo_assignments_data(request):
    path = MEDIA_ROOT + "/assignmentsAnon.csv"
    data = raw_data(None,path)
    #data.parse_accepted()
    graph_list, rows, cols = data.parse_assignments()
    settings = Conference.objects.get(user = request.user, pk=request.session['conf'])
    settings.paper_graph_string = str(graph_list)

    request.session['float_msg'] = "Successfully loaded reviewer biddings for " + str(rows) + " papers by " + str(cols) + " reviewers"
    # conf = Conference.objects.get(user=request.user, pk=request.session['conf'])
    settings.reviewer_biddings_string = "Reviewer biddings: " + str(rows) + " papers by " + str(cols) + " reviewers"
    #conf.reviewer_biddings_string = "a"
    # print(conf.reviewer_biddings_string)
    # conf.save()
    # print(conf.reviewer_biddings_string)
    # conf = Conference.objects.get(user=request.user, pk=request.session['conf'])
    # print(conf.reviewer_biddings_string)
    settings.save()

    file_model = UpoladedFile()
    file_model.file = File(open(path, 'r'))
    file_model.user = request.user
    file_model.save()
    return redirect('/app/index')


def write_paper_data(request):
    papers = Paper.objects.filter(user=request.user, conference=request.session['conf']).order_by('submission_id')
    folder = MEDIA_ROOT + "/" + request.user.username + "/temp/"
    path = folder + "paper_data.xls"
    if not os.path.exists(folder):
        os.makedirs(folder)
    data = raw_data(None, None)
    workbook = data.write_accepted(papers)
    workbook.save(path)
    file = open(path, 'rb')
    response = HttpResponse(file, content_type='application/vnd.ms-excel')
    response['Content-Disposition'] = 'attachment; filename=paper_data.xls'
    return response


def write_assignments_data(request):
    latest_files = app.models.UpoladedFile.objects.filter(user=request.user).order_by('-id')
    if len(latest_files) < 1:
        return redirect('/app/index')
    latest_file = latest_files[0]
    file = latest_file.file
    response = HttpResponse(file, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=reviewer_biddings.csv'
    return response



"""
def check_graph_validity(graph_list, papers):
    for connection in graph_list:
        graph_id = connection[0]
        ok = False
        for paper in papers:
            if graph_id == paper.submission_id:
                ok = True
                break
        if ok == False:
            return False
    return True
"""
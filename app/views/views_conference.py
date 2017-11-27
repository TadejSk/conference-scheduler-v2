import ast
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from app.models import Conference, Paper

__author__ = 'Tadej'

@login_required
def conference_list(request):
    conferences = Conference.objects.filter(user=request.user).order_by('id')
    form_data = [(c.title,c.id) for c in conferences]
    num_conferences = conferences.count()
    return render(request, 'app/conference_list.html', {'conferences':conferences, 'num_conferences':num_conferences,
                                                        'form_data':form_data})

@login_required
def create_conference(request):
    conference = Conference()
    conference.user = request.user
    conference.title = request.POST.get('title', 'Unnamed conference')
    if conference.title == '':
        conference.title = 'Unnamed conference'
    conference.save()
    return redirect('/app/conference/list')

@login_required
def delete_conference(request):
    conference = Conference.objects.get(pk=request.POST.get('conference', None), user=request.user)
    conference.delete()
    return redirect('/app/conference/list')

@login_required
def copy_conference(request):
    # Create conference base
    new_conference = Conference()
    new_conference.user = request.user
    copied_conference = Conference.objects.get(pk=request.POST.get('conference', None), user=request.user)
    new_conference.title = "Copy of " + copied_conference.title
    # Copy conference structure
    settings_str = copied_conference.settings_string
    new_conference.settings_string = settings_str
    # Copy names
    names_str = copied_conference.names_string
    new_conference.names_string = names_str
    # Create new empty list of papers
    settings_list = ast.literal_eval(settings_str)
    new_list = []
    for d in settings_list:
        day = []
        for r in d:
            row = []
            for c in r:
                col = []
                row.append(col)
            day.append(row)
        new_list.append(day)
    new_conference.schedule_string = str(new_list)
    new_conference.num_days = copied_conference.num_days
    new_conference.slot_length = copied_conference.slot_length
    new_conference.save()
    return redirect('/app/conference/list')

@login_required
def rename_conference(request):
    old_conference = Conference.objects.get(pk=request.POST.get('conference', None), user=request.user)
    id = old_conference.id
    old_name = old_conference.title
    return render(request, 'app/conference_rename.html', {'old_name':old_name, 'id':id})

@login_required
def rename_conference_action(request):
    old_conference = Conference.objects.get(pk=request.POST.get('conference', None), user=request.user)
    old_conference.title = request.POST.get('newname', 'Unnamed conference')
    old_conference.save()
    return redirect('/app/conference/list')

@login_required
def export_schedule(request):
    conf = request.session.get('conf', {})
    if conf=={}:
        return redirect('/app/index/')
    
    conf = Conference.objects.get(pk=request.session['conf'], user=request.user)
    schedule = ast.literal_eval(conf.settings_string)
    max_slots = []
    for index, day in enumerate(schedule):
        max_slots.append(0)
        for row in day:
            if len(row) > max_slots[index]:
                max_slots[index] = len(row)
    # Generate starting times
    start_times = ast.literal_eval(conf.start_times)
    times = []
    for day, time in enumerate(start_times):
        day_times = []
        hour = int(time.split(':')[0])
        minute = int(time.split(':')[1])
        length = 0
        for row in schedule[day]:
            row_times = []
            length = row[0]
            for i in range(length//15):
                row_times.append(str(hour) + ":" + str(minute))
                minute += 15
                if minute >= 60:
                    minute -= 60
                    hour += 1
            day_times.append(row_times)
        times.append(day_times)
    # Saves titles
    names = ast.literal_eval(conf.names_string)
    print(names)
    names_list = []
    for day in names:
        day_list = []
        for row in day:
            row_names=[]
            for name in row:
                row_names.append(name)
            day_list.append(row_names)
        names_list.append(day_list)
    # Save paper titles
    ids = ast.literal_eval(conf.schedule_string)
    papers = []
    for day in ids:
        day_list = []
        for row in day:
            row_list = []
            for col in row:
                col_papers = []
                for paper_id in col:
                    col_papers.append(Paper.objects.get(pk=paper_id).title)
                row_list.append(col_papers)
            day_list.append(row_list)
        papers.append(day_list)
    # print("----------")
    # print(papers)
    # print(names_list)
    return render(request, 'app/export_schedule.html', {'schedule':schedule, 'max_slots':max_slots, 'times':times,
                                                        'names':names, 'paper_names': papers, 'paper_slots':names_list})
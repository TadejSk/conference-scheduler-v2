from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import render, redirect
from ..models import Conference
from ..classes import schedule_settings_class, schedule_manager_class
import ast
__author__ = 'Tadej'

@login_required
def schedule_settings(request):
    conf = request.session.get('conf', {})
    if conf == {}:
        return redirect('/app/index/')        
    # error = request.session['parallel_error']
    error = request.session.get('parallel_error', '')
    time_error = request.session.get('time_error', "")
    settings = Conference.objects.get(user=request.user, pk=request.session['conf'])
    num_days = None
    base_time = None
    if settings is not None:
        num_days = settings.num_days
        base_time = settings.slot_length
    day = int(request.GET.get('day',0))
    if day >= num_days and num_days >=1:
        raise Http404('The selected day is too high')
    settings_list = ast.literal_eval(settings.settings_string)[int(request.GET.get('day',0))]
    print(settings.start_times)
    start_time = ast.literal_eval(settings.start_times)[int(request.GET.get('day',0))]
    names = ast.literal_eval(settings.names_string)
    return render(request, 'app/settings/schedule.html', {'num_days':num_days, 'base_time':base_time, 'day':day,
                                                          'settings_list':settings_list, 'error':error, 'start_time':start_time,
                                                          'time_error':time_error, 'names':names, 'conference_title':settings.title})

@login_required
def save_simple_schedule_settings(request):
    num_days = request.POST.get('num_days',None)
    base_time = request.POST.get('base_time',None)
    if not Conference.objects.filter(user=request.user, pk=request.session['conf']).exists():
        settings=Conference()
        settings.num_days=num_days
        settings.slot_length=base_time
        settings.user=request.user
        settings.start_times = ['7:00' for x in range(num_days)]
        settings.save()
    else:
        settings=Conference.objects.get(user=request.user, pk=request.session['conf'])
        settings.num_days=num_days
        settings.slot_length=base_time
        # Update settings string
        s = schedule_settings_class(settings.settings_string, int(settings.num_days))
        settings.settings_string = str(s)
        # Update schedule string and starting times string
        schedule = ast.literal_eval(settings.schedule_string)
        names = ast.literal_eval(settings.names_string)
        times = ast.literal_eval(settings.start_times)
        if len(schedule) < int(num_days):
            for i in range(int(num_days)-len(schedule)):
                    schedule.append([])
                    names.append([])
                    times.append('7:00')
        if len(schedule) > int(num_days):
            schedule = schedule[:int(num_days)]
            times = times[:int(num_days)]
            names = names[:int(num_days)]
        settings.schedule_string = str(schedule)
        settings.names_string = str(names)
        settings.start_times = str(times)
        # If we added extra days, the settings string needs to be updated
        settings.save()
    return redirect('/app/settings/schedule/')

@login_required
def change_start_time(request):
    request.session['time_error'] = ""
    day = int(request.POST.get('day', None))
    time = request.POST.get('time', None)
    # Validate starting time
    if not ':' in time:
        request.session['time_error'] = "Starting time must be in format Hour:Minutes (ie. 7:00)"
        return redirect('/app/settings/schedule/?day='+request.POST.get('day'))
    try:
        hour = int(time.split(':')[0])
        if hour < 0 or hour > 24:
            request.session['time_error'] = "Invalid hour"
            return redirect('/app/settings/schedule/?day='+request.POST.get('day'))
        minute = int(time.split(':')[1])
        if minute < 0 or minute > 59:
            request.session['time_error'] = "Invalid minute"
            return redirect('/app/settings/schedule/?day='+request.POST.get('day'))
    except:
        request.session['time_error'] = "Starting time must be in format Hour:Minutes (ie. 7:00)"
        return redirect('/app/settings/schedule/?day='+request.POST.get('day'))
    conf = Conference.objects.get(user=request.user, pk=request.session['conf'])
    start_times = ast.literal_eval(conf.start_times)
    start_times[day] = time
    conf.start_times = start_times
    conf.save()
    return redirect('/app/settings/schedule/?day='+request.POST.get('day'))

@login_required
def schedule_add_slot(request):
    settings_model = Conference.objects.get(user=request.user, pk=request.session['conf'])
    schedule = schedule_manager_class()
    if settings_model.schedule_string == "[]":
        schedule.set_settings(settings_model.settings_string)
        schedule.create_empty_list_from_settings()
    else:
        schedule.import_paper_schedule(settings_model.schedule_string)
    settings = schedule_settings_class(settings_model.settings_string, settings_model.num_days)
    settings.add_slot_to_day(int(request.POST.get('day')), settings_model.slot_length)
    settings_model.settings_string = str(settings)
    schedule.add_slot_to_day(int(request.POST.get('day')))
    settings_model.schedule_string = schedule.papers
    # Add name
    names = ast.literal_eval(settings_model.names_string)
    names[int(request.POST.get('day'))].append(['Slot'])
    settings_model.names_string = str(names)
    settings_model.save()
    return redirect('/app/settings/schedule/?day='+request.POST.get('day'))

@login_required
def schedule_add_parallel_slots(request):
    request.session['parallel_error']=""
    if not request.POST.get('num_slots').isdigit():
        request.session['parallel_error'] = "Enter a valid integer"
        return redirect('/app/settings/schedule/?day='+request.POST.get('day'))
    settings_model = Conference.objects.get(user=request.user, pk=request.session['conf'])
    settings = schedule_settings_class(settings_model.settings_string, settings_model.num_days)
    settings.add_parallel_slots_to_day(int(request.POST.get('day')), settings_model.slot_length,
                                       int(request.POST.get('num_slots')))
    settings_model.settings_string = str(settings)
    schedule = schedule_manager_class()
    schedule.set_settings(settings_model.settings_string)
    if settings_model.schedule_string == "[]":
        schedule.set_settings(settings_model.settings_string)
        schedule.create_empty_list_from_settings()
    else:
        schedule.import_paper_schedule(settings_model.schedule_string)
    schedule.add_parallel_slots_to_day(int(request.POST.get('day')),int(request.POST.get('num_slots')))
    settings_model.schedule_string = schedule.papers
    # Add names
    names_to_add = []
    for i in range(0,int(request.POST.get('num_slots'))):
        names_to_add.append('Slot')
    names = ast.literal_eval(settings_model.names_string)
    names[int(request.POST.get('day'))].append(names_to_add)
    settings_model.names_string = str(names)
    settings_model.save()
    return redirect('/app/settings/schedule/?day='+request.POST.get('day'))

@login_required
def schedule_change_slot_time(request):
    settings_model = Conference.objects.get(user=request.user, pk=request.session['conf'])
    settings = schedule_settings_class(settings_model.settings_string, settings_model.num_days)
    settings.change_slot_time(day=int(request.POST.get('day')), row=int(request.POST.get('row')),
                              col=int(request.POST.get('col')),new_len=int(request.POST.get('len')))
    settings_model.settings_string = str(settings)
    settings_model.save()
    return redirect('/app/settings/schedule/?day='+request.POST.get('day'))

@login_required
def delete_slot(request):
    day = int(request.POST.get('day'))
    row = int(request.POST.get('row'))
    col = int(request.POST.get('col'))
    settings_model = Conference.objects.get(user=request.user, pk=request.session['conf'])
    settings = schedule_settings_class(settings_model.settings_string, settings_model.num_days)
    schedule = schedule_manager_class()
    schedule.import_paper_schedule(settings_model.schedule_string)
    settings.delete_slot(day,row,col)
    schedule.delete_slot(day,row,col)
    settings_model.settings_string = str(settings)
    settings_model.schedule_string = str(schedule)
    names = ast.literal_eval(settings_model.names_string)
    # Delete name
    names[day][row][col] = None
    names[day][row] = [ x for x in names[day][row] if x != None]
    # If a parallel group of slots contains no more slots, the group should also be deleted
    if names[day][row] == []:
        names[day][row] = None
        names[day] = [ x for x in names[day] if x != None]
    settings_model.names_string = str(names)
    settings_model.save()
    return redirect('/app/settings/schedule/?day='+request.POST.get('day'))

@login_required
def move_slot_up(request):
    day = int(request.POST.get('day'))
    slot = int(request.POST.get('slot'))-1
    if slot == 0:
        return redirect('/app/index/')
    conf = Conference.objects.get(user=request.user, pk=request.session['conf'])
    settings = ast.literal_eval(conf.settings_string)
    schedule = ast.literal_eval(conf.schedule_string)
    names = ast.literal_eval(conf.names_string)
    settings_day = settings[day]
    schedule_day = schedule[day]
    names_day = names[day]
    settings_day = swap_slots(slot, slot-1, settings_day)
    schedule_day = swap_slots(slot, slot-1, schedule_day)
    names_day = swap_slots(slot, slot-1, names_day)
    settings[day] = settings_day
    schedule[day] = schedule_day
    names[day] = names_day
    conf.settings_string = str(settings)
    conf.schedule_string = str(schedule)
    conf.names_string = str(names)
    conf.save()
    return redirect('/app/index/')

@login_required
def move_slot_down(request):
    day = int(request.POST.get('day'))
    slot = int(request.POST.get('slot'))-1
    conf = Conference.objects.get(user=request.user, pk=request.session['conf'])
    settings = ast.literal_eval(conf.settings_string)
    schedule = ast.literal_eval(conf.schedule_string)
    settings_day = settings[day]
    schedule_day = schedule[day]
    if slot == len(settings_day) - 1 or slot == len(schedule_day) - 1:
        return redirect('/app/index/')
    settings_day = swap_slots(slot, slot+1, settings_day)
    schedule_day = swap_slots(slot, slot+1, schedule_day)
    settings[day] = settings_day
    schedule[day] = schedule_day
    conf.settings_string = str(settings)
    conf.schedule_string = str(schedule)
    conf.save()
    return redirect('/app/index/')

def swap_slots(slot1_index, slot2_index, list):
    temp_slot = list[slot2_index]
    list[slot2_index] = list[slot1_index]
    list[slot1_index] = temp_slot
    return list

@login_required
def rename_slot(request):
    day = int(request.POST.get('day'))
    col = int(request.POST.get('col'))
    row = int(request.POST.get('row'))
    conf = Conference.objects.get(user=request.user, pk=request.session['conf'])
    names = ast.literal_eval(conf.names_string)
    names[day][row][col] = request.POST.get('name')
    conf.names_string = str(names)
    conf.save()
    return redirect('/app/settings/schedule/?day='+request.POST.get('day'))



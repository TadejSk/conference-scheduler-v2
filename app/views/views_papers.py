import ast
from django.forms import modelformset_factory
from django.http import Http404
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from app.models import Paper, Conference
from ..classes import schedule_manager_class


__author__ = 'Tadej'

def view_paper(request):
    id = request.GET.get('id',0)
    if id != 0:
        if Paper.objects.get(pk=id).user_id != request.user.pk:
            raise Http404("The selected view does not exist")
    PaperFormSet = modelformset_factory(Paper,extra=1,max_num=1, fields=('title', 'abstract', 'length'), can_delete=True)
    paper_id = request.GET.get('id', 0)
    formset = PaperFormSet(queryset=Paper.objects.filter(pk=paper_id))
    for form in formset:
        form.fields['length'].label = "Length in minutes"
    return render(request, 'app/view_paper.html', {'formset':formset})

def update_paper(request):
    if request.method == "POST":
        PaperFormSet = modelformset_factory(Paper,extra=0, fields=('title', 'abstract', 'length'), can_delete=True)
        formset = PaperFormSet(request.POST, request.FILES)
        for form in formset:
            form.is_valid()
        objs = formset.save(commit=False)
        for to_delete in formset.deleted_objects:
            # Remove paper from schedule
            set = schedule_manager_class()
            settings=Conference.objects.get(user=request.user, pk=request.session['conf'])
            if settings.schedule_string == "[]":
                set.create_empty_list(settings.settings_string)
            else:
                set.import_paper_schedule(settings.schedule_string)
            id = to_delete.id
            # Since papers not in schedule do not display if they are locked, they should be unlocked when removed from schedule
            paper = Paper.objects.get(pk=id)
            paper.is_locked = False
            paper.save()
            set.remove_paper(id)
            settings.schedule_string = str(set.papers)
            settings.save()
            to_delete.delete()
            return redirect('/app/index/')
        for obj in objs:
            if obj.user_id == None:
                obj.user_id = request.user.pk
            if obj.conference_id == None:
                obj.conference_id = request.session['conf']
            if obj.submission_id == None:
                obj.submission_id = 0
            obj.save()
        return redirect('/app/index/')
    else:
        raise Http404

@csrf_exempt
def add_paper_to_schedule(request):
    set = schedule_manager_class()
    settings=Conference.objects.get(user=request.user, pk=request.session['conf'])
    if settings.schedule_string == "[]":
        set.create_empty_list(settings.settings_string)
    else:
        set.import_paper_schedule(settings.schedule_string)
    row = request.POST.get('row')
    col = request.POST.get('col')
    id = request.POST.get('id')
    day = request.POST.get('day')
    set.set_settings(settings.settings_string)
    set.assign_paper(int(id), int(day), int(row), int(col))
    settings.schedule_string = str(set.papers)
    settings.save()
    return redirect('/app/index/')

@csrf_exempt
def remove_paper_from_schedule(request):
    set = schedule_manager_class()
    settings=Conference.objects.get(user=request.user, pk=request.session['conf'])
    if settings.schedule_string == "[]":
        set.create_empty_list(settings.settings_string)
    else:
        set.import_paper_schedule(settings.schedule_string)
    id = int(request.POST.get('id'))
    # Since papers not in schedule do not display if they are locked, they should be unlocked when removed from schedule
    paper = Paper.objects.get(pk=id)
    paper.is_locked = False
    paper.save()
    set.remove_paper(id)
    settings.schedule_string = str(set.papers)
    settings.save()
    return redirect('/app/index')

@csrf_exempt
def lock_paper(request):
    id = int(request.POST.get('id'))
    paper = Paper.objects.get(pk=id)
    if paper.user != request.user:
        raise Http404
    paper.is_locked = not paper.is_locked
    paper.save()
    return redirect('/app/index')
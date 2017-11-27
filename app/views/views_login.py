from django.contrib.auth.models import User

__author__ = 'Tadej'
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as django_login, logout
from app.models import Conference, Paper

def login(request):
     return render(request, 'app/login.html')


def login_register(request):
    return render(request, 'app/login_register.html')


def login_register_action(request):
    # Get user info from the form
    username = request.POST.get('username','')
    password = request.POST.get('password','')
    password2 = request.POST.get('confirm_password','')
    errors = []
    # Error checking
    if not password == password2:
        errors.append("Passwords do not match")
    if len(username) < 3:
        errors.append("Username is too short")
    if len(password) < 3:
        errors.append("Password is too short")
    if User.objects.filter(username=username).exists():
        errors.append("Username already exixts")
    if len(errors) > 0:
        return render(request, 'app/login_register.html', {'errors': errors, 'error_flag': True})
    user = User.objects.create_user(username, '', password)
    user.save()
    return redirect('/app/login')

def login_action(request):
    username = request.POST.get('username','')
    password = request.POST.get('password','')
    user = authenticate(username=username, password=password)
    if user is not None:
        if user.is_active:
            django_login(request, user)
            return redirect('/app/index')
        else:
            print('not active')
    else:
        return render(request, 'app/login.html', {'error_message':'Invalid username or password.'})

def logout_view(request):
    logout(request)
    return redirect('/app/login')


def login_as_guest(request):
    if request.user.is_authenticated():
        # if the user 'guestXYZ' is logged in, don't reset his conference objects,
        #   simply redirect him to the main page
        # if another (normal) user is already logged in, redirect him to the main page
        return redirect('/app/conference/list/')
    else:
        # Create new guest user
        suffix = User.objects.make_random_password(length=5, allowed_chars='1234567890')
        username = "guest" + suffix
        while User.objects.filter(username=username).exists():
            suffix = User.objects.make_random_password(length=5, allowed_chars='1234567890')
            username = "guest" + suffix
        password = ''
        user = User.objects.create_user(username, '', password)
        user.save()
        guest_user = authenticate(username=username, password=password)
        django_login(request, guest_user)
        c = Conference()
        c.user = user
        c.title = "Demo Conference " + suffix
        c.num_days = 2
        c.schedule_string = '[[[[]], [[]], [[], []]], []]'
        c.settings_string = '[[[60], [60], [60, 60]], []]'
        c.names_string = "[[['Slot1'], ['Slot2'], ['Slot 3a', 'Slot 3b']]]"
        c.start_times = "['7:00', '7:00']"
        c.save()
        return redirect('/app/conference/list/')

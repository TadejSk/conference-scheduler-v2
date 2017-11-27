# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('app', '0018_auto_20150806_1828'),
    ]

    operations = [
        migrations.CreateModel(
            name='Conference',
            fields=[
                ('id', models.AutoField(verbose_name='ID', auto_created=True, serialize=False, primary_key=True)),
                ('settings_string', models.CharField(max_length=100000)),
                ('schedule_string', models.CharField(max_length=100000, default=[])),
                ('slot_length', models.IntegerField()),
                ('num_days', models.IntegerField()),
                ('paper_graph_string', models.TextField(max_length=100000000, default='[]')),
                ('user', models.ForeignKey(to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.RemoveField(
            model_name='schedulesettings',
            name='user',
        ),
        migrations.DeleteModel(
            name='ScheduleSettings',
        ),
    ]

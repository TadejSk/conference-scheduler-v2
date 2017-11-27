# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0004_author_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='ScheduleSettings',
            fields=[
                ('id', models.AutoField(primary_key=True, auto_created=True, verbose_name='ID', serialize=False)),
                ('settings_string', models.CharField(max_length=100000)),
            ],
        ),
    ]

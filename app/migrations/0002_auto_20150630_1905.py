# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='author',
            name='paper',
        ),
        migrations.AddField(
            model_name='author',
            name='paper',
            field=models.ManyToManyField(to='app.Paper'),
        ),
    ]

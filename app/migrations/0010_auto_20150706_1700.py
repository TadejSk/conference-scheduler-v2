# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0009_schedulesettings_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='cluster',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paper',
            name='length',
            field=models.IntegerField(default=60),
        ),
    ]

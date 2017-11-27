# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0011_schedulesettings_schedule_string'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='is_locked',
            field=models.BooleanField(default=False),
        ),
    ]

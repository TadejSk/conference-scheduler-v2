# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0025_auto_20150820_1624'),
    ]

    operations = [
        migrations.AlterField(
            model_name='conference',
            name='start_times',
            field=models.CharField(default="['7:00']", max_length=100000),
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0023_paper_conference'),
    ]

    operations = [
        migrations.AddField(
            model_name='conference',
            name='start_times',
            field=models.CharField(max_length=100000, default='[[7:00]]'),
        ),
    ]

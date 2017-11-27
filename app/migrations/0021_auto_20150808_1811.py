# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0020_conference_title'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='simple_cluster',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='conference',
            name='num_days',
            field=models.IntegerField(default=1),
        ),
        migrations.AlterField(
            model_name='conference',
            name='schedule_string',
            field=models.CharField(max_length=100000, default='[[]]'),
        ),
        migrations.AlterField(
            model_name='conference',
            name='settings_string',
            field=models.CharField(max_length=100000, default='[[]]'),
        ),
        migrations.AlterField(
            model_name='conference',
            name='slot_length',
            field=models.IntegerField(default=60),
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0021_auto_20150808_1811'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='simple_visual_x',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='paper',
            name='simple_visual_y',
            field=models.FloatField(default=0),
        ),
    ]

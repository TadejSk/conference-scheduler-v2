# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0017_schedulesettings_paper_graph_string'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='visual_x',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='paper',
            name='visual_y',
            field=models.FloatField(default=0),
        ),
    ]

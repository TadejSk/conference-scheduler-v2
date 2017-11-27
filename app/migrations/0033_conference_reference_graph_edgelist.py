# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0032_paper_uses_advanced_features'),
    ]

    operations = [
        migrations.AddField(
            model_name='conference',
            name='reference_graph_edgelist',
            field=models.TextField(max_length=100000000, default='[]'),
        ),
    ]

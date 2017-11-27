# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0033_conference_reference_graph_edgelist'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='paper_references',
            field=models.TextField(default='', max_length=1000000),
        ),
    ]

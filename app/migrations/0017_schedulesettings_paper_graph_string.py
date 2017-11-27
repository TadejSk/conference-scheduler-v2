# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0016_upoladedfile'),
    ]

    operations = [
        migrations.AddField(
            model_name='schedulesettings',
            name='paper_graph_string',
            field=models.TextField(default='[]', max_length=100000000),
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0013_paper_submission_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='add_to_col',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paper',
            name='add_to_day',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paper',
            name='add_to_row',
            field=models.IntegerField(default=0),
        ),
    ]

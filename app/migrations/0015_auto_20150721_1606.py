# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0014_auto_20150721_1553'),
    ]

    operations = [
        migrations.AlterField(
            model_name='paper',
            name='add_to_col',
            field=models.IntegerField(default=-1),
        ),
        migrations.AlterField(
            model_name='paper',
            name='add_to_day',
            field=models.IntegerField(default=-1),
        ),
        migrations.AlterField(
            model_name='paper',
            name='add_to_row',
            field=models.IntegerField(default=-1),
        ),
    ]

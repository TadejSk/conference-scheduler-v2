# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0026_auto_20150820_1626'),
    ]

    operations = [
        migrations.AddField(
            model_name='conference',
            name='names_string',
            field=models.CharField(max_length=100000, default='[[]]'),
        ),
    ]

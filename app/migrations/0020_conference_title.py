# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0019_auto_20150807_1545'),
    ]

    operations = [
        migrations.AddField(
            model_name='conference',
            name='title',
            field=models.CharField(max_length=1000, default='Unnamed conference'),
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0022_auto_20150808_1823'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='conference',
            field=models.ForeignKey(to='app.Conference', null=True, default=None),
        ),
    ]

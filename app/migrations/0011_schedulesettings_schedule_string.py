# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0010_auto_20150706_1700'),
    ]

    operations = [
        migrations.AddField(
            model_name='schedulesettings',
            name='schedule_string',
            field=models.CharField(max_length=100000, default=[]),
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_schedulesettings'),
    ]

    operations = [
        migrations.AddField(
            model_name='schedulesettings',
            name='settings_string2',
            field=models.CharField(max_length=100000, default=''),
            preserve_default=False,
        ),
    ]

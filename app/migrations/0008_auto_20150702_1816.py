# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0007_remove_schedulesettings_settings_string2'),
    ]

    operations = [
        migrations.AddField(
            model_name='schedulesettings',
            name='num_days',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='schedulesettings',
            name='slot_length',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
    ]

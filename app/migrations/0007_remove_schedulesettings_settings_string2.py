# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0006_schedulesettings_settings_string2'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='schedulesettings',
            name='settings_string2',
        ),
    ]

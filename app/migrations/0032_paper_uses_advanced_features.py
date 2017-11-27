# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0031_auto_20170313_1736'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='uses_advanced_features',
            field=models.BooleanField(default=False),
        ),
    ]

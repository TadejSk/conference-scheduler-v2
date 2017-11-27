# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_auto_20150630_1905'),
    ]

    operations = [
        migrations.RenameField(
            model_name='author',
            old_name='paper',
            new_name='papers',
        ),
    ]

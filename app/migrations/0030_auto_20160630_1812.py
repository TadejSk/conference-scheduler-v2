# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0029_conference_reviewer_biddings_string'),
    ]

    operations = [
        migrations.AlterField(
            model_name='paper',
            name='length',
            field=models.IntegerField(default=15),
        ),
    ]

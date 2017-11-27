# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0028_upoladedfile_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='conference',
            name='reviewer_biddings_string',
            field=models.CharField(default='No reviewer biddings uploaded', max_length=1000),
        ),
    ]

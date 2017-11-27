# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0030_auto_20160630_1812'),
    ]

    operations = [
        migrations.AddField(
            model_name='paper',
            name='extracted_terms',
            field=models.CharField(default='[]', max_length=100000),
        ),
        migrations.AddField(
            model_name='paper',
            name='ground_truth_class',
            field=models.CharField(default='unclassfied', max_length=100000),
        ),
        migrations.AddField(
            model_name='paper',
            name='pos_tagged_text_content',
            field=models.TextField(default='', max_length=1000000),
        ),
        migrations.AddField(
            model_name='paper',
            name='text_content',
            field=models.TextField(default='', max_length=1000000),
        ),
        migrations.AlterField(
            model_name='conference',
            name='reviewer_biddings_string',
            field=models.CharField(default='Reviewer biddings: 0', max_length=1000),
        ),
    ]

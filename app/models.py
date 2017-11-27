from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class Paper(models.Model):
    title = models.CharField(max_length=1000)
    abstract = models.TextField(max_length=1000000)
    user = models.ForeignKey(User)
    conference = models.ForeignKey('app.Conference', default=None, null=True)
    cluster = models.IntegerField(default=0)
    simple_cluster = models.IntegerField(default=0)
    add_to_day = models.IntegerField(default = -1)
    add_to_row = models.IntegerField(default = -1)
    add_to_col = models.IntegerField(default = -1)
    length = models.IntegerField(default=15)
    is_locked = models.BooleanField(default=False)
    submission_id = models.IntegerField()   # The id imported from the xls file - used for constructing graphs
    visual_x = models.FloatField(default = 0)
    simple_visual_x = models.FloatField(default = 0)
    simple_visual_y = models.FloatField(default = 0)
    visual_y = models.FloatField(default = 0)
    # Stuff for more advanced NLP methods
    uses_advanced_features = models.BooleanField(default=False)
    text_content = models.TextField(max_length=1000000, default = "")
    paper_references = models.TextField(max_length=1000000, default = "")
    pos_tagged_text_content = models.TextField(max_length=1000000, default = "")
    extracted_terms = models.CharField(max_length=100000, default="[]")
    ground_truth_class = models.CharField(max_length=100000, default="unclassfied")
    def __str__(self):
        return self.title+" ("+self.user.username+")"

class Author(models.Model):
    name = models.CharField(max_length=255)
    papers = models.ManyToManyField(Paper)
    user = models.ForeignKey(User)
    def __str__(self):
        return self.name+" ("+self.user.username+")"

class Conference(models.Model):
    title = models.CharField(max_length=1000, default="Unnamed conference")
    settings_string = models.CharField(max_length=100000, default='[[]]')
    schedule_string = models.CharField(max_length=100000, default='[[]]')
    names_string = models.CharField(max_length=100000, default='[[]]')
    slot_length = models.IntegerField(default=60)
    num_days = models.IntegerField(default=1)
    reviewer_biddings_string = models.CharField(max_length=1000, default="Reviewer biddings: 0")
    paper_graph_string = models.TextField(max_length=100000000, default = '[]')
    reference_graph_edgelist = models.TextField(max_length=100000000, default = '[]')
    user = models.ForeignKey(User)
    start_times=models.CharField(max_length=100000, default = "['7:00']")
    def __str__(self):
        return self.settings_string

class UpoladedFile(models.Model):
    file = models.FileField(upload_to='uploads/%Y%m%d')
    user = models.ForeignKey(User)

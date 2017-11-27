__author__ = 'Tadej'
from django import template

register = template.Library()

@register.assignment_tag
def get_paper(value, arg1, arg2, arg3):
    return value[arg1][arg2][arg3]
from app.models import Paper

__author__ = 'Tadej'
import ast

"""
The class is used to serialize and deserialize schedule information between a python list and a settings string
The settings are stored in the following format
settings_list: [[dayx_list]*]
dayx_list:[slot*]
slot:sequential_slots* | [parallel_slots*]
sequential_slot: [paper_id*],
parallel_slot: [paper_id*],
paper_id: int
Example:
    The list [[[1], [2], [3,8] [[4], [5,6], [7]] would create the folowing timetable

                    SLOT1: paper1
                    SLOT2: paper2
                    SLOT3: paper3 and paper8
    SLOT4:paper 4   SLOT4: paper5 and paper6   SLOT4:paper7

    Where each slot is 60 minutes long and occurs on day 1
"""

class schedule_manager_class(object):
    papers=[]
    settings=[]

    def __init__(self):
        return

    """
    def __init__(self, paper_string, num_days:int):
        if paper_string == "":
            self.papers = []
            for i in range(num_days):
                self.papers.append([])
        else:
            self.papers = ast.literal_eval(paper_string)
        return
    """
    def import_paper_schedule(self, schedule_string:str):
        self.papers = ast.literal_eval(schedule_string)

    def assign_paper(self, paper:int, day:int, slot_row:int, slot_col:int):
        self.remove_paper(paper)
        free_time = self.get_slot_free_time(day, slot_row, slot_col)
        day = self.papers[day]
        row = day[slot_row]
        col = row[slot_col]
        if Paper.objects.get(pk=paper).length > free_time:
            return False
        col.append(paper)
        return True


    def remove_paper(self, paper:int):
        for di,day_list in enumerate(self.papers):
            for ri,row in enumerate(day_list):
                for ci,col in enumerate(row):
                    for i in range(0,len(col)):
                        if col[i] == paper:
                            col = col[0:i] + col[i+1:len(col)]
                            self.papers[di][ri][ci] = col
                            return True
        return False

    def get_slot_free_time(self, day:int, row:int, col:int):
        max_length = self.settings[day][row][col]
        slot = self.papers[day][row][col]
        total_length = 0
        for paper_id in slot:
            paper = Paper.objects.get(pk=paper_id)
            total_length += paper.length
        return max_length - total_length

    def set_settings(self, settings_string):
        self.settings = ast.literal_eval(settings_string)
        return

    def create_empty_list_from_settings(self):
        schedule = []
        for settings_day in self.settings:
            schedule_day = []
            for settings_row in settings_day:
                schedule_row = []
                for settings_col in settings_row:
                    schedule_row.append([])
                schedule_day.append(schedule_row)
            schedule.append(schedule_day)
        self.papers = schedule

    def add_slot_to_day(self, day:int):
        day_schedule = self.papers[day]
        day_schedule.append([[]])
        return

    def add_parallel_slots_to_day(self, day:int, num_slots:int):
        day_schedule = self.papers[day]
        parallel_slots = []
        for i in range(num_slots):
            parallel_slots.append([])
        day_schedule.append(parallel_slots)
        return

    def delete_slot(self, day:int, row:int, col:int):
        self.papers[day][row][col] = None
        self.papers[day][row] = [ x for x in self.papers[day][row] if x != None]
        # If a parallel group of slots contains no more slots, the group should also be deleted
        if self.papers[day][row] == []:
            self.papers[day][row] = None
            self.papers[day] = [ x for x in self.papers[day] if x != None]
        return


    def __str__(self):
        return str(self.papers)


__author__ = 'Tadej'
from . import views
from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^login/',  views.login, name='login'),
    url(r'^login_action/',  views.login_action, name='login_action'),
    url(r'^login_register/',  views.login_register, name='login_register'),
    url(r'^login_as_guest/',  views.login_as_guest, name='login_as_guest'),
    url(r'^login_register_action/',  views.login_register_action, name='login_register_action'),
    url(r'^index/', views.index, name='index'),
    url(r'^$', views.index, name='index0'), ##
    url(r'^import_page/load_raw_data', views.load_raw_data, name='import_page_load_raw_data'),
    url(r'^import_page/pos_tag_data', views.pos_tag_data, name='import_page_pos_tag_data'),
    url(r'^import_page/find_terms', views.find_terms, name='import_page_find_terms'),
    url(r'^import_page/create_reference_graph', views.create_reference_graph, name='import_page_create_reference_graph'),
    url(r'^import_page/run_clustering', views.run_clustering, name='import_page_run_clustering'),
    url(r'^import_page/', views.import_page, name='import_page'),
    url(r'^logout_view/', views.logout_view, name='logout'),
    url(r'^import_data/', views.import_data, name='import_data'),
    url(r'^import_demo_data/', views.import_demo_data, name='import_demo_data'),
    url(r'^import_assignments_data/', views.import_assignments_data, name='import_assignments_data'),
    url(r'^import_demo_assignments_data/', views.import_demo_assignments_data, name='import_demo_assignments_data'),
    url(r'^write_data/', views.write_paper_data, name='write_data'),
    url(r'^write_assignments_data/', views.write_assignments_data, name='write_assignments_data'),
    url(r'^settings/schedule/save_simple/', views.save_simple_schedule_settings, name='save_simple_schedule_settings'),
    url(r'^settings/schedule/add_slot/', views.schedule_add_slot, name='schedule_add_slot'),
    url(r'^settings/schedule/add_parallel_slots/', views.schedule_add_parallel_slots, name='schedule_add_parallel_slots'),
    url(r'^settings/schedule/change_slot_length/', views.schedule_change_slot_time, name='schedule_change_slot_length'),
    url(r'^settings/schedule/delete_slot/', views.delete_slot, name='schedule_delete_slot'),
    url(r'^settings/schedule/change_start_time', views.change_start_time, name='change_start_time'),
    url(r'^settings/schedule/rename_slot', views.rename_slot, name='rename_slot'),
    url(r'^settings/schedule/up/', views.move_slot_up, name='move_slot_up'),
    url(r'^settings/schedule/down/', views.move_slot_down, name='move_slot_down'),
    url(r'^settings/schedule/', views.schedule_settings, name='schedule_settings'),
    url(r'^papers/view/', views.view_paper, name='view_papers'),
    url(r'^papers/update/', views.update_paper, name='update_paper'),
    url(r'^papers/add_to_schedule/', views.add_paper_to_schedule, name='add_paper_to_schedule'),
    url(r'^papers/remove_from_schedule/', views.remove_paper_from_schedule, name='remove_paper_from_schedule'),
    url(r'^papers/change_lock/', views.lock_paper, name='lock_paper'),
    url(r'^clustering/basic/', views.basic_clustering, name='basic_clustering'),
    url(r'^clustering/settings/', views.clustering_settings, name='clustering_settings'),
    url(r'^clustering/results/overview', views.clustering_results_overview, name='clustering_results_overview'),
    url(r'^clustering/results/all', views.clustering_results, name='clustering_results_all'),
    url(r'^clustering/results/assigned', views.clustering_results_assigned, name='clustering_results_assigned'),
    url(r'^conference/list/', views.conference_list, name='conference_list'),
    url(r'^conference/create/', views.create_conference, name='create_conference'),
    url(r'^conference/delete/', views.delete_conference, name='delete_conference'),
    url(r'^conference/copy/', views.copy_conference, name='copy_conference'),
    url(r'^conference/rename/action', views.rename_conference_action, name='rename_conference_action'),
    url(r'^conference/rename/', views.rename_conference, name='rename_conference'),
    url(r'^conference/export/', views.export_schedule, name='export_schedule'),

]
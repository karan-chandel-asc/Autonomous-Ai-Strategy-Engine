from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('home/',                               views.HomeView.as_view(),                  name='home'),
    path('dashboard/',                          views.DashboardView.as_view(),             name='dashboard'),
    path('strategy/',                           views.StrategyView.as_view(),              name='strategy'),
    path('report/',                             views.ReportView.as_view(),                name='report'),
    path('profile/',                            views.ProfileView.as_view(),               name='profile'),
    path('generate-thread-id/',                 views.GenerateThreadID.as_view(),          name='generate_thread_id'),
    path('api/validate-query/',                  views.ValidateQueryView.as_view(),         name='validate_query'),
    path('pipeline/start/',                     views.StartPipelineView.as_view(),         name='pipeline_start'),
    path('pipeline/stream/<str:thread_id>/',    views.StreamPipelineView.as_view(),        name='pipeline_stream'),
    # Reports & Dashboard
    path('api/dashboard/stats/',                 views.DashboardStatsView.as_view(), name='dashboard_stats'),
    path('api/reports/',                         views.ReportListView.as_view(),    name='report_list'),
    path('api/report/<str:thread_id>/',           views.ReportDetailView.as_view(),  name='report_detail'),
    # Knowledge Base
    path('knowledge-base/',                     views.KnowledgeBaseDashboardView.as_view(), name='kb_dashboard'),
    path('api/knowledge-base/',                 views.KnowledgeBaseView.as_view(),          name='kb_list'),
    path('api/knowledge-base/<uuid:doc_id>/',   views.KnowledgeBaseDetailView.as_view(),    name='kb_detail'),

    path('api/knowledge-base-for-strategy/',   views.KnowledgeBaseForDashboardView.as_view(), name='kb_for_dashboard'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

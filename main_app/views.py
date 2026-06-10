import json
import redis as redis_lib
from django.http import StreamingHttpResponse
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from Ai_strategy_engine.logger import logger
from .chains import ParallelStrategicAnalysis, AggregatedStrategicAnalysis
from .helper import *
from django.shortcuts import render, redirect
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from auth_app.schemas import *
from .models import Document, KnowledgeBase, Thread
from .serializers import ReportSerializer, ThreadSerializer
from .pagination import StandardPagination


class HomeView(APIView):
    def get(self, request):
        return render(request, 'ase_home.html')


class DashboardView(APIView):
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/auth-api/user_login/')
        return render(request, 'ase_dashboard.html')


class StrategyView(APIView):
    permission_classes = []

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/auth-api/user_login/')
        return render(request, 'ase_strategy.html')


class ReportView(APIView):
    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/auth-api/user_login/')
        return render(request, 'ase_report.html')


class ProfileView(APIView):
    permission_classes = []

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect('/auth-api/user_login/')
        return render(request, 'ase_profile.html')



class GenerateThreadID(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        logger.info("Request comes for generating thread id")
        try:
            thread_service = ThreadService()
            thread_id = thread_service.generate_thread_id()
            new_thread_id = thread_service.create_thread(request.user, thread_id)
            return Response(success_response(message="Thread ID generated successfully", data={"thread_id": new_thread_id}), status=status.HTTP_200_OK)
        except Exception as e:
            return Response(error_response(message=str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ValidateQueryView(APIView):
    """Lightweight pre-flight: classify the query before firing the expensive pipeline."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage as LCHuman

        query = (request.data.get("input_query") or "").strip()
        if not query:
            return Response(
                success_response(data={"valid": False, "message": "Please enter a strategic objective before launching the pipeline."}),
                status=status.HTTP_200_OK,
            )

        SYSTEM = (
            "You validate queries for a business strategy AI engine called Nexus. "
            "Return ONLY JSON — no explanation, no markdown fences.\n"
            "Return {\"valid\": true} for any specific business, product, market, or strategic question.\n"
            "Return {\"valid\": false, \"message\": \"...\"} for: greetings (hi/hello/hey/thanks), casual chat, "
            "single words, test strings, math, or anything unrelated to business or strategy.\n"
            "If invalid, the message must be exactly 2 sentences:\n"
            "  Sentence 1: Directly reply to what the user said — if they greeted, greet them back warmly; "
            "if they asked something off-topic, acknowledge it briefly.\n"
            "  Sentence 2: Tell them what Nexus is built for and give one short example of a good query.\n"
            "Keep it warm and natural. Never lecture."
        )
        try:
            llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0,
                max_tokens=150,
            )
            resp = llm.invoke([SystemMessage(content=SYSTEM), LCHuman(content=query)])
            raw = (getattr(resp, "content", "") or "").strip()
            raw = re.sub(r'^```[^\n]*\n?', '', raw).rstrip('`').strip()
            parsed = json.loads(raw)
            return Response(success_response(data=parsed), status=status.HTTP_200_OK)
        except Exception as e:
            logger.warning(f"[ValidateQuery] fallback allow: {e}")
            return Response(success_response(data={"valid": True, "message": ""}), status=status.HTTP_200_OK)


class StartPipelineView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from .tasks import run_pipeline_task
        logger.info("StartPipeline request received")
        try:
            input_validation_service = InputValidationService()
            is_valid, result = input_validation_service.validate_input(request.data)
            if not is_valid:
                return Response(error_response(message=result), status=status.HTTP_400_BAD_REQUEST)

            validated_data = result
            thread_id      = validated_data.thread_id
            input_query    = validated_data.input_query

            # Accept optional list of already-indexed KB document IDs
            document_ids = request.data.getlist("document_ids") or []

            run_pipeline_task.delay(thread_id, input_query, document_ids=document_ids)

            return Response(
                success_response(message="Pipeline started", data={"thread_id": thread_id}),
                status=status.HTTP_202_ACCEPTED,
            )
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return Response(error_response(message=str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── Knowledge Base views ───────────────────────────────────────────────────────
class KnowledgeBaseForDashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs         = KnowledgeBase.objects.filter(owner=request.user, status='indexed')
        paginator  = StandardPagination()
        page       = paginator.paginate_queryset(qs, request)
        data = [
            {"id": str(d.id), "name": d.name, "status": d.status,
             "chunk_count": d.chunk_count, "error": d.error,
             "created_at": d.created_at.isoformat()}
            for d in page
        ]
        return paginator.get_paginated_response(data)


class KnowledgeBaseView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs        = KnowledgeBase.objects.filter(owner=request.user)
        paginator = StandardPagination()
        page      = paginator.paginate_queryset(qs, request)
        data = [
            {"id": str(d.id), "name": d.name, "status": d.status,
             "chunk_count": d.chunk_count, "error": d.error,
             "created_at": d.created_at.isoformat()}
            for d in page
        ]
        return paginator.get_paginated_response(data)

    def post(self, request):
        """Upload a document and trigger async indexing into Pinecone."""
        from .tasks import index_document_task

        file = request.FILES.get("file")
        if not file:
            return Response(error_response(message="No file provided"), status=status.HTTP_400_BAD_REQUEST)

        allowed = {".pdf", ".txt", ".docx", ".md"}
        ext = "." + file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
        if ext not in allowed:
            return Response(
                error_response(message=f"Unsupported file type. Allowed: {', '.join(allowed)}"),
                status=status.HTTP_400_BAD_REQUEST,
            )

        doc = KnowledgeBase.objects.create(
            owner=request.user,
            name=file.name,
            file=file,
            status="pending",
        )
        index_document_task.delay(str(doc.id))

        return Response(
            success_response(
                message="Document uploaded — indexing started",
                data={"id": str(doc.id), "name": doc.name, "status": doc.status},
            ),
            status=status.HTTP_202_ACCEPTED,
        )


class KnowledgeBaseDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, doc_id):
        """Remove a document from the KB and delete its Pinecone vectors."""
        from .pinecone_service import PineconeService

        try:
            doc = KnowledgeBase.objects.get(id=doc_id, owner=request.user)
        except KnowledgeBase.DoesNotExist:
            return Response(error_response(message="Document not found"), status=status.HTTP_404_NOT_FOUND)

        try:
            PineconeService().delete_document(str(doc.id))
        except Exception as e:
            logger.warning(f"[KB] Pinecone delete failed for {doc_id}: {e}")

        doc.file.delete(save=False)
        doc.delete()
        return Response(success_response(message="Document deleted"), status=status.HTTP_200_OK)


class KnowledgeBaseDashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return render(request, "ase_knowledge_base.html")


class DashboardStatsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from .models import AgentResponse
        from django.db.models import Avg, Count
        threads = Thread.objects.filter(owner=request.user)
        total     = threads.count()
        completed = threads.filter(status='complete').count()
        running   = threads.filter(status='running').count()
        failed    = threads.filter(status='failed').count()
        queued    = threads.filter(status='queued').count()
        denom     = completed + failed
        success_rate = round((completed / denom) * 100, 1) if denom else 0
        avg_ms = threads.filter(status='complete', runtime_ms__isnull=False).aggregate(a=Avg('runtime_ms'))['a']
        avg_s  = round(avg_ms / 1000, 1) if avg_ms else 0

        agent_labels = {
            'executive_summary':     'Executive Summary',
            'market_analysis':       'Market Analysis',
            'competitive_landscape': 'Competitive Landscape',
            'monetization_strategy': 'Monetization Strategy',
            'risk_assessment':       'Risk Assessment',
            'product_roadmap':       'Product Roadmap',
            'weakness_review':       'Weakness Review',
        }
        module_usage = []
        for key, label in agent_labels.items():
            done = AgentResponse.objects.filter(thread__owner=request.user, agent_name=key, status='complete').count()
            module_usage.append({'key': key, 'label': label, 'complete': done, 'total': total})

        return Response(success_response(data={
            'total': total, 'completed': completed, 'running': running,
            'failed': failed, 'queued': queued,
            'success_rate': success_rate, 'avg_runtime_s': avg_s,
            'module_usage': module_usage,
        }), status=status.HTTP_200_OK)


class ReportListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs         = Thread.objects.filter(owner=request.user)
        paginator  = StandardPagination()
        page       = paginator.paginate_queryset(qs, request)
        serializer = ThreadSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)


class ReportDetailView(APIView):
    """GET /api/report/<thread_id>/ — full report with agents + final strategy."""
    permission_classes = [IsAuthenticated]

    def get(self, request, thread_id):
        try:
            thread = Thread.objects.get(thread_id=thread_id, owner=request.user)
        except Thread.DoesNotExist:
            return Response(error_response(message="Report not found"), status=status.HTTP_404_NOT_FOUND)
        serializer = ReportSerializer(thread)
        return Response(success_response(data=serializer.data), status=status.HTTP_200_OK)

    def delete(self, request, thread_id):
        try:
            thread = Thread.objects.get(thread_id=thread_id, owner=request.user)
        except Thread.DoesNotExist:
            return Response(error_response(message="Report not found"), status=status.HTTP_404_NOT_FOUND)
        thread.delete()
        return Response(success_response(message="Session deleted"), status=status.HTTP_200_OK)


class StreamPipelineView(View):
    def get(self, request, thread_id):
        def event_stream():
            r = redis_lib.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            key = f"pipeline:{thread_id}"
            yield f"data: {json.dumps({'step': 'connected', 'message': 'Connected to pipeline stream'})}\n\n"
            while True:
                result = r.blpop(key, timeout=25)
                if result is None:
                    yield ": keepalive\n\n"
                    continue
                _, raw = result
                yield f"data: {raw}\n\n"
                data = json.loads(raw)
                if data.get("done") or data.get("error"):
                    break

        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"]     = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response


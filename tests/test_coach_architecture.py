import unittest
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base
from Logic.agent_runtime import (
    AgentHandoff,
    build_initial_agent_state,
    complete_agent_run,
    normalize_agent_role,
    normalize_handoff_status,
    normalize_message_type,
    record_agent_handoff,
    record_agent_messages,
    record_agent_step,
    record_agent_tool_calls,
    runtime_summary,
    start_agent_run,
)
from Logic.coach.evaluation_suite import SCENARIOS, run_offline_coach_evaluation
from Logic.coach.attachments import prepare_attachments
from Logic.coach.mastery_engine import build_active_mastery_profile
from Logic.coach.multimodal_learning import infer_diagram_specs, parse_formula_signals
from Logic.coach.source_metadata import build_source_bundle
from Logic.coach.specialist_tools import calculator, diagram_helper, formula_checker
from Logic.coach.unified_orchestrator import build_orchestration_plan
from Logic.coach.lead_orchestrator import build_lead_coach_decision
from Logic.coach.retrieval_gate import evaluate_retrieval_gate
from Logic.coach.growth_loop import evaluate_turn_growth
from Logic.coach.llm_router import LLMRouter
from Logic.coach.model_gateway import ModelGateway
from Logic.coach.tool_gateway import ToolGateway
from Logic.coach.mastery_store import build_mastery_signal, build_student_memory_update, persist_mastery_signal
from Logic.coach.quality_scorer import score_coach_answer
from Logic.coach.query_understanding import understand_query
from Logic.coach.intent_scenarios import (
    build_conversation_response,
    build_scenario_intent_profile,
    rank_intent_scenarios,
)
from Logic.coach.response_planner import (
    ResponsePlannerOutput,
    build_response_plan,
    build_response_plan_instruction,
)
from Logic.coach.answer_repair import decide_answer_repair, mark_repair_applied
from Logic.coach.turn_engine import (
    build_adaptive_answer_blocks,
    parse_semantic_event,
    resolve_hybrid_query,
    semantic_event,
)
from Logic.agents.coach_agent import (
    _material_supports_question,
    _safe_tutor_fallback,
    coach_agent_stream,
)
from Logic.analytics_engine import get_user_analytics
from Logic.agent_event_bus import AgentEvent
from main import (
    GenerateMCQRequest,
    SectionAIRequest,
    _conversation_rows_for_user,
    _group_conversation_rows,
    _serialize_coach_conversation,
    format_test_session,
    require_owned_study_session,
    session_id_belongs_to_user,
)
from Logic.observability_store import (
    get_observability_events_since,
    get_observability_summary,
    persist_coach_trace,
    persist_observability_event,
)
from models import (
    AICoachMemory,
    AICoachInteraction,
    AICoachProfile,
    AgentRuntimeHandoff,
    AgentRuntimeMessage,
    AgentRuntimeRun,
    AgentRuntimeStep,
    AgentRuntimeToolCall,
    ModelToolTrace,
    ObservabilityEvent,
    TestHistory,
    UserProgress,
)


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary route failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered tutor response"))],
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class AlwaysFailCompletions:
    def create(self, **_kwargs):
        raise RuntimeError("provider unavailable")


class StaticCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0
        self.requests = []

    def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
        )


class StaticStreamCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if kwargs.get("stream"):
            return [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk))])
                for chunk in self.chunks
            ]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="".join(self.chunks)))]
        )


class ProviderClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class FakeVisionRouter:
    def complete(self, **_kwargs):
        return """
{
  "visible_text": "Find force when mass = 2 kg and acceleration = 3 m/s^2. CH4 is methane.",
  "handwritten_work": "F = ma",
  "math_lines": ["F = m × a", "F = 2 × 3"],
  "formulas": ["F = ma", "CH4"],
  "diagram_labels": ["object", "force arrow", "mass"],
  "likely_topic": "Newton's second law",
  "confidence": 0.86
}
""".strip()


class CoachArchitectureTests(unittest.TestCase):
    def test_selected_topic_quick_actions_do_not_false_fail_coverage(self):
        retrieved = {
            "context": (
                "Atomic mass is the mass of an atom relative to one twelfth "
                "of the mass of a carbon-12 atom."
            ),
            "section_id": "atomic_mass",
            "scope": {
                "topic": "Atomic Mass of an Element",
                "chapter": "Some Basic Concepts of Chemistry",
            },
            "retrieval_question": (
                "Explain this concept from the basics with one simple example."
            ),
        }

        self.assertTrue(
            _material_supports_question(retrieved, {"learning_context": {}}, {})
        )

    def test_semantic_paraphrase_can_pass_scoped_coverage_gate(self):
        retrieved = {
            "context": "The electron occupies quantised energy levels around the nucleus.",
            "section_id": "bohr_model",
            "scope": {
                "topic": "Bohr's Model",
                "chapter": "Structure of Atom",
            },
            "retrieval_question": "Why are only certain orbits allowed?",
            "best_semantic_similarity": 0.43,
        }

        self.assertTrue(
            _material_supports_question(retrieved, {"learning_context": {}}, {})
        )

    def test_unrelated_question_still_fails_selected_topic_coverage(self):
        retrieved = {
            "context": "Atomic mass is measured relative to carbon-12.",
            "section_id": "atomic_mass",
            "scope": {
                "topic": "Atomic Mass of an Element",
                "chapter": "Some Basic Concepts of Chemistry",
            },
            "retrieval_question": "Explain photosynthesis and chlorophyll.",
            "best_semantic_similarity": 0.0,
        }

        self.assertFalse(
            _material_supports_question(retrieved, {"learning_context": {}}, {})
        )

    def _response_plan_for(
        self,
        prompt,
        *,
        declared_intent="general",
        has_history=False,
        scope=None,
        attachments=(),
        adaptive_context=None,
        mode="coach",
    ):
        query = understand_query(prompt, declared_intent=declared_intent, has_history=has_history)
        answer_format = {
            "id": query.answer_format,
            "label": query.answer_format.replace("_", " ").title(),
            "sections": [],
            "rules": [],
        }
        return build_response_plan(
            question=prompt,
            query=query,
            answer_format=answer_format,
            mode=mode,
            retrieval_policy=query.retrieval_policy,
            selected_scope=scope or {},
            attachments=attachments,
            adaptive_context=adaptive_context or {},
            conversation_context={"is_follow_up": query.is_follow_up},
        )

    def test_response_planner_dynamic_student_answer_styles(self):
        short = self._response_plan_for("Photosynthesis means?")
        self.assertEqual(short.answer_length, "short")
        self.assertEqual(short.format_style, "plain")
        self.assertFalse(short.include_examples)

        medium = self._response_plan_for("What is photosynthesis?")
        self.assertEqual(medium.answer_length, "medium")
        self.assertTrue(medium.include_examples)

        deep = self._response_plan_for("Explain photosynthesis deeply.")
        self.assertIn(deep.answer_length, {"detailed", "long"})
        self.assertEqual(deep.tone, "deep_teaching")
        self.assertTrue(deep.include_summary)

        exam = self._response_plan_for("Give 5 marks answer on photosynthesis.")
        self.assertEqual(exam.format_style, "exam_answer")
        self.assertEqual(exam.tone, "exam_focused")
        self.assertEqual(exam.mode, "exam")

        mcq = self._response_plan_for(
            "Give MCQs from this chapter",
            scope={"chapter": "Photosynthesis", "section_id": "photosynthesis"},
        )
        self.assertEqual(mcq.format_style, "quiz")
        self.assertEqual(mcq.mode, "practice")
        self.assertTrue(mcq.use_rag)
        self.assertTrue(mcq.grounding_required)

        source_only = self._response_plan_for("Explain photosynthesis from my notes only")
        self.assertTrue(source_only.grounding_required)
        self.assertTrue(source_only.use_rag)

        simpler_follow_up = self._response_plan_for("I did not understand", has_history=True)
        self.assertEqual(simpler_follow_up.tone, "simple")
        self.assertEqual(simpler_follow_up.student_level, "beginner")
        self.assertIn("easier words", simpler_follow_up.special_instruction)

        code = self._response_plan_for("Explain this Python code line by line")
        self.assertEqual(code.mode, "coding_help")
        self.assertTrue(code.include_code)
        self.assertIn(code.format_style, {"numbered_steps", "code"})

        image = self._response_plan_for(
            "Solve this image question",
            attachments=[{"name": "question.png", "mime_type": "image/png"}],
        )
        self.assertEqual(image.mode, "upload_explanation")
        self.assertEqual(image.format_style, "numbered_steps")
        self.assertTrue(image.grounding_required)

        numerical = self._response_plan_for("Calculate density if mass is 20 g and volume is 5 cm3")
        self.assertEqual(numerical.format_style, "numbered_steps")
        self.assertTrue(numerical.include_formula)

        answer_only = self._response_plan_for("Only final value")
        self.assertEqual(answer_only.answer_length, "one_line")
        self.assertFalse(answer_only.ask_follow_up)

        hinglish = self._response_plan_for("Explain photosynthesis in Hinglish")
        self.assertIn("Hinglish", hinglish.special_instruction)

    def test_response_planner_handles_common_student_edge_cases(self):
        compare = self._response_plan_for("Compare mitosis and meiosis")
        self.assertEqual(compare.format_style, "table")

        no_table = self._response_plan_for("Compare mitosis and meiosis, no table")
        self.assertEqual(no_table.format_style, "bullets")

        derivation = self._response_plan_for("Derive the formula for kinetic energy")
        self.assertEqual(derivation.format_style, "derivation")
        self.assertTrue(derivation.include_formula)

        formula_only = self._response_plan_for("No derivation, just formula")
        self.assertEqual(formula_only.answer_length, "short")
        self.assertEqual(formula_only.format_style, "plain")
        self.assertTrue(formula_only.include_formula)
        self.assertFalse(formula_only.include_examples)

        one_line = self._response_plan_for("Newton's first law one line")
        self.assertEqual(one_line.answer_length, "one_line")

        hint = self._response_plan_for("Give hint only")
        self.assertEqual(hint.answer_length, "short")
        self.assertIn("hint only", hint.special_instruction)

        formula_list = self._response_plan_for("Give formula list for motion")
        self.assertTrue(formula_list.include_formula)
        self.assertEqual(formula_list.mode, "revision")

        selected_revision = self._response_plan_for(
            "Give revision notes",
            scope={"chapter": "Hydrocarbons", "topic": "Alkanes"},
        )
        self.assertTrue(selected_revision.use_rag)
        self.assertTrue(selected_revision.grounding_required)

    def test_response_planner_instruction_exposes_strict_contract(self):
        plan = ResponsePlannerOutput(
            answer_length="short",
            format_style="plain",
            include_examples=False,
            ask_follow_up=False,
            special_instruction="Return definition only.",
        )
        instruction = build_response_plan_instruction(plan)
        self.assertIn("answer_length: short", instruction)
        self.assertIn("format_style: plain", instruction)
        self.assertIn("Do not add examples", instruction)

    def test_scenario_bank_routes_social_closure_to_conversation_responder(self):
        message = "Okay thanks, you are the best and I understood the concept easily"
        profile = build_scenario_intent_profile(message, has_history=True)
        self.assertEqual(profile.primary_intent, "social_closure")
        self.assertEqual(profile.dialogue_act, "gratitude_acknowledgement")
        self.assertFalse(profile.requires_tutor_answer)
        self.assertEqual(profile.expected_route, "conversation_responder")
        self.assertGreaterEqual(profile.confidence, 0.85)

        reply = build_conversation_response(profile)
        self.assertIsNotNone(reply)
        self.assertIn("welcome", reply.lower())

        query = understand_query(message, has_history=True)
        self.assertEqual(query.intent, "conversation")
        self.assertEqual(query.answer_format, "conversation")
        self.assertTrue(query.is_conversational)
        self.assertFalse(query.is_follow_up)
        self.assertFalse(query.needs_retrieval)
        self.assertFalse(query.needs_quality_review)
        self.assertEqual(query.reasoning_mode, "conversation")
        self.assertEqual(query.scenario_profile["primary_intent"], "social_closure")

        orchestration = build_orchestration_plan(query, message)
        self.assertEqual(orchestration["tools"], [])
        self.assertFalse(orchestration["socratic"])

        decision = build_lead_coach_decision(
            query=query,
            answer_format={"id": "conversation", "label": "Conversation"},
            orchestration_plan=orchestration,
            retrieval_policy=query.retrieval_policy,
            strict_grounding=False,
            material_supported=True,
        )
        self.assertIn("conversation_responder", decision.agent_sequence)
        self.assertNotIn("tutor_model", decision.agent_sequence)
        self.assertNotIn("answer_reviewer", decision.agent_sequence)
        self.assertNotIn("quality_verifier", decision.agent_sequence)

        response_plan = build_response_plan(
            question=message,
            query=query,
            answer_format={"id": "conversation", "label": "Conversation"},
            retrieval_policy=query.retrieval_policy,
            conversation_context={"is_follow_up": query.is_follow_up},
        )
        self.assertEqual(response_plan.answer_length, "short")
        self.assertEqual(response_plan.format_style, "plain")
        self.assertFalse(response_plan.ask_follow_up)

    def test_scenario_bank_preserves_real_follow_up_after_thanks(self):
        message = "Thanks but explain it again"
        profile = build_scenario_intent_profile(message, has_history=True)
        self.assertTrue(profile.requires_tutor_answer)
        self.assertEqual(profile.primary_intent, "clarification")

        query = understand_query(message, has_history=True)
        self.assertEqual(query.intent, "clarification")
        self.assertEqual(query.answer_format, "stuck")
        self.assertFalse(query.is_conversational)
        self.assertTrue(query.is_follow_up)
        self.assertTrue(query.needs_quality_review)

    def test_scenario_bank_protects_conversation_route_from_llm_override(self):
        message = "Okay thanks, you are the best and I understood the concept easily"
        calls = []

        def bad_classifier(_messages):
            calls.append("called")
            return '{"intent":"concept","answer_format":"concept","is_follow_up":true,"retrieval_policy":"none"}'

        query = resolve_hybrid_query(message, has_history=True, classifier=bad_classifier)
        self.assertEqual(query.intent, "conversation")
        self.assertTrue(query.is_conversational)
        self.assertEqual(calls, [])

    def test_intent_scenario_bank_retrieves_learning_and_social_counterexamples(self):
        social = rank_intent_scenarios("I understood everything thanks", limit=3)
        self.assertTrue(social)
        self.assertFalse(social[0].scenario.requires_tutor_answer)

        learning = rank_intent_scenarios("Thanks but give me one more example", limit=3)
        self.assertTrue(learning)
        self.assertTrue(learning[0].scenario.requires_tutor_answer)

    def test_stream_social_closure_bypasses_tutor_after_learning_turn(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        uid = "student_social_stream"
        session_id = f"coach-{uid}-conv"
        coach = AICoachProfile(coach_id="coach_social_stream", user_id=uid, coach_name="Aria")
        db.add(coach)
        db.add_all([
            AICoachInteraction(
                coach_id=coach.coach_id,
                user_id=uid,
                role="user",
                message="What is photosynthesis?",
                intent="definition",
                mode="coach",
                metadata_json={"session_id": session_id},
            ),
            AICoachInteraction(
                coach_id=coach.coach_id,
                user_id=uid,
                role="assistant",
                message="Photosynthesis is the process by which plants make food.",
                intent="definition",
                mode="coach",
                metadata_json={"session_id": session_id},
            ),
        ])
        db.commit()

        request = SimpleNamespace(
            user_id=uid,
            session_id=session_id,
            question="Okay thanks, you are the best and I understood the concept easily",
            intent="study_advice",
            mode="coach",
            attachments=[],
        )
        completed = {}
        stream = coach_agent_stream(request, db=db)
        try:
            for frame in stream:
                event = parse_semantic_event(frame)
                if event.get("event") == "answer.completed":
                    completed = event
        finally:
            stream.close()

        self.assertTrue(completed)
        answer = completed["answer"]
        metadata = completed["metadata"]
        self.assertIn("welcome", answer.lower())
        self.assertNotIn("photosynthesis is the process", answer.lower())
        self.assertEqual(metadata["intent"], "conversation")
        self.assertEqual(metadata["query"]["scenario_profile"]["primary_intent"], "social_closure")
        sequence = metadata["lead_orchestrator"]["agent_sequence"]
        self.assertIn("conversation_responder", sequence)
        self.assertNotIn("tutor_model", sequence)
        self.assertNotIn("answer_reviewer", sequence)
        db.close()
        engine.dispose()

    def test_stream_disconnect_persists_user_turn(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        uid = "student_disconnect"
        session_id = f"coach-{uid}-conv"
        request = SimpleNamespace(
            user_id=uid,
            session_id=session_id,
            question="Okay thanks, that helped me a lot today",
            intent="study_advice",
            mode="coach",
            attachments=[],
        )

        stream = coach_agent_stream(request, db=db)
        saw_drafting = False
        for frame in stream:
            event = parse_semantic_event(frame)
            if event.get("stage") == "drafting":
                saw_drafting = True
                break
        # Simulate the client disconnecting mid-stream.
        stream.close()

        self.assertTrue(saw_drafting)
        rows = (
            db.query(AICoachInteraction)
            .filter(AICoachInteraction.user_id == uid)
            .all()
        )
        self.assertTrue(rows, "interrupted stream must still persist the student turn")
        user_rows = [row for row in rows if row.role == "user"]
        self.assertEqual(len(user_rows), 1)
        self.assertEqual(user_rows[0].message, request.question)
        self.assertTrue(user_rows[0].metadata_json.get("stream_interrupted"))
        db.close()
        engine.dispose()

    def test_llm_judge_parses_scores_and_respects_sample_gate(self):
        from Logic.coach import llm_judge

        class JudgeGateway:
            def complete(self, *args, **kwargs):
                return (
                    '{"factual_accuracy": 9, "pedagogy": 7, "grounding_fidelity": 10,'
                    ' "verdict": "pass", "main_issue": ""}'
                )

        report = llm_judge.judge_coach_answer(
            JudgeGateway(),
            question="What is a mole?",
            answer="A mole is the SI unit for amount of substance.",
            retrieved_context="",
            intent="definition",
        )
        self.assertIsNotNone(report)
        self.assertEqual(report["factual_accuracy"], 9.0)
        self.assertEqual(report["verdict"], "pass")
        self.assertAlmostEqual(report["overall"], 8.67, places=2)

        with patch.dict("os.environ", {"COACH_JUDGE_SAMPLE_RATE": "0"}):
            self.assertFalse(llm_judge.should_judge_turn())
        with patch.dict("os.environ", {"COACH_JUDGE_SAMPLE_RATE": "1.0"}):
            self.assertTrue(llm_judge.should_judge_turn())

        class BrokenGateway:
            def complete(self, *args, **kwargs):
                raise RuntimeError("provider down")

        self.assertIsNone(
            llm_judge.judge_coach_answer(
                BrokenGateway(), question="q", answer="a"
            )
        )

    def test_session_summary_includes_replay_contract(self):
        test = SimpleNamespace(
            id=42,
            total_questions=2,
            score=1,
            time_spent_seconds=75,
            date=date(2026, 6, 4),
            started_at=datetime(2026, 6, 4, 8, 0),
            completed_at=datetime(2026, 6, 4, 8, 2),
            confidence_before=40,
            confidence_after=65,
            topic="alkanes",
            xp_earned=10,
            focus_score=88,
            accuracy_rate=50,
            session_type="exam",
            response_latency_ms=320,
            hint_count=1,
            retry_count=2,
            details=SimpleNamespace(
                replay_data={
                    "questions": [
                        {
                            "question": "What is methane?",
                            "correct_answer": "CH4",
                            "user_answer": "CH4",
                        }
                    ]
                }
            ),
        )

        payload = format_test_session(test)

        self.assertTrue(payload["has_replay"])
        self.assertEqual(payload["replay_question_count"], 1)
        self.assertEqual(payload["replay_data"]["questions"][0]["correct_answer"], "CH4")

    def test_session_id_ownership_matches_frontend_patterns(self):
        uid = "student123"
        self.assertTrue(session_id_belongs_to_user(f"coach-{uid}-conv_a", uid))
        self.assertTrue(session_id_belongs_to_user(f"revision-{uid}-alkanes-summary", uid))
        self.assertTrue(session_id_belongs_to_user(f"exam-{uid}-alkanes-123", uid))
        self.assertTrue(session_id_belongs_to_user(f"probable-{uid}-alkanes-123", uid))
        self.assertFalse(session_id_belongs_to_user("coach-other-conv_a", uid))

    def test_legacy_study_session_guard_blocks_cross_user_sessions(self):
        uid = "student123"

        self.assertEqual(
            require_owned_study_session(f"revision-{uid}-alkanes-summary", {"uid": uid}),
            uid,
        )
        self.assertEqual(
            require_owned_study_session("exam-other-alkanes-123", {"uid": uid, "admin": True}),
            uid,
        )
        with self.assertRaises(Exception) as raised:
            require_owned_study_session("exam-other-alkanes-123", {"uid": uid})
        self.assertEqual(getattr(raised.exception, "status_code", None), 403)

    def test_legacy_study_requests_validate_student_payload_size(self):
        with self.assertRaises(Exception):
            SectionAIRequest(question="", section_id="alkanes", session_id="revision-student123-alkanes-summary")
        with self.assertRaises(Exception):
            GenerateMCQRequest(topic="alkanes", session_id="exam-student123-alkanes-1", count=20)

    def test_stream_wrapper_returns_controlled_error_event_on_unhandled_failure(self):
        request = SimpleNamespace(user_id="student123", session_id="coach-student123-conv_a", question="Hello")
        with patch("Logic.agents.coach_agent._coach_agent_stream_impl", side_effect=RuntimeError("boom")):
            frames = list(coach_agent_stream(request, db=None))

        parsed = [parse_semantic_event(frame) for frame in frames]
        events = [event.get("event") for event in parsed if event]
        self.assertIn("turn.error", events)
        self.assertIn("answer.completed", events)
        completed = next(event for event in parsed if event.get("event") == "answer.completed")
        self.assertTrue(str(completed.get("answer") or "").strip())
        self.assertEqual(frames[-1], "data: [DONE]\n\n")

    def test_persisted_coach_interactions_serialize_as_conversation(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        uid = "student123"
        coach = AICoachProfile(coach_id="coach_conv", user_id=uid)
        db.add(coach)
        db.add_all([
            AICoachInteraction(
                coach_id=coach.coach_id,
                user_id=uid,
                role="user",
                message="What is matter?",
                intent="definition",
                mode="coach",
                metadata_json={"session_id": f"coach-{uid}-conv_a"},
            ),
            AICoachInteraction(
                coach_id=coach.coach_id,
                user_id=uid,
                role="assistant",
                message="Matter has mass and occupies space.",
                intent="definition",
                mode="coach",
                metadata_json={
                    "session_id": f"coach-{uid}-conv_a",
                    "answer_blocks": [{"kind": "explanation", "title": "Core idea", "content": "Matter has mass."}],
                    "sources": {"grounded": False, "indicator": "General tutor reasoning", "citations": []},
                    "conversation_pinned": True,
                },
            ),
        ])
        db.commit()

        rows = _conversation_rows_for_user(db, coach, uid)
        grouped = _group_conversation_rows(rows)
        payload = _serialize_coach_conversation(f"coach-{uid}-conv_a", grouped[f"coach-{uid}-conv_a"])

        self.assertEqual(payload["id"], "conv_a")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][1]["role"], "coach")
        self.assertEqual(payload["messages"][1]["sources"]["indicator"], "General tutor reasoning")
        self.assertTrue(payload["pinned"])

    def test_offline_routing_suite(self):
        report = run_offline_coach_evaluation()
        self.assertTrue(report["passed"], report)
        self.assertGreaterEqual(len(SCENARIOS), 150)

    def test_growth_loop_scores_stable_and_repair_turns(self):
        stable = evaluate_turn_growth({
            "quality": {"score": 0.9},
            "retrieval": {"gate": {"grounding_status": "grounded"}},
            "repair": {"final": {"action": "deliver", "repair_applied": False}},
            "model_calls": [{"status": "success"}],
            "tool_gateway": [{"status": "success"}],
            "student_memory_update": {"support_style": "steady_guided"},
        })
        self.assertEqual(stable["readiness"], "excellent")
        self.assertIn("stable baseline", stable["recommendations"][0])

        repair = evaluate_turn_growth({
            "quality": {"score": 0.62},
            "retrieval": {"gate": {"grounding_status": "missing_required_source"}},
            "repair": {"final": {"action": "deliver", "repair_applied": True}},
            "model_calls": [{"status": "error"}, {"status": "success"}],
            "tool_gateway": [{"status": "error"}],
            "student_memory_update": {"support_style": "simplify_and_check"},
        })
        self.assertIn(repair["readiness"], {"watch", "repair"})
        self.assertGreaterEqual(len(repair["recommendations"]), 3)
        self.assertEqual(repair["signals"]["model_errors"], 1)
        self.assertEqual(repair["signals"]["tool_errors"], 1)

    def test_conversational_thanks_does_not_reopen_lesson(self):
        query = understand_query("Thank you", has_history=True)
        self.assertEqual(query.intent, "conversation")
        self.assertFalse(query.needs_retrieval)
        self.assertFalse(query.requires_grounding)

    def test_required_grounding_only_when_requested(self):
        regular = understand_query("Explain alkanes simply")
        grounded = understand_query("Explain alkanes from my notes only")
        self.assertEqual(regular.retrieval_policy, "none")
        self.assertEqual(grounded.retrieval_policy, "required")

    def test_quality_scorer_handles_conversation_and_grounding_guard(self):
        casual = score_coach_answer(
            question="Thanks",
            answer="You are welcome. Send the next doubt when you are ready.",
            strict_grounding=False,
            intent="conversation",
            answer_format="conversation",
        )
        unsupported = score_coach_answer(
            question="Explain alkanes from my notes",
            answer="Alkanes are saturated hydrocarbons.",
            strict_grounding=True,
            intent="definition",
            answer_format="definition",
        )
        self.assertTrue(casual.passed, casual.to_dict())
        self.assertGreaterEqual(unsupported.hallucination_risk, 0.65)

    def test_answer_repair_decision_replaces_only_when_required(self):
        unsupported = score_coach_answer(
            question="Explain alkanes from my notes",
            answer="Alkanes are saturated hydrocarbons.",
            strict_grounding=True,
            intent="definition",
            answer_format="definition",
        )
        missing_gate = evaluate_retrieval_gate(
            policy="required",
            retrieved_material={},
            strict_grounding=True,
            material_supported=False,
        )
        replace = decide_answer_repair(
            quality=unsupported,
            verification={"passed": False, "issues": ["unsupported_without_retrieval"]},
            retrieval_gate=missing_gate,
            strict_grounding=True,
        )
        self.assertEqual(replace.action, "replace_with_material_not_found")
        self.assertEqual(replace.required_action, "request_or_select_study_material")

        repaired_quality = score_coach_answer(
            question="Explain alkanes from my notes",
            answer="I could not find this in your study material. Please upload or select the correct chapter/data.",
            strict_grounding=True,
            intent="definition",
            answer_format="definition",
        )
        applied = mark_repair_applied(original=replace, quality=repaired_quality)
        self.assertTrue(applied.repair_applied)
        self.assertEqual(applied.action, "deliver")

        good = score_coach_answer(
            question="Define matter",
            answer="Matter is anything that has mass and occupies space.",
            strict_grounding=False,
            intent="definition",
            answer_format="definition",
        )
        deliver = decide_answer_repair(quality=good, verification={"passed": True}, strict_grounding=False)
        self.assertEqual(deliver.action, "deliver")

    def test_adaptive_answer_blocks_and_semantic_event(self):
        answer = "Core Idea:\nMatter has mass.\n\nExample:\n- Water occupies space.\n\nQuick Check:\n- Does air occupy space?"
        blocks = build_adaptive_answer_blocks(answer)
        event = parse_semantic_event(semantic_event("answer.completed", turn_id="turn_test", answer=answer, blocks=blocks))
        self.assertEqual(event["event"], "answer.completed")
        self.assertTrue(any(block["kind"] == "example" for block in blocks))
        self.assertTrue(any(block["kind"] == "checkpoint" for block in blocks))

    def test_agent_state_tracks_safe_shared_turn_contract(self):
        request = SimpleNamespace(
            raw_message="Explain alkanes from my notes only",
            original_message="Explain alkanes from my notes only",
        )
        query = understand_query("Explain alkanes from my notes only")
        state = build_initial_agent_state(
            request=request,
            turn_id="turn_state",
            user_id="student_state",
            session_id="coach-student_state-conv",
            question="Explain alkanes from my notes only",
            mode="coach",
            query=query,
            answer_format={"id": "definition"},
            adaptive_context={
                "student_state": {"emotional_state": "confused", "level": "beginner"},
                "adaptive_strategy": {"answer_style": "simple"},
                "learning_context": {},
            },
        )
        state.apply_conversation_context({
            "is_follow_up": True,
            "last_student_question": "What are hydrocarbons?",
            "recent_thread": "Tutor: " + ("Alkanes are saturated hydrocarbons. " * 80),
            "durable_memory": "Student prefers simple examples.",
        })
        state.apply_scope({"topic": "alkanes", "section_id": "alkanes"})
        state.apply_retrieval(
            {
                "context": "Alkanes are saturated hydrocarbons. " * 100,
                "section_id": "alkanes",
                "source": "markdown",
                "paragraphs_found": 4,
                "keywords_used": ["alkane", "formula"],
            },
            "required",
            True,
        )
        state.apply_attachments(SimpleNamespace(
            image_count=1,
            document_count=1,
            warnings=["Image extraction is model-assisted"],
            has_material=True,
            vision_summary="Visible formula CH4 and handwritten note.",
            multimodal={
                "confidence": 0.84,
                "math_lines": ["F = ma"],
                "formulas": [{"raw": "CH4"}],
                "diagram_specs": [{"diagram_type": "chemistry_structure"}],
            },
        ))
        state.apply_tools(
            {"formula_checker": {"used": True, "formulas": ["CH4"], "directive": "Keep chemistry exact."}},
            ["formula_checker", "answer_verifier"],
        )
        state.apply_answer(
            draft="Alkanes are saturated hydrocarbons.",
            final_answer="Alkanes are saturated hydrocarbons with single bonds.",
            next_best_action="Try one easy alkane question.",
        )
        quality = score_coach_answer(
            question="Explain alkanes from my notes only",
            answer="Alkanes are saturated hydrocarbons with single bonds.",
            retrieved_context="Alkanes are saturated hydrocarbons with single covalent bonds.",
            strict_grounding=True,
            intent="definition",
            answer_format="definition",
        )
        state.apply_quality(quality, {"passed": True})
        trace = state.to_trace_dict()

        self.assertEqual(trace["detected_intent"], "concept")
        self.assertEqual(trace["retrieval_policy"], "required")
        self.assertEqual(trace["grounding_status"], "grounded")
        self.assertEqual(trace["detected_topic"], "alkanes")
        self.assertEqual(trace["attachment_summary"]["multimodal"]["formulas"], 1)
        self.assertIn("retrieved_context_excerpt", trace)
        self.assertNotIn("retrieved_context", trace)
        self.assertNotIn("final_answer", trace)
        self.assertLessEqual(len(trace["conversation_history"]["recent_thread_excerpt"]), 903)
        self.assertTrue(trace["agent_messages"])

    def test_agent_contract_normalizes_messages_and_handoffs(self):
        self.assertEqual(normalize_agent_role("Lead Coach Orchestrator"), "lead_coach_orchestrator")
        self.assertEqual(normalize_agent_role("???"), "unknown_agent")
        self.assertEqual(normalize_message_type("Verification Result"), "verification_result")
        self.assertEqual(normalize_message_type("made up event"), "status")
        self.assertEqual(normalize_handoff_status("needs-review"), "needs_review")

        request = SimpleNamespace(raw_message="Explain matter", original_message="Explain matter")
        state = build_initial_agent_state(
            request=request,
            turn_id="turn_contract",
            user_id="student_contract",
            session_id="coach-student_contract-conv",
            question="Explain matter",
            mode="coach",
            query=understand_query("Explain matter"),
            answer_format={"id": "definition"},
            adaptive_context={"student_state": {}, "adaptive_strategy": {}, "learning_context": {}},
        )
        state.add_message(
            sender_agent="Lead Coach Orchestrator",
            receiver_agent="not a real agent",
            message_type="Strange Custom Event",
            task="x" * 700,
            evidence=["bad"],
            confidence=3.7,
            result="bad",
        )
        message = state.agent_messages[-1].to_dict()
        self.assertEqual(message["sender_agent"], "lead_coach_orchestrator")
        self.assertEqual(message["receiver_agent"], "unknown_agent")
        self.assertEqual(message["message_type"], "status")
        self.assertEqual(message["confidence"], 1.0)
        self.assertEqual(message["evidence"], {})
        self.assertEqual(message["result"], {})
        self.assertLessEqual(len(message["task"]), 503)

        handoff = AgentHandoff(
            from_agent="Lead Coach Orchestrator",
            to_agent="Tutor Model",
            reason="Send draft task",
            status="DONE",
            input_payload="bad",
            result_payload={"accepted": True},
            confidence=-4,
        ).to_dict()
        self.assertEqual(handoff["from_agent"], "lead_coach_orchestrator")
        self.assertEqual(handoff["to_agent"], "tutor_model")
        self.assertEqual(handoff["status"], "requested")
        self.assertEqual(handoff["input_payload"], {})
        self.assertEqual(handoff["result_payload"], {"accepted": True})
        self.assertEqual(handoff["confidence"], 0.0)

    def test_agent_runtime_store_persists_controlled_run_history(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        request = SimpleNamespace(raw_message="Explain matter", original_message="Explain matter")
        query = understand_query("Explain matter")
        state = build_initial_agent_state(
            request=request,
            turn_id="turn_runtime",
            user_id="student_runtime",
            session_id="coach-student_runtime-conv",
            question="Explain matter",
            mode="coach",
            query=query,
            answer_format={"id": "definition"},
            adaptive_context={"student_state": {}, "adaptive_strategy": {}, "learning_context": {}},
        )
        state.add_message(
            sender_agent="intent_profiler",
            receiver_agent="lead_coach_orchestrator",
            message_type="profile_result",
            task="Classify the student request.",
            confidence=0.82,
            result={"intent": query.intent, "retrieval_policy": query.retrieval_policy},
        )

        run = start_agent_run(db, state=state, metadata={"endpoint": "test"})
        self.assertIsNotNone(run)
        record_agent_step(
            db,
            run_id=state.turn_id,
            step_name="intent_profiled",
            agent_name="intent_profiler",
            output_data={"intent": query.intent},
        )
        record_agent_handoff(
            db,
            run_id=state.turn_id,
            from_agent="lead_coach_orchestrator",
            to_agent="intent_profiler",
            reason="Classify the request.",
            result_data={"intent": query.intent},
        )
        record_agent_tool_calls(
            db,
            run_id=state.turn_id,
            tools=[
                {"name": "knowledge_search", "paragraphs_found": 2, "source": "notes"},
                {"name": "answer_verifier", "result": {"passed": True, "issues": []}},
            ],
            agent_name="tool_gateway",
        )
        state.apply_answer(
            draft="Matter has mass and occupies space.",
            final_answer="Matter is anything that has mass and occupies space.",
        )
        quality = score_coach_answer(
            question="Explain matter",
            answer=state.final_answer,
            strict_grounding=False,
            intent=query.intent,
            answer_format="definition",
        )
        state.apply_quality(quality, {"passed": True})
        stored_messages = record_agent_messages(db, run_id=state.turn_id, messages=state.agent_messages)
        complete_agent_run(db, state=state, status="success", latency_ms=321)

        summary = runtime_summary(db, state.turn_id)
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["messages"], stored_messages)
        self.assertEqual(summary["tool_calls"], 2)
        self.assertEqual(summary["handoffs"], 1)
        self.assertEqual(db.query(AgentRuntimeRun).count(), 1)
        self.assertEqual(db.query(AgentRuntimeStep).first().step_order, 1)
        self.assertEqual(db.query(AgentRuntimeMessage).count(), stored_messages)
        self.assertEqual(db.query(AgentRuntimeHandoff).count(), 1)
        search_call = db.query(AgentRuntimeToolCall).filter(AgentRuntimeToolCall.tool_name == "knowledge_search").one()
        self.assertEqual(search_call.output_json["paragraphs_found"], 2)
        completed = db.query(AgentRuntimeRun).filter(AgentRuntimeRun.run_id == state.turn_id).one()
        self.assertIn("final_answer", completed.state_json)
        self.assertEqual(completed.latency_ms, 321)
        db.close()

    def test_router_uses_fallback_and_records_route(self):
        with patch.dict("os.environ", {"GROQ_FALLBACK_MODEL": "test/fallback"}, clear=False):
            router = LLMRouter()
            router._groq = FakeClient()
            router.begin_turn("turn_router")
            answer = router.complete("tutor", [{"role": "user", "content": "Explain matter"}])
            records = router.records()
        self.assertEqual(answer, "Recovered tutor response")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[1]["status"], "success")
        self.assertTrue(records[1]["fallback"])
        self.assertGreater(records[1]["estimated_input_tokens"], 0)
        self.assertGreater(records[1]["estimated_output_tokens"], 0)
        self.assertGreaterEqual(records[1]["estimated_cost_usd"], 0)

    def test_router_fails_over_across_providers(self):
        with patch.dict(
            "os.environ",
            {
                "COACH_PROVIDER_ORDER": "groq,openrouter",
                "COACH_LLM_MAX_ATTEMPTS": "2",
                "COACH_BUDGET_ROUTING": "false",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_TUTOR_MODEL": "openrouter/test-tutor",
            },
            clear=False,
        ):
            router = LLMRouter()
            router._provider_clients["groq"] = ProviderClient(AlwaysFailCompletions())
            router._provider_clients["openrouter"] = ProviderClient(StaticCompletions("OpenRouter recovered"))
            router.begin_turn("turn_provider_failover")

            answer = router.complete("tutor", [{"role": "user", "content": "Explain force"}], max_tokens=120)
            records = router.records()

        self.assertEqual(answer, "OpenRouter recovered")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["provider"], "groq")
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[1]["provider"], "openrouter")
        self.assertEqual(records[1]["status"], "success")
        self.assertTrue(records[1]["fallback"])

    def test_model_gateway_enriches_fallback_records_with_agent_task_metadata(self):
        with patch.dict(
            "os.environ",
            {
                "COACH_PROVIDER_ORDER": "groq,openrouter",
                "COACH_LLM_MAX_ATTEMPTS": "2",
                "COACH_BUDGET_ROUTING": "false",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_TUTOR_MODEL": "openrouter/test-tutor",
            },
            clear=False,
        ):
            router = LLMRouter()
            router._provider_clients["groq"] = ProviderClient(AlwaysFailCompletions())
            router._provider_clients["openrouter"] = ProviderClient(StaticCompletions("Gateway recovered"))
            gateway = ModelGateway(router)
            gateway.begin_turn("turn_gateway")

            answer = gateway.complete(
                "tutor",
                [{"role": "user", "content": "Explain inertia"}],
                agent_name="Tutor Model",
                task="Draft a student-friendly explanation.",
                student_visible=False,
                max_tokens=120,
            )
            records = gateway.records()

        self.assertEqual(answer, "Gateway recovered")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[1]["status"], "success")
        self.assertEqual(records[0]["gateway_agent"], "tutor_model")
        self.assertEqual(records[1]["gateway_agent"], "tutor_model")
        self.assertEqual(records[0]["gateway_task"], "Draft a student-friendly explanation.")
        self.assertEqual(records[0]["gateway_call_id"], records[1]["gateway_call_id"])
        self.assertFalse(records[1]["student_visible"])

    def test_router_stream_fails_over_across_providers(self):
        stream = StaticStreamCompletions(["Better", " answer"])
        with patch.dict(
            "os.environ",
            {
                "COACH_PROVIDER_ORDER": "groq,openrouter",
                "COACH_LLM_MAX_ATTEMPTS": "2",
                "COACH_BUDGET_ROUTING": "false",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_REVIEW_MODEL": "openrouter/reviewer",
            },
            clear=False,
        ):
            router = LLMRouter()
            router._provider_clients["groq"] = ProviderClient(AlwaysFailCompletions())
            router._provider_clients["openrouter"] = ProviderClient(stream)
            router.begin_turn("turn_stream_failover")

            chunks = list(router.stream("reviewer", [{"role": "user", "content": "Review this"}], max_tokens=120))
            records = router.records()

        text = "".join(chunk.choices[0].delta.content for chunk in chunks)
        self.assertEqual(text, "Better answer")
        self.assertEqual(records[0]["provider"], "groq")
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[1]["provider"], "openrouter")
        self.assertEqual(records[1]["mode"], "stream")
        self.assertEqual(records[1]["status"], "success")
        self.assertGreater(records[1]["estimated_output_tokens"], 0)

    def test_retired_groq_models_resolve_to_a_live_fallback(self):
        router = LLMRouter()
        retired = "meta-llama/llama-4-scout-17b-16e-instruct"
        deployed_settings = SimpleNamespace(fast_model=retired, vision_model=retired)
        with patch("Logic.coach.llm_router.coach_settings", deployed_settings):
            self.assertEqual(router.model_for("profiler", "fast"), "openai/gpt-oss-120b")
            self.assertEqual(router.model_for("vision", "fast"), "")

    def test_empty_model_answer_has_grounded_or_honest_fallback(self):
        grounded = _safe_tutor_fallback(
            question="What is matter?",
            retrieved_material={"context": "Matter has mass and occupies space."},
            strict_grounding=True,
        )
        open_tutor = _safe_tutor_fallback(
            question="Explain an unavailable concept",
            retrieved_material=None,
            strict_grounding=False,
        )

        self.assertIn("Verified material", grounded)
        self.assertIn("Matter has mass", grounded)
        self.assertTrue(open_tutor.strip())
        self.assertIn("please try again", open_tutor.lower())

    def test_router_prefers_lowest_cost_when_configured(self):
        cheap = StaticCompletions("Cheap route selected")
        with patch.dict(
            "os.environ",
            {
                "COACH_PROVIDER_ORDER": "groq,openrouter",
                "COACH_LLM_MAX_ATTEMPTS": "2",
                "COACH_BUDGET_ROUTING": "true",
                "COACH_ROUTE_PREFERENCE": "lowest_cost",
                "COACH_DAILY_BUDGET_USD": "0",
                "COACH_TURN_BUDGET_USD": "0",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_TUTOR_MODEL": "openrouter/cheap",
                "COACH_MODEL_PRICES_PER_1M": (
                    "groq:openai/gpt-oss-120b=10:10;"
                    "openrouter:openrouter/cheap=0.001:0.001"
                ),
            },
            clear=False,
        ):
            router = LLMRouter()
            router._provider_clients["groq"] = ProviderClient(StaticCompletions("Expensive route"))
            router._provider_clients["openrouter"] = ProviderClient(cheap)
            router.begin_turn("turn_lowest_cost")

            answer = router.complete("tutor", [{"role": "user", "content": "Explain work"}], max_tokens=120)
            records = router.records()

        self.assertEqual(answer, "Cheap route selected")
        self.assertEqual(records[0]["provider"], "openrouter")
        self.assertEqual(records[0]["budget_action"], "allowed")
        self.assertEqual(cheap.calls, 1)

    def test_router_skips_over_budget_route_and_uses_cheaper_fallback_provider(self):
        cheap = StaticCompletions("Budget safe answer")
        with patch.dict(
            "os.environ",
            {
                "COACH_PROVIDER_ORDER": "groq,openrouter",
                "COACH_LLM_MAX_ATTEMPTS": "2",
                "COACH_BUDGET_ROUTING": "true",
                "COACH_ROUTE_PREFERENCE": "balanced",
                "COACH_DAILY_BUDGET_USD": "0",
                "COACH_TURN_BUDGET_USD": "0.0001",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_TUTOR_MODEL": "openrouter/cheap",
                "COACH_MODEL_PRICES_PER_1M": (
                    "groq:openai/gpt-oss-120b=10:10;"
                    "openrouter:openrouter/cheap=0.001:0.001"
                ),
            },
            clear=False,
        ):
            router = LLMRouter()
            router._provider_clients["groq"] = ProviderClient(StaticCompletions("Should be skipped"))
            router._provider_clients["openrouter"] = ProviderClient(cheap)
            router.begin_turn("turn_budget_skip")

            answer = router.complete("tutor", [{"role": "user", "content": "Explain pressure"}], max_tokens=120)
            records = router.records()

        self.assertEqual(answer, "Budget safe answer")
        self.assertEqual(records[0]["provider"], "groq")
        self.assertEqual(records[0]["status"], "skipped")
        self.assertEqual(records[0]["budget_action"], "skipped")
        self.assertEqual(records[1]["provider"], "openrouter")
        self.assertEqual(records[1]["status"], "success")
        self.assertEqual(cheap.calls, 1)

    def test_unified_orchestrator_routes_specialists_selectively(self):
        simple = build_orchestration_plan(understand_query("Define matter"), "Define matter")
        chemistry = build_orchestration_plan(understand_query("Explain C2H6"), "Explain C2H6")
        numerical = build_orchestration_plan(understand_query("Calculate 20 / 5"), "Calculate 20 / 5")
        homework = build_orchestration_plan(understand_query("Help me solve this homework"), "Help me solve this homework")
        direct = build_orchestration_plan(understand_query("Help me solve this homework"), "Help me solve this homework", direct_answer=True)
        self.assertEqual(simple["tools"], ["answer_verifier"])
        self.assertIn("formula_checker", chemistry["tools"])
        self.assertIn("calculator", numerical["tools"])
        self.assertIn("socratic_tutor", homework["tools"])
        self.assertNotIn("socratic_tutor", direct["tools"])

    def test_lead_coach_orchestrator_declares_agent_sequence_and_safety_gates(self):
        query = understand_query("Explain alkanes from my notes only")
        plan = build_orchestration_plan(query, "Explain alkanes from my notes only")
        decision = build_lead_coach_decision(
            query=query,
            answer_format={"id": "concept", "label": "Concept Builder"},
            orchestration_plan=plan,
            retrieval_policy=query.retrieval_policy,
            strict_grounding=query.requires_grounding,
            material_supported=False,
            attachment_summary={"has_material": False},
            mastery_profile={"route": "simplify_and_reinforce"},
        )

        self.assertEqual(decision.primary_agent, "lead_coach_orchestrator")
        self.assertIn("context_retriever", decision.agent_sequence)
        self.assertIn("tool_gateway", decision.agent_sequence)
        self.assertIn("quality_verifier", decision.agent_sequence)
        self.assertIn("memory_mastery_engine", decision.agent_sequence)
        self.assertIn("strict_grounding", decision.safety_gates)
        self.assertIn("required_material_missing", decision.safety_gates)
        self.assertIn("answer_verifier", decision.safety_gates)
        self.assertIn("student_friendly_format", decision.safety_gates)

        casual_query = understand_query("thanks")
        casual_plan = build_orchestration_plan(casual_query, "thanks")
        casual_decision = build_lead_coach_decision(
            query=casual_query,
            answer_format={"id": "conversation", "label": "Conversation"},
            orchestration_plan=casual_plan,
            retrieval_policy="none",
            strict_grounding=False,
            material_supported=True,
        )
        self.assertNotIn("context_retriever", casual_decision.agent_sequence)
        self.assertNotIn("quality_verifier", casual_decision.agent_sequence)

    def test_text_attachment_becomes_source_context(self):
        payload = "data:text/plain;base64,TWF0dGVyIGhhcyBtYXNzIGFuZCBvY2N1cGllcyBzcGFjZS4="
        bundle = prepare_attachments([{"name": "notes.txt", "mime_type": "text/plain", "data_url": payload}], "Define matter")
        self.assertIn("Matter has mass", bundle.context)
        self.assertEqual(bundle.document_count, 1)
        self.assertEqual(bundle.citations[0]["source"], "Uploaded material")
        self.assertIn("multimodal", bundle.to_dict())

    def test_attachment_validation_rejects_mime_mismatch(self):
        payload = "data:application/pdf;base64,TWF0dGVyIGhhcyBtYXNzLg=="
        bundle = prepare_attachments([{"name": "fake.txt", "mime_type": "text/plain", "data_url": payload}], "Define matter")
        self.assertFalse(bundle.safe_attachments)
        self.assertTrue(any("does not match" in warning for warning in bundle.warnings))

    def test_active_mastery_profile_simplifies_repeated_confusion(self):
        memory = SimpleNamespace(metadata_json={
            "topic": "alkanes",
            "observations": 3,
            "support_count": 2,
            "average_confidence": 42,
            "last_observed_at": "2026-05-30T00:00:00",
        })
        profile = build_active_mastery_profile([memory], {"topic": "alkanes"})
        self.assertEqual(profile["route"], "simplify_and_reinforce")
        unrelated = build_active_mastery_profile([memory], {"topic": "fractions"})
        self.assertEqual(unrelated["route"], "baseline")

    def test_student_memory_update_combines_mastery_repair_and_retrieval(self):
        mastery_signal = {
            "stored": True,
            "topic": "alkanes",
            "needs_support": True,
            "confidence": 42,
            "quality_score": 0.58,
        }
        repair_report = {
            "initial": {"action": "replace_with_material_not_found"},
            "final": {"action": "deliver", "repair_applied": True},
        }
        retrieval_gate = evaluate_retrieval_gate(
            policy="required",
            retrieved_material={},
            strict_grounding=True,
            material_supported=False,
        )

        update = build_student_memory_update(
            mastery_signal=mastery_signal,
            mastery_profile={"route": "simplify_and_reinforce"},
            repair_report=repair_report,
            retrieval_gate=retrieval_gate,
            recommendation="Revise alkanes with one example.",
        )

        self.assertTrue(update["stored"])
        self.assertEqual(update["support_style"], "simplify_and_check")
        self.assertIn("answer_needed_repair", update["guardrails"])
        self.assertIn("needs_source_selection", update["guardrails"])
        self.assertEqual(update["mastery_route"], "simplify_and_reinforce")

    def test_specialist_calculator_and_formula_checker_are_bounded(self):
        calculation = calculator("Calculate 20 / 5")
        formulas = formula_checker("Explain CH4 and C2H6. Carbon remains a normal word.")
        unsafe = calculator("Run import os")
        self.assertEqual(calculation["result"], 4.0)
        self.assertEqual(formulas["formulas"], ["CH4", "C2H6"])
        self.assertEqual(formulas["formatted"][0]["display"], "CH₄")
        self.assertFalse(unsafe["used"])

    def test_tool_gateway_records_success_and_safe_failure(self):
        gateway = ToolGateway()
        gateway.begin_turn("turn_tools")

        calculation = gateway.run(
            "calculator",
            agent_name="Tool Gateway",
            task="Calculate bounded arithmetic.",
            question="Calculate 20 / 5",
        )
        missing = gateway.run(
            "missing_tool",
            agent_name="Tool Gateway",
            task="Try an unavailable optional tool.",
        )
        records = gateway.records()

        self.assertEqual(calculation["result"], 4.0)
        self.assertTrue(missing["tool_failed"])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["name"], "calculator")
        self.assertEqual(records[0]["agent_name"], "tool_gateway")
        self.assertEqual(records[0]["gateway_task"], "Calculate bounded arithmetic.")
        self.assertEqual(records[0]["status"], "success")
        self.assertEqual(records[1]["status"], "error")
        self.assertTrue(records[1]["result"]["tool_failed"])
        with self.assertRaises(KeyError):
            gateway.run("missing_tool", fail_open=False)

    def test_multimodal_image_extracts_ocr_math_formulas_and_diagram_specs(self):
        one_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        bundle = prepare_attachments(
            [{
                "name": "force-doubt.png",
                "mime_type": "image/png",
                "data_url": f"data:image/png;base64,{one_pixel_png}",
            }],
            "Solve this handwritten force question with a diagram",
            llm_router=FakeVisionRouter(),
        )

        self.assertEqual(bundle.image_count, 1)
        self.assertIn("MULTIMODAL EXTRACTION", bundle.context)
        self.assertIn("F = ma", bundle.context)
        self.assertTrue(bundle.multimodal["math_lines"])
        self.assertIn("CH4", [item["raw"] for item in bundle.multimodal["formulas"]])
        self.assertTrue(bundle.multimodal["diagram_specs"])

        sources = build_source_bundle(None, bundle)
        self.assertTrue(any(item["id"] == "upload-multimodal-extraction" for item in sources["citations"]))

    def test_source_bundle_distinguishes_general_reasoning_from_notes(self):
        general = build_source_bundle(None, None, retrieval_policy="none", material_supported=True)
        self.assertFalse(general["grounded"])
        self.assertEqual(general["answer_basis"], "general_reasoning")
        self.assertEqual(general["indicator"], "General tutor reasoning")

        notes = build_source_bundle(
            {
                "context": "Alkanes are saturated hydrocarbons with single covalent bonds.",
                "section_id": "alkanes",
                "source": "Hydrocarbon notes",
                "scope": {"topic": "Alkanes", "chapter": "Hydrocarbon"},
            },
            None,
            retrieval_policy="required",
            material_supported=True,
        )
        self.assertTrue(notes["grounded"])
        self.assertEqual(notes["answer_basis"], "notes")
        self.assertEqual(notes["indicator"], "Based on your notes")
        self.assertEqual(notes["citations"][0]["kind"], "notes")

    def test_retrieval_gate_controls_required_optional_and_hybrid_sources(self):
        missing = evaluate_retrieval_gate(
            policy="required",
            retrieved_material={"context": "", "error": "No source"},
            strict_grounding=True,
            material_supported=False,
            attachment_summary={"has_material": False},
        )
        self.assertFalse(missing.can_answer)
        self.assertEqual(missing.grounding_status, "missing_required_source")
        self.assertEqual(missing.required_action, "request_or_select_study_material")

        optional = evaluate_retrieval_gate(
            policy="optional",
            retrieved_material=None,
            strict_grounding=False,
            material_supported=False,
        )
        self.assertTrue(optional.can_answer)
        self.assertEqual(optional.grounding_status, "optional_no_source")

        hybrid = evaluate_retrieval_gate(
            policy="required",
            retrieved_material={
                "context": "Matter has mass.",
                "source": "chapter_notes",
                "section_id": "matter",
                "paragraphs_found": 2,
            },
            strict_grounding=True,
            material_supported=True,
            attachment_summary={"has_material": True, "documents": 1, "images": 1},
        )
        self.assertTrue(hybrid.can_answer)
        self.assertEqual(hybrid.grounding_status, "grounded")
        self.assertIn("platform_notes", hybrid.source_mix)
        self.assertIn("uploaded_document", hybrid.source_mix)
        self.assertIn("uploaded_image", hybrid.source_mix)
        self.assertIn("hybrid", hybrid.source_mix)

    def test_multimodal_formula_parser_and_diagram_helper_are_student_safe(self):
        formulas = parse_formula_signals("Can carbon form CH4? Use v^2 = u^2 + 2as. Class remains text.")
        raw_values = [formula.raw for formula in formulas]
        self.assertIn("CH4", raw_values)
        self.assertIn("v^2 = u^2 + 2as", raw_values)
        self.assertFalse(any(value in {"Can", "Class"} for value in raw_values))
        self.assertTrue(any(formula.kind == "math" for formula in formulas))

        diagrams = infer_diagram_specs("Explain photosynthesis with a diagram", "CO2 + H2O -> C6H12O6 + O2")
        self.assertEqual(diagrams[0].diagram_type, "process_flow")
        helper = diagram_helper("Draw a circuit with battery and resistor")
        self.assertEqual(helper["diagram_specs"][0]["diagram_type"], "circuit_diagram")

    def test_mastery_memory_is_deduplicated(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        coach = AICoachProfile(coach_id="coach_test", user_id="student_test")
        db.add(coach)
        db.commit()

        query = understand_query("I am confused about alkanes")
        signal = build_mastery_signal(
            query=query,
            adaptive_context={
                "student_state": {"emotional_state": "confused", "confidence": 35},
                "adaptive_strategy": {"weak_signals": ["confusion"], "answer_style": "simple explanation"},
            },
            scope={"topic": "alkanes"},
            quality={"score": 0.86},
            answer_blocks=[{"kind": "explanation"}],
        )
        persist_mastery_signal(db, coach, signal)
        persist_mastery_signal(db, coach, signal)
        memories = db.query(AICoachMemory).all()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].metadata_json["observations"], 2)
        self.assertEqual(memories[0].metadata_json["support_count"], 2)
        db.close()

    def test_learning_telemetry_uses_measured_session_signals(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        started_at = datetime(2026, 6, 3, 10, 0, 0)
        completed_at = started_at + timedelta(seconds=142)
        db.add(UserProgress(user_id="student_telemetry", total_tests=1, total_questions=1, total_correct=1, xp=10))
        db.add(
            TestHistory(
                user_id="student_telemetry",
                date=date.today(),
                topic="alkanes",
                score=1,
                total_questions=1,
                xp_earned=10,
                time_spent_seconds=142,
                accuracy_rate=100,
                focus_score=84,
                session_type="autonomous_mission",
                started_at=started_at,
                completed_at=completed_at,
                response_latency_ms=780,
                hint_count=1,
                retry_count=2,
                confidence_before=45,
                confidence_after=70,
            )
        )
        db.commit()

        analytics = get_user_analytics(db, "student_telemetry")

        self.assertEqual(analytics["learning_telemetry"]["measured_sessions"], 1)
        self.assertEqual(analytics["learning_telemetry"]["avg_session_seconds"], 142)
        self.assertEqual(analytics["learning_telemetry"]["avg_response_latency_ms"], 780)
        self.assertEqual(analytics["learning_telemetry"]["total_hints_used"], 1)
        self.assertEqual(analytics["learning_telemetry"]["total_retries"], 2)
        self.assertEqual(analytics["learning_telemetry"]["avg_confidence_change"], 25)
        db.close()

    def test_durable_observability_stores_events_and_trace_costs(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        event = AgentEvent(
            version=7,
            timestamp=datetime.utcnow().isoformat(),
            agent_id="coach",
            event_type="metric",
            data={
                "message": "Coach answer scored and persisted.",
                "latency_ms": 640,
                "estimated_cost_usd": 0.00009,
            },
            session_id="session_obs",
        )
        persist_observability_event(db, event)
        events = get_observability_events_since(db, 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["version"], 7)
        self.assertEqual(db.query(ObservabilityEvent).count(), 1)

        metrics = persist_coach_trace(
            db,
            user_id="student_obs",
            session_id="session_obs",
            turn_id="turn_obs",
            observability={
                "latency_ms": 1200,
                "query": {"intent": "definition"},
                "retrieval": {"policy": "none"},
                "plan": {"strategy": "simple"},
                "quality": {"passed": True, "score": 0.91},
                "trace": {
                    "phases_ms": {"received": 10, "answering": 1000},
                    "tools": [{"name": "answer_verifier", "passed": True}],
                },
                "model_calls": [
                    {
                        "role": "tutor",
                        "model": "openai/gpt-oss-120b",
                        "provider": "groq",
                        "mode": "complete",
                        "status": "success",
                        "latency_ms": 520,
                        "attempt": 1,
                        "estimated_input_tokens": 220,
                        "estimated_output_tokens": 95,
                        "estimated_cost_usd": 0.000101,
                    }
                ],
            },
        )

        self.assertEqual(metrics["model_calls"], 1)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(db.query(ModelToolTrace).count(), 3)
        summary = get_observability_summary(db)
        self.assertEqual(summary["model_calls"], 1)
        self.assertEqual(summary["tool_calls"], 1)
        self.assertEqual(summary["turns"], 1)
        self.assertGreater(summary["estimated_cost_usd"], 0)
        db.close()


if __name__ == "__main__":
    unittest.main()

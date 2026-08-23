# Logic/analytics_engine.py

from models import TopicPerformance, TestHistory, UserProgress
from datetime import datetime, timedelta
import math


def _average(values):
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return 0.0
    return sum(usable) / len(usable)


def _session_duration_seconds(session):
    seconds = int(session.time_spent_seconds or 0)
    if seconds > 0:
        return seconds
    started_at = getattr(session, "started_at", None)
    completed_at = getattr(session, "completed_at", None)
    if started_at and completed_at:
        return max(0, int((completed_at - started_at).total_seconds()))
    return 0


def summarize_learning_telemetry(sessions):
    durations = [_session_duration_seconds(session) for session in sessions]
    response_latencies = [
        int(getattr(session, "response_latency_ms", 0) or 0)
        for session in sessions
        if int(getattr(session, "response_latency_ms", 0) or 0) > 0
    ]
    confidence_changes = [
        float(session.confidence_after) - float(session.confidence_before)
        for session in sessions
        if getattr(session, "confidence_before", None) is not None
        and getattr(session, "confidence_after", None) is not None
    ]
    measured_sessions = sum(1 for value in durations if value > 0)

    return {
        "measured_sessions": measured_sessions,
        "avg_session_seconds": round(_average([value for value in durations if value > 0]), 1),
        "avg_response_latency_ms": round(_average(response_latencies), 1),
        "total_hints_used": sum(int(getattr(session, "hint_count", 0) or 0) for session in sessions),
        "total_retries": sum(int(getattr(session, "retry_count", 0) or 0) for session in sessions),
        "avg_hints_per_session": round(_average([int(getattr(session, "hint_count", 0) or 0) for session in sessions]), 2),
        "avg_retries_per_session": round(_average([int(getattr(session, "retry_count", 0) or 0) for session in sessions]), 2),
        "avg_confidence_change": round(_average(confidence_changes), 1),
    }

# =====================================================
# UPDATE PERFORMANCE (ENHANCED)
# =====================================================
def update_topic_performance(db, user_id, topic, correct_answers, total_questions, time_spent=0):
    record = db.query(TopicPerformance).filter(
        TopicPerformance.user_id == user_id,
        TopicPerformance.topic == topic
    ).first()

    if not record:
        record = TopicPerformance(
            user_id=user_id,
            topic=topic,
            attempts=0,
            correct=0,
            weak=False,
            avg_time_per_question=0.0,
            trend_score=0.0
        )
        db.add(record)
        db.commit()
        db.refresh(record)

    # Calculate new accuracy
    old_accuracy = record.accuracy
    record.attempts += total_questions
    record.correct += correct_answers
    new_accuracy = record.accuracy
    
    # Update trend score (improvement vs decline)
    record.trend_score = new_accuracy - old_accuracy
    
    # Update time per question
    if total_questions > 0:
        current_time_per_q = time_spent / total_questions
        if record.avg_time_per_question == 0:
            record.avg_time_per_question = current_time_per_q
        else:
            # Moving average
            record.avg_time_per_question = (record.avg_time_per_question * 0.7) + (current_time_per_q * 0.3)
    
    record.last_practiced = datetime.utcnow()
    record.weak = new_accuracy < 60.0

    db.commit()
    return record

# =====================================================
# COGNITIVE METRICS CALCULATOR
# =====================================================
def calculate_cognitive_metrics(db, user_id):
    """
    Calculates advanced behavioral metrics:
    - Focus Score: Based on consistency of answer times
    - Consistency Index: Frequency of sessions
    - Learning Efficiency: Accuracy / Time spent
    """
    sessions = db.query(TestHistory).filter(TestHistory.user_id == user_id).order_by(TestHistory.date.desc()).limit(10).all()
    
    if not sessions:
        return {
            "focus_score": 0.0,
            "consistency_index": 0.0,
            "learning_efficiency": 0.0,
            "avg_session_seconds": 0.0,
            "avg_response_latency_ms": 0.0,
            "avg_confidence_change": 0.0,
        }
    
    # 1. Focus Score (Avg of session focus scores)
    avg_focus = sum(s.focus_score for s in sessions) / len(sessions)
    
    # 2. Consistency Index (Sessions per week)
    last_10_days = (datetime.now().date() - sessions[-1].date).days if sessions else 0
    consistency = min(100.0, (len(sessions) / max(1, last_10_days)) * 20)
    
    # 3. Learning Efficiency (Accuracy / Time)
    total_acc = sum(s.accuracy_rate for s in sessions) / len(sessions)
    avg_time = sum(_session_duration_seconds(s) for s in sessions) / len(sessions)
    efficiency = min(100.0, (total_acc / max(1, avg_time)) * 100)
    telemetry = summarize_learning_telemetry(sessions)
    
    # Update UserProgress
    user = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
    if user:
        user.focus_score = round(avg_focus, 2)
        user.consistency_index = round(consistency, 2)
        user.learning_efficiency = round(efficiency, 2)
        db.commit()
        
    return {
        "focus_score": round(avg_focus, 2),
        "consistency_index": round(consistency, 2),
        "learning_efficiency": round(efficiency, 2),
        "avg_session_seconds": telemetry["avg_session_seconds"],
        "avg_response_latency_ms": telemetry["avg_response_latency_ms"],
        "avg_confidence_change": telemetry["avg_confidence_change"],
    }

# =====================================================
# 🔥 BLOOMBERG INTELLIGENCE ENGINE
# =====================================================
def get_user_analytics(db, user_id):
    """
    Generates high-density, actionable insights.
    """
    # 1. Fetch Data
    topics = db.query(TopicPerformance).filter(TopicPerformance.user_id == user_id).all()
    sessions = db.query(TestHistory).filter(TestHistory.user_id == user_id).order_by(TestHistory.date.desc()).limit(20).all()
    learning_telemetry = summarize_learning_telemetry(sessions)
    
    # 2. Cognitive Metrics
    cog_metrics = calculate_cognitive_metrics(db, user_id)
    
    # 3. Topic Heatmap & Weak Areas
    heatmap = []
    weak_areas = []
    for t in topics:
        heatmap.append({
            "topic": t.topic,
            "value": round(t.accuracy, 1),
            "attempts": t.attempts,
            "trend": round(t.trend_score, 1)
        })
        if t.weak:
            weak_areas.append({
                "topic": t.topic,
                "accuracy": round(t.accuracy, 1),
                "avg_time": round(t.avg_time_per_question, 1)
            })
            
    # 4. Performance Trends (Last 10 sessions)
    trends = []
    topic_evidence = []
    for s in reversed(sessions[:10]):
        trends.append({
            "date": s.date.isoformat(),
            "accuracy": s.accuracy_rate,
            "xp": s.xp_earned,
            "focus": s.focus_score,
            "duration_seconds": _session_duration_seconds(s),
            "response_latency_ms": int(getattr(s, "response_latency_ms", 0) or 0),
            "hints": int(getattr(s, "hint_count", 0) or 0),
            "retries": int(getattr(s, "retry_count", 0) or 0),
        })
        if str(getattr(s, "topic", "") or "").strip() and int(s.total_questions or 0) > 0:
            topic_evidence.append({
                "topic": s.topic,
                "session_type": str(getattr(s, "session_type", "") or "assessment"),
                "attempts": int(s.total_questions or 0),
                "accuracy": float(s.accuracy_rate or 0),
                "confidence_after": getattr(s, "confidence_after", None),
            })
        
    # 5. AI Insights Generation
    insights = []
    
    # Performance Insights
    if weak_areas:
        insights.append({
            "type": "performance",
            "message": f"Critical weakness detected in {weak_areas[0]['topic']}. Accuracy is below 60%.",
            "severity": "warning",
            "action_label": "REVISE NOW",
            "action_trigger": f"revision_{weak_areas[0]['topic']}"
        })
        
    # Behavioral Insights
    if cog_metrics['focus_score'] < 70:
        insights.append({
            "type": "behavior",
            "message": "Your focus score is declining. Consider shorter, high-intensity study blocks.",
            "severity": "info"
        })
        
    # Predictive Insights
    user = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
    if user and user.xp > 0:
        days_to_level = math.ceil((100 - (user.xp % 100)) / max(1, user.xp / max(1, user.total_tests)))
        insights.append({
            "type": "predictive",
            "message": f"At your current pace, you will reach Level {user.level + 1} in approximately {days_to_level} days.",
            "severity": "success"
        })

    return {
        "summary": {
            "total_topics": len(topics),
            "avg_accuracy": round(sum(t.accuracy for t in topics) / len(topics), 1) if topics else 0.0,
            "total_xp": user.xp if user else 0,
            "streak": user.streak if user else 0
        },
        "topic_heatmap": heatmap,
        "performance_trends": trends,
        "topic_evidence": topic_evidence,
        "weak_areas": weak_areas[:3],
        "insights": insights,
        "cognitive_metrics": cog_metrics,
        "learning_telemetry": learning_telemetry,
        "predictive_stats": {
            "days_to_next_level": days_to_level if 'days_to_level' in locals() else 0,
            "next_milestone": f"Level {user.level + 1}" if user else "N/A"
        }
    }

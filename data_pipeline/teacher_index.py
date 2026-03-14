import json
import os
import pickle
import re
from collections import Counter


PLACEHOLDER_TEACHER_NAMES = {
    "",
    "-",
    "staff",
    "tba",
    "to be announced",
    "instructor tba",
    "no instructor listed",
    "no instructor",
}


def normalize_teacher_name(text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_course_name(text):
    return normalize_teacher_name(text)


def is_placeholder_teacher(name):
    return normalize_teacher_name(name) in PLACEHOLDER_TEACHER_NAMES


def get_session_latest_enrollment(session):
    points = session.get("points") or []
    if not points:
        return 0

    latest_point = points[-1]
    enrollment_total = latest_point.get("enrollment_total")
    if enrollment_total is None:
        return 0
    return int(enrollment_total)


def resolve_course_index(course, data_to_index_dict, course_data_dict, topic_class_map):
    subject = str(course.get("subject") or "").strip()
    catalog_number = str(course.get("catalog_number") or "").strip()

    if not subject or not catalog_number:
        return None

    direct_match = data_to_index_dict.get((subject, catalog_number))
    if direct_match is not None:
        return direct_match

    topic = str(course.get("topic") or "").strip()
    if not topic:
        return None

    topic_key = (subject, catalog_number)
    candidate_course_keys = topic_class_map.get(topic_key, [])
    if not candidate_course_keys:
        return None

    desired_name = normalize_course_name(f"{course.get('descr', '')} - {topic}")
    desired_topic = normalize_course_name(topic)

    for candidate_key in candidate_course_keys:
        course_index = data_to_index_dict.get(candidate_key)
        if course_index is None:
            continue

        course_name = normalize_course_name(course_data_dict[course_index].get("name", ""))
        if course_name == desired_name:
            return course_index
        if desired_topic and desired_topic in course_name:
            return course_index

    return None


def build_teacher_course_index(history_dir, data_to_index_dict, course_data_dict, topic_class_map):
    teacher_index = {}
    semester_dirs = sorted(
        [name for name in os.listdir(history_dir) if name.isdigit()],
        key=int,
    )

    for semester in semester_dirs:
        semester_value = int(semester)
        semester_dir = os.path.join(history_dir, semester)

        for filename in os.listdir(semester_dir):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(semester_dir, filename)
            with open(file_path, "r") as history_file:
                history_payload = json.load(history_file)

            for course in history_payload.get("courses", {}).values():
                course_index = resolve_course_index(
                    course,
                    data_to_index_dict=data_to_index_dict,
                    course_data_dict=course_data_dict,
                    topic_class_map=topic_class_map,
                )
                if course_index is None:
                    continue

                for session in course.get("sessions", {}).values():
                    for instructor in session.get("instructors") or []:
                        teacher_name = str(instructor.get("name") or "").strip()
                        normalized_name = normalize_teacher_name(teacher_name)
                        if not normalized_name or is_placeholder_teacher(normalized_name):
                            continue

                        teacher_entry = teacher_index.setdefault(
                            normalized_name,
                            {
                                "display_name_counts": Counter(),
                                "courses": {},
                            },
                        )
                        teacher_entry["display_name_counts"][teacher_name] += 1

                        course_entry = teacher_entry["courses"].setdefault(
                            course_index,
                            {
                                "strms": set(),
                                "enrollment_by_strm": {},
                                "historical_enrollment_total": 0,
                            },
                        )
                        course_entry["strms"].add(semester_value)

                        session_enrollment = get_session_latest_enrollment(session)
                        course_entry["historical_enrollment_total"] += session_enrollment
                        course_entry["enrollment_by_strm"][semester_value] = (
                            course_entry["enrollment_by_strm"].get(semester_value, 0)
                            + session_enrollment
                        )

    finalized_index = {}
    for normalized_name, teacher_entry in teacher_index.items():
        display_name = teacher_entry["display_name_counts"].most_common(1)[0][0]
        courses = {}
        for course_index, course_entry in teacher_entry["courses"].items():
            sorted_semesters = sorted(course_entry["strms"], reverse=True)
            latest_taught_strm = sorted_semesters[0]
            courses[course_index] = {
                "latest_taught_strm": latest_taught_strm,
                "semester_count": len(sorted_semesters),
                "strms": sorted_semesters,
                "historical_enrollment_total": course_entry["historical_enrollment_total"],
                "latest_enrollment_total": course_entry["enrollment_by_strm"].get(latest_taught_strm, 0),
            }

        finalized_index[normalized_name] = {
            "display_name": display_name,
            "courses": courses,
        }

    return finalized_index


def write_teacher_course_index(output_path, teacher_course_index):
    with open(output_path, "wb") as output_file:
        pickle.dump(teacher_course_index, output_file)
